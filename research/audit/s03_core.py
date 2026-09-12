"""Source 03 (Fed quadrant / 'Fed chessboard') audit -- shared machinery.

Builds a strictly causal monthly quadrant label out of FRED WALCL + FEDFUNDS,
aligns it to month-end ETF proxy prices, and provides Newey-West and
non-overlapping inference helpers.  No scipy / statsmodels in this venv, so the
HAC variance is implemented directly.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
from core.journal import two_sided_p  # noqa: E402

START = "2002-01-01"
END = "2026-09-11"

# archetype -> (primary proxy, alternates)
ARCHETYPES = {
    "D": ["IWO", "XBI", "ARKK"],   # unprofitable high-growth / speculative
    "C": ["QQQ", "IWF"],           # growth
    "B": ["SPY"],                  # quality / blend
    "A": ["IWD", "VTV"],           # low-multiple value, low leverage
}
PRIMARY = {"D": "IWO", "C": "QQQ", "B": "SPY", "A": "IWD"}
ALL_PROXIES = ["IWO", "XBI", "ARKK", "QQQ", "IWF", "SPY", "IWD", "VTV", "SPLV", "USMV"]

# source's claim: quadrant -> preferred archetype
PREFERRED = {"Q1": "D", "Q2": "B", "Q3": "C", "Q4": "A"}
# source's stated max share of cash deployed
MAX_WEIGHT = {"Q1": 1.00, "Q2": 0.50, "Q3": 0.50, "Q4": 0.20}

QUAD_LABEL = {
    "Q1": "rates falling & QE",
    "Q2": "rates rising & QE",
    "Q3": "rates falling & QT",
    "Q4": "rates rising & QT",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_macro() -> pd.DataFrame:
    df = pd.read_csv(A.RESULTS / "s03_macro.csv", index_col=0, parse_dates=True)
    return df


def load_prices() -> pd.DataFrame:
    cols = {}
    for sym in ALL_PROXIES:
        try:
            b = A.bars(sym, START, END, "1Day", source="yfinance", eastern=False)
            cols[sym] = b["close"]
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {sym}: {exc}")
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.DatetimeIndex(px.index).normalize()
    return px


def month_end_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Last traded close in each calendar month, indexed by that trade date."""
    g = px.groupby([px.index.year, px.index.month])
    idx = [grp.index.max() for _, grp in g]
    return px.loc[sorted(idx)]


# ---------------------------------------------------------------------------
# quadrant construction (strictly causal)
# ---------------------------------------------------------------------------
def build_quadrants(macro: pd.DataFrame,
                    obs_dates: pd.DatetimeIndex,
                    walcl_lag_days: int = 7,
                    ff_lag_months: int = 1,
                    rate_lookback_m: int = 3,
                    bs_lookback_w: int = 13,
                    rate_band: float = 0.10,
                    flat_is_state: bool = False,
                    signal_shift_months: int = 0) -> pd.DataFrame:
    """One row per observation date with the quadrant visible AT that date.

    walcl_lag_days : only WALCL observations dated <= t - this many days are used
                     (WALCL for Wednesday W is released Thursday W+1; 7d is
                     conservative).
    ff_lag_months  : FEDFUNDS for month M is published early in month M+1, so at
                     month-end t only months <= t - ff_lag_months are used.
    signal_shift_months : >0 shifts the signal FORWARD in time (deliberate
                     LOOKAHEAD, used only for the self-test).
    """
    walcl = macro["WALCL"].dropna()
    ff = macro["FEDFUNDS"].dropna()

    rows = []
    prev_rate = None
    for t in obs_dates:
        if signal_shift_months:
            asof = t + pd.DateOffset(months=signal_shift_months)
        else:
            asof = t

        # ---- balance sheet ------------------------------------------------
        cut = asof - pd.Timedelta(days=walcl_lag_days)
        hist = walcl[walcl.index <= cut]
        if len(hist) <= bs_lookback_w:
            rows.append({"date": t, "rate_dir": None, "bs_dir": None, "quad": None,
                         "walcl": np.nan, "walcl_chg": np.nan,
                         "ff": np.nan, "ff_chg": np.nan})
            continue
        w_now = float(hist.iloc[-1])
        w_then = float(hist.iloc[-1 - bs_lookback_w])
        bs_chg = w_now - w_then
        bs_dir = "QE" if bs_chg > 0 else "QT"

        # ---- policy rate --------------------------------------------------
        ff_cut = (asof - pd.DateOffset(months=ff_lag_months)).replace(day=1)
        fh = ff[ff.index <= ff_cut]
        if len(fh) <= rate_lookback_m:
            rows.append({"date": t, "rate_dir": None, "bs_dir": bs_dir, "quad": None,
                         "walcl": w_now, "walcl_chg": bs_chg,
                         "ff": np.nan, "ff_chg": np.nan})
            continue
        f_now = float(fh.iloc[-1])
        f_then = float(fh.iloc[-1 - rate_lookback_m])
        ff_chg = f_now - f_then
        if ff_chg > rate_band:
            rate_dir = "rising"
        elif ff_chg < -rate_band:
            rate_dir = "falling"
        else:
            rate_dir = "flat" if flat_is_state else prev_rate
        if rate_dir in ("rising", "falling"):
            prev_rate = rate_dir

        if rate_dir is None:
            quad = None
        elif rate_dir == "flat":
            quad = "F_QE" if bs_dir == "QE" else "F_QT"
        elif rate_dir == "falling" and bs_dir == "QE":
            quad = "Q1"
        elif rate_dir == "rising" and bs_dir == "QE":
            quad = "Q2"
        elif rate_dir == "falling" and bs_dir == "QT":
            quad = "Q3"
        else:
            quad = "Q4"

        rows.append({"date": t, "rate_dir": rate_dir, "bs_dir": bs_dir, "quad": quad,
                     "walcl": w_now, "walcl_chg": bs_chg,
                     "ff": f_now, "ff_chg": ff_chg})

    out = pd.DataFrame(rows).set_index("date")
    return out


def forward_returns(mep: pd.DataFrame, horizon_m: int) -> pd.DataFrame:
    """Simple forward return over `horizon_m` month-end steps."""
    return mep.shift(-horizon_m) / mep - 1.0


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def newey_west_mean_t(x: np.ndarray, lag: int) -> tuple[float, float, float]:
    """t-stat for mean(x) = 0 with a Newey-West (Bartlett) HAC variance.

    Returns (mean, se, t).
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else np.nan, np.nan, np.nan)
    mu = x.mean()
    e = x - mu
    lag = int(min(lag, n - 1))
    gamma0 = float(e @ e) / n
    s = gamma0
    for L in range(1, lag + 1):
        g = float(e[L:] @ e[:-L]) / n
        s += 2.0 * (1.0 - L / (lag + 1.0)) * g
    if s <= 0:
        s = gamma0  # HAC can go negative in tiny samples; fall back
    se = np.sqrt(s / n)
    return float(mu), float(se), float(mu / se) if se > 0 else np.nan


def effective_n(x: np.ndarray, lag: int) -> float:
    """n / variance-inflation factor -- roughly how many independent draws."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return float(n)
    mu, se, _ = newey_west_mean_t(x, lag)
    naive_se = x.std(ddof=1) / np.sqrt(n)
    if not np.isfinite(se) or se <= 0 or naive_se <= 0:
        return float(n)
    vif = (se / naive_se) ** 2
    return float(n / max(vif, 1e-9))


def naive_t(x) -> float:
    return A.tstat(x)


def episodes(quad: pd.Series) -> pd.DataFrame:
    """Contiguous runs of the same quadrant label."""
    q = quad.dropna()
    if q.empty:
        return pd.DataFrame(columns=["quad", "start", "end", "months"])
    grp = (q != q.shift()).cumsum()
    rows = []
    for _, seg in q.groupby(grp):
        rows.append({"quad": seg.iloc[0], "start": seg.index[0], "end": seg.index[-1],
                     "months": len(seg)})
    return pd.DataFrame(rows)


def stats_row(x, horizon_m: int, label: str = "") -> dict:
    x = pd.Series(x).dropna()
    n = len(x)
    if n == 0:
        return {"label": label, "n": 0}
    tn = naive_t(x)
    mu, se, tnw = newey_west_mean_t(x.values, lag=max(horizon_m - 1, 1))
    eff = effective_n(x.values, lag=max(horizon_m - 1, 1))
    return {
        "label": label,
        "n": n,
        "mean_pct": 100.0 * float(x.mean()),
        "median_pct": 100.0 * float(x.median()),
        "t_naive": tn,
        "p_naive": two_sided_p(tn),
        "t_nw": tnw,
        "p_nw": two_sided_p(tnw) if np.isfinite(tnw) else np.nan,
        "n_eff": eff,
        "hit_rate": float((x > 0).mean()),
    }
