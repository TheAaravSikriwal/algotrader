"""Source 01 audit -- the family of "strong technical momentum" readings.

The claim under test is verbatim:

    "Swing Trading Liquid Stocks. Buy stocks with high trading volume showing
    'strong technical momentum', aiming to sell for a small profit 'within 2
    to 5 days.'"

"Strong technical momentum" is not defined. This module therefore encodes a
*family* of reasonable readings rather than one, and every reading is measured
the same way so the distribution of outcomes is comparable.

Measurement contract (identical to core.engine.run_backtest):
    a signal computed from bar t is FILLED AT THE OPEN OF BAR t+1.
So a hold of h bars is the open-to-open return open[t+1+h] / open[t+1] - 1.
s01_panel.py cross-checks this event-study arithmetic against the actual
engine's trade log, per symbol, and the two must agree to a fraction of a bp.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402,F401

from core import indicators as IND  # noqa: E402
from core.panel import Panel  # noqa: E402

HOLDS = (1, 2, 3, 4, 5, 10)


# ---------------------------------------------------------------------------
# forward returns, strictly causal
# ---------------------------------------------------------------------------
def forward_open_returns(panel: Panel, hold: int) -> pd.DataFrame:
    """Return earned by a signal on bar t: buy open[t+1], sell open[t+1+hold].

    Indexed at t, so it lines up with a signal known at t's close. Nothing in
    the value is observable at t -- which is the point.
    """
    opens = panel.opens
    entry = opens.shift(-1)
    exit_ = opens.shift(-(1 + hold))
    return (exit_ / entry - 1.0) * 100.0          # percent of notional


# ---------------------------------------------------------------------------
# per-symbol indicator signals
# ---------------------------------------------------------------------------
def _ts_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Time-series readings of 'strong technical momentum' for one symbol.

    Every one uses only bars up to and including t.
    """
    close, vol = df["close"], df["volume"]
    out: dict[str, pd.Series] = {}

    sma20, sma50 = IND.sma(close, 20), IND.sma(close, 50)
    out["b1_ma20_up"] = (close > sma20) & (sma20 > sma20.shift(1))
    out["b1_ma20_up_INV"] = (close < sma20) & (sma20 < sma20.shift(1))
    out["b2_ma50_up"] = (close > sma50) & (sma50 > sma50.shift(1))
    out["b2_ma50_up_INV"] = (close < sma50) & (sma50 < sma50.shift(1))

    r = IND.rsi(close, 14)
    out["c1_rsi_gt60"] = r > 60
    out["c1_rsi_gt60_INV"] = r < 40          # source 11's reversal reading
    out["c2_rsi_gt70"] = r > 70
    out["c2_rsi_gt70_INV"] = r < 30          # classic "oversold -> bounce"

    line, sig, _ = IND.macd(close)
    out["d_macd_cross_up"] = (line > sig) & (line.shift(1) <= sig.shift(1))
    out["d_macd_cross_up_INV"] = (line < sig) & (line.shift(1) >= sig.shift(1))

    lo20, hi20 = IND.donchian(df, 20)
    out["e_20d_high"] = close > hi20
    out["e_20d_high_INV"] = close < lo20

    relvol = vol / IND.sma(vol, 20)
    ret1 = close.pct_change()
    out["f_relvol_up"] = (relvol > 1.5) & (ret1 > 0)
    out["f_relvol_up_INV"] = (relvol > 1.5) & (ret1 < 0)

    adx, pdi, mdi = IND.adx(df, 14)
    out["g_adx_bull"] = (adx > 25) & (pdi > mdi)
    out["g_adx_bull_INV"] = (adx > 25) & (mdi > pdi)

    return {k: v.fillna(False).astype(bool) for k, v in out.items()}


TS_NAMES = [
    "b1_ma20_up", "b1_ma20_up_INV",
    "b2_ma50_up", "b2_ma50_up_INV",
    "c1_rsi_gt60", "c1_rsi_gt60_INV",
    "c2_rsi_gt70", "c2_rsi_gt70_INV",
    "d_macd_cross_up", "d_macd_cross_up_INV",
    "e_20d_high", "e_20d_high_INV",
    "f_relvol_up", "f_relvol_up_INV",
    "g_adx_bull", "g_adx_bull_INV",
]

XS_NAMES = [
    "a1_ret5_topdec", "a1_ret5_topdec_INV",
    "a2_ret10_topdec", "a2_ret10_topdec_INV",
    "a3_ret20_topdec", "a3_ret20_topdec_INV",
]

ALL_NAMES = XS_NAMES + TS_NAMES

LABELS = {
    "a1_ret5_topdec": "(a) top-decile 5d return",
    "a2_ret10_topdec": "(a) top-decile 10d return",
    "a3_ret20_topdec": "(a) top-decile 20d return",
    "b1_ma20_up": "(b) close>SMA20 & SMA20 rising",
    "b2_ma50_up": "(b) close>SMA50 & SMA50 rising",
    "c1_rsi_gt60": "(c) RSI(14)>60",
    "c2_rsi_gt70": "(c) RSI(14)>70",
    "d_macd_cross_up": "(d) MACD bullish crossover",
    "e_20d_high": "(e) new 20-day high",
    "f_relvol_up": "(f) relvol>1.5 & up day",
    "g_adx_bull": "(g) ADX>25 & +DI>-DI",
    "a1_ret5_topdec_INV": "(a-inv) bottom-decile 5d return",
    "a2_ret10_topdec_INV": "(a-inv) bottom-decile 10d return",
    "a3_ret20_topdec_INV": "(a-inv) bottom-decile 20d return",
    "b1_ma20_up_INV": "(b-inv) close<SMA20 & SMA20 falling",
    "b2_ma50_up_INV": "(b-inv) close<SMA50 & SMA50 falling",
    "c1_rsi_gt60_INV": "(c-inv) RSI(14)<40",
    "c2_rsi_gt70_INV": "(c-inv) RSI(14)<30",
    "d_macd_cross_up_INV": "(d-inv) MACD bearish crossover",
    "e_20d_high_INV": "(e-inv) new 20-day low",
    "f_relvol_up_INV": "(f-inv) relvol>1.5 & down day",
    "g_adx_bull_INV": "(g-inv) ADX>25 & -DI>+DI",
}


def build_signals(panel: Panel, bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """All readings as symbol-by-date boolean frames on the panel's index."""
    per_symbol: dict[str, dict[str, pd.Series]] = {}
    for sym in panel.symbols:
        per_symbol[sym] = _ts_signals(bars[sym])

    sigs: dict[str, pd.DataFrame] = {}
    for name in TS_NAMES:
        frame = pd.DataFrame({s: per_symbol[s][name] for s in panel.symbols})
        sigs[name] = frame.reindex(panel.index).fillna(False).astype(bool)

    # (a) cross-sectional rank readings: top / bottom decile of that day's
    # cross-section of trailing N-day return. Uses closes up to t only.
    closes = panel.closes
    for name, n in (("a1_ret5_topdec", 5), ("a2_ret10_topdec", 10), ("a3_ret20_topdec", 20)):
        trailing = closes / closes.shift(n) - 1.0
        rank = trailing.rank(axis=1, pct=True, ascending=True)
        breadth = trailing.notna().sum(axis=1)
        ok = breadth >= 20
        sigs[name] = ((rank >= 0.9) & ok.to_numpy()[:, None]).fillna(False)
        sigs[name + "_INV"] = ((rank <= 0.1) & ok.to_numpy()[:, None]).fillna(False)

    return sigs


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def newey_west_t(series: pd.Series, lag: int) -> float:
    """t-stat of the mean with a Newey-West HAC standard error.

    A pooled per-trade t-stat over overlapping h-day holds across 60-odd
    correlated large caps is wrong by a large factor: the observations are
    neither independent across names (one market factor) nor across days (the
    windows overlap). Collapsing to one equal-weighted number per entry date
    kills the cross-sectional correlation; a Newey-West correction with lag h
    handles what overlap is left. This is the t-stat worth reading.
    """
    x = pd.Series(series, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 10:
        return 0.0
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        g = float(e[L:] @ e[:-L]) / n
        var += 2.0 * (1.0 - L / (lag + 1.0)) * g
    if var <= 0:
        return 0.0
    return float(mu / np.sqrt(var / n))


def daily_mean(values: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean return of the names signalling on each date."""
    v = values.where(mask)
    return v.mean(axis=1, skipna=True)


def _t(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def cell_stats(fwd: pd.DataFrame, mask: pd.DataFrame, hold: int,
               name: str = "", cost_bps: float = 0.0,
               base: dict | None = None) -> dict:
    """Everything reported for one (entry, hold) cell.

    `fwd` is percent-of-notional open-to-open return; `mask` is the boolean
    entry signal; both indexed by date x symbol.
    """
    F = fwd.to_numpy(dtype=float)
    M = mask.reindex(index=fwd.index, columns=fwd.columns).fillna(False).to_numpy(dtype=bool)
    valid = np.isfinite(F)
    m = M & valid
    sel = F[m]

    n = int(sel.size)
    if n < 30:
        return {"entry": name, "hold": hold, "n": n}

    comp = F[valid & ~M]
    if base is None:
        base = {"mu": float(F[valid].mean()),
                "daily": pd.Series(np.where(valid, F, np.nan),
                                   index=fwd.index).pipe(lambda _: None)}
    base_mu = base["mu"]

    net = sel - cost_bps / 100.0

    # date-clustered series: one equal-weighted number per entry date
    num = np.where(m, np.nan_to_num(F), 0.0).sum(axis=1)
    cnt = m.sum(axis=1)
    d_sig = pd.Series(np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan),
                      index=fwd.index).dropna()
    d_all = base["daily"].reindex(d_sig.index)
    d_exc = (d_sig - d_all).dropna()

    # Welch two-sample t on signal vs non-signal (naive: ignores dependence)
    m1, m2 = sel.mean(), comp.mean()
    se = np.sqrt(sel.var(ddof=1) / sel.size + comp.var(ddof=1) / comp.size)
    t_welch = float((m1 - m2) / se) if se > 0 else 0.0

    t_pool = _t(sel)
    t_pool_net = _t(net)
    t_dc = newey_west_t(d_sig, hold)
    t_dc_net = newey_west_t(d_sig - cost_bps / 100.0, hold)
    t_exc_dc = newey_west_t(d_exc, hold)
    sel = pd.Series(sel)

    return {
        "entry": name,
        "label": LABELS.get(name, name),
        "hold": hold,
        "n": n,
        "n_days": int(len(d_sig)),
        "mean_pct": float(sel.mean()),
        "median_pct": float(sel.median()),
        "win_rate": float((sel > 0).mean()),
        "std_pct": float(sel.std(ddof=1)),
        "t_pooled": t_pool,
        "p_pooled": A.two_sided_p(t_pool),
        "t_dateclust": t_dc,
        "p_dateclust": A.two_sided_p(t_dc),
        "base_rate_pct": base_mu,
        "excess_pct": float(sel.mean()) - base_mu,
        "t_excess_welch": t_welch,
        "p_excess_welch": A.two_sided_p(t_welch),
        "t_excess_dateclust": t_exc_dc,
        "p_excess_dateclust": A.two_sided_p(t_exc_dc),
        f"mean_net{int(cost_bps)}bps": float(net.mean()),
        f"t_net{int(cost_bps)}bps_pooled": t_pool_net,
        f"t_net{int(cost_bps)}bps_dateclust": t_dc_net,
        "breakeven_bps": A.breakeven_bps(sel),
        "signals_per_day": float(cnt.mean()),
    }


def base_for(fwd: pd.DataFrame) -> dict:
    """The unconditional benchmark for one hold: mean over every (symbol, day).

    This is the control that matters. A momentum entry returning +0.20% over
    3 days says nothing if a coin flip over the same universe and period
    returns +0.18%.
    """
    F = fwd.to_numpy(dtype=float)
    valid = np.isfinite(F)
    allobs = F[valid]
    num = np.where(valid, np.nan_to_num(F), 0.0).sum(axis=1)
    cnt = valid.sum(axis=1)
    daily = pd.Series(np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan),
                      index=fwd.index)
    return {"mu": float(allobs.mean()), "daily": daily,
            "n": int(allobs.size), "median": float(np.median(allobs)),
            "win_rate": float((allobs > 0).mean()),
            "std": float(allobs.std(ddof=1))}
