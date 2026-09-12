"""Shared helpers for the independent audit of research/sources/*.

Nothing in here modifies repo files. It wraps core.data with intraday-specific
handling the existing loader deliberately leaves alone:

  * Alpaca returns UTC timestamps; core.data._normalise converts to UTC then
    drops the tzinfo. So the naive index IS UTC. `to_eastern` re-attaches UTC
    and converts to America/New_York, which is what every rule in sources 05,
    09 and 10 is implicitly written in.
  * Alpaca intraday bars include pre- and post-market. `rth` filters to
    09:30-15:59 inclusive of the 09:30 bar, exclusive of any 16:00 stub.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.env import load_env  # noqa: E402

load_env()

from core import data  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent
CACHE = AUDIT_DIR / "cache"
CACHE.mkdir(exist_ok=True)
RESULTS = AUDIT_DIR / "results"
RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# time handling
# --------------------------------------------------------------------------
def to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Naive-UTC index -> naive US/Eastern index."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert("America/New_York").tz_localize(None)
    out.index.name = "timestamp"
    return out


def rth(df: pd.DataFrame) -> pd.DataFrame:
    """Regular trading hours only: bars stamped 09:30 .. 15:59 Eastern."""
    t = df.index.time
    import datetime as _dt
    mask = (t >= _dt.time(9, 30)) & (t < _dt.time(16, 0))
    return df[mask]


def premarket(df: pd.DataFrame) -> pd.DataFrame:
    import datetime as _dt
    t = df.index.time
    return df[(t >= _dt.time(4, 0)) & (t < _dt.time(9, 30))]


def sessions(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(set(pd.DatetimeIndex(df.index).normalize()))


# --------------------------------------------------------------------------
# data loading (cached under research/audit/cache, never touching data_cache/)
# --------------------------------------------------------------------------
def bars(symbol: str, start: str, end: str, timeframe: str = "5Min",
         source: str = "alpaca", eastern: bool = True) -> pd.DataFrame:
    tag = f"{source}_{symbol.upper()}_{timeframe}_{start}_{end}.parquet"
    path = CACHE / tag
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = data.load_bars(symbol, start, end, timeframe=timeframe,
                            source=source, use_cache=False)
        df.to_parquet(path)
    return to_eastern(df) if eastern else df


def bars_chunked(symbol: str, start: str, end: str, timeframe: str = "5Min",
                 source: str = "alpaca", months: int = 12) -> pd.DataFrame:
    """Fetch a long span in chunks so each chunk caches independently."""
    edges = pd.date_range(start=start, end=end, freq=pd.DateOffset(months=months))
    edges = list(edges) + [pd.Timestamp(end)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if a >= b:
            continue
        try:
            out.append(bars(symbol, str(a.date()), str(b.date()), timeframe, source))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{symbol} {a.date()}..{b.date()}] {type(exc).__name__}: {exc}")
    if not out:
        raise RuntimeError(f"no data for {symbol}")
    df = pd.concat(out)
    return df[~df.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def tstat(x) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def two_sided_p(t: float) -> float:
    from core.journal import two_sided_p as _p
    return _p(t)


def describe_trades(pnl_pct, label: str = "", extra: dict | None = None) -> dict:
    """pnl_pct: per-trade return in PERCENT of capital at risk (or of notional)."""
    s = pd.Series(pnl_pct, dtype=float).dropna()
    n = len(s)
    if n == 0:
        return {"label": label, "n": 0}
    t = tstat(s)
    d = {
        "label": label,
        "n": n,
        "mean_pct": float(s.mean()),
        "median_pct": float(s.median()),
        "std_pct": float(s.std(ddof=1)),
        "win_rate": float((s > 0).mean()),
        "total_pct": float(s.sum()),
        "t": t,
        "p": two_sided_p(t),
        "best": float(s.max()),
        "worst": float(s.min()),
    }
    if extra:
        d.update(extra)
    return d


def cost_sweep(pnl_pct, round_trip_bps_list=(0, 2, 5, 10, 20, 40)) -> pd.DataFrame:
    """pnl_pct must be in percent of NOTIONAL, so bps map directly."""
    s = pd.Series(pnl_pct, dtype=float).dropna()
    rows = []
    for bps in round_trip_bps_list:
        adj = s - bps / 100.0
        rows.append({"round_trip_bps": bps, "n": len(adj),
                     "mean_pct": adj.mean(), "total_pct": adj.sum(),
                     "win_rate": (adj > 0).mean(), "t": tstat(adj),
                     "p": two_sided_p(tstat(adj))})
    return pd.DataFrame(rows)


def breakeven_bps(pnl_pct) -> float:
    """Round-trip cost in bps of notional that zeroes the average trade."""
    s = pd.Series(pnl_pct, dtype=float).dropna()
    if s.empty:
        return float("nan")
    return float(s.mean() * 100.0)


def save(obj, name: str):
    path = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        pd.DataFrame(obj).to_csv(path, index=False)
    return path
