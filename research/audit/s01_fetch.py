"""Source 01 audit -- data fetch.

Pulls daily bars for core.universe.CANDIDATE_POOL from 2011-01-01 so that the
tradability screen has real pre-start history to measure `as_of` 2014-12-31,
and the backtest window itself starts 2015-01-01.

Caches into research/audit/cache/ under an s01_ prefix. Nothing is written to
data_cache/ and no repo file is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

from core import data as _data  # noqa: E402
from core.universe import CANDIDATE_POOL  # noqa: E402

START = "2011-01-01"
END = "2026-09-11"
CACHE = A.CACHE


def cache_path(symbol: str) -> Path:
    return CACHE / f"s01_{symbol.upper()}_1Day_{START}_{END}.parquet"


def fetch_one(symbol: str) -> pd.DataFrame | None:
    path = cache_path(symbol)
    if path.exists():
        return pd.read_parquet(path)
    try:
        df = _data.load_bars(symbol, START, END, timeframe="1Day",
                             source="yfinance", use_cache=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  {symbol}: {type(exc).__name__}: {exc}")
        return None
    if df is None or df.empty:
        return None
    df.to_parquet(path)
    return df


def load_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(CANDIDATE_POOL):
        df = fetch_one(sym)
        if df is not None and len(df) > 400:
            out[sym.upper()] = df
        if verbose and (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(CANDIDATE_POOL)} ({len(out)} usable)")
    return out


if __name__ == "__main__":
    bars = load_all()
    print(f"fetched {len(bars)} / {len(CANDIDATE_POOL)} symbols")
    spans = pd.Series({s: (d.index[0], d.index[-1], len(d)) for s, d in bars.items()})
    lens = pd.Series({s: len(d) for s, d in bars.items()})
    print("bar counts: min", lens.min(), "median", lens.median(), "max", lens.max())
    first = pd.Series({s: d.index[0] for s, d in bars.items()})
    print("latest first-bar:", first.max(), "for", first.idxmax())
