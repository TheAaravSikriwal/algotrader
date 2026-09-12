"""Pull the intraday bars the audit needs and cache them under research/audit/cache.

Alpaca minute/5-minute history goes back years (verified to 2016); yfinance
intraday does not (1m ~30d, 5m ~60d), so everything here is Alpaca.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN"]
START = "2021-01-01"
END = "2026-09-01"


def main(timeframe: str = "5Min", symbols=None, start=START, end=END):
    symbols = symbols or SYMBOLS
    out = {}
    for sym in symbols:
        t0 = time.time()
        try:
            df = A.bars_chunked(sym, start, end, timeframe=timeframe, months=12)
        except Exception as exc:  # noqa: BLE001
            print(f"{sym}: FAILED {type(exc).__name__}: {exc}")
            continue
        r = A.rth(df)
        days = len(set(pd.DatetimeIndex(r.index).normalize()))
        print(f"{sym}: {len(df)} bars total, {len(r)} RTH, {days} sessions, "
              f"{r.index.min()} -> {r.index.max()}  ({time.time()-t0:.1f}s)")
        out[sym] = r
    return out


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "5Min"
    syms = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    main(tf, syms)
