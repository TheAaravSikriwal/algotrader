"""Connectivity / shape probe for the s02 leg. Reads only."""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

KEY = os.environ["ALPACA_API_KEY_ID"]
SEC = os.environ["ALPACA_API_SECRET_KEY"]
print("keys present:", bool(KEY), bool(SEC))

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockQuotesRequest  # noqa: E402
from alpaca.data.enums import DataFeed  # noqa: E402

dc = StockHistoricalDataClient(KEY, SEC)


def attempt(label, **kw):
    for i in range(3):
        try:
            t0 = time.time()
            r = dc.get_stock_quotes(StockQuotesRequest(**kw))
            df = r.df
            print(f"OK  {label}: rows={len(df)} in {time.time()-t0:.1f}s")
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"ERR {label} try{i}: {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(2)
    return None


for date, tm, secs, lim, feed in [
    ("2026-09-04", "11:00:00", 30, 500, None),
    ("2026-09-04", "11:00:00", 60, None, None),
    ("2022-06-13", "11:00:00", 60, 500, None),
    ("2022-06-13", "09:31:00", 60, 500, DataFeed.SIP),
    ("2021-03-10", "11:00:00", 60, 500, None),
]:
    s = pd.Timestamp(f"{date} {tm}", tz="America/New_York")
    kw = dict(symbol_or_symbols="SPY", start=s.to_pydatetime(),
              end=(s + pd.Timedelta(seconds=secs)).to_pydatetime())
    if lim:
        kw["limit"] = lim
    if feed:
        kw["feed"] = feed
    df = attempt(f"{date} {tm} {secs}s lim={lim} feed={feed}", **kw)
    if df is not None and len(df):
        print("   cols:", list(df.columns))
        print(df.head(2).to_string())
        break
