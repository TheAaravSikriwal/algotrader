"""Source 05 audit -- connectivity probe. Read-only. Writes nothing."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

KEY = os.getenv("ALPACA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY")
print("keys present:", bool(KEY), bool(SEC))

# 1. assets API
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    tc = TradingClient(KEY, SEC, paper=True)
    assets = tc.get_all_assets(GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
    print("assets returned:", len(assets))
    a = assets[0]
    print("sample:", a.symbol, a.exchange, a.tradable, a.shortable, a.fractionable)
    ex = pd.Series([str(x.exchange) for x in assets]).value_counts()
    print(ex.head(10))
    tradable = [x for x in assets if x.tradable]
    print("tradable:", len(tradable))
except Exception as exc:  # noqa: BLE001
    print("ASSETS FAILED:", type(exc).__name__, exc)

# 2. multi-symbol daily bars
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    dc = StockHistoricalDataClient(KEY, SEC)
    r = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=["AAPL", "F", "PLUG", "SNDL"],
        timeframe=TimeFrame.Day,
        start=pd.Timestamp("2024-01-02").to_pydatetime(),
        end=pd.Timestamp("2024-01-10").to_pydatetime(),
    ))
    df = r.df
    print("multi bars shape:", df.shape)
    print(df.head())
except Exception as exc:  # noqa: BLE001
    print("BARS FAILED:", type(exc).__name__, exc)

# 3. quotes
try:
    from alpaca.data.requests import StockQuotesRequest

    q = dc.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols="PLUG",
        start=pd.Timestamp("2024-01-03 14:30:00").to_pydatetime(),
        end=pd.Timestamp("2024-01-03 14:32:00").to_pydatetime(),
        limit=50,
    ))
    qd = q.df
    print("quotes shape:", qd.shape)
    print(qd.head())
except Exception as exc:  # noqa: BLE001
    print("QUOTES FAILED:", type(exc).__name__, exc)

# 4. how far back does minute data go
try:
    b = A.bars("AAPL", "2016-01-04", "2016-01-06", "1Min", source="alpaca")
    print("2016 minute bars:", b.shape, b.index.min(), b.index.max())
except Exception as exc:  # noqa: BLE001
    print("2016 MIN FAILED:", type(exc).__name__, exc)
