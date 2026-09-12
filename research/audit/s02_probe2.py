"""Probe 2: is the quotes 504 an outage, an entitlement problem, or a window problem?
Keys are sent in HEADERS only and never printed."""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

H = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
     "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
BASE = "https://data.alpaca.markets"


def hit(path, params, label, timeout=60):
    t0 = time.time()
    try:
        r = requests.get(BASE + path, headers=H, params=params, timeout=timeout)
        body = r.text[:200]
        print(f"{label}: HTTP {r.status_code} in {time.time()-t0:.1f}s  {body}")
        return r
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: EXC {type(exc).__name__} {str(exc)[:120]}")
        return None


s = pd.Timestamp("2026-09-04 11:00:00", tz="America/New_York")
e = s + pd.Timedelta(seconds=10)

hit("/v2/stocks/SPY/quotes/latest", {}, "latest quote SPY")
hit("/v2/stocks/bars", {"symbols": "SPY", "start": "2026-09-04T14:00:00Z",
                        "end": "2026-09-04T15:00:00Z", "timeframe": "1Min",
                        "feed": "sip", "limit": 10}, "bars SPY sip")
hit("/v2/stocks/trades", {"symbols": "SPY", "start": s.isoformat(),
                          "end": e.isoformat(), "limit": 20}, "trades SPY 10s")
hit("/v2/stocks/quotes", {"symbols": "SPY", "start": s.isoformat(),
                          "end": e.isoformat(), "limit": 20}, "quotes SPY 10s")
hit("/v2/stocks/quotes", {"symbols": "SPY", "start": s.isoformat(),
                          "end": e.isoformat(), "limit": 20, "feed": "sip"},
    "quotes SPY 10s sip")
hit("/v2/stocks/quotes", {"symbols": "SPY", "start": s.isoformat(),
                          "end": e.isoformat(), "limit": 20, "feed": "iex"},
    "quotes SPY 10s iex")
hit("/v2/stocks/SPY/quotes", {"start": s.isoformat(), "end": e.isoformat(),
                              "limit": 20}, "quotes SPY (v2 single-symbol path)")
hit("/v2/stocks/bars", {"symbols": "SPY", "start": "2026-09-04T14:00:00Z",
                        "end": "2026-09-04T14:05:00Z", "timeframe": "1Min",
                        "limit": 5}, "bars again (control)")
