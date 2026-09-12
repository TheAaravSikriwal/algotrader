"""Source 05 audit -- STEP 2a: fetch 1-minute bars for every selected gap event.

One Alpaca request per trading date covering that date's <=5 selected symbols,
04:00-16:00 ET (pre-market included so the pre-market-anchored VWAP variant and
the 9MA warm-up are both computable). Cached one parquet per date.

Also fetches NATIVE 5Min bars for a random audit sample so the 1Min->5Min
resample used by the backtest can be verified against Alpaca's own aggregation.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

KEY = os.getenv("ALPACA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY")
OUT = A.CACHE / "s05_min"
OUT.mkdir(exist_ok=True)
OUT5 = A.CACHE / "s05_min5_check"
OUT5.mkdir(exist_ok=True)

COLS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(KEY, SEC)


def fetch_day(dc, date: str, symbols: list[str], outdir, tf_minutes: int):
    path = outdir / f"{date}.parquet"
    if path.exists():
        return True
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame.Minute if tf_minutes == 1 else TimeFrame(tf_minutes, TimeFrameUnit.Minute)
    # 04:00 ET .. 20:00 ET expressed in UTC-naive terms via tz-aware timestamps
    s = pd.Timestamp(f"{date} 04:00", tz="America/New_York")
    e = pd.Timestamp(f"{date} 16:05", tz="America/New_York")
    for attempt in range(4):
        try:
            r = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbols, timeframe=tf,
                start=s.to_pydatetime(), end=e.to_pydatetime()))
            df = r.df
            break
        except Exception as exc:  # noqa: BLE001
            print(f"    retry {attempt} {date} {type(exc).__name__}: {str(exc)[:110]}",
                  flush=True)
            time.sleep(2 * (attempt + 1))
    else:
        return False
    if df is None or len(df) == 0:
        pd.DataFrame(columns=COLS).to_parquet(path, index=False)
        return False
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df = df[COLS]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df.to_parquet(path, index=False)
    return False


def main():
    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    by_date = ev.groupby("date")["symbol"].apply(lambda s: sorted(set(s))).to_dict()
    dates = sorted(by_date)
    print(f"{len(dates)} dates, {len(ev)} events", flush=True)
    dc = _client()
    t0 = time.time()
    for i, d in enumerate(dates):
        cached = fetch_day(dc, d, by_date[d], OUT, 1)
        if not cached and (i % 25 == 0):
            print(f"  {i + 1}/{len(dates)} {d} ({time.time() - t0:.0f}s)", flush=True)

    # native 5Min audit sample: every 40th date
    sample = dates[::40]
    print(f"native-5Min audit sample: {len(sample)} dates", flush=True)
    for d in sample:
        fetch_day(dc, d, by_date[d], OUT5, 5)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
