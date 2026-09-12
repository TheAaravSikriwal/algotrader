"""Verify the clock and the completeness of the intraday feed BEFORE trusting any
intraday backtest built on it. Run this first; every s09_/s10_ result depends on it.

Three things are checked:

  1. CLOCK. core/data.py:_normalise calls tz_convert(None), which converts to UTC
     and then drops the tzinfo. So the naive index returned for Alpaca (and for
     yfinance INTRADAY) is UTC, not Eastern. A `between_time("09:30","16:00")`
     on that index selects 05:30-12:00 ET -- mostly pre-market. auditlib.to_eastern
     re-localises to UTC and converts to America/New_York, which is correct across
     DST because it uses the tz database rather than a constant offset.

  2. FEED. Alpaca's free tier can serve IEX-only data, which is ~2-3% of consolidated
     volume and would make every volume-conditioned rule meaningless. Summing RTH
     minute volume and comparing it to the daily bar distinguishes SIP from IEX.

  3. SESSION SHAPE. 78 five-minute or 390 one-minute bars per regular session, the
     first bar's open equal to the daily open, the last bar's close near the daily
     close.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402


def check_clock(symbol="AAPL", start="2026-09-01", end="2026-09-10"):
    print("=" * 72)
    print("1. CLOCK")
    print("=" * 72)
    raw = A.bars(symbol, start, end, "1Min", eastern=False)   # naive, as core.data gives it
    east = A.to_eastern(raw)

    day = pd.Timestamp("2026-09-08")
    raw_d = raw[pd.DatetimeIndex(raw.index).normalize() == day]
    east_d = east[pd.DatetimeIndex(east.index).normalize() == day]

    print(f"\n{symbol} {day.date()} volume by hour on the RAW (naive) index:")
    print(raw_d.groupby(raw_d.index.hour)["volume"].sum().to_string())
    print(f"\nsame day, volume by hour AFTER to_eastern():")
    print(east_d.groupby(east_d.index.hour)["volume"].sum().to_string())

    # the naive index must be UTC: Sept is EDT (UTC-4), so 09:30 ET == 13:30 UTC
    raw_peak = raw_d.groupby(raw_d.index.hour)["volume"].sum().idxmax()
    east_peak = east_d.groupby(east_d.index.hour)["volume"].sum().idxmax()
    print(f"\nbusiest hour raw={raw_peak}:00   after to_eastern={east_peak}:00")
    print("EDT = UTC-4, so a raw busiest hour of 19-20 and an Eastern one of 15-16 "
          "(the closing hour) both confirm the naive index is UTC.")

    # the trap, demonstrated
    naive_filter = raw_d.between_time("09:30", "16:00")
    print(f"\nTHE TRAP: between_time('09:30','16:00') on the RAW index captures "
          f"{naive_filter['volume'].sum() / raw_d['volume'].sum():.1%} of the day's volume")
    correct = A.rth(east_d)
    print(f"CORRECT:  A.rth(A.to_eastern(df))          captures "
          f"{correct['volume'].sum() / east_d['volume'].sum():.1%} of the day's volume")

    # DST: a constant offset would break across the boundary
    print("\nDST check -- first RTH bar of the session either side of the Nov 2025 boundary:")
    for s, e in [("2025-10-27", "2025-10-31"), ("2025-11-05", "2025-11-08")]:
        d2 = A.rth(A.bars(symbol, s, e, "5Min"))
        for day2 in sorted(set(pd.DatetimeIndex(d2.index).normalize()))[:1]:
            sl = d2[pd.DatetimeIndex(d2.index).normalize() == day2]
            print(f"  {day2.date()}: first={sl.index[0].time()} last={sl.index[-1].time()} "
                  f"bars={len(sl)}")
    print("  Both must read 09:30:00 / 15:55:00. A constant -4h shift would not.")


def check_feed(symbols=("AAPL", "SPY"), start="2024-03-01", end="2024-03-15"):
    print()
    print("=" * 72)
    print("2. FEED COMPLETENESS  (SIP vs IEX)")
    print("=" * 72)
    rows = []
    for sym in symbols:
        minute = A.rth(A.bars(sym, start, end, "1Min"))
        daily = A.bars(sym, start, end, "1Day", eastern=False)
        daily.index = pd.DatetimeIndex(daily.index).normalize()
        by_day = minute.groupby(pd.DatetimeIndex(minute.index).normalize())["volume"].sum()
        for d, v in by_day.items():
            if d in daily.index:
                dv = float(daily.loc[d, "volume"])
                rows.append({"symbol": sym, "date": d.date(),
                             "minute_rth_volume": v, "daily_volume": dv,
                             "ratio": v / dv if dv else float("nan")})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nmedian ratio (RTH minute volume / daily bar volume) = {df['ratio'].median():.3f}")
    print("IEX-only would be ~0.02-0.03. A ratio near 0.90-1.00 means consolidated (SIP).")
    print("It sits slightly below 1.00 because the daily bar includes extended hours.")
    return df


def check_session_shape(symbol="SPY", start="2024-01-02", end="2024-01-20"):
    print()
    print("=" * 72)
    print("3. SESSION SHAPE")
    print("=" * 72)
    for tf, expect in [("5Min", 78), ("1Min", 390)]:
        r = A.rth(A.bars(symbol, start, end, tf))
        counts = r.groupby(pd.DatetimeIndex(r.index).normalize()).size()
        print(f"{tf}: bars per session -> {sorted(set(counts))} (expected {expect}); "
              f"{(counts == expect).mean():.0%} of sessions exact")
    daily = A.bars(symbol, start, end, "1Day", eastern=False)
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    r = A.rth(A.bars(symbol, start, end, "5Min"))
    g = r.groupby(pd.DatetimeIndex(r.index).normalize())
    comp = pd.DataFrame({"intraday_open": g["open"].first(), "intraday_close": g["close"].last(),
                         "intraday_high": g["high"].max(), "intraday_low": g["low"].min()})
    comp = comp.join(daily[["open", "close", "high", "low"]], rsuffix="_daily")
    comp["open_diff_bps"] = (comp["intraday_open"] / comp["open"] - 1) * 1e4
    comp["close_diff_bps"] = (comp["intraday_close"] / comp["close"] - 1) * 1e4
    print()
    print(comp[["intraday_open", "open", "open_diff_bps",
                "intraday_close", "close", "close_diff_bps"]].to_string())
    print(f"\nmedian |open diff| = {comp['open_diff_bps'].abs().median():.2f} bps "
          f"-- the RTH 09:30 bar open must equal the daily open")
    return comp


if __name__ == "__main__":
    check_clock()
    feed = check_feed()
    shape = check_session_shape()
    A.save(feed, "verify_feed_completeness.csv")
