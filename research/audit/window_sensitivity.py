"""Where in the session does this rule's edge actually live?

The audit measured the effect over the whole session and then recommended
restricting to 10:30-15:30 on cost grounds, because SPY's spread is 3.78 bps
in the first fifteen minutes against 1.59 mid-day. Those are two separate
measurements, and nothing checked that the edge survives the restriction.

If the gross edge is concentrated in the opening hour, the recommendation is
self-defeating: the window that makes the cost affordable also removes the
thing being paid for.
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from core.env import load_env  # noqa: E402

load_env()

import pandas as pd  # noqa: E402

from core.data import load_bars, session  # noqa: E402
from core.daytrade import DayTradeConfig, backtest, summarise  # noqa: E402
from core.marketclock import MarketCalendar  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "GOOGL"]
OUT = Path(__file__).resolve().parent / "results"

WINDOWS = [
    ("full session   09:30-15:55", time(9, 30), time(15, 55)),
    ("open only      09:30-10:30", time(9, 30), time(10, 30)),
    ("first 15 min   09:30-09:45", time(9, 30), time(9, 45)),
    ("recommended    10:30-15:30", time(10, 30), time(15, 30)),
    ("midday         11:00-14:00", time(11, 0), time(14, 0)),
    ("close          15:00-15:55", time(15, 0), time(15, 55)),
]

# Measured effective spread by window, from the audit (bps, round trip).
SPREAD = {"09:30-09:45": 3.78, "09:30-10:30": 2.34, "10:30-15:30": 1.59,
          "11:00-14:00": 1.59, "15:00-15:55": 1.79, "09:30-15:55": 2.10}


def main():
    cal = MarketCalendar.load()
    bars = {}
    for sym in SYMBOLS:
        frames = []
        for year in range(2021, 2027):
            lo = pd.Timestamp(f"{year}-01-01")
            hi = min(pd.Timestamp(f"{year + 1}-01-01"), pd.Timestamp("2026-09-01"))
            if lo >= hi:
                continue
            try:
                frames.append(load_bars(sym, lo.date(), hi.date(), "5Min", "alpaca"))
            except Exception:
                pass
        if frames:
            df = pd.concat(frames)
            bars[sym] = session(df[~df.index.duplicated(keep="last")].sort_index())

    rows = []
    print(f"{'window':30s} {'n':>6} {'mean bps':>9} {'t':>7} {'win':>7} "
          f"{'spread':>7} {'net':>8}")
    print("-" * 80)
    for label, lo, hi in WINDOWS:
        cfg = DayTradeConfig(window_start=lo, window_end=hi,
                             max_trades_per_symbol_per_day=99)
        t = backtest(bars, cfg, cal, cost_bps=0.0)
        s = summarise(t)
        if not s.get("trades"):
            continue
        key = f"{lo:%H:%M}-{hi:%H:%M}"
        spread = SPREAD.get(key, 2.0)
        net = s["mean_ret_bps"] - spread
        rows.append({"window": label, "spread_bps": spread, "net_bps": net, **s})
        print(f"{label:30s} {s['trades']:>6} {s['mean_ret_bps']:>9.2f} "
              f"{s['t_stat']:>7.2f} {s['win_rate']:>6.1%} {spread:>7.2f} {net:>8.2f}")

    pd.DataFrame(rows).to_csv(OUT / "window_sensitivity.csv", index=False)

    print()
    print("SPY and QQQ alone, the two symbols actually recommended:")
    print("-" * 80)
    duo = {k: v for k, v in bars.items() if k in ("SPY", "QQQ")}
    for label, lo, hi in WINDOWS:
        cfg = DayTradeConfig(window_start=lo, window_end=hi,
                             max_trades_per_symbol_per_day=99)
        s = summarise(backtest(duo, cfg, cal, cost_bps=0.0))
        if s.get("trades"):
            print(f"{label:30s} {s['trades']:>6} {s['mean_ret_bps']:>9.2f} "
                  f"{s['t_stat']:>7.2f}")


if __name__ == "__main__":
    main()
