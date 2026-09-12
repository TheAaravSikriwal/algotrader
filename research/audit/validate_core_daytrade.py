"""Does core/daytrade.py reproduce the audit's result on real bars?

The audit's finding lived in research/audit/s10_*.py against its own cache,
written under the old UTC clock. This re-runs the promoted implementation on
freshly fetched Eastern-clock bars. If the two disagree badly, one of them is
wrong and the strategy should not be traded.
"""
from __future__ import annotations

import sys
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
START, END = "2021-01-01", "2026-09-01"
SPLIT = pd.Timestamp("2024-06-01")          # same in/out split the audit used
OUT = Path(__file__).resolve().parent / "results"


def fetch(symbol: str) -> pd.DataFrame:
    """Year by year -- one multi-year 5-minute request is refused."""
    frames = []
    for year in range(2021, 2027):
        lo = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(START))
        hi = min(pd.Timestamp(f"{year + 1}-01-01"), pd.Timestamp(END))
        if lo >= hi:
            continue
        try:
            frames.append(load_bars(symbol, lo.date(), hi.date(), "5Min", "alpaca"))
        except Exception as exc:
            print(f"  {symbol} {year}: {type(exc).__name__} {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return session(df[~df.index.duplicated(keep="last")].sort_index())


def main():
    cal = MarketCalendar.load()
    cfg = DayTradeConfig()

    bars = {}
    for sym in SYMBOLS:
        df = fetch(sym)
        if not df.empty:
            bars[sym] = df
            print(f"{sym:6s} {len(df):>8,} session bars  "
                  f"{df.index.min().date()} .. {df.index.max().date()}")

    trades = backtest(bars, cfg, cal, cost_bps=0.0)
    if trades.empty:
        print("no trades")
        return

    OUT.mkdir(exist_ok=True)
    trades.to_csv(OUT / "core_daytrade_trades.csv", index=False)

    print()
    print("=" * 74)
    print("FULL SAMPLE, frictionless")
    print("=" * 74)
    s = summarise(trades)
    for k, v in s.items():
        print(f"  {k:22s} {v:,.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    print()
    print("=" * 74)
    print("IN SAMPLE vs OUT OF SAMPLE")
    print("=" * 74)
    for label, sub in [("in  (< 2024-06)", trades[trades["entry_ts"] < SPLIT]),
                       ("out (>= 2024-06)", trades[trades["entry_ts"] >= SPLIT])]:
        st = summarise(sub)
        print(f"  {label}: n={st.get('trades', 0):>5}  "
              f"mean={st.get('mean_ret_bps', float('nan')):>7.2f} bps  "
              f"t={st.get('t_stat', float('nan')):>6.2f}  "
              f"R={st.get('mean_r', float('nan')):>6.3f}")

    print()
    print("=" * 74)
    print("COST SWEEP -- measured spreads: SPY 1.59 bps, QQQ 2.12 mid-day")
    print("=" * 74)
    rows = []
    for bps in [0, 1, 2, 3, 5, 10]:
        t = backtest(bars, cfg, cal, cost_bps=bps)
        st = summarise(t, cost_bps=bps)
        rows.append({"cost_bps": bps, **st})
        print(f"  {bps:>3} bps: mean={st['mean_ret_bps']:>7.2f} bps  "
              f"t={st['t_stat']:>6.2f}  win={st['win_rate']:.1%}")
    pd.DataFrame(rows).to_csv(OUT / "core_daytrade_costsweep.csv", index=False)

    print()
    print("=" * 74)
    print("PER SYMBOL, frictionless")
    print("=" * 74)
    for sym, sub in trades.groupby("symbol"):
        st = summarise(sub)
        print(f"  {sym:6s} n={st['trades']:>5}  mean={st['mean_ret_bps']:>7.2f} bps  "
              f"t={st['t_stat']:>6.2f}  breakeven={st['breakeven_bps']:>6.2f}")


if __name__ == "__main__":
    main()
