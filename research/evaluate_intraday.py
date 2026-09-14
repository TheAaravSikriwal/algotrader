"""Backtest every strategy as a DAY TRADE, on real five-minute bars.

`research/evaluate_all.py` measures daily-bar strategies against buy-and-hold.
This measures the same rules as intraday trades: flat by the close, confined
to the trading window, costs charged on both sides of every turn.

That distinction matters because the app recommends a rule "for right now".
Recommending on daily-bar evidence would be answering an intraday question
with an answer measured over months.

The output feeds the modular mode: one row per strategy-symbol pair with its
expectancy, its win rate, the bet size the edge supports, and how many trades
that rests on.

    python research/evaluate_intraday.py             # default
    python research/evaluate_intraday.py --quick     # 2 symbols, 6 months
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.data import DataError, load_bars, session as rth  # noqa: E402
from core.env import load_env  # noqa: E402
from core.expectancy import summarise as expectancy_of  # noqa: E402
from core.intraday import IntradayConfig, combine, run  # noqa: E402
from core.journal import bonferroni_bar, two_sided_p  # noqa: E402
from core.marketclock import MarketCalendar  # noqa: E402
from core.strategy import REGISTRY  # noqa: E402
import strategies  # noqa: F401,E402

load_env()

OUT = REPO / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESULT = OUT / "evaluate_intraday.csv"

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "AMD", "TSLA", "MSFT"]
QUICK = ["SPY", "QQQ"]

#: Measured mid-day effective spread, round trip, by symbol. Anything not
#: listed uses the fallback, which is deliberately not optimistic.
SPREAD_BPS = {"SPY": 1.59, "QQQ": 2.12, "IWM": 2.10}
FALLBACK_SPREAD = 4.0

MIN_TRADES = 100


def load(symbol: str, months: int, timeframe: str) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=31 * months)
    frames = []
    # Year by year: one multi-year five-minute request is refused.
    cursor = start
    while cursor < end:
        stop = min(date(cursor.year + 1, 1, 1), end)
        try:
            frames.append(load_bars(symbol, cursor, stop, timeframe, "alpaca"))
        except (DataError, Exception):
            pass
        cursor = stop
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return rth(df[~df.index.duplicated(keep="last")].sort_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--timeframe", default="5Min")
    args = ap.parse_args()

    symbols = QUICK if args.quick else UNIVERSE
    months = 6 if args.quick else args.months
    calendar = MarketCalendar.load()

    print(f"Universe: {', '.join(symbols)}   {months} months of "
          f"{args.timeframe} bars")
    print("Costs are the measured mid-day spread per symbol, charged both "
          "sides.\n")

    bars = {}
    for s in symbols:
        df = load(s, months, args.timeframe)
        if len(df) > 500:
            bars[s] = df
            print(f"  {s:<6} {len(df):>7,} session bars  "
                  f"{df.index.min().date()} .. {df.index.max().date()}")
    if not bars:
        print("no data")
        return

    names = sorted(n for n, c in REGISTRY.items()
                   if not c.requires_features and not n.startswith("ZZ"))
    rows = []
    for i, name in enumerate(names, 1):
        print(f"[{i:>2}/{len(names)}] {name}")
        results = []
        for sym, df in bars.items():
            cfg = IntradayConfig(
                timeframe=args.timeframe,
                cost_bps=SPREAD_BPS.get(sym, FALLBACK_SPREAD))
            r = run(df, name, sym, cfg=cfg, calendar=calendar)
            if r.trades.empty:
                continue
            results.append(r)

            e = expectancy_of(r.trade_returns)
            gross = expectancy_of(r.trades["gross_pct"].tolist())
            rows.append({
                "strategy": name, "symbol": sym,
                "sessions": r.sessions, "trades": e.trades,
                "trades_per_session": e.trades / max(r.sessions, 1),
                "win_rate": e.win_rate,
                "expectancy_pct": e.expectancy_pct,
                "expectancy_bps": e.expectancy_pct * 100.0,
                "gross_bps": gross.expectancy_pct * 100.0,
                "cost_bps": cfg.cost_bps,
                "payoff_ratio": e.payoff_ratio,
                "profit_factor": e.profit_factor,
                "t_stat": e.t_stat,
                "p_value": two_sided_p(e.t_stat) if np.isfinite(e.t_stat) else np.nan,
                "kelly": e.kelly_fraction,
                "suggested_size": e.suggested_fraction,
                "stdev_pct": e.stdev_pct,
                "trades_needed": e.trades_needed(),
                "avg_bars_held": float(r.trades["bars_held"].mean()),
            })

        both = combine(results)
        if not both.empty:
            e = expectancy_of(both["return_pct"].tolist())
            rows.append({
                "strategy": name, "symbol": "ALL",
                "sessions": sum(r.sessions for r in results),
                "trades": e.trades,
                "trades_per_session": e.trades / max(
                    sum(r.sessions for r in results), 1),
                "win_rate": e.win_rate,
                "expectancy_pct": e.expectancy_pct,
                "expectancy_bps": e.expectancy_pct * 100.0,
                "gross_bps": expectancy_of(
                    both["gross_pct"].tolist()).expectancy_pct * 100.0,
                "cost_bps": np.nan,
                "payoff_ratio": e.payoff_ratio,
                "profit_factor": e.profit_factor,
                "t_stat": e.t_stat,
                "p_value": two_sided_p(e.t_stat) if np.isfinite(e.t_stat) else np.nan,
                "kelly": e.kelly_fraction,
                "suggested_size": e.suggested_fraction,
                "stdev_pct": e.stdev_pct,
                "trades_needed": e.trades_needed(),
                "avg_bars_held": float(both["bars_held"].mean()),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing traded")
        return
    df.to_csv(RESULT, index=False)

    tested = df[(df["symbol"] != "ALL") & (df["trades"] >= MIN_TRADES)]
    bar = bonferroni_bar(max(len(tested), 1))
    df["bonferroni_t_bar"] = bar
    df["clears_bar"] = (df["t_stat"] > bar) & (df["trades"] >= MIN_TRADES)
    df.to_csv(RESULT, index=False)

    agg = df[df["symbol"] == "ALL"].sort_values("expectancy_bps", ascending=False)
    print()
    print("=" * 92)
    print(f"ALL SYMBOLS POOLED, net of measured spread. "
          f"{len(tested)} pairs tested, Bonferroni |t| > {bar:.2f}")
    print("=" * 92)
    print(f"{'strategy':<34}{'trades':>8}{'/session':>9}{'gross':>8}"
          f"{'net bps':>9}{'t':>7}{'win':>6}{'size':>7}")
    print("-" * 92)
    for _, r in agg.iterrows():
        print(f"{r['strategy']:<34}{int(r['trades']):>8}"
              f"{r['trades_per_session']:>9.1f}{r['gross_bps']:>8.2f}"
              f"{r['expectancy_bps']:>9.2f}{r['t_stat']:>7.2f}"
              f"{r['win_rate']:>6.0%}{r['suggested_size']:>7.2%}")

    winners = df[df["clears_bar"]]
    print()
    print(f"Cleared the corrected bar: {len(winners)} of {len(tested)}")
    if len(winners):
        print(winners[["strategy", "symbol", "expectancy_bps", "t_stat",
                       "trades", "suggested_size"]].to_string(index=False))
    positive = agg[agg["expectancy_bps"] > 0]
    print(f"Positive net expectancy pooled: {len(positive)} of {len(agg)} "
          f"strategies")
    print(f"\nWritten to {RESULT}")


if __name__ == "__main__":
    main()
