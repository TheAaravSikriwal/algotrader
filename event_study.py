"""What actually happens after an event, pooled across companies.

    # every product announcement across the mega caps
    python event_study.py --universe megacap --event product

    # a keyword, which is how you test a specific idea
    python event_study.py --universe sp30 --keyword "ad campaign" --post 20

    # only strongly positive news carried by several outlets
    python event_study.py --universe sp30 --event analyst --min-sentiment 0.3

    # a price condition rather than news
    python event_study.py --universe sp30 --condition gap-down

One company's history of a rare event is an anecdote. The same event pooled
across a hundred companies over a decade is evidence. This turns the first into
the second, and tells you when it still is not enough.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import load_bars
from core.env import load_env
from core.eventstudy import (EventWindow, events_from_condition,
                             events_from_news, interpret, run_event_study)
from core.news import NewsError, load_news, to_frame
from core.sentiment import get_scorer, score_frame

load_env()

UNIVERSES = {
    "megacap": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                "TSLA", "JPM", "V", "WMT", "XOM"],
    "sp30": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
             "JPM", "V", "WMT", "XOM", "JNJ", "PG", "MA", "HD", "CVX", "MRK",
             "ABBV", "KO", "PEP", "COST", "ADBE", "CRM", "NFLX", "AMD", "INTC",
             "CSCO", "QCOM", "TXN"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL"],
}

CONDITIONS = {
    "gap-down": lambda df: df["open"] / df["close"].shift(1) - 1 < -0.03,
    "gap-up": lambda df: df["open"] / df["close"].shift(1) - 1 > 0.03,
    "52w-high": lambda df: df["close"] >= df["close"].rolling(252).max(),
    "52w-low": lambda df: df["close"] <= df["close"].rolling(252).min(),
    "big-drop": lambda df: df["close"].pct_change() < -0.05,
    "volume-spike": lambda df: df["volume"] > 3 * df["volume"].rolling(50).mean(),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", choices=sorted(UNIVERSES), default="megacap")
    p.add_argument("--symbols", help="comma separated, overrides --universe")
    p.add_argument("--start", default=str(date.today() - timedelta(days=365 * 4)))
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--benchmark", default="SPY")

    p.add_argument("--event", help="news event tag, e.g. product, analyst, mna")
    p.add_argument("--keyword", help="literal phrase to match in the article text")
    p.add_argument("--condition", choices=sorted(CONDITIONS),
                   help="price condition instead of news")
    p.add_argument("--min-sentiment", type=float)
    p.add_argument("--max-sentiment", type=float)
    p.add_argument("--min-sources", type=int, default=1)
    p.add_argument("--scorer", default="lexicon", choices=["lexicon", "finbert"])

    p.add_argument("--pre", type=int, default=5)
    p.add_argument("--post", type=int, default=20)
    p.add_argument("--estimation", type=int, default=200)
    p.add_argument("--gap", type=int, default=10)
    p.add_argument("--model", default="market",
                   choices=["market", "market_adjusted", "raw"])
    p.add_argument("--save", metavar="PATH")
    return p


def gather_events(args, symbols, bars) -> pd.DataFrame:
    if args.condition:
        return events_from_condition(bars, CONDITIONS[args.condition])

    if not (args.event or args.keyword or args.min_sentiment is not None):
        raise SystemExit("Pick something to study: --event, --keyword, "
                         "--min-sentiment or --condition.")

    items = load_news(symbols, args.start, args.end)
    if not items:
        raise SystemExit("No news returned for that universe and window.")
    print(f"  {len(items):,} articles fetched")

    scored = score_frame(to_frame(items), get_scorer(args.scorer))
    return events_from_news(
        scored, event_type=args.event, keyword=args.keyword,
        min_sentiment=args.min_sentiment, max_sentiment=args.max_sentiment,
        min_sources=args.min_sources)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else UNIVERSES[args.universe])

    label = (args.condition or args.event or
             (f'"{args.keyword}"' if args.keyword else "sentiment filter"))
    print(f"Event study: {label} across {len(symbols)} symbols, "
          f"{args.start} to {args.end}\n")

    bars = {}
    for symbol in symbols:
        try:
            df = load_bars(symbol, args.start, args.end, "1Day", args.source)
            if len(df) > 60:
                bars[symbol] = df
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: skipped ({exc})")
    if len(bars) < 2:
        print("Not enough symbols with price history.")
        return 1

    benchmark = None
    try:
        benchmark = load_bars(args.benchmark, args.start, args.end, "1Day",
                              args.source)["close"]
    except Exception:  # noqa: BLE001
        print(f"  ({args.benchmark} unavailable -- falling back to raw returns)")

    try:
        events = gather_events(args, symbols, bars)
    except NewsError as exc:
        print(f"News unavailable: {exc}")
        return 1

    if events.empty:
        print("No events matched. Loosen the filter or widen the window.")
        return 1
    print(f"  {len(events):,} candidate events across "
          f"{events['symbol'].nunique()} symbols\n")

    window = EventWindow(pre=args.pre, post=args.post,
                         estimation=args.estimation, gap=args.gap)
    try:
        result = run_event_study(events, bars, benchmark, window, args.model)
    except ValueError as exc:
        print(f"Could not run the study: {exc}")
        return 1

    s = result.summary
    print("=" * 72)
    print(f"{s['events']:,} usable events · {s['symbols']} symbols · "
          f"{s['first_event']} to {s['last_event']} · {s['model']} model")
    dropped = {k: v for k, v in s["skipped"].items() if v}
    if dropped:
        print(f"skipped: {dropped}")
    print("=" * 72)

    table = result.daily.copy()
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print(f"\nEvent day itself: {s['event_day_%']:+.3f}%  "
          f"(t = {s['t_event_day']:+.2f})")
    if args.condition:
        print("  A condition screen selects on this number, so it restates the "
              "filter\n  rather than telling you anything. Not tradeable.")

    print("\nCumulative abnormal return from the NEXT bar onward:")
    for day in (1, 3, 5, 10, 20):
        if f"car_{day}d_%" in s:
            flag = "  <-- clears |t| >= 2" if abs(s[f"t_{day}d"]) >= 2 else ""
            print(f"  {day:>2} day(s): {s[f'car_{day}d_%']:+7.3f}%   "
                  f"t = {s[f't_{day}d']:+6.2f}{flag}")

    print("\nReading it:")
    for note in interpret(result):
        print(f"  * {note}")

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.events.to_csv(path, index=False)
        print(f"\nWrote {len(result.events)} events to {path}")

    print("\nAbnormal returns are measured against the benchmark, so a rising "
          "market\nis already accounted for. Costs are not -- a 0.3% edge does "
          "not survive\ncrossing the spread twice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
