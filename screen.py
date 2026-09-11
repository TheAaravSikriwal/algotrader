"""Pick what to trade, by what the strategy type actually needs.

    python screen.py --profile short_term
    python screen.py --profile long_term --top-n 30
    python screen.py --profile reasoning --with-news
    python screen.py --profile all

Screens on tradability -- liquidity, spread proxy, history, coverage -- and
never on past returns. Filtering to names that went up is survivorship bias in
a screener's clothing; liquidity says whether a trade is executable, not
whether it profits, which is what makes it safe to select on.
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
from core.news import NewsError, load_news
from core.taxonomy import top_categories
from core.universe import (CANDIDATE_POOL, PROFILES, describe_profile,
                           profile_frame, screen)

load_env()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="all",
                   choices=sorted(PROFILES) + ["all"])
    p.add_argument("--symbols", help="candidate pool, comma separated")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--window", type=int, default=252,
                   help="bars used to measure liquidity")
    p.add_argument("--top-n", type=int, default=0, help="override the profile")
    p.add_argument("--with-news", action="store_true",
                   help="measure coverage too (needs Alpaca keys, and is slow)")
    p.add_argument("--news-days", type=int, default=90)
    p.add_argument("--as-of", metavar="YYYY-MM-DD",
                   help="screen using only data up to this date. Set it to the "
                        "start of the backtest that follows -- without it, "
                        "volatility and range are measured over the very "
                        "window you are about to test on.")
    p.add_argument("--save", metavar="PATH")
    return p


def gather_news(symbols, days: int) -> tuple[dict, dict]:
    """Coverage volume and, per symbol, what that coverage is actually about."""
    from core.taxonomy import profile_texts

    end = date.today()
    start = end - timedelta(days=days)
    counts, profiles = {}, {}
    for symbol in symbols:
        try:
            items = load_news([symbol], start, end)
        except (NewsError, Exception):  # noqa: BLE001
            counts[symbol], profiles[symbol] = 0.0, profile_texts([])
            continue
        counts[symbol] = len(items) / max(days, 1)
        profiles[symbol] = profile_texts([i.text for i in items])
    return counts, profiles


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pool = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            if args.symbols else CANDIDATE_POOL)

    start = date.today() - timedelta(days=365 * args.years)
    print(f"Screening {len(pool)} candidates over {args.years} years...\n")

    bars, failed = {}, []
    for symbol in pool:
        try:
            df = load_bars(symbol, start, date.today(), "1Day", args.source)
            if len(df) > 30:
                bars[symbol] = df
            else:
                failed.append(symbol)
        except Exception:  # noqa: BLE001
            failed.append(symbol)

    if failed:
        print(f"  no data for {len(failed)}: {', '.join(failed[:12])}"
              f"{' ...' if len(failed) > 12 else ''}\n")
    if len(bars) < 5:
        print("Too few candidates returned data to screen.")
        return 1

    news_counts, news_profiles = None, None
    if args.with_news:
        print(f"  measuring news coverage over {args.news_days} days...")
        news_counts, news_profiles = gather_news(list(bars), args.news_days)

    if not args.as_of:
        print("  NOTE: screening as of today. Volatility and daily range are\n"
              "  therefore measured over whatever window you backtest next,\n"
              "  which flatters the result. Pass --as-of with the backtest's\n"
              "  start date to screen on what was knowable then.\n")

    frame = profile_frame(bars, news_counts, args.window, news_profiles,
                          as_of=args.as_of)
    profiles = sorted(PROFILES) if args.profile == "all" else [args.profile]
    results = {}

    for name in profiles:
        prof = PROFILES[name]
        if args.top_n:
            prof = type(prof)(**{**prof.__dict__, "top_n": args.top_n})

        print("=" * 74)
        print(f"{name.upper().replace('_', ' ')}")
        print(f"  {describe_profile(prof)}")
        print("=" * 74)

        picked = screen(frame, prof)
        if picked.empty:
            if prof.min_news_per_day > 0 and not news_counts:
                print("  nothing cleared these thresholds -- but coverage was "
                      "never measured.\n  This profile screens on news volume; "
                      "re-run with --with-news.\n")
            else:
                print("  nothing cleared these thresholds.\n")
            continue

        show = picked[["price", "dollar_volume", "amihud", "range_pct",
                       "ann_vol", "history_days"]].copy()
        show["dollar_volume"] = (show["dollar_volume"] / 1e6).round(1)
        show["range_pct"] = (show["range_pct"] * 100).round(2)
        show["ann_vol"] = (show["ann_vol"] * 100).round(1)
        show.columns = ["price", "$vol (M)", "amihud", "range %", "vol %", "bars"]
        if news_counts:
            show["news/day"] = picked["news_per_day"].round(2)
            column = f"{prof.bucket}_news_per_day" if prof.bucket else ""
            if column and column in picked:
                show[f"{prof.bucket}/day"] = picked[column].round(2)

        print(show.to_string(float_format=lambda x: f"{x:,.2f}"))
        print(f"\n  {len(picked)} names: {','.join(picked.index)}\n")

        if news_profiles:
            bucket = prof.bucket or ("macro" if "etf" in prof.asset_types
                                     else "company")
            print(f"  what that coverage is actually about ({bucket}):")
            for symbol in picked.index[:10]:
                top = top_categories(news_profiles.get(symbol, {}), bucket, 3)
                shown = ", ".join(f"{k} {v:.0%}" for k, v in top if v > 0)
                print(f"    {symbol:<7} {shown or 'nothing classified'}")
            print()
        results[name] = picked

    if results:
        print("=" * 74)
        print("Using these:")
        for name, picked in results.items():
            flag = "--universe" if name != "reasoning" else "--symbols"
            print(f"  {name:<12} {','.join(picked.index[:12])}"
                  f"{' ...' if len(picked) > 12 else ''}")

    print("\nScreened on tradability only -- no return ranking entered this.")
    if args.as_of:
        print(f"Measured as of {args.as_of}, so the universe is one you could")
        print("actually have picked on that date.")
    else:
        print("But measured as of today: realised volatility and daily range are")
        print("past-return quantities, and relative to a backtest that follows")
        print("they are FUTURE quantities. Re-run with --as-of before trusting")
        print("any result built on this universe.")
    print("Neither fixes the candidate pool: every name in it survived to be")
    print("listed today, so the companies that failed are already missing.")

    if args.save and results:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([df.assign(profile=name) for name, df in results.items()])
        combined.to_csv(path)
        print(f"\nWrote {len(combined)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
