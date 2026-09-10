"""News pipeline: fetch, score, analyse, backtest.

    # is there any signal here at all, and at what horizon?
    python news_backtest.py --symbols AAPL --decay

    # sweep the holding period to find where the edge lives
    python news_backtest.py --symbols AAPL,MSFT,NVDA --sweep-horizons

    # walk-forward tournament over the news strategies
    python news_backtest.py --symbols AAPL,MSFT --tournament

    # no Alpaca keys yet? see the whole thing run on synthetic news
    python news_backtest.py --demo

Fetching news needs Alpaca keys in .env. Scoring defaults to the built-in
lexicon, which needs nothing; pass --scorer finbert once you have installed
transformers and torch.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import load_bars
from core.engine import BacktestConfig, run_backtest
from core.env import load_env
from core.metrics import summarise
from core.news import NewsError, load_news, to_frame
from core.newsfeatures import build_news_features, decay_profile
from core.sentiment import get_scorer
from core.strategy import get_strategy
from core.tournament import WalkForwardConfig, leaders, run_tournament

load_env()
import strategies  # noqa: F401

NEWS_STRATEGIES = ["News sentiment", "News drift", "News event drift"]
HORIZONS = [1, 2, 3, 5, 10, 20]


def synthetic_market(symbol="DEMO", n=900, alpha=0.006, seed=17, horizon=5):
    """Prices that genuinely respond to sentiment, for demonstrating the flow."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2021-01-04", periods=n)
    sentiment = rng.uniform(-1, 1, n)
    quiet = rng.random(n) < 0.45                # most days have no real news
    sentiment[quiet] = 0.0

    returns = rng.normal(0.0002, 0.009, n)
    for lag in range(1, horizon + 1):
        returns[lag:] += alpha * sentiment[:-lag] / horizon

    prices = 100 * np.exp(np.cumsum(returns))
    bars = pd.DataFrame({"open": prices, "high": prices * 1.004,
                         "low": prices * 0.996, "close": prices,
                         "volume": np.full(n, 1e6)}, index=sessions)

    active = ~quiet
    scored = pd.DataFrame({
        "session": sessions[active], "symbol": symbol,
        "sentiment": sentiment[active], "confidence": 0.8,
        "events": [["earnings"]] * int(active.sum()),
        "text": ["synthetic story"] * int(active.sum()),
    })
    return bars, scored


def enrich(symbol: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bars joined with news features, plus the scored article frame."""
    bars = load_bars(symbol, args.start, args.end, "1Day", args.source)
    items = load_news([symbol], args.start, args.end)
    if not items:
        raise NewsError(f"no news returned for {symbol}")

    frame = to_frame(items)
    scored = get_scorer(args.scorer)
    from core.sentiment import score_frame
    scored_frame_ = score_frame(frame, scored)
    features = build_news_features(scored_frame_, bars, symbol,
                                   carry_forward=args.carry_forward)
    return bars.join(features), scored_frame_


def show_decay(symbol: str, features_bars: pd.DataFrame,
               benchmark: pd.Series | None = None):
    table = decay_profile(features_bars, features_bars, horizons=tuple(HORIZONS),
                          benchmark=benchmark)
    label = "excess return vs SPY" if benchmark is not None else "forward return"
    print(f"\n--- {symbol}: mean {label} by sentiment bucket ---")
    if table.empty:
        print("  not enough news days to bucket")
        return

    print(table.to_string(index=False, float_format=lambda x: f"{x:,.3f}",
                          na_rep=""))

    spread_rows = table[table["bucket"] == "top-bottom"]
    tstat_rows = table[table["bucket"] == "t-stat"]
    if spread_rows.empty or tstat_rows.empty:
        return

    spread, tstats = spread_rows.iloc[0], tstat_rows.iloc[0]
    significant = [(h, spread.get(f"fwd_{h}_%", 0), tstats.get(f"fwd_{h}_%", 0))
                   for h in HORIZONS
                   if abs(tstats.get(f"fwd_{h}_%", 0) or 0) >= 2.0]

    if significant:
        print("\n  Statistically distinguishable from noise (|t| >= 2):")
        for h, value, t in significant:
            print(f"    {h:>2} day(s): {value:+.3f}%  (t = {t:+.2f})")
    else:
        best = max(HORIZONS, key=lambda h: abs(tstats.get(f"fwd_{h}_%", 0) or 0))
        print(f"\n  NO horizon reaches |t| >= 2. Strongest is {best} day(s) at "
              f"t = {tstats.get(f'fwd_{best}_%', 0):+.2f}.")
        print("  Whatever spread you see above is within what random buckets "
              "would produce.")


def sweep_horizons(symbol: str, enriched: pd.DataFrame, args):
    rows = []
    cfg = BacktestConfig(initial_cash=args.cash, slippage_bps=args.slippage_bps)
    for hold in HORIZONS:
        strat = get_strategy("News sentiment")(
            threshold=args.threshold, hold_bars=hold, min_articles=args.min_articles)
        result = run_backtest(enriched, strat.generate_signals(enriched), cfg)
        stats = summarise(result)
        rows.append({
            "hold_bars": hold,
            "return_%": stats["Total return"] * 100,
            "sharpe": stats["Sharpe"],
            "max_dd_%": stats["Max drawdown"] * 100,
            "trades": stats["Trades"],
            "win_rate_%": stats["Win rate"] * 100,
            "time_in_mkt_%": stats["Time in market"] * 100,
        })

    table = pd.DataFrame(rows)
    print(f"\n--- {symbol}: holding-period sweep ---")
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    bh = summarise(run_backtest(enriched, pd.Series(1.0, index=enriched.index), cfg))
    print(f"\n  Buy and hold over the same window: "
          f"{bh['Total return'] * 100:+.2f}%  Sharpe {bh['Sharpe']:.2f}")
    best = table.loc[table["sharpe"].idxmax()]
    print(f"  Best holding period by Sharpe: {int(best['hold_bars'])} bars "
          f"({best['sharpe']:.2f})")
    if best["sharpe"] <= bh["Sharpe"]:
        print("  ...which still does not beat doing nothing. That is a result, "
              "not a bug.")

    # A "winner" that holds continuously has stopped being a news strategy and
    # become an expensive imitation of buy-and-hold.
    if best["time_in_mkt_%"] > 95 or best["trades"] < 10:
        print(f"  WARNING: that setting holds {best['time_in_mkt_%']:.0f}% of the "
              f"time across {int(best['trades'])} trade(s).")
        print("  It has degenerated into buy-and-hold -- the news signal is not "
              "driving it,\n  and a handful of trades cannot support any "
              "conclusion either way.")

    tradeable = table[(table["trades"] >= 20) & (table["time_in_mkt_%"] < 95)]
    if not tradeable.empty:
        pick = tradeable.loc[tradeable["sharpe"].idxmax()]
        print(f"  Best setting that actually trades: {int(pick['hold_bars'])} bars, "
              f"{pick['return_%']:+.2f}% over {int(pick['trades'])} trades "
              f"(Sharpe {pick['sharpe']:.2f})")
    return table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="AAPL")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"],
                   help="price source; news always comes from Alpaca")
    p.add_argument("--start", default=str(date.today() - timedelta(days=365 * 3)))
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--scorer", default="lexicon", choices=["lexicon", "finbert"])
    p.add_argument("--carry-forward", type=int, default=0,
                   help="keep a day's news features alive for N extra bars")
    p.add_argument("--benchmark", default="SPY",
                   help="measure excess returns against this symbol; "
                        "pass an empty string for raw returns")

    p.add_argument("--decay", action="store_true", help="forward-return profile")
    p.add_argument("--sweep-horizons", action="store_true")
    p.add_argument("--tournament", action="store_true",
                   help="walk-forward ranking of the news strategies")
    p.add_argument("--demo", action="store_true",
                   help="run on synthetic news, no keys needed")

    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--min-articles", type=int, default=1)
    p.add_argument("--cash", type=float, default=10_000)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--train", type=int, default=378)
    p.add_argument("--test", type=int, default=126)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.decay or args.sweep_horizons or args.tournament):
        args.decay = args.sweep_horizons = True

    enriched_by_symbol: dict[str, pd.DataFrame] = {}

    if args.demo:
        print("DEMO MODE -- synthetic news with a planted 5-day signal.\n"
              "Real runs need Alpaca keys in .env.")
        bars, scored = synthetic_market()
        features = build_news_features(scored, bars, "DEMO",
                                       carry_forward=args.carry_forward)
        enriched_by_symbol["DEMO"] = bars.join(features)
    else:
        for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
            try:
                enriched, scored = enrich(symbol, args)
                enriched_by_symbol[symbol] = enriched
                days = int((enriched["news_count"] > 0).sum())
                # `scored` holds one row per (article, symbol) pair, so count
                # only the rows belonging to this symbol
                n_articles = int((scored["symbol"] == symbol).sum())
                coverage = days / len(enriched) * 100 if len(enriched) else 0
                print(f"{symbol}: {len(enriched):,} bars, {n_articles:,} articles "
                      f"on {days:,} days ({coverage:.0f}% of bars, "
                      f"{args.scorer} scorer)")
            except (NewsError, Exception) as exc:  # noqa: BLE001
                print(f"{symbol}: skipped -- {exc}")

    if not enriched_by_symbol:
        print("\nNothing to analyse. Add Alpaca keys to .env, or try --demo.")
        return 1

    benchmark = None
    if args.benchmark and not args.demo:
        try:
            benchmark = load_bars(args.benchmark, args.start, args.end,
                                  "1Day", args.source)["close"]
            print(f"\nMeasuring excess returns against {args.benchmark}.")
        except Exception as exc:  # noqa: BLE001
            print(f"\nBenchmark {args.benchmark} unavailable ({exc}); "
                  "reporting raw returns.")

    for symbol, enriched in enriched_by_symbol.items():
        if args.decay:
            show_decay(symbol, enriched, benchmark)
        if args.sweep_horizons:
            sweep_horizons(symbol, enriched, args)

    if args.tournament:
        print("\n" + "=" * 78)
        print("WALK-FORWARD TOURNAMENT -- news strategies, out-of-sample")
        print("=" * 78)
        table = run_tournament(
            enriched_by_symbol, [get_strategy(n) for n in NEWS_STRATEGIES],
            wf=WalkForwardConfig(train_bars=args.train, test_bars=args.test,
                                 max_grid=120),
            backtest=BacktestConfig(initial_cash=args.cash,
                                    slippage_bps=args.slippage_bps),
            progress=lambda d, t, l: print(f"  [{d}/{t}] {l}", flush=True))

        if table.empty:
            print("Nothing evaluated -- try a longer history or a smaller --train.")
        else:
            cols = [c for c in ("strategy", "symbol", "oos_return_%", "oos_sharpe",
                                "oos_max_dd_%", "folds", "fold_win_rate",
                                "params_at_edge") if c in table.columns]
            print(table[cols].to_string(index=False,
                                        float_format=lambda x: f"{x:,.2f}"))
            top = leaders(table, n=5)
            print("\nQualifying strategies:"
                  if not top.empty else
                  "\nNone qualified out-of-sample. That is the honest answer.")
            if not top.empty:
                for _, row in top.iterrows():
                    print(f"  {row['strategy']} on {row['symbol']}: {row['params']}")

    print("\nHypothetical results on historical news. Sentiment scoring is applied "
          "to\nhistorical text with today's model -- fine for research, but it is "
          "not the\nsame as having scored it live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
