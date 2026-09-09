"""Run every strategy against every symbol and rank them out-of-sample.

    python tournament.py --symbols SPY,QQQ,AAPL,MSFT,NVDA
    python tournament.py --symbols SPY --train 504 --test 126 --top 5
    python tournament.py --symbols SPY,QQQ --save runs/leaderboard.csv

Rankings come from walk-forward testing: parameters are chosen on a training
window and scored on the window that follows, which the optimiser never saw.
An in-sample leaderboard would just rank how well each strategy memorised the
past.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import load_bars
from core.engine import BacktestConfig
from core.env import load_env
from core.strategy import available, get_strategy
from core.tournament import WalkForwardConfig, leaders, run_tournament

load_env()
import strategies  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="SPY,QQQ", help="comma-separated")
    p.add_argument("--strategies", default="",
                   help="comma-separated; default is every registered strategy")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--start", default=str(date.today() - timedelta(days=365 * 8)))
    p.add_argument("--end", default=str(date.today()))

    p.add_argument("--train", type=int, default=504, help="training bars per fold")
    p.add_argument("--test", type=int, default=126, help="held-out bars per fold")
    p.add_argument("--anchored", action="store_true",
                   help="expanding training window instead of rolling")
    p.add_argument("--metric", default="Sharpe",
                   choices=["Sharpe", "Sortino", "Calmar", "Total return", "CAGR"])
    p.add_argument("--max-grid", type=int, default=240)

    p.add_argument("--cash", type=float, default=10_000)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--min-folds", type=int, default=3)
    p.add_argument("--save", metavar="PATH", help="write the full leaderboard to CSV")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    names = ([s.strip() for s in args.strategies.split(",") if s.strip()]
             or available())
    strats = [get_strategy(n) for n in names]

    print(f"Loading {len(symbols)} symbol(s)...")
    data = {}
    for sym in symbols:
        try:
            data[sym] = load_bars(sym, args.start, args.end, args.timeframe, args.source)
            print(f"  {sym}: {len(data[sym]):,} bars")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: skipped ({exc})")

    if not data:
        print("No data loaded.")
        return 1

    wf = WalkForwardConfig(train_bars=args.train, test_bars=args.test,
                           anchored=args.anchored, rank_metric=args.metric,
                           max_grid=args.max_grid)
    bt = BacktestConfig(initial_cash=args.cash, slippage_bps=args.slippage_bps)

    print(f"\nWalk-forward: {args.train} train / {args.test} test bars per fold, "
          f"ranking on {args.metric}")
    print(f"{len(data) * len(strats)} combinations to evaluate. This takes a while.\n")

    def progress(done, total, label):
        print(f"  [{done:>3}/{total}] {label}", flush=True)

    table = run_tournament(data, strats, wf=wf, backtest=bt, progress=progress)

    if table.empty:
        print("\nNothing evaluated -- try a longer date range or smaller --train.")
        return 1

    show = [c for c in ("strategy", "symbol", "oos_return_%", "oos_sharpe",
                        "oos_max_dd_%", "folds", "fold_win_rate", "param_stability",
                        "params_at_edge")
            if c in table.columns]

    print("\n" + "=" * 78)
    print("FULL LEADERBOARD (out-of-sample)")
    print("=" * 78)
    print(table[show].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    top = leaders(table, n=args.top, min_folds=args.min_folds)
    print("\n" + "=" * 78)
    print(f"TOP {args.top} worth paper trading")
    print("=" * 78)
    if top.empty:
        print("None qualified. Every candidate was either too thinly tested,\n"
              "profitable in fewer than half its folds, or negative out-of-sample.\n"
              "That is a real answer, not a failure -- none of these earned money.")
    else:
        print(top[show].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("\nParameters to trade:")
        for _, row in top.iterrows():
            print(f"  {row['strategy']} on {row['symbol']}: {row['params']}")

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"\nFull leaderboard written to {path}")

    print("\nThese are out-of-sample results, which is the honest test -- but they "
          "are still\nhistorical. Paper trade the survivors before funding them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
