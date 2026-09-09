"""Command-line backtest -- same engine as the dashboard, no browser.

    python cli.py --symbol AAPL --strategy "SMA crossover" --fast 20 --slow 50
    python cli.py --list
    python cli.py --symbol SPY --strategy "RSI mean reversion" --sweep period=5:30:5
"""
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.bundle import (bundle_equity, build_bundle, describe, load_bundle,
                          replay, save_bundle)
from core.data import bars_per_year, load_bars
from core.engine import BacktestConfig, run_backtest
from core.metrics import summarise
from core.strategy import available, get_strategy

from core.env import load_env

load_env()
import strategies  # noqa: F401


def parse_sweep(spec: str) -> tuple[str, list]:
    """`period=5:30:5` -> ("period", [5, 10, 15, 20, 25, 30])"""
    name, _, rng = spec.partition("=")
    lo, hi, step = (float(x) for x in rng.split(":"))
    values, v = [], lo
    while v <= hi + 1e-9:
        values.append(int(v) if float(v).is_integer() else round(v, 6))
        v += step
    return name, values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest a strategy over historical bars.")
    p.add_argument("--list", action="store_true", help="show available strategies and exit")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--strategy", default="SMA crossover")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--timeframe", default="1Day",
                   choices=["1Day", "1Hour", "15Min", "5Min", "1Min"])
    p.add_argument("--start", default=str(date.today() - timedelta(days=365 * 5)))
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--cash", type=float, default=10_000)
    p.add_argument("--position-size", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--commission-pct", type=float, default=0.0)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--stop-loss", type=float, default=0.0, help="percent, 0 disables")
    p.add_argument("--take-profit", type=float, default=0.0, help="percent, 0 disables")
    p.add_argument("--sweep", action="append", default=[],
                   metavar="NAME=LO:HI:STEP", help="grid-search a parameter; repeatable")
    p.add_argument("--save-trades", metavar="PATH", help="write the trade log to CSV")
    p.add_argument("--save-bundle", metavar="PATH",
                   help="write strategy + settings + results to a .json bundle")
    p.add_argument("--replay", metavar="PATH",
                   help="re-run a saved bundle and report any drift from it")
    return p


def show_report(stats: dict, title: str):
    print(f"\n{title}")
    print("-" * len(title))
    pct_keys = ("return", "CAGR", "drawdown", "rate", "Volatility", "Time in market", "DD")
    for k, v in stats.items():
        if isinstance(v, float) and any(w in k for w in pct_keys):
            shown = f"{v * 100:>10,.2f}%"
        elif isinstance(v, float):
            shown = f"{v:>11,.2f}"
        else:
            shown = f"{v:>11,}"
        print(f"  {k:<26}{shown}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # strategy flags are only known once --strategy has been read, so take two passes
    args, rest = parser.parse_known_args(argv)

    if args.replay:
        from core import engine as engine_mod
        bundle = load_bundle(args.replay)
        print(f"Replaying: {describe(bundle)}\n")
        result, strategy, drift = replay(bundle, load_bars, engine_mod, get_strategy)
        show_report(summarise(result, bars_per_year(result.equity.index)),
                    f"{bundle['run']['symbol']} — {strategy}")
        if drift is None:
            print("\nBundle carried no equity curve to compare against.")
        elif abs(drift) < 0.01:
            print("\nReproduced exactly -- final equity matches the saved run.")
        else:
            saved = bundle_equity(bundle).iloc[-1]
            print(f"\nDRIFT: final equity differs by ${drift:,.2f} "
                  f"(saved ${saved:,.2f}). The data or the engine has changed "
                  "since this bundle was made.")
        return 0

    if args.list:
        print("Available strategies:\n")
        for name in available():
            cls = get_strategy(name)
            print(f"  {name}")
            if cls.description:
                print(f"      {cls.description}")
            for p in cls.params:
                print(f"      --{p.name.replace('_', '-'):<22} default {p.default}")
            print()
        return 0

    cls = get_strategy(args.strategy)

    # strategy params arrive as extra flags derived from the class definition
    sub = argparse.ArgumentParser(add_help=False)
    for p in cls.params:
        flag = f"--{p.name.replace('_', '-')}"
        if p.kind == "bool":
            sub.add_argument(flag, dest=p.name, action="store_true", default=p.default)
        else:
            sub.add_argument(flag, dest=p.name, type=float if p.kind == "float" else int,
                             default=p.default)
    known, unknown = sub.parse_known_args(rest)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}\n"
                     f"(run --list to see the flags {args.strategy!r} accepts)")
    params = {p.name: getattr(known, p.name) for p in cls.params}

    df = load_bars(args.symbol, args.start, args.end, args.timeframe, args.source)
    print(f"{args.symbol}: {len(df):,} {args.timeframe} bars, "
          f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")

    cfg = BacktestConfig(
        initial_cash=args.cash, position_size=args.position_size,
        slippage_bps=args.slippage_bps, commission_pct=args.commission_pct / 100.0,
        allow_short=args.allow_short, stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    ppy = bars_per_year(df.index)

    if args.sweep:
        grids = [parse_sweep(s) for s in args.sweep]
        rows = []
        for combo in itertools.product(*(v for _, v in grids)):
            trial = dict(params)
            trial.update(dict(zip((n for n, _ in grids), combo)))
            try:
                res = run_backtest(df, cls(**trial).generate_signals(df), cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {trial}: {exc}")
                continue
            s = summarise(res, ppy)
            rows.append({**trial, "return_%": s["Total return"] * 100,
                         "CAGR_%": s["CAGR"] * 100, "sharpe": s["Sharpe"],
                         "max_dd_%": s["Max drawdown"] * 100, "trades": s["Trades"]})
        table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        print("\nParameter sweep, best Sharpe first:\n")
        print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("\nNote: the top row of a sweep is the best fit to *this* history. "
              "Re-test it on a window you did not search over before believing it.")
        return 0

    strategy = cls(**params)
    result = run_backtest(df, strategy.generate_signals(df), cfg)
    show_report(summarise(result, ppy), f"{args.symbol} — {strategy}")

    if args.save_trades:
        result.trades.to_csv(args.save_trades, index=False)
        print(f"\nWrote {len(result.trades)} trades to {args.save_trades}")

    if args.save_bundle:
        path = save_bundle(build_bundle(
            symbol=args.symbol, start=args.start, end=args.end,
            timeframe=args.timeframe, source=args.source,
            strategy=strategy, result=result, stats=summarise(result, ppy),
        ), args.save_bundle)
        print(f"\nWrote bundle to {path}  (reload with --replay {path})")

    print("\nHypothetical results. Fills are modelled at the next bar's open and ignore "
          "gaps, partial fills, borrow costs and taxes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
