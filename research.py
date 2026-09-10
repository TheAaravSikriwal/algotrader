"""The research loop -- test, record, and let the evidence accumulate.

    # run a hypothesis and journal it
    python research.py test --strategy "Faber TAA" --symbols SPY --months 10 \
        --hypothesis "A 10-month trend filter beats buy and hold on SPY"

    # what has been tried, and what survives the corrected bar
    python research.py log
    python research.py summary

    # lock a specification to be scored on data that does not exist yet
    python research.py register --strategy "Faber TAA" --symbols SPY \
        --hypothesis "Holds up on genuinely unseen data" --months 10

    # score any registration that has aged enough
    python research.py score

The point of the ledger is the two things a one-off backtest cannot do: raise
the significance bar as your test count grows, and hold you to a prediction
made before the data existed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import bars_per_year, load_bars
from core.engine import BacktestConfig, run_backtest
from core.env import load_env
from core.journal import (Experiment, ForwardTest, Journal, benjamini_hochberg,
                          critical_t)
from core.metrics import equity_stats
from core.panel import Panel, PanelConfig, run_panel_backtest
from core.strategy import available, get_strategy

load_env()
import strategies  # noqa: F401
from strategies.cross_sectional import available_xs, get_xs_strategy  # noqa: E402


def paired_tstat(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> tuple[float, int]:
    a = strategy_equity.pct_change().dropna()
    b = benchmark_equity.reindex(a.index).pct_change().dropna()
    diff = (a - b).dropna()
    if len(diff) < 30:
        return 0.0, len(diff)
    sd = diff.std(ddof=1)
    if sd <= 0:
        return 0.0, len(diff)
    return float(diff.mean() / (sd / len(diff) ** 0.5)), len(diff)


def strategy_params(cls, extra: list[str]) -> dict:
    sub = argparse.ArgumentParser(add_help=False)
    for p in cls.params:
        flag = f"--{p.name.replace('_', '-')}"
        if p.kind == "bool":
            sub.add_argument(flag, dest=p.name, action="store_true", default=p.default)
        else:
            sub.add_argument(flag, dest=p.name,
                             type=float if p.kind == "float" else int, default=p.default)
    known, _ = sub.parse_known_args(extra)
    return {p.name: getattr(known, p.name) for p in cls.params}


def evaluate(name: str, symbols: list[str], start: str, end: str,
             params: dict, source: str, cash: float, slippage: float):
    """Run a strategy and return (metrics, tstat, observations)."""
    is_panel = name in available_xs()

    if is_panel:
        bars = {}
        for symbol in symbols:
            try:
                bars[symbol] = load_bars(symbol, start, end, "1Day", source)
            except Exception as exc:  # noqa: BLE001
                print(f"  {symbol}: skipped ({exc})")
        if len(bars) < 3:
            raise SystemExit("a cross-sectional test needs at least 3 symbols")

        panel = Panel.from_bars(bars)
        weights = get_xs_strategy(name)(**params).generate_weights(panel)
        # A dollar-neutral strategy asks for negative weights. Running it under
        # a long-only config silently clips the short leg and journals a number
        # that belongs to a different strategy, so infer the setting.
        needs_shorts = bool((weights < -1e-9).any().any())
        cfg = PanelConfig(initial_cash=cash, slippage_bps=slippage,
                          allow_short=needs_shorts)
        result = run_panel_backtest(panel, weights, cfg)
        benchmark = run_panel_backtest(
            panel, get_xs_strategy("Equal weight all")().generate_weights(panel), cfg
        ).equity
        equity = result.equity
        trades = len(result.trades)
        bench_label = "equal_weight"
    else:
        df = load_bars(symbols[0], start, end, "1Day", source)
        cfg = BacktestConfig(initial_cash=cash, slippage_bps=slippage)
        result = run_backtest(df, get_strategy(name)(**params).generate_signals(df), cfg)
        equity = result.equity
        benchmark = result.benchmark
        trades = len(result.trades)
        bench_label = "buy_and_hold"

    stats = equity_stats(equity, bars_per_year(equity.index))
    tstat, observations = paired_tstat(equity, benchmark)
    metrics = {
        "return_%": round(stats.get("Total return", 0) * 100, 2),
        "cagr_%": round(stats.get("CAGR", 0) * 100, 2),
        "sharpe": round(stats.get("Sharpe", 0), 3),
        "max_dd_%": round(stats.get("Max drawdown", 0) * 100, 2),
        "trades": trades,
        "benchmark_return_%": round(
            (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100, 2),
    }
    return metrics, tstat, observations, bench_label


def cmd_test(args, extra) -> int:
    journal = Journal()
    name = args.strategy
    cls = get_xs_strategy(name) if name in available_xs() else get_strategy(name)
    params = strategy_params(cls, extra)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    metrics, tstat, observations, bench = evaluate(
        name, symbols, args.start, args.end, params, args.source,
        args.cash, args.slippage_bps)

    experiment = Experiment(
        hypothesis=args.hypothesis or f"{name} beats its benchmark",
        strategy=name, params=params, symbols=symbols,
        universe=args.universe or "", start=args.start, end=args.end,
        benchmark=bench, window=args.window, metrics=metrics,
        tstat=round(tstat, 3), observations=observations, notes=args.notes or "")

    # the bar reflects every test BEFORE this one, plus this one
    bar_before = journal.bar(args.alpha)
    journal.record(experiment)
    bar_now = journal.bar(args.alpha)
    verdict = experiment.verdict(bar_now)

    print(f"\n{name} on {','.join(symbols)}  {args.start} to {args.end}")
    print(f"  params      {params}")
    for k, v in metrics.items():
        print(f"  {k:<20}{v}")
    print(f"  {'t vs ' + bench:<20}{tstat:+.2f}  (p = {experiment.pvalue:.4f}, "
          f"n = {observations:,})")
    print(f"\n  Tests in the journal: {journal.count()}")
    print(f"  Uncorrected bar |t| >= {critical_t(args.alpha):.2f}, "
          f"corrected |t| >= {bar_now:.2f}")
    if bar_now > bar_before:
        print(f"  (this test raised the bar from {bar_before:.2f})")
    print(f"  VERDICT: {verdict.upper()}")

    if verdict == "null" and abs(tstat) >= critical_t(args.alpha):
        print("\n  It clears the naive bar but not the corrected one. With "
              f"{journal.count()} tests\n  run, a result this size is what "
              "searching produces on its own.")
    return 0


def cmd_log(args, _extra) -> int:
    frame = Journal().frame(args.alpha)
    if frame.empty:
        print("Nothing recorded yet. Run: python research.py test --strategy ...")
        return 0
    columns = ["recorded", "strategy", "universe", "window", "return_%",
               "sharpe", "t", "p", "verdict", "survives_fdr"]
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    return 0


def cmd_summary(args, _extra) -> int:
    journal = Journal()
    summary = journal.summary(args.alpha)
    if not summary["tests_run"]:
        print("Nothing recorded yet.")
        return 0

    print("Research journal\n" + "-" * 60)
    for key, value in summary.items():
        print(f"  {key.replace('_', ' '):<40}{value}")

    print(f"\n  With {summary['tests_run']} tests at alpha {args.alpha}, roughly "
          f"{summary['expected_false_positives_uncorrected']} would clear the")
    print("  naive bar by chance alone. That is why the corrected bar exists.")

    frame = journal.frame(args.alpha)
    winners = frame[frame["verdict"] == "significant"]
    if winners.empty:
        print("\n  Nothing has beaten its benchmark at the corrected bar.")
    else:
        print("\n  Cleared the corrected bar:")
        for _, row in winners.iterrows():
            print(f"    {row['strategy']} on {row['universe']}: "
                  f"t {row['t']:+.2f}, {row['return_%']:+.2f}%")
        print("\n  Register these as forward tests before believing them:")
        print("    python research.py register --strategy ...")

    losers = frame[frame["verdict"] == "significant-negative"]
    if not losers.empty:
        print("\n  Significantly WORSE than benchmark:")
        for _, row in losers.iterrows():
            print(f"    {row['strategy']} on {row['universe']}: t {row['t']:+.2f}")
    return 0


def cmd_register(args, extra) -> int:
    journal = Journal()
    name = args.strategy
    is_panel = name in available_xs()
    cls = get_xs_strategy(name) if is_panel else get_strategy(name)
    params = strategy_params(cls, extra)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    test = ForwardTest(
        hypothesis=args.hypothesis or f"{name} keeps working on unseen data",
        strategy=name, params=params, symbols=symbols,
        kind="panel" if is_panel else "single",
        min_days=args.min_days, notes=args.notes or "")

    try:
        journal.register(test)
    except ValueError as exc:
        print(exc)
        return 1

    print(f"Registered forward test {test.id}")
    print(f"  {name}({params}) on {','.join(symbols)}")
    print(f"  registered {test.registered}, scoreable after {test.min_days} days "
          f"({date.fromisoformat(test.registered) + timedelta(days=test.min_days)})")
    print("\n  Scoring will use only bars after today. Nothing you have already")
    print("  searched can contaminate it -- which makes this the only test here")
    print("  that cannot be gamed.")
    return 0


def cmd_score(args, _extra) -> int:
    journal = Journal()
    open_tests = journal.open_forward_tests()
    if not open_tests:
        print("No open forward tests. Register one with: python research.py register")
        return 0

    print(f"{len(open_tests)} open forward test(s)\n" + "-" * 60)
    scored = 0
    for test in open_tests:
        elapsed = test.days_elapsed
        if not test.ready and not args.force:
            print(f"  {test.id}  {test.strategy:<28} {elapsed}/{test.min_days} days "
                  f"-- too early")
            continue

        start = test.registered
        end = date.today().isoformat()
        try:
            metrics, tstat, observations, bench = evaluate(
                test.strategy, test.symbols, start, end, test.params,
                args.source, args.cash, args.slippage_bps)
        except Exception as exc:  # noqa: BLE001
            print(f"  {test.id}  failed to score: {exc}")
            continue

        experiment = Experiment(
            hypothesis=test.hypothesis, strategy=test.strategy, params=test.params,
            symbols=test.symbols, start=start, end=end, benchmark=bench,
            window="forward", metrics=metrics, tstat=round(tstat, 3),
            observations=observations,
            notes=f"forward test {test.id}, registered {test.registered}")
        journal.record(experiment)
        journal.close_forward_test(test.id)
        scored += 1

        verdict = experiment.verdict(journal.bar(args.alpha))
        print(f"\n  {test.id}  {test.strategy} on {','.join(test.symbols)}")
        print(f"    {elapsed} days of genuinely unseen data "
              f"({observations:,} observations)")
        print(f"    return {metrics['return_%']:+.2f}%  vs benchmark "
              f"{metrics['benchmark_return_%']:+.2f}%")
        print(f"    t {tstat:+.2f}  ->  {verdict.upper()}")

    if scored:
        print(f"\nScored {scored} forward test(s) and recorded them in the journal.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
        sp.add_argument("--cash", type=float, default=100_000)
        sp.add_argument("--slippage-bps", type=float, default=5.0)
        sp.add_argument("--alpha", type=float, default=0.05)

    t = sub.add_parser("test", help="run a hypothesis and record it")
    t.add_argument("--strategy", required=True)
    t.add_argument("--symbols", default="SPY")
    t.add_argument("--universe", default="")
    t.add_argument("--hypothesis", default="")
    t.add_argument("--notes", default="")
    t.add_argument("--window", default="holdout",
                   choices=["in-sample", "holdout", "forward"])
    t.add_argument("--start", default=str(date.today() - timedelta(days=365 * 10)))
    t.add_argument("--end", default=str(date.today()))
    common(t)

    log = sub.add_parser("log", help="everything tested so far")
    common(log)

    s = sub.add_parser("summary", help="what survives the corrected bar")
    common(s)

    r = sub.add_parser("register", help="lock a spec for future scoring")
    r.add_argument("--strategy", required=True)
    r.add_argument("--symbols", default="SPY")
    r.add_argument("--hypothesis", default="")
    r.add_argument("--notes", default="")
    r.add_argument("--min-days", type=int, default=180)
    common(r)

    sc = sub.add_parser("score", help="score registrations that have aged enough")
    sc.add_argument("--force", action="store_true",
                    help="score even if too little time has passed (weakens the test)")
    common(sc)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, extra = build_parser().parse_known_args(argv)
    return {
        "test": cmd_test, "log": cmd_log, "summary": cmd_summary,
        "register": cmd_register, "score": cmd_score,
    }[args.command](args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
