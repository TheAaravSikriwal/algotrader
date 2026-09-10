"""Cross-sectional backtests -- rank a universe, hold the best.

    python panel_backtest.py --universe megacap
    python panel_backtest.py --symbols AAPL,MSFT,NVDA,JNJ,KO,XOM,JPM,PG --top-n 3
    python panel_backtest.py --universe sectors --split 0.5

Deliberately runs each strategy at its **published default parameters** rather
than searching for the best ones. Searching a parameter grid and reporting the
winner is how a cross-sectional backtest flatters itself; the textbook 12-month
momentum window is a prior, not a fitted value. Use --split to hold out a period
you never look at while forming an opinion.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import bars_per_year, load_bars
from core.env import load_env
from core.metrics import equity_stats
from core.panel import Panel, PanelConfig, run_panel_backtest
from strategies.cross_sectional import available_xs, get_xs_strategy

load_env()

UNIVERSES = {
    "megacap": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                "TSLA", "JPM", "V", "WMT", "XOM"],
    "sectors": ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB",
                "XLU", "XLRE", "XLC"],
    "dow": ["AAPL", "MSFT", "JNJ", "KO", "XOM", "JPM", "PG", "WMT", "CVX",
            "MRK", "HD", "CAT", "IBM", "CSCO", "VZ", "MCD", "NKE", "AXP"],
    "factor": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
               "DBC", "VNQ"],
}


def describe(stats: dict, turnover: float = 0.0, tstat: float | None = None) -> str:
    line = (f"{stats.get('Total return', 0) * 100:>9.2f}% "
            f"{stats.get('CAGR', 0) * 100:>8.2f}% "
            f"{stats.get('Sharpe', 0):>8.2f} "
            f"{stats.get('Max drawdown', 0) * 100:>9.2f}% "
            f"{turnover:>9.2f}x")
    if tstat is None:
        return line + f"{'':>9}"
    flag = "*" if abs(tstat) >= 2.0 else " "
    return line + f"{tstat:>8.2f}{flag}"


def paired_tstat(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> float:
    """Is this strategy's daily edge over the benchmark bigger than its noise?

    A paired t-test on daily return differences. Comparing two Sharpes by eye
    is how a 0.16 gap that a coin flip would produce gets mistaken for skill --
    the same trap the news pipeline fell into before significance testing.
    """
    a = strategy_equity.pct_change().dropna()
    b = benchmark_equity.reindex(a.index).pct_change().dropna()
    diff = (a - b).dropna()
    if len(diff) < 30:
        return 0.0
    sd = diff.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(diff.mean() / (sd / (len(diff) ** 0.5)))


def evaluate(panel: Panel, name: str, cfg: PanelConfig, **params):
    strategy = get_xs_strategy(name)(**params)
    weights = strategy.generate_weights(panel)
    result = run_panel_backtest(panel, weights, cfg)
    stats = equity_stats(result.equity, bars_per_year(result.equity.index))
    return strategy, result, stats


def slice_panel(panel: Panel, start: int, end: int) -> Panel:
    return Panel(opens=panel.opens.iloc[start:end], highs=panel.highs.iloc[start:end],
                 lows=panel.lows.iloc[start:end], closes=panel.closes.iloc[start:end],
                 volumes=panel.volumes.iloc[start:end])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", choices=sorted(UNIVERSES), default="megacap")
    p.add_argument("--symbols", help="comma-separated, overrides --universe")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--start", default=str(date.today() - timedelta(days=365 * 10)))
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=0,
                   help="override every strategy's holding count")
    p.add_argument("--split", type=float, default=0.0,
                   help="fraction of history to use as an in-sample window; the "
                        "rest is reported separately as held-out")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--save", metavar="PATH")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else UNIVERSES[args.universe])

    bars = {}
    for symbol in symbols:
        try:
            df = load_bars(symbol, args.start, args.end, "1Day", args.source)
            if len(df) > 30:
                bars[symbol] = df
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: skipped ({exc})")

    if len(bars) < 3:
        print("Need at least 3 symbols with data for a cross-section.")
        return 1

    panel = Panel.from_bars(bars)
    print(f"Universe: {len(panel.symbols)} symbols, {len(panel):,} bars, "
          f"{panel.index[0]:%Y-%m-%d} to {panel.index[-1]:%Y-%m-%d}")

    for note in panel.survivorship_warning():
        print(f"\n  SURVIVORSHIP: {note}")

    benchmark_equity = None
    try:
        bench_bars = load_bars(args.benchmark, args.start, args.end, "1Day",
                               args.source)
        bench = bench_bars["close"].reindex(panel.index).ffill()
        benchmark_equity = args.cash * bench / bench.iloc[0]
    except Exception:  # noqa: BLE001
        print(f"  ({args.benchmark} benchmark unavailable)")

    cfg = PanelConfig(initial_cash=args.cash, slippage_bps=args.slippage_bps)

    windows = [("Full period", 0, len(panel))]
    if 0 < args.split < 1:
        cut = int(len(panel) * args.split)
        windows = [("In-sample", 0, cut), ("HELD OUT", cut, len(panel))]

    rows = []
    for label, start, end in windows:
        window = slice_panel(panel, start, end)
        if len(window) < 60:
            continue

        print(f"\n{'=' * 78}\n{label}: {window.index[0]:%Y-%m-%d} to "
              f"{window.index[-1]:%Y-%m-%d} ({len(window):,} bars)\n{'=' * 78}")
        print(f"{'strategy':<30}{'return':>10}{'CAGR':>9}{'sharpe':>9}"
              f"{'max DD':>10}{'turnover':>10}{'t vs EW':>9}")
        print("-" * 87)

        # the equal-weight basket is the yardstick a ranking has to clear
        ew_equity = None
        try:
            _, ew_result, _ = evaluate(window, "Equal weight all", cfg)
            ew_equity = ew_result.equity
        except Exception:  # noqa: BLE001
            pass

        for name in available_xs():
            params = {}
            if args.top_n > 0:
                cls = get_xs_strategy(name)
                if any(p.name == "top_n" for p in cls.params):
                    params["top_n"] = args.top_n
            try:
                _, result, stats = evaluate(window, name, cfg, **params)
            except Exception as exc:  # noqa: BLE001
                print(f"{name:<30}  failed: {exc}")
                continue

            tstat = (None if name == "Equal weight all" or ew_equity is None
                     else paired_tstat(result.equity, ew_equity))
            print(f"{name:<30}{describe(stats, result.annual_turnover, tstat)}")
            rows.append({"window": label, "strategy": name,
                         "t_vs_equal_weight": tstat,
                         "return_%": stats.get("Total return", 0) * 100,
                         "cagr_%": stats.get("CAGR", 0) * 100,
                         "sharpe": stats.get("Sharpe", 0),
                         "max_dd_%": stats.get("Max drawdown", 0) * 100,
                         "turnover": result.annual_turnover,
                         "trades": len(result.trades)})

        if benchmark_equity is not None:
            bench_window = benchmark_equity.iloc[start:end]
            bench_window = bench_window / bench_window.iloc[0] * args.cash
            bstats = equity_stats(bench_window, bars_per_year(bench_window.index))
            print("-" * 87)
            btstat = (None if ew_equity is None
                      else paired_tstat(bench_window, ew_equity))
            print(f"{args.benchmark + ' (buy and hold)':<30}"
                  f"{describe(bstats, 0.0, btstat)}")

    print(f"\n{'=' * 78}")
    print("Reading this honestly:")
    print("  * 'Equal weight all' is the benchmark that matters. Beating SPY may")
    print("    only mean your universe outperformed; beating equal weight means")
    print("    the RANKING added something.")
    print("  * Turnover is annual portfolio churn. A 5x strategy pays the spread")
    print("    five times over per year, and that is already in these numbers.")
    if 0 < args.split < 1:
        print("  * Compare the two windows. A strategy that leads in-sample and")
        print("    lags out-of-sample was fitted to the first window, by you or")
        print("    by whoever published the parameters.")
    print("  * These are all long-only, equal-weighted, on a universe you chose")
    print("    today. That last part is the survivorship problem and no amount of")
    print("    careful backtesting fixes it.")

    if args.save and rows:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\nWrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
