"""Replay history through the real trading loop, fast.

    python replay.py --symbols SPY --strategy "SMA crossover" --fast 20 --slow 50
    python replay.py --symbols SPY,QQQ --strategy "MACD trend" --compare
    python replay.py --symbols SPY --strategy "SMA crossover" --stop-loss 5

Unlike a backtest, this drives `core.trader.Trader` -- the same object that
talks to Alpaca -- against historical bars. Order submission, position
reconciliation, the working-order guard, buying-power checks, the daily-loss
rail and the HALT file all execute for real. Ten years takes seconds.

`--compare` also runs a plain backtest of the same strategy and reports the
difference. They should agree closely; a large gap means the live loop and the
simulator disagree about something, which is exactly what you want to find
before real money is involved.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brokers.replay import ReplayBroker
from core.data import bars_per_year, load_bars
from core.engine import BacktestConfig, run_backtest
from core.env import load_env
from core.metrics import summarise
from core.strategy import get_strategy
from core.trader import Halted, LiveConfig, Trader

load_env()
import strategies  # noqa: F401

log = logging.getLogger("replay")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="SPY")
    p.add_argument("--strategy", default="SMA crossover")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--start", default=str(date.today() - timedelta(days=365 * 5)))
    p.add_argument("--end", default=str(date.today()))

    p.add_argument("--cash", type=float, default=10_000)
    p.add_argument("--position-size", type=float, default=0.95)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--warmup", type=int, default=250,
                   help="bars reserved for indicator warm-up before trading starts")
    p.add_argument("--max-daily-loss", type=float, default=100.0,
                   help="percent; the default effectively disables the rail so a "
                        "long replay is not cut short on its first bad day")
    p.add_argument("--stop-loss", type=float, default=0.0)
    p.add_argument("--allow-short", action="store_true")

    p.add_argument("--compare", action="store_true",
                   help="also run a plain backtest and report the difference")
    p.add_argument("--save-fills", metavar="PATH")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    cls = get_strategy(args.strategy)
    sub = argparse.ArgumentParser(add_help=False)
    for prm in cls.params:
        flag = f"--{prm.name.replace('_', '-')}"
        if prm.kind == "bool":
            sub.add_argument(flag, dest=prm.name, action="store_true", default=prm.default)
        else:
            sub.add_argument(flag, dest=prm.name,
                             type=float if prm.kind == "float" else int,
                             default=prm.default)
    known, unknown = sub.parse_known_args(rest)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    strategy = cls(**{prm.name: getattr(known, prm.name) for prm in cls.params})

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)-7s %(message)s")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bars = {}
    for symbol in symbols:
        try:
            bars[symbol] = load_bars(symbol, args.start, args.end, "1Day", args.source)
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: skipped ({exc})")
    if not bars:
        print("No data loaded.")
        return 1

    if not cls.can_run_on(next(iter(bars.values()))):
        print(f"'{cls.name}' needs {cls.requires_features}, which plain bars do "
              f"not carry. Use news_backtest.py for news strategies.")
        return 1

    broker = ReplayBroker(bars, cash=args.cash, slippage_bps=args.slippage_bps,
                          warmup=args.warmup)

    cfg = LiveConfig(
        symbols=symbols, dry_run=False, allow_short=args.allow_short,
        position_size=args.position_size,
        max_daily_loss_pct=args.max_daily_loss,
        require_market_open=True,
        # every replay bar is a completed bar, so there is no partial bar to drop
        use_closed_bars_only=False,
        flatten_on_halt=True,
        log_dir="logs/replay",
    )
    trader = Trader(broker, strategy, cfg)

    total = len(broker.timeline) - args.warmup - 1
    print(f"Replaying {', '.join(symbols)} | {strategy}")
    print(f"{total:,} bars from {broker.now.date()} to {broker.timeline[-1].date()}\n")

    started = time.time()
    cycles, halted = 0, None
    while True:
        try:
            trader.run_once()
        except Halted as exc:
            halted = str(exc)
            break
        except Exception as exc:  # noqa: BLE001
            log.exception("cycle failed: %s", exc)
        cycles += 1
        if not broker.advance():
            break

    elapsed = max(time.time() - started, 1e-9)
    equity = broker.equity_series()
    if equity.empty:
        print("Replay produced no equity curve -- check the warmup length.")
        return 1

    final = equity.iloc[-1]
    print(f"Replayed {cycles:,} cycles in {elapsed:,.1f}s "
          f"({cycles / elapsed:,.0f} bars/sec)")
    if halted:
        print(f"Stopped early: {halted}")

    stats = summarise_equity(equity, broker, args.cash)
    print("\n--- Replay through the live trading loop ---")
    for k, v in stats.items():
        print(f"  {k:<24}{v}")

    if args.compare:
        symbol = symbols[0]
        bt = run_backtest(
            bars[symbol].loc[broker.timeline[args.warmup]:],
            strategy.generate_signals(bars[symbol]).loc[broker.timeline[args.warmup]:],
            BacktestConfig(initial_cash=args.cash, position_size=args.position_size,
                           slippage_bps=args.slippage_bps,
                           allow_short=args.allow_short,
                           stop_loss_pct=args.stop_loss))
        bt_final = bt.equity.iloc[-1]
        gap = final - bt_final
        print("\n--- Plain backtest, same strategy and window ---")
        print(f"  {'Final equity':<24}${bt_final:,.2f}")
        print(f"  {'Difference':<24}${gap:,.2f} "
              f"({gap / args.cash * 100:+.2f}% of starting cash)")
        if abs(gap) / args.cash < 0.02:
            print("\n  The live loop and the simulator agree. Order handling, "
                  "reconciliation\n  and sizing all behave the same way against "
                  "real history.")
        else:
            print("\n  They DISAGREE by more than 2%. Worth investigating before "
                  "trusting either:\n  multi-symbol capital splitting, whole-share "
                  "rounding and the working-order\n  guard are the usual causes.")

    if args.save_fills:
        path = Path(args.save_fills)
        path.parent.mkdir(parents=True, exist_ok=True)
        broker.fills_frame().to_csv(path, index=False)
        print(f"\nWrote {len(broker.fills)} fills to {path}")

    return 0


def summarise_equity(equity: pd.Series, broker: ReplayBroker, cash: float) -> dict:
    from core.metrics import equity_stats

    stats = equity_stats(equity, bars_per_year(equity.index))
    fills = broker.fills_frame()
    return {
        "Final equity": f"${equity.iloc[-1]:,.2f}",
        "Total return": f"{stats.get('Total return', 0) * 100:+.2f}%",
        "CAGR": f"{stats.get('CAGR', 0) * 100:+.2f}%",
        "Sharpe": f"{stats.get('Sharpe', 0):.2f}",
        "Max drawdown": f"{stats.get('Max drawdown', 0) * 100:.2f}%",
        "Fills": f"{len(fills):,}",
        "Commissions paid": f"${fills['fee'].sum():,.2f}" if not fills.empty else "$0.00",
    }


if __name__ == "__main__":
    raise SystemExit(main())
