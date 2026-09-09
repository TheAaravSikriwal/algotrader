"""Run a strategy against a broker.

Defaults are deliberately timid: paper account, dry run, market-hours only.
You have to ask for anything riskier, twice.

    # see what it would do, without sending anything
    python run_live.py --symbols SPY --strategy "SMA crossover" --once

    # actually trade the paper account
    python run_live.py --symbols SPY,QQQ --strategy "SMA crossover" --execute --once

    # keep it running, checking every 5 minutes
    python run_live.py --symbols SPY --strategy "SMA crossover" --execute --interval 300

Stop it at any time by creating a file called HALT in this directory, or Ctrl-C.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.broker import BrokerError
from core.strategy import available, get_strategy
from core.trader import Halted, LiveConfig, Trader

from core.env import load_env

load_env()
import strategies  # noqa: F401


def setup_logging(log_dir: Path, verbose: bool):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "trader.log", encoding="utf-8")],
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="SPY", help="comma-separated, e.g. SPY,QQQ")
    p.add_argument("--strategy", default="SMA crossover")
    p.add_argument("--timeframe", default="1Day",
                   choices=["1Day", "1Hour", "15Min", "5Min", "1Min"])
    p.add_argument("--lookback", type=int, default=400,
                   help="bars fetched per cycle; must cover your slowest indicator")

    p.add_argument("--execute", action="store_true",
                   help="actually send orders (without this it is a dry run)")
    p.add_argument("--live", action="store_true",
                   help="use the LIVE account instead of paper. Requires --execute "
                        "and --i-understand-this-is-real-money")
    p.add_argument("--i-understand-this-is-real-money", dest="confirmed",
                   action="store_true", help=argparse.SUPPRESS)

    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--interval", type=int, default=300, help="seconds between cycles")
    p.add_argument("--max-cycles", type=int, default=None)

    p.add_argument("--position-size", type=float, default=0.95,
                   help="fraction of equity deployed across all symbols")
    p.add_argument("--max-daily-loss", type=float, default=3.0,
                   help="percent; flattens and halts past this")
    p.add_argument("--max-order-notional", type=float, default=None,
                   help="dollar cap on a single order. Default derives it from "
                        "account size (1.5x a full position)")
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--ignore-market-hours", action="store_true",
                   help="trade even when the market is closed (orders will queue)")
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
                             type=float if prm.kind == "float" else int, default=prm.default)
    known, unknown = sub.parse_known_args(rest)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    strategy = cls(**{prm.name: getattr(known, prm.name) for prm in cls.params})

    # --- the two gates in front of real money ---------------------------
    if args.live and not (args.execute and args.confirmed):
        parser.error(
            "--live also requires --execute and --i-understand-this-is-real-money.\n"
            "Before you use it: run the same strategy on the paper account for "
            "weeks, and read logs/activity.jsonl to confirm it did what you expected."
        )

    cfg = LiveConfig(
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        timeframe=args.timeframe,
        lookback=args.lookback,
        position_size=args.position_size,
        allow_short=args.allow_short,
        dry_run=not args.execute,
        max_daily_loss_pct=args.max_daily_loss,
        max_order_notional=args.max_order_notional,
        require_market_open=not args.ignore_market_hours,
    )

    setup_logging(Path(cfg.log_dir), args.verbose)
    logger = logging.getLogger("trader")

    if Path(cfg.halt_file).exists():
        logger.error("a %s file is present -- remove it before trading", cfg.halt_file)
        return 1

    try:
        from brokers.alpaca import AlpacaBroker
        broker = AlpacaBroker(paper=not args.live)
    except BrokerError as exc:
        logger.error("%s", exc)
        return 1

    try:
        account = broker.get_account()
    except Exception as exc:  # noqa: BLE001 -- the SDK raises its own error types
        detail = str(exc)
        if "unauthorized" in detail.lower() or "forbidden" in detail.lower():
            logger.error(
                "Alpaca rejected these credentials. Check that the keys in .env are "
                "%s keys -- paper and live keys are different and are not "
                "interchangeable.", "paper" if broker.is_paper else "live")
        else:
            logger.error("could not reach Alpaca: %s", detail)
        return 1

    mode = "DRY RUN" if cfg.dry_run else ("PAPER" if broker.is_paper else "*** LIVE ***")
    logger.info("%s | equity $%s | cash $%s | %s",
                mode, f"{account.equity:,.2f}", f"{account.cash:,.2f}", strategy)
    if account.blocked:
        logger.error("the broker has this account blocked for trading")
        return 1

    trader = Trader(broker, strategy, cfg)
    try:
        if args.once:
            summary = trader.run_once()
            if summary.get("skipped"):
                logger.info("nothing to do: %s", summary["reason"])
            else:
                for i in summary["intents"]:
                    logger.info("  %-6s signal %+.2f | holding %g -> target %g",
                                i["symbol"], i["signal"], i["current_shares"],
                                i["target_shares"])
                logger.info("%d order(s) this cycle", len(summary["orders"]))
        else:
            trader.run_forever(args.interval, args.max_cycles)
    except Halted as exc:
        logger.error("stopped: %s", exc)
        return 2

    logger.info("done -- see %s", Path(cfg.log_dir) / "activity.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
