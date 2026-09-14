"""Put everything back to a clean start, and say exactly what it did.

Four things, in this order, because the order matters:

  1. **Cancel working orders.** A resting order that fills while you are
     closing positions leaves you holding something you just sold.
  2. **Close every position.**
  3. **Clear the app's own files** -- cycle log, published state, queued
     instructions, fill history.
  4. **Report** what the account looks like afterwards.

It does not touch the account's cash balance. Alpaca can reset a paper account
to a starting figure, but that is irreversible, wipes the trade history the
fill measurements depend on, and may hand back a different starting balance
than the one you chose. If that is what you want, do it deliberately from
Alpaca's own dashboard.

    python reset.py            show what would happen, change nothing
    python reset.py --yes      actually do it
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from core.env import load_env  # noqa: E402

load_env()

from brokers.alpaca import AlpacaBroker  # noqa: E402
from core.broker import BrokerError  # noqa: E402

FILES = [
    REPO / "live",                                 # every instance's state
    REPO / "logs" / "daytrade_fills.jsonl",
    REPO / "logs" / "daytrader_state.json",
    REPO / "logs" / "plumbing_fills.jsonl",
]


def clearable() -> list[Path]:
    out: list[Path] = []
    for target in FILES:
        if target.is_dir():
            out.extend(sorted(p for p in target.glob("*")
                              if p.is_file() and p.suffix in {".json", ".jsonl"}))
        elif target.exists():
            out.append(target)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="actually do it; without this nothing changes")
    args = ap.parse_args()

    broker = AlpacaBroker(paper=True)
    account = broker.get_account()
    if not account.is_paper:
        print("Refusing: this is not a paper account.")
        return 1

    positions = broker.get_positions()
    orders = broker.get_open_orders()
    files = clearable()

    print(f"account      ${account.equity:,.2f} equity, "
          f"${account.cash:,.2f} cash")
    print(f"positions    {len(positions)}")
    for sym, p in positions.items():
        print(f"               {p.qty:g} {sym} @ {p.avg_price:,.2f}  "
              f"unrealised ${p.unrealized_pl:+,.2f}")
    print(f"orders       {len(orders)}")
    for o in orders:
        print(f"               {o.side} {o.qty:g} {o.symbol} "
              f"limit {o.limit_price}")
    print(f"files        {len(files)}")
    for f in files:
        print(f"               {f.relative_to(REPO)}  "
              f"({f.stat().st_size:,} bytes)")

    if not args.yes:
        print("\nNothing changed. Run again with --yes to do it.")
        return 0

    print("\nresetting...")

    # Cancel first. A resting order that fills midway through closing leaves
    # you holding something you have just sold.
    if orders:
        try:
            n = broker.cancel_all_orders()
            print(f"  cancelled {n} order(s)")
            time.sleep(1.5)
        except BrokerError as exc:
            print(f"  cancel failed: {exc}")

    for sym in list(positions):
        try:
            broker.close_position(sym)
            print(f"  closing {sym}")
        except BrokerError as exc:
            print(f"  could not close {sym}: {exc}")

    for _ in range(10):
        time.sleep(1.5)
        if not broker.get_positions():
            break

    for f in files:
        try:
            f.unlink()
            print(f"  cleared {f.relative_to(REPO)}")
        except OSError as exc:
            print(f"  could not clear {f.name}: {exc}")

    final = broker.get_account()
    left = broker.get_positions()
    still = broker.get_open_orders()
    print()
    print(f"account      ${final.equity:,.2f} equity, ${final.cash:,.2f} cash")
    print(f"positions    {len(left)}")
    print(f"orders       {len(still)}")
    clean = not left and not still
    print("\nClean." if clean else "\nSomething is still open -- check above.")
    print("The cash balance is untouched. Alpaca can reset a paper account to "
          "a starting figure, but that is irreversible and wipes the trade "
          "history, so it is left to you to do deliberately.")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
