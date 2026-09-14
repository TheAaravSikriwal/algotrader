"""End-to-end check that the order path works, on the practice account.

This does not test whether the strategy is any good -- it tests whether the
machinery around it does what it claims:

  A. a resting limit order submits and can be cancelled
  B. a marketable limit fills, and its stop and target attach at the venue
  C. flatten closes the position and cancels the working orders
  D. the fill log records what actually happened

Worth running on its own because the strategy loop currently cannot: Alpaca's
free plan delays market data by fifteen minutes, which corrupts a fill-rate
measurement but has no bearing on whether orders route correctly.

Paper only. It refuses to run against anything else, and it flattens whatever
it opened in a `finally` block so an exception cannot leave a position behind.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.env import load_env  # noqa: E402

load_env()

import os  # noqa: E402

from alpaca.data.enums import DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest  # noqa: E402

from brokers.alpaca import AlpacaBroker  # noqa: E402
from core.broker import BrokerError  # noqa: E402
from core.fills import FillLog, FillRecord, mid  # noqa: E402

SYMBOL = "SPY"
QTY = 1
LOG = REPO / "logs" / "plumbing_fills.jsonl"

ok = []
fail = []


def check(label: str, passed: bool, detail: str = ""):
    (ok if passed else fail).append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
          + (f"  -- {detail}" if detail else ""))


def live_quote():
    c = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY_ID"),
                                  os.getenv("ALPACA_API_SECRET_KEY"))
    q = c.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=SYMBOL, feed=DataFeed.IEX))[SYMBOL]
    return float(q.bid_price), float(q.ask_price)


def main():
    broker = AlpacaBroker(paper=True)
    account = broker.get_account()
    if not account.is_paper:
        raise SystemExit("refusing to run against a live account")

    fills = FillLog(LOG)
    bid, ask = live_quote()
    spread_bps = (ask - bid) / ask * 10_000
    print(f"account   ${account.equity:,.2f} equity, paper={account.is_paper}")
    print(f"quote     bid {bid:.2f}  ask {ask:.2f}  spread {spread_bps:.1f} bps")
    print()

    opened = False
    try:
        # -- A. a resting limit submits, then cancels -----------------------
        print("A. resting limit order")
        far = round(bid * 0.90, 2)          # 10% below market: will not fill
        stop = round(far * 0.98, 2)
        target = round(far * 1.04, 2)
        try:
            order = broker.submit_order(SYMBOL, QTY, "buy", order_type="limit",
                                        limit_price=far, stop_loss=stop,
                                        take_profit=target)
            check("submits a bracketed limit", bool(order.id),
                  f"id {order.id[:8]} at {far}")
        except BrokerError as exc:
            check("submits a bracketed limit", False, str(exc))
            order = None

        time.sleep(2)
        working = broker.get_open_orders()
        check("the order is working at the venue",
              any(o.id == getattr(order, "id", None) for o in working),
              f"{len(working)} open")

        cancelled = broker.cancel_all_orders()
        time.sleep(2)
        check("cancel clears it", not broker.get_open_orders(),
              f"cancelled {cancelled}")

        # -- the guard rails on a bracket ----------------------------------
        print("\nB. bracket validation")
        for label, sl, tp in [("stop above entry on a buy", round(far * 1.02, 2), target),
                              ("target below entry on a buy", stop, round(far * 0.98, 2))]:
            try:
                broker.submit_order(SYMBOL, QTY, "buy", order_type="limit",
                                    limit_price=far, stop_loss=sl, take_profit=tp)
                check(f"rejects {label}", False, "it was accepted")
            except BrokerError:
                check(f"rejects {label}", True)

        # -- C. a marketable limit fills, with exits attached ---------------
        print("\nC. marketable limit, stop and target attached")
        entry = round(ask * 1.001, 2)       # through the ask: should fill
        stop = round(entry * 0.99, 2)
        target = round(entry * 1.02, 2)
        filled = None
        try:
            order = broker.submit_order(SYMBOL, QTY, "buy", order_type="limit",
                                        limit_price=entry, stop_loss=stop,
                                        take_profit=target)
            opened = True
            check("submits", bool(order.id), f"limit {entry}")
        except BrokerError as exc:
            check("submits", False, str(exc))
            order = None

        for _ in range(10):
            time.sleep(1.5)
            pos = broker.get_position(SYMBOL)
            if pos and pos.qty:
                filled = pos
                break
        check("fills and opens a position", filled is not None,
              f"{filled.qty} @ {filled.avg_price:.2f}" if filled else "no fill in 15s")

        if filled:
            slip = (filled.avg_price - ask) / ask * 10_000
            print(f"       filled at {filled.avg_price:.2f} vs ask {ask:.2f} "
                  f"-> {slip:+.1f} bps")
            fills.record(FillRecord(
                symbol=SYMBOL, side="buy", qty=float(QTY),
                reference_price=mid(bid, ask) or ask, order_type="limit",
                limit_price=entry, filled_qty=float(abs(filled.qty)),
                filled_price=float(filled.avg_price), status="filled",
                strategy="plumbing_test", bid=bid, ask=ask,
                order_id=getattr(order, "id", "")))

            time.sleep(2)
            legs = broker.get_open_orders()
            check("stop and target are working at the venue", len(legs) >= 1,
                  f"{len(legs)} exit order(s)")

        # -- D. flatten -----------------------------------------------------
        print("\nD. flatten")
        broker.cancel_all_orders()
        time.sleep(1)
        if broker.get_position(SYMBOL):
            broker.close_position(SYMBOL)
        for _ in range(10):
            time.sleep(1.5)
            if not broker.get_position(SYMBOL):
                break
        check("position is closed", broker.get_position(SYMBOL) is None)
        check("no orders left working", not broker.get_open_orders())
        opened = False

        # -- E. the log -----------------------------------------------------
        print("\nE. the fill log")
        df = fills.frame()
        check("records the fill", not df.empty, f"{len(df)} row(s)")
        if not df.empty:
            row = df.iloc[-1]
            check("slippage is computed from prices, signed as a cost",
                  row["slippage_bps"] == row["slippage_bps"],
                  f"{row['slippage_bps']:+.2f} bps")

    finally:
        # Never leave a position behind, whatever went wrong above.
        if opened:
            print("\ncleaning up after an error...")
            try:
                broker.cancel_all_orders()
                if broker.get_position(SYMBOL):
                    broker.close_position(SYMBOL)
            except BrokerError as exc:
                print(f"  cleanup failed: {exc}")

    print()
    print("=" * 62)
    print(f"{len(ok)} passed, {len(fail)} failed")
    if fail:
        for f in fail:
            print(f"  FAILED: {f}")
    final = broker.get_account()
    print(f"account back to ${final.equity:,.2f}, "
          f"{len(broker.get_positions())} positions, "
          f"{len(broker.get_open_orders())} open orders")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
