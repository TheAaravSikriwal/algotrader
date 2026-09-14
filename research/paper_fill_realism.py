"""Can the paper account answer the fill-quality question at all?

The whole point of paper trading this strategy was to test one assumption:
the backtest fills a limit order whenever price trades through it, and real
markets do not -- you sit in a queue behind everyone who was there first, and
you get filled when someone wants the other side, which correlates with you
being wrong.

That test only works if the paper engine is *more* realistic than the
backtest. If it fills any limit the price touches, it implements the same
assumption, and paper trading confirms the backtest by construction.

Three probes, from most passive to most aggressive:

  1. a limit BEHIND the bid    -- should not fill until price comes down
  2. a limit AT the bid        -- the queue question: a real venue puts you
                                  last in line at that price
  3. a limit THROUGH the ask   -- must fill; the control

Paper only. Cancels everything it opens.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.env import load_env  # noqa: E402

load_env()

from alpaca.data.enums import DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest  # noqa: E402

from brokers.alpaca import AlpacaBroker  # noqa: E402
from core.broker import BrokerError  # noqa: E402

SYMBOL = "SPY"
WAIT_SECONDS = 25


def quote(client):
    q = client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=SYMBOL, feed=DataFeed.IEX))[SYMBOL]
    return float(q.bid_price), float(q.ask_price)


def probe(broker, client, label: str, price_of, note: str) -> dict:
    bid, ask = quote(client)
    price = round(price_of(bid, ask), 2)
    print(f"\n{label}")
    print(f"  bid {bid:.2f} / ask {ask:.2f}  ->  buy limit at {price:.2f}   ({note})")

    try:
        order = broker.submit_order(SYMBOL, 1, "buy", order_type="limit",
                                    limit_price=price, time_in_force="day")
    except BrokerError as exc:
        print(f"  submit failed: {exc}")
        return {"label": label, "filled": None, "error": str(exc)}

    filled_at, waited = None, 0.0
    touched_low = bid
    while waited < WAIT_SECONDS:
        time.sleep(2.5)
        waited += 2.5
        b, a = quote(client)
        touched_low = min(touched_low, b)
        pos = broker.get_position(SYMBOL)
        if pos and pos.qty:
            filled_at = float(pos.avg_price)
            break

    # Clean up before reporting, so a failure cannot leave anything behind.
    try:
        broker.cancel_all_orders()
        time.sleep(1)
        if broker.get_position(SYMBOL):
            broker.close_position(SYMBOL)
            time.sleep(2)
    except BrokerError as exc:
        print(f"  cleanup warning: {exc}")

    if filled_at:
        print(f"  FILLED at {filled_at:.2f} after {waited:.0f}s")
    else:
        print(f"  not filled in {waited:.0f}s (bid fell to {touched_low:.2f})")
    return {"label": label, "limit": price, "filled": filled_at,
            "bid_at_entry": bid, "lowest_bid": touched_low, "waited": waited}


def main():
    broker = AlpacaBroker(paper=True)
    acct = broker.get_account()
    if not acct.is_paper:
        raise SystemExit("paper only")
    client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY_ID"),
                                       os.getenv("ALPACA_API_SECRET_KEY"))

    print(f"{datetime.now():%H:%M:%S}  paper account ${acct.equity:,.2f}")
    results = [
        probe(broker, client, "1. BEHIND the bid",
              lambda b, a: b - 0.05, "2 cents below -- needs the market to come to it"),
        probe(broker, client, "2. AT the bid",
              lambda b, a: b, "the queue question"),
        probe(broker, client, "3. THROUGH the ask",
              lambda b, a: a + 0.05, "control -- must fill"),
    ]

    print()
    print("=" * 66)
    at_bid = results[1]
    control = results[2]
    if control["filled"] is None:
        print("The control did not fill, so nothing here is conclusive.")
    elif at_bid["filled"] is not None:
        print("A limit AT THE BID filled.")
        print()
        print("On a real venue that order joins the back of the queue at that")
        print("price and fills only after everyone ahead of it. Filling here")
        print("means the simulator is using roughly the same rule as the")
        print("backtest -- fill if price reaches the level -- so paper trading")
        print("would confirm the backtest's assumption by construction rather")
        print("than testing it.")
    else:
        print("A limit AT THE BID did not fill while the control did.")
        print()
        print("That is the behaviour you want: the simulator is modelling")
        print("something beyond 'price touched the level', so a paper fill rate")
        print("carries real information about queue position.")
    print("=" * 66)

    final = broker.get_account()
    print(f"\nback to ${final.equity:,.2f}, {len(broker.get_positions())} positions, "
          f"{len(broker.get_open_orders())} open orders")


if __name__ == "__main__":
    main()
