"""An in-memory broker for testing the trading loop without a network call.

Fills market orders instantly at the last close. It is deliberately naive --
its job is to prove the *loop* reconciles positions correctly, not to model
market microstructure. Realistic cost modelling lives in the backtest engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import count

import pandas as pd

from core.broker import Account, Broker, BrokerError, Clock, Order, Position


class FakeBroker(Broker):
    name = "fake"
    is_paper = True

    def __init__(self, bars: dict[str, pd.DataFrame], cash: float = 10_000.0,
                 market_open: bool = True):
        self._bars = {k.upper(): v for k, v in bars.items()}
        self._cash = float(cash)
        self._positions: dict[str, Position] = {}
        self._market_open = market_open
        self._ids = count(1)
        self.submitted: list[Order] = []      # every order the loop sent
        self.pending: list[Order] = []        # orders a test wants left "working"

    # ---- state ----------------------------------------------------------
    def last_price(self, symbol: str) -> float:
        return float(self._bars[symbol.upper()]["close"].iloc[-1])

    def get_account(self) -> Account:
        holdings = sum(p.qty * self.last_price(s) for s, p in self._positions.items())
        return Account(cash=self._cash, equity=self._cash + holdings,
                       buying_power=max(self._cash, 0.0), is_paper=True)

    def get_positions(self) -> dict[str, Position]:
        out = {}
        for s, p in self._positions.items():
            if abs(p.qty) < 1e-12:
                continue
            px = self.last_price(s)
            out[s] = Position(s, p.qty, p.avg_price, p.qty * px,
                              (px - p.avg_price) * p.qty)
        return out

    def get_clock(self) -> Clock:
        return Clock(timestamp=datetime.now(timezone.utc), is_open=self._market_open)

    def set_market_open(self, is_open: bool):
        self._market_open = is_open

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 300) -> pd.DataFrame:
        if symbol.upper() not in self._bars:
            raise BrokerError(f"no bars loaded for {symbol}")
        return self._bars[symbol.upper()].tail(limit).copy()

    # ---- orders ---------------------------------------------------------
    def submit_order(self, symbol: str, qty: float, side: str,
                     order_type: str = "market", limit_price: float | None = None,
                     time_in_force: str = "day") -> Order:
        if qty <= 0:
            raise BrokerError(f"order qty must be positive, got {qty}")

        symbol = symbol.upper()
        px = self.last_price(symbol)
        signed = qty if side.lower() == "buy" else -qty

        held = self._positions.get(symbol, Position(symbol, 0.0, 0.0))
        new_qty = held.qty + signed
        # weighted average only while adding to the same side
        if held.qty == 0 or (held.qty > 0) == (signed > 0):
            basis = abs(held.qty) * held.avg_price + abs(signed) * px
            avg = basis / abs(new_qty) if new_qty else 0.0
        else:
            avg = held.avg_price if abs(new_qty) > 1e-12 else 0.0

        self._positions[symbol] = Position(symbol, new_qty, avg)
        self._cash -= signed * px

        order = Order(id=f"fake-{next(self._ids)}", symbol=symbol, qty=qty,
                      side=side.lower(), status="filled", filled_qty=qty,
                      filled_price=px, submitted_at=datetime.now(timezone.utc))
        self.submitted.append(order)
        return order

    def get_open_orders(self) -> list[Order]:
        return list(self.pending)     # tests push into this to simulate a working order

    def cancel_all_orders(self) -> int:
        cancelled = len(self.pending)
        self.pending.clear()
        return cancelled

    def close_position(self, symbol: str) -> Order | None:
        pos = self._positions.get(symbol.upper())
        if pos is None or abs(pos.qty) < 1e-12:
            return None
        return self.submit_order(symbol, abs(pos.qty),
                                 "sell" if pos.qty > 0 else "buy")
