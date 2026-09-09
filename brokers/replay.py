"""A broker backed by historical bars, stepped forward one at a time.

This is not the backtester. The backtester tests a *strategy* -- signals in,
equity out. This tests the *system*: the real `Trader` loop, real order
submission, real position reconciliation, the risk rails and the kill switch,
all driven against history at whatever speed your CPU manages. Ten years takes
seconds instead of ten years.

The property that makes it worth anything: **`get_bars` cannot see past the
current bar.** A replay that leaks future bars tells you nothing, so the clock
is enforced in one place and tested directly.

Orders submitted after bar *t* closes fill at the open of bar *t+1*, matching
the backtest engine exactly -- which is why a replay and a backtest of the same
strategy should agree, and why a disagreement means one of them has a bug.
"""
from __future__ import annotations

from datetime import datetime
from itertools import count

import numpy as np
import pandas as pd

from core.broker import Account, Broker, BrokerError, Clock, Order, Position


class ReplayBroker(Broker):
    name = "replay"
    is_paper = True

    def __init__(self, bars: dict[str, pd.DataFrame], cash: float = 10_000.0,
                 slippage_bps: float = 5.0, commission_pct: float = 0.0,
                 warmup: int = 200):
        if not bars:
            raise BrokerError("replay needs at least one symbol of bars")

        self._bars = {k.upper(): v.sort_index() for k, v in bars.items()}

        timeline = sorted(set().union(*(df.index for df in self._bars.values())))
        self.timeline = pd.DatetimeIndex(timeline)
        if len(self.timeline) <= warmup + 1:
            raise BrokerError(
                f"only {len(self.timeline)} bars, which is not enough for a "
                f"{warmup}-bar warmup plus a replay")

        self._i = int(warmup)          # index of the bar whose close we are at
        self._start_i = self._i
        self._cash = float(cash)
        self._slip = slippage_bps / 10_000.0
        self._commission_pct = commission_pct
        self._positions: dict[str, Position] = {}
        self._pending: list[Order] = []
        self._ids = count(1)

        self.initial_cash = float(cash)
        self.fills: list[dict] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    # ---- the clock ------------------------------------------------------
    @property
    def now(self) -> pd.Timestamp:
        return self.timeline[self._i]

    @property
    def finished(self) -> bool:
        return self._i >= len(self.timeline) - 1

    @property
    def progress(self) -> float:
        span = len(self.timeline) - 1 - self._start_i
        return (self._i - self._start_i) / span if span > 0 else 1.0

    def advance(self) -> bool:
        """Step to the next bar, filling anything submitted at this one.

        Returns False once history runs out.
        """
        if self.finished:
            return False
        self._i += 1
        self._fill_pending()
        self.equity_curve.append((self.now, self.get_account().equity))
        return True

    # ---- price access, strictly no lookahead ----------------------------
    def _history(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        if symbol not in self._bars:
            raise BrokerError(f"no bars loaded for {symbol}")
        df = self._bars[symbol]
        return df.loc[:self.now]        # inclusive of the current bar, nothing after

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 300):
        return self._history(symbol).tail(limit).copy()

    def last_price(self, symbol: str) -> float:
        history = self._history(symbol)
        if history.empty:
            raise BrokerError(f"{symbol} has no bars at or before {self.now}")
        return float(history["close"].iloc[-1])

    def _next_open(self, symbol: str) -> float | None:
        """The price a market order submitted now would actually fill at."""
        df = self._bars[symbol.upper()]
        future = df.loc[df.index > self.now]
        return float(future["open"].iloc[0]) if len(future) else None

    # ---- account state --------------------------------------------------
    def get_account(self) -> Account:
        holdings = 0.0
        for symbol, pos in self._positions.items():
            try:
                holdings += pos.qty * self.last_price(symbol)
            except BrokerError:
                continue
        equity = self._cash + holdings
        return Account(cash=self._cash, equity=equity,
                       buying_power=max(equity, 0.0), is_paper=True)

    def get_positions(self) -> dict[str, Position]:
        out = {}
        for symbol, pos in self._positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            try:
                price = self.last_price(symbol)
            except BrokerError:
                continue
            out[symbol] = Position(symbol, pos.qty, pos.avg_price,
                                   pos.qty * price,
                                   (price - pos.avg_price) * pos.qty)
        return out

    def get_clock(self) -> Clock:
        return Clock(timestamp=self.now.to_pydatetime(), is_open=True)

    # ---- orders ---------------------------------------------------------
    def submit_order(self, symbol: str, qty: float, side: str,
                     order_type: str = "market", limit_price: float | None = None,
                     time_in_force: str = "day") -> Order:
        if qty <= 0:
            raise BrokerError(f"order qty must be positive, got {qty}")
        symbol = symbol.upper()
        if symbol not in self._bars:
            raise BrokerError(f"no bars loaded for {symbol}")

        order = Order(id=f"replay-{next(self._ids)}", symbol=symbol, qty=float(qty),
                      side=side.lower(), status="new",
                      submitted_at=self.now.to_pydatetime())
        self._pending.append(order)
        return order

    def get_open_orders(self) -> list[Order]:
        return list(self._pending)

    def cancel_all_orders(self) -> int:
        cancelled = len(self._pending)
        self._pending.clear()
        return cancelled

    def close_position(self, symbol: str) -> Order | None:
        pos = self._positions.get(symbol.upper())
        if pos is None or abs(pos.qty) < 1e-12:
            return None
        return self.submit_order(symbol, abs(pos.qty),
                                 "sell" if pos.qty > 0 else "buy")

    def _fill_pending(self):
        """Fill every queued order at this new bar's open."""
        if not self._pending:
            return

        for order in self._pending:
            df = self._bars[order.symbol]
            if self.now not in df.index:
                continue                      # symbol has no bar today; leave queued
            price = float(df.loc[self.now, "open"])
            signed = order.qty if order.side == "buy" else -order.qty
            exec_price = price * (1 + self._slip) if signed > 0 else price * (1 - self._slip)
            fee = abs(signed * exec_price) * self._commission_pct

            held = self._positions.get(order.symbol, Position(order.symbol, 0.0, 0.0))
            new_qty = held.qty + signed
            if held.qty == 0 or (held.qty > 0) == (signed > 0):
                basis = abs(held.qty) * held.avg_price + abs(signed) * exec_price
                avg = basis / abs(new_qty) if abs(new_qty) > 1e-12 else 0.0
            else:
                avg = held.avg_price if abs(new_qty) > 1e-12 else 0.0

            self._positions[order.symbol] = Position(order.symbol, new_qty, avg)
            self._cash -= signed * exec_price + fee

            order.status = "filled"
            order.filled_qty = order.qty
            order.filled_price = exec_price
            self.fills.append({
                "timestamp": self.now, "symbol": order.symbol, "side": order.side,
                "qty": order.qty, "price": exec_price, "fee": fee,
                "equity_after": self.get_account().equity,
            })

        self._pending = [o for o in self._pending if o.status != "filled"]

    # ---- results --------------------------------------------------------
    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        stamps, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(stamps), name="equity")

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fills, columns=["timestamp", "symbol", "side",
                                                 "qty", "price", "fee",
                                                 "equity_after"])
