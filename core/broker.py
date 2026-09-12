"""Broker abstraction.

Every venue implements this same interface, so the trading loop never knows or
cares which one it is talking to. That is the whole point: you exercise the
interface for weeks against Alpaca paper, and switching to real money changes a
config flag, not the logic. `brokers/fake.py` implements it too, which is how
the loop gets tested without a network.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Account:
    cash: float
    equity: float
    # Intraday buying power. Since FINRA retired the Pattern Day Trader rule and
    # Alpaca moved to its intraday margin framework (4 June 2026), this is the
    # authoritative "how much can I trade" number -- `daytrade_count` and
    # `pattern_day_trader` were removed from the API on 6 July 2026. Orders that
    # would create a margin deficit are now rejected pre-trade, so sizing has to
    # respect this rather than equity alone.
    buying_power: float
    currency: str = "USD"
    is_paper: bool = True
    blocked: bool = False          # trading_blocked / account_blocked at the venue


@dataclass
class Position:
    symbol: str
    qty: float                     # negative when short
    avg_price: float
    market_value: float = 0.0
    unrealized_pl: float = 0.0

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else ("short" if self.qty < 0 else "flat")


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: str                      # buy | sell
    status: str
    filled_qty: float = 0.0
    filled_price: float | None = None
    submitted_at: datetime | None = None

    @property
    def is_filled(self) -> bool:
        return self.status in {"filled", "closed"}


@dataclass
class Clock:
    timestamp: datetime
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    """The contract the trading loop depends on."""

    name: str = "broker"
    is_paper: bool = True

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Keyed by symbol. Symbols with no position are simply absent."""

    @abstractmethod
    def get_clock(self) -> Clock: ...

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Most recent `limit` bars, same shape as core.data.load_bars."""

    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str,
                     order_type: str = "market", limit_price: float | None = None,
                     time_in_force: str = "day",
                     stop_loss: float | None = None,
                     take_profit: float | None = None) -> Order:
        """Send one order.

        Passing `stop_loss` and/or `take_profit` submits a bracket: the exits
        are attached at the venue when the entry fills, and the two are
        one-cancels-other. That matters more than it looks. A stop that only
        exists inside a Python loop is not a stop -- it protects nothing if
        the process dies, the laptop sleeps, or the network drops while a
        position is open."""

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Orders still working at the venue.

        The loop needs these: it reconciles against *filled* positions, so an
        unfilled order from the previous cycle would otherwise look like the
        target was never reached, and it would order the same thing again.
        """

    @abstractmethod
    def cancel_all_orders(self) -> int: ...

    @abstractmethod
    def close_position(self, symbol: str) -> Order | None: ...

    def get_position(self, symbol: str) -> Position | None:
        return self.get_positions().get(symbol.upper())

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} paper={self.is_paper}>"
