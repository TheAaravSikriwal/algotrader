"""What a position is worth, and what you would actually get for it.

Three numbers, and the third is the one people get wrong:

  * **Put in** -- what the shares cost, at the price they actually filled.
  * **Worth now** -- the same shares at the current mid.
  * **If you sold now** -- what would land in the account after selling.

The third is not the second minus the first. Selling a long means hitting the
**bid**, not the mid, so half the spread is gone the moment you act. On a
position a few dollars in profit that difference decides the sign, and a panel
showing "worth now minus cost" as profit is quietly optimistic exactly when it
matters most.

Shorts are the mirror: you close by lifting the **ask**.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Valuation:
    """A position priced three ways."""
    symbol: str
    qty: float                    # negative when short
    avg_price: float
    bid: float
    ask: float

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def shares(self) -> float:
        return abs(self.qty)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def put_in(self) -> float:
        """What the shares cost at the fill price."""
        return self.shares * self.avg_price

    @property
    def worth_now(self) -> float:
        """The same shares at the midpoint -- a fair mark, not an exit price."""
        return self.shares * self.mid

    @property
    def exit_price(self) -> float:
        """The price you would actually get: bid to sell, ask to cover."""
        return self.bid if self.is_long else self.ask

    @property
    def if_sold_now(self) -> float:
        """Proceeds of closing at the price the market is showing you."""
        return self.shares * self.exit_price

    @property
    def profit_if_sold(self) -> float:
        """The number that matters. Long: sell at the bid. Short: buy the ask."""
        if self.is_long:
            return self.shares * (self.exit_price - self.avg_price)
        return self.shares * (self.avg_price - self.exit_price)

    @property
    def profit_at_mid(self) -> float:
        """The flattering version, kept only to show the gap between them."""
        if self.is_long:
            return self.shares * (self.mid - self.avg_price)
        return self.shares * (self.avg_price - self.mid)

    @property
    def spread_cost(self) -> float:
        """What crossing the spread to get out takes off the table."""
        return abs(self.profit_at_mid - self.profit_if_sold)

    @property
    def profit_pct(self) -> float:
        return (self.profit_if_sold / self.put_in * 100.0) if self.put_in else 0.0

    @property
    def breakeven_price(self) -> float:
        """Where the bid has to be for selling to break even.

        Above the entry for a long, because the entry already paid the spread
        once. A position sitting exactly at its entry price is down, not flat.
        """
        return self.avg_price + (self.ask - self.bid) if self.is_long \
            else self.avg_price - (self.ask - self.bid)

    @property
    def move_to_breakeven_pct(self) -> float:
        if not self.avg_price:
            return 0.0
        gap = self.breakeven_price - self.exit_price
        return (gap if self.is_long else -gap) / self.avg_price * 100.0


def value(position, bid: float, ask: float) -> Valuation | None:
    """Price a broker position. None when the quote is unusable.

    A crossed or half-missing quote produces no valuation rather than a
    confident wrong one -- the whole point of these numbers is to be trusted
    at a glance.
    """
    if position is None or not getattr(position, "qty", 0):
        return None
    if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return Valuation(symbol=position.symbol, qty=float(position.qty),
                     avg_price=float(position.avg_price),
                     bid=float(bid), ask=float(ask))
