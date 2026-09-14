"""Where a trade is, in one word, and what happens next.

The app could already show every number about a position and still leave you
asking "so what is actually going on right now?". A cycle strip of grey blocks
answers "has it been running" and nothing else.

This answers the question directly. A trade is always at exactly one of five
stages, and each one knows what comes next and what would end it:

    WATCHING  ->  ORDER PLACED  ->  HOLDING  ->  CLOSED
        |               |
        |               +-- cancels itself if it never fills
        +-- most cycles stop here, and that is normal

Nothing here decides anything. It reads the broker and describes it, so the
description cannot drift from what is really in the account.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

WATCHING = "watching"
ORDER_PLACED = "order_placed"
HOLDING = "holding"
CLOSED = "closed"
BLOCKED = "blocked"

#: In the order they happen, for drawing a stepper.
SEQUENCE = [WATCHING, ORDER_PLACED, HOLDING, CLOSED]

TITLES = {
    WATCHING: "Watching",
    ORDER_PLACED: "Order placed",
    HOLDING: "Holding",
    CLOSED: "Closed",
    BLOCKED: "Standing down",
}


@dataclass
class Stage:
    """What is happening, in plain words, with the numbers behind it."""
    key: str
    headline: str = ""
    detail: str = ""
    next_step: str = ""
    money_in: float = 0.0
    worth_now: float = 0.0
    if_sold_now: float = 0.0
    exit_plan: list[str] = field(default_factory=list)
    symbol: str = ""

    @property
    def title(self) -> str:
        return TITLES.get(self.key, self.key)

    @property
    def index(self) -> int:
        """Position in the sequence, for the stepper. Blocked sits at the start."""
        return SEQUENCE.index(self.key) if self.key in SEQUENCE else 0


def _money(x: float) -> str:
    return f"${x:,.2f}"


def describe(position=None, order=None, valuation=None, quote=None,
             blocks: list[str] | None = None, flatten_at=None,
             expiry_bars: int = 12, bar_minutes: int = 5,
             closed_today: bool = False) -> Stage:
    """Work out the stage from what the broker actually reports.

    Order of checks matters: a held position outranks a resting order, which
    outranks a block. Someone holding something wants to know about that
    first, even if the loop is also standing down for another symbol.
    """
    blocks = blocks or []
    flat_by = f" It closes anything still open at {flatten_at}." if flatten_at else ""

    # -- holding something ------------------------------------------------
    if position is not None and getattr(position, "qty", 0) and valuation:
        v = valuation
        long = v.qty > 0
        plan = []
        if flatten_at:
            plan.append(f"Closes automatically at {flatten_at} — nothing is "
                        f"held overnight.")
        plan.append(f"Break even once the {'bid reaches' if long else 'ask falls to'} "
                    f"{v.breakeven_price:,.2f} "
                    f"({abs(v.move_to_breakeven_pct):.3f}% away).")
        plan.append("The stop and target sit at the venue, so they still work "
                    "if you close the app.")
        return Stage(
            key=HOLDING, symbol=v.symbol,
            headline=f"Holding {abs(v.qty):g} {v.symbol}",
            detail=(f"Bought at {v.avg_price:,.2f}. "
                    f"{_money(v.put_in)} is in the market. "
                    f"Now worth {_money(v.worth_now)}; selling this second "
                    f"would {'make' if v.profit_if_sold >= 0 else 'lose'} "
                    f"{_money(abs(v.profit_if_sold))}."),
            next_step=("It sells when the target or the stop is hit, or at the "
                       "end of the day, whichever comes first."),
            money_in=v.put_in, worth_now=v.worth_now,
            if_sold_now=v.profit_if_sold, exit_plan=plan)

    # -- an order resting -------------------------------------------------
    if order is not None:
        limit = float(getattr(order, "limit_price", 0) or 0)
        qty = float(getattr(order, "qty", 0) or 0)
        buying = str(getattr(order, "side", "")).lower() == "buy"
        cost = limit * qty
        bid, ask = (quote or (0.0, 0.0))
        facing = ask if buying else bid
        gap = (facing - limit) if buying else (limit - bid)
        minutes = expiry_bars * bar_minutes
        detail = (f"{'Buying' if buying else 'Selling'} {qty:g} "
                  f"{order.symbol} at {limit:,.2f} or better. "
                  + (f"That would put {_money(cost)} to work."
                     if buying else
                     f"That is {_money(cost)} of {order.symbol}."))
        if facing:
            # Which way the market has to move depends on the side, not on
            # the sign alone: a buy needs the ask to come DOWN to the limit,
            # a sell needs the bid to come UP to it. Reading the sign without
            # the side prints the wrong direction on every short.
            needs_down = (gap > 0) if buying else (gap < 0)
            detail += (f" The market is at {facing:,.2f}, so it needs to move "
                       f"{abs(gap):,.2f} "
                       f"({'down' if needs_down else 'up'}) before you are "
                       f"filled.")
        return Stage(
            key=ORDER_PLACED, symbol=order.symbol, headline="Order placed, waiting",
            detail=detail,
            next_step=(f"Nothing has been bought yet. If it has not filled "
                       f"within about {minutes} minutes it cancels itself, "
                       f"and no money changes hands.{flat_by}"),
            money_in=0.0,
            exit_plan=["A stop and a target go live the moment it fills."])

    # -- stood down -------------------------------------------------------
    if blocks:
        return Stage(
            key=BLOCKED, headline="Standing down", detail=blocks[0],
            next_step="Nothing will be bought until that clears.")

    if closed_today:
        return Stage(
            key=CLOSED, headline="Closed out",
            detail="The position is finished and you are back to cash.",
            next_step="The loop keeps watching for the next setup.")

    return Stage(
        key=WATCHING, headline="Watching",
        detail="No setup on the bar that just closed.",
        next_step=("Most checks find nothing — the rule is selective. It "
                   "places an order only when its exact pattern appears."))
