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


@dataclass(frozen=True)
class Step:
    """One box in the workflow, including the ones not reached yet.

    A stepper that only describes the stage you are on tells you where you
    are and not where you are going. Every box carries its own "what has to
    happen" and "what the app will do", so hovering a later one answers
    "and then what?" without waiting to get there.
    """
    n: int                 #: 1-based, for the numbered badge
    key: str
    title: str
    label: str             #: one short line under the title
    waiting_for: str       #: what has to happen to leave this stage
    plan: str              #: what the app does while it is here
    state: str             #: "done" | "current" | "upcoming"

    @property
    def is_current(self) -> bool:
        return self.state == "current"


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
    steps: list["Step"] = field(default_factory=list)

    @property
    def title(self) -> str:
        return TITLES.get(self.key, self.key)

    @property
    def index(self) -> int:
        """Position in the sequence, for the stepper. Blocked sits at the start."""
        return SEQUENCE.index(self.key) if self.key in SEQUENCE else 0


def _money(x: float) -> str:
    return f"${x:,.2f}"


#: What each box says when nothing more specific is known. Every one is
#: phrased as a thing that has to *happen*, not a state -- "waiting for X"
#: with no X is the thing this whole module exists to stop.
LABELS = {
    WATCHING: "checking each bar",
    ORDER_PLACED: "limit resting, nothing bought",
    HOLDING: "money in the market",
    CLOSED: "back to cash",
}


def build_steps(current: str, symbols=(), window=("10:30", "15:30"),
                flatten_at=None, expiry_minutes: int = 60,
                order_facts: dict | None = None,
                valuation=None) -> list[Step]:
    """The four boxes, each knowing its own trigger and plan.

    `current` is the stage key the account is actually at. BLOCKED is not one
    of the four, so it shows as sitting at the first box -- which is true: a
    blocked loop is watching and refusing to act.
    """
    where = SEQUENCE.index(current) if current in SEQUENCE else 0
    names = " or ".join(str(x).upper() for x in symbols) or "a watched symbol"
    w0, w1 = window
    f = order_facts or {}

    # 1 -- watching
    watch_for = (f"A three-bar gap on {names}: the first bar's high finishing "
                 f"below the third bar's low, leaving a price range nothing "
                 f"traded in. It only counts between {w0} and {w1}.")
    watch_plan = ("Reads one 5-minute bar as it closes and checks the last "
                  "three. Almost every check finds nothing — that is the rule "
                  "being selective, not the app being idle.")

    # 2 -- order placed
    if f.get("limit"):
        side = "buy" if f["buying"] else "sell"
        order_for = (f"A {side} limit is resting at {f['limit']:,.2f}. "
                     f"It fills when the {f['facing_name']} reaches it — "
                     f"that is {f['facing']:,.2f} now, so {f['gap']:,.2f} "
                     f"{f['direction']} to go.")
    else:
        order_for = ("The market to come to the limit price. The order rests "
                     "at the midpoint of the gap and waits to be hit, rather "
                     "than paying the spread to cross.")
    order_plan = (f"Nothing has been bought yet and no money has moved. If it "
                  f"has not filled within about {expiry_minutes} minutes it "
                  f"cancels itself and the loop goes back to watching.")

    # 3 -- holding
    if valuation is not None:
        closing = "sell" if valuation.qty > 0 else "buy it back"
        hold_for = (f"One of three exits: the target, the stop, or the "
                    f"end-of-day close — all three {closing}. You are "
                    f"break-even once the "
                    f"{'bid reaches' if valuation.qty > 0 else 'ask falls to'} "
                    f"{valuation.breakeven_price:,.2f}.")
    else:
        hold_for = ("One of three exits: the profit target at 2R, the stop, "
                    "or the end-of-day close — whichever comes first.")
    hold_plan = ("The stop and target sit at the venue the moment the order "
                 "fills, so they still work if you close the app."
                 + (f" Anything still open is closed at {flatten_at}."
                    if flatten_at else ""))

    # 4 -- closed
    closed_for = "Nothing — this trade is finished and the money is back as cash."
    closed_plan = ("The result is written to the fill log, and the loop "
                   "returns to box 1 to look for the next setup.")

    content = {
        WATCHING: (watch_for, watch_plan),
        ORDER_PLACED: (order_for, order_plan),
        HOLDING: (hold_for, hold_plan),
        CLOSED: (closed_for, closed_plan),
    }

    out = []
    for i, key in enumerate(SEQUENCE):
        waiting, plan = content[key]
        out.append(Step(
            n=i + 1, key=key, title=TITLES[key], label=LABELS[key],
            waiting_for=waiting, plan=plan,
            state="current" if i == where else ("done" if i < where else "upcoming")))
    return out


def describe(position=None, order=None, valuation=None, quote=None,
             blocks: list[str] | None = None, flatten_at=None,
             expiry_bars: int = 12, bar_minutes: int = 5,
             closed_today: bool = False, symbols=(),
             window=("10:30", "15:30")) -> Stage:
    """Work out the stage from what the broker actually reports.

    Order of checks matters: a held position outranks a resting order, which
    outranks a block. Someone holding something wants to know about that
    first, even if the loop is also standing down for another symbol.
    """
    blocks = blocks or []
    flat_by = f" It closes anything still open at {flatten_at}." if flatten_at else ""
    minutes = expiry_bars * bar_minutes

    # Worked out once. The headline and box 2 of the workflow both quote these,
    # and a second copy of the arithmetic is a second chance to disagree.
    facts: dict = {}
    if order is not None:
        limit = float(getattr(order, "limit_price", 0) or 0)
        qty = float(getattr(order, "qty", 0) or 0)
        buying = str(getattr(order, "side", "")).lower() == "buy"
        bid, ask = (quote or (0.0, 0.0))
        facing = ask if buying else bid
        gap = (facing - limit) if buying else (limit - bid)
        # Which way the market has to move depends on the side, not on the
        # sign alone: a buy needs the ask to come DOWN to the limit, a sell
        # needs the bid to come UP to it. Reading the sign without the side
        # printed the wrong direction on every short.
        needs_down = (gap > 0) if buying else (gap < 0)
        facts = {"limit": limit, "qty": qty, "buying": buying,
                 "cost": limit * qty, "facing": facing,
                 "facing_name": "ask" if buying else "bid",
                 "gap": abs(gap), "direction": "down" if needs_down else "up"}

    def _with_steps(stage: Stage) -> Stage:
        """Attach the four boxes. Every branch goes through here, so the
        workflow can never describe a different stage than the headline."""
        stage.steps = build_steps(
            stage.key, symbols=symbols, window=window, flatten_at=flatten_at,
            expiry_minutes=minutes, order_facts=facts, valuation=valuation)
        return stage

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
        return _with_steps(Stage(
            key=HOLDING, symbol=v.symbol,
            headline=(f"Holding {abs(v.qty):g} {v.symbol}" if long else
                      f"Short {abs(v.qty):g} {v.symbol}"),
            # A short was sold, not bought, and it is closed by buying it
            # back. Calling both sides "bought" and "sold" is the kind of
            # wrong that someone learning has no way to catch.
            detail=(f"{'Bought' if long else 'Sold short'} at "
                    f"{v.avg_price:,.2f}. "
                    f"{_money(v.put_in)} is in the market. "
                    f"Now worth {_money(v.worth_now)}; "
                    f"{'selling' if long else 'buying it back'} this second "
                    f"would {'make' if v.profit_if_sold >= 0 else 'lose'} "
                    f"{_money(abs(v.profit_if_sold))}."),
            next_step=(f"It {'sells' if long else 'buys back'} when the target "
                       f"or the stop is hit, or at the end of the day, "
                       f"whichever comes first."),
            money_in=v.put_in, worth_now=v.worth_now,
            if_sold_now=v.profit_if_sold, exit_plan=plan))

    # -- an order resting -------------------------------------------------
    if order is not None:
        limit, qty, buying = facts["limit"], facts["qty"], facts["buying"]
        detail = (f"{'Buying' if buying else 'Selling'} {qty:g} "
                  f"{order.symbol} at {limit:,.2f} or better. "
                  + (f"That would put {_money(facts['cost'])} to work."
                     if buying else
                     f"That is {_money(facts['cost'])} of {order.symbol}."))
        if facts["facing"]:
            detail += (f" The market is at {facts['facing']:,.2f}, so it needs "
                       f"to move {facts['gap']:,.2f} ({facts['direction']}) "
                       f"before you are filled.")
        return _with_steps(Stage(
            key=ORDER_PLACED, symbol=order.symbol, headline="Order placed, waiting",
            detail=detail,
            next_step=(f"Nothing has been bought yet. If it has not filled "
                       f"within about {minutes} minutes it cancels itself, "
                       f"and no money changes hands.{flat_by}"),
            money_in=0.0,
            exit_plan=["A stop and a target go live the moment it fills."]))

    # -- stood down -------------------------------------------------------
    if blocks:
        return _with_steps(Stage(
            key=BLOCKED, headline="Standing down", detail=blocks[0],
            next_step="Nothing will be bought until that clears."))

    if closed_today:
        return _with_steps(Stage(
            key=CLOSED, headline="Closed out",
            detail="The position is finished and you are back to cash.",
            next_step="The loop keeps watching for the next setup."))

    return _with_steps(Stage(
        key=WATCHING, headline="Watching",
        detail="No setup on the bar that just closed.",
        next_step=("Most checks find nothing — the rule is selective. It "
                   "places an order only when its exact pattern appears.")))
