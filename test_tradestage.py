"""Which stage a trade is at, and what it says about it.

The point of this module is that someone reading the page knows what is
happening without interpreting anything, so the tests are mostly about what
the words actually say.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.positionvalue import Valuation
from core.tradestage import (
    BLOCKED,
    CLOSED,
    HOLDING,
    ORDER_PLACED,
    SEQUENCE,
    WATCHING,
    describe,
)


@dataclass
class FakePosition:
    symbol: str
    qty: float
    avg_price: float


@dataclass
class FakeOrder:
    symbol: str
    qty: float
    side: str
    limit_price: float


def _held(qty=2, entry=100.00, bid=100.50, ask=100.60):
    pos = FakePosition("SPY", qty, entry)
    val = Valuation("SPY", qty, entry, bid, ask)
    return pos, val


# -- watching -------------------------------------------------------------

def test_nothing_happening_says_so_and_says_it_is_normal():
    """The most common outcome by far. If it reads like a failure, someone
    starts overriding the rule to make it do something."""
    s = describe()
    assert s.key == WATCHING
    assert s.title == "Watching"
    assert "selective" in s.next_step


# -- an order resting -----------------------------------------------------

def test_a_resting_order_says_nothing_has_been_bought_yet():
    """The confusion this whole module exists for: 'placed an order' and
    'bought something' are not the same, and the money panel stays empty."""
    s = describe(order=FakeOrder("QQQ", 1, "buy", 712.08), quote=(712.28, 712.34))
    assert s.key == ORDER_PLACED
    assert "Nothing has been bought yet" in s.next_step
    assert s.money_in == 0.0


def test_a_resting_order_shows_what_it_would_cost():
    s = describe(order=FakeOrder("QQQ", 2, "buy", 700.00), quote=(700.10, 700.20))
    assert "$1,400.00" in s.detail


def test_it_measures_the_gap_against_the_side_that_would_fill_you():
    """A buy fills when the ask comes down. Comparing against the bid would
    say 'almost there' on an order that is not close."""
    s = describe(order=FakeOrder("QQQ", 1, "buy", 712.00), quote=(712.28, 712.34))
    assert "0.34" in s.detail          # ask 712.34 - limit 712.00
    assert "down" in s.detail


def test_a_sell_order_needs_the_bid_to_come_up():
    s = describe(order=FakeOrder("QQQ", 1, "sell", 713.00), quote=(712.28, 712.34))
    assert "up" in s.detail


def test_it_says_when_the_order_gives_up():
    s = describe(order=FakeOrder("QQQ", 1, "buy", 712.00), quote=(712.1, 712.2),
                 expiry_bars=12, bar_minutes=5)
    assert "60 minutes" in s.next_step
    assert "no money changes hands" in s.next_step


# -- holding --------------------------------------------------------------

def test_holding_says_what_is_in_and_what_it_is_worth():
    pos, val = _held(qty=2, entry=100.00, bid=100.50, ask=100.60)
    s = describe(position=pos, valuation=val)
    assert s.key == HOLDING
    assert s.headline == "Holding 2 SPY"
    assert "$200.00 is in the market" in s.detail
    assert s.money_in == pytest.approx(200.0)


def test_holding_reports_the_sold_number_not_the_marked_one():
    """Selling means hitting the bid. A page that quotes the mid is telling
    you about a price you cannot get."""
    pos, val = _held(qty=2, entry=100.00, bid=100.50, ask=100.60)
    s = describe(position=pos, valuation=val)
    assert s.if_sold_now == pytest.approx(1.0)      # 2 x (100.50 - 100.00)
    assert "make $1.00" in s.detail


def test_a_losing_position_is_described_as_losing():
    pos, val = _held(qty=2, entry=101.00, bid=100.50, ask=100.60)
    s = describe(position=pos, valuation=val)
    assert "lose $1.00" in s.detail


def test_the_exit_plan_names_all_three_ways_out():
    pos, val = _held()
    s = describe(position=pos, valuation=val, flatten_at="15:55")
    joined = " ".join(s.exit_plan) + s.next_step
    assert "15:55" in joined
    assert "target" in joined and "stop" in joined
    assert "break even" in joined.lower()


def test_the_exit_plan_says_the_brackets_survive_the_app_closing():
    """Worth stating: the 15:55 flatten needs the app, the stop does not."""
    pos, val = _held()
    s = describe(position=pos, valuation=val, flatten_at="15:55")
    assert any("close the app" in p for p in s.exit_plan)


# -- precedence -----------------------------------------------------------

def test_holding_outranks_a_resting_order():
    """Someone holding something wants to know that first."""
    pos, val = _held()
    s = describe(position=pos, valuation=val,
                 order=FakeOrder("QQQ", 1, "buy", 700.0), quote=(700.1, 700.2))
    assert s.key == HOLDING


def test_a_resting_order_outranks_a_block():
    s = describe(order=FakeOrder("QQQ", 1, "buy", 700.0), quote=(700.1, 700.2),
                 blocks=["market data is 20 minutes behind"])
    assert s.key == ORDER_PLACED


def test_a_block_is_reported_when_there_is_nothing_else():
    s = describe(blocks=["market data is 20 minutes behind"])
    assert s.key == BLOCKED
    assert "20 minutes behind" in s.detail


def test_closed_is_only_reported_when_nothing_else_is_happening():
    assert describe(closed_today=True).key == CLOSED
    pos, val = _held()
    assert describe(position=pos, valuation=val, closed_today=True).key == HOLDING


# -- the stepper ----------------------------------------------------------

def test_each_stage_knows_where_it_sits_in_the_sequence():
    assert describe().index == SEQUENCE.index(WATCHING)
    s = describe(order=FakeOrder("Q", 1, "buy", 1.0), quote=(1.0, 1.1))
    assert s.index == SEQUENCE.index(ORDER_PLACED)


def test_a_block_sits_at_the_start_rather_than_off_the_scale():
    """It has to render somewhere on a four-step bar."""
    assert describe(blocks=["nope"]).index == 0


def test_direction_is_read_from_the_side_not_the_sign():
    """A buy needs the ask to come down; a sell needs the bid to come up.

    Both can have the same sign on the gap, so reading the sign alone printed
    "down" on every short order.
    """
    buy = describe(order=FakeOrder("Q", 1, "buy", 700.00), quote=(700.10, 700.20))
    sell = describe(order=FakeOrder("Q", 1, "sell", 701.00), quote=(700.10, 700.20))
    assert "(down)" in buy.detail
    assert "(up)" in sell.detail


def test_a_sell_is_not_described_as_putting_money_to_work():
    s = describe(order=FakeOrder("Q", 1, "sell", 701.00), quote=(700.10, 700.20))
    assert "to work" not in s.detail
