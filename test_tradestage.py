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


# -- the workflow boxes ---------------------------------------------------

def test_every_box_says_what_it_is_waiting_for_and_what_it_will_do():
    """The point of the whole workflow. A box with an empty tooltip is the
    'grey blocks with no legend' problem in a new shape."""
    for s in (describe(),
              describe(order=FakeOrder("Q", 1, "buy", 700.0), quote=(700.1, 700.2)),
              describe(position=_held()[0], valuation=_held()[1]),
              describe(closed_today=True),
              describe(blocks=["data is stale"])):
        assert len(s.steps) == 4
        for box in s.steps:
            assert box.title and box.label
            assert len(box.waiting_for) > 25, f"{box.key} has no trigger"
            assert len(box.plan) > 25, f"{box.key} has no plan"


def test_the_boxes_are_numbered_one_to_four_in_order():
    boxes = describe().steps
    assert [b.n for b in boxes] == [1, 2, 3, 4]
    assert [b.key for b in boxes] == SEQUENCE


def test_exactly_one_box_is_lit_and_it_is_the_stage_you_are_on():
    s = describe(order=FakeOrder("Q", 1, "buy", 700.0), quote=(700.1, 700.2))
    lit = [b for b in s.steps if b.is_current]
    assert len(lit) == 1
    assert lit[0].key == ORDER_PLACED


def test_boxes_before_the_current_one_are_done_and_later_ones_are_not():
    s = describe(position=_held()[0], valuation=_held()[1])
    by_key = {b.key: b.state for b in s.steps}
    assert by_key[WATCHING] == "done"
    assert by_key[ORDER_PLACED] == "done"
    assert by_key[HOLDING] == "current"
    assert by_key[CLOSED] == "upcoming"


def test_a_block_lights_the_first_box_because_it_is_still_watching():
    """Standing down is not a fifth box. A blocked loop is watching and
    refusing to act, which is box one."""
    s = describe(blocks=["market data is 20 minutes behind"])
    assert s.steps[0].is_current


def test_box_one_names_the_symbols_and_the_window_it_actually_watches():
    s = describe(symbols=("SPY", "QQQ"), window=("10:30", "15:30"))
    watching = s.steps[0].waiting_for
    assert "SPY" in watching and "QQQ" in watching
    assert "10:30" in watching and "15:30" in watching


def test_box_one_still_reads_as_a_sentence_with_no_symbols_configured():
    assert "symbol" in describe().steps[0].waiting_for


def test_box_two_quotes_the_same_gap_the_headline_does():
    """Two copies of the arithmetic is two chances to disagree, and the one
    on the tooltip is the one nobody would check."""
    s = describe(order=FakeOrder("QQQ", 1, "buy", 712.00), quote=(712.28, 712.34))
    assert "0.34" in s.detail and "0.34" in s.steps[1].waiting_for
    assert "down" in s.steps[1].waiting_for


def test_box_two_reads_the_direction_from_the_side_here_too():
    s = describe(order=FakeOrder("QQQ", 1, "sell", 713.00), quote=(712.28, 712.34))
    assert "up" in s.steps[1].waiting_for
    assert "bid" in s.steps[1].waiting_for


def test_box_two_promises_the_order_expires_rather_than_resting_forever():
    s = describe(expiry_bars=12, bar_minutes=5)
    assert "60 minutes" in s.steps[1].plan
    assert "cancels itself" in s.steps[1].plan


def test_box_three_quotes_the_break_even_price_when_something_is_held():
    pos, val = _held(qty=2, entry=100.00, bid=100.50, ask=100.60)
    s = describe(position=pos, valuation=val)
    assert f"{val.breakeven_price:,.2f}" in s.steps[2].waiting_for


def test_box_three_names_all_three_exits_before_you_are_in_it():
    """Hovering box three while still watching has to answer 'and then what?'
    -- that is the reason later boxes carry text at all."""
    three = describe().steps[2]
    assert "target" in three.waiting_for and "stop" in three.waiting_for
    assert "close" in three.waiting_for


def test_box_three_says_the_brackets_survive_the_app_closing():
    s = describe(flatten_at="15:55")
    assert "close the app" in s.steps[2].plan
    assert "15:55" in s.steps[2].plan


def test_box_four_does_not_claim_to_be_waiting_for_anything():
    assert "Nothing" in describe().steps[3].waiting_for


# -- shorts ---------------------------------------------------------------

def _shorted(qty=-1, entry=711.11, bid=711.02, ask=711.08):
    pos = FakePosition("QQQ", qty, entry)
    val = Valuation("QQQ", qty, entry, bid, ask)
    return pos, val


def test_a_short_is_not_described_as_having_been_bought():
    """The app sold QQQ short and the page said 'Bought at 711.11'. Someone
    learning has no way to catch that."""
    pos, val = _shorted()
    s = describe(position=pos, valuation=val)
    assert "Sold short at 711.11" in s.detail
    assert "Bought" not in s.detail


def test_a_short_is_closed_by_buying_it_back_not_by_selling():
    pos, val = _shorted()
    s = describe(position=pos, valuation=val)
    assert "buying it back" in s.detail
    assert "buys back" in s.next_step
    assert "selling this second" not in s.detail


def test_a_short_headline_says_short_rather_than_holding():
    pos, val = _shorted()
    assert describe(position=pos, valuation=val).headline == "Short 1 QQQ"


def test_box_three_says_a_short_is_closed_by_buying():
    pos, val = _shorted()
    s = describe(position=pos, valuation=val)
    assert "buy it back" in s.steps[2].waiting_for


def test_a_long_still_reads_the_way_it_did():
    pos, val = _held()
    s = describe(position=pos, valuation=val)
    assert s.headline == "Holding 2 SPY"
    assert "Bought at 100.00" in s.detail
    assert "selling this second" in s.detail
