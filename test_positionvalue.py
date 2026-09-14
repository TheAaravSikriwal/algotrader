"""Pricing a live position three ways.

The case that matters is the third number. "Worth now minus what I put in" is
not what you would get, because selling a long means hitting the bid. On a
position a few dollars up, that difference decides the sign.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.positionvalue import Valuation, value


@dataclass
class FakePosition:
    symbol: str
    qty: float
    avg_price: float


def _long(bid=100.10, ask=100.20, qty=3, entry=100.00):
    return Valuation("SPY", qty, entry, bid, ask)


# -- the three numbers ----------------------------------------------------

def test_put_in_is_shares_times_the_fill_price():
    assert _long().put_in == pytest.approx(300.0)


def test_worth_now_marks_at_the_mid():
    v = _long(bid=100.10, ask=100.20)
    assert v.mid == pytest.approx(100.15)
    assert v.worth_now == pytest.approx(300.45)


def test_selling_a_long_hits_the_bid_not_the_mid():
    """The number people get wrong.

    Marked at the mid this position is up 45 cents. Sold, it is up 30 --
    the difference is half the spread, gone the moment you act.
    """
    v = _long(bid=100.10, ask=100.20, entry=100.00, qty=3)
    assert v.profit_at_mid == pytest.approx(0.45)
    assert v.profit_if_sold == pytest.approx(0.30)
    assert v.spread_cost == pytest.approx(0.15)


def test_the_spread_can_flip_a_small_profit_negative():
    """A position that looks green at the mid and is red when sold.

    This is the whole reason the panel shows the sold number.
    """
    v = _long(bid=99.99, ask=100.05, entry=100.00, qty=10)
    assert v.profit_at_mid > 0
    assert v.profit_if_sold < 0


def test_a_short_closes_by_lifting_the_ask():
    v = Valuation("QQQ", -2, 200.00, bid=199.80, ask=199.90)
    assert not v.is_long
    assert v.exit_price == pytest.approx(199.90)
    assert v.profit_if_sold == pytest.approx(0.20)   # 2 x (200.00 - 199.90)


def test_a_short_that_moved_against_you_loses():
    v = Valuation("QQQ", -2, 200.00, bid=201.00, ask=201.10)
    assert v.profit_if_sold == pytest.approx(-2.20)


def test_shares_is_always_positive_even_when_short():
    assert Valuation("X", -5, 10.0, 9.0, 9.1).shares == 5


# -- break-even -----------------------------------------------------------

def test_sitting_at_your_entry_price_is_down_not_flat():
    """The entry already paid the spread once. Getting back out pays it again.

    A panel that calls this 'flat' is why people hold a losing trade waiting
    for it to 'get back to even'.
    """
    v = _long(bid=100.00, ask=100.10, entry=100.00)
    assert v.profit_if_sold == pytest.approx(0.0)
    assert v.breakeven_price == pytest.approx(100.10)
    assert v.move_to_breakeven_pct > 0


def test_break_even_for_a_short_sits_below_the_entry():
    v = Valuation("X", -1, 100.00, bid=99.95, ask=100.05)
    assert v.breakeven_price == pytest.approx(99.90)


def test_profit_percent_is_against_what_you_put_in():
    v = _long(bid=101.00, ask=101.10, entry=100.00, qty=2)
    assert v.profit_if_sold == pytest.approx(2.0)
    assert v.profit_pct == pytest.approx(1.0)


# -- refusing a bad quote -------------------------------------------------

@pytest.mark.parametrize("bid,ask", [
    (0, 100.0), (100.0, 0), (100.20, 100.10), (-1, 100.0), (None, 100.0),
])
def test_an_unusable_quote_gives_no_valuation(bid, ask):
    """Better no number than a confident wrong one on a panel read at a glance."""
    assert value(FakePosition("SPY", 3, 100.0), bid, ask) is None


def test_no_position_gives_no_valuation():
    assert value(None, 100.0, 100.1) is None
    assert value(FakePosition("SPY", 0, 100.0), 100.0, 100.1) is None


def test_a_good_quote_prices_the_position():
    v = value(FakePosition("SPY", 3, 762.65), 762.66, 762.79)
    assert v is not None
    assert v.put_in == pytest.approx(2287.95)
    assert v.profit_if_sold == pytest.approx(0.03, abs=0.001)


# -- saying it the right way round ----------------------------------------

def _a_long():
    return Valuation("SPY", 2, 100.00, 100.50, 100.60)


def _a_short():
    return Valuation("QQQ", -1, 711.11, 710.91, 710.94)


def test_a_short_is_closed_by_buying_it_back():
    assert _a_short().close_verb == "buy back"
    assert _a_long().close_verb == "sell"


def test_closing_a_short_costs_money_rather_than_paying_you():
    """The page said 'You would receive $710.94' about money you are about to
    hand over. That inverts the trade in the reader's head."""
    assert _a_short().cash_direction == "You would pay"
    assert _a_long().cash_direction == "You would receive"


def test_closing_a_short_lifts_the_ask_not_the_bid():
    assert _a_short().closing_phrase == "buying back at the ask"
    assert _a_long().closing_phrase == "selling at the bid"


def test_the_number_behind_the_wording_was_always_right():
    """Worth pinning: the arithmetic was correct throughout, only the labels
    were wrong. Sold at 711.11, bought back at the ask 710.94."""
    v = _a_short()
    assert v.exit_price == pytest.approx(710.94)
    assert v.profit_if_sold == pytest.approx(0.17)
    assert v.if_sold_now == pytest.approx(710.94)


def test_a_short_breaks_even_when_the_ask_falls_not_when_the_bid_rises():
    """A short is closed by buying at the ask, so the ask is the price that
    has to come to you. Naming the bid points at the wrong number."""
    assert _a_short().breakeven_phrase == "ask falls to"
    assert _a_long().breakeven_phrase == "bid reaches"
