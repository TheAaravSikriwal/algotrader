"""Order construction in the Alpaca adapter.

These build the request objects and stop short of sending them. The point is
the guards: Alpaca rejects a malformed bracket with a 422, and a rejection
discovered at the venue during a live session is a rejection discovered at the
worst possible moment.
"""
from __future__ import annotations

import pytest

from core.broker import BrokerError


class _Captured(Exception):
    """Carries the request out of submit_order without sending it."""

    def __init__(self, req):
        self.req = req


def _broker(monkeypatch):
    pytest.importorskip("alpaca")
    from brokers.alpaca import AlpacaBroker

    b = AlpacaBroker.__new__(AlpacaBroker)      # no network, no credentials
    b.is_paper = True

    class FakeTrading:
        def submit_order(self, order_data):
            raise _Captured(order_data)

    b._trading = FakeTrading()
    return b


def _send(b, **kw):
    base = dict(symbol="SPY", qty=10, side="buy", order_type="limit",
                limit_price=100.0)
    base.update(kw)
    try:
        b.submit_order(**base)
    except _Captured as c:
        return c.req
    raise AssertionError("submit_order did not reach the venue call")


def test_a_plain_limit_carries_no_bracket(monkeypatch):
    req = _send(_broker(monkeypatch))
    assert getattr(req, "order_class", None) in (None, "")
    assert float(req.limit_price) == pytest.approx(100.0)


def test_a_bracket_attaches_both_exits(monkeypatch):
    req = _send(_broker(monkeypatch), stop_loss=99.0, take_profit=102.0)
    assert req.order_class is not None
    assert float(req.stop_loss.stop_price) == pytest.approx(99.0)
    assert float(req.take_profit.limit_price) == pytest.approx(102.0)


def test_prices_are_rounded_to_the_penny(monkeypatch):
    """Sub-penny prices are rejected outright for most US equities."""
    req = _send(_broker(monkeypatch), limit_price=100.123456,
                stop_loss=99.987654, take_profit=102.555555)
    assert float(req.limit_price) == pytest.approx(100.12)
    assert float(req.stop_loss.stop_price) == pytest.approx(99.99)
    assert float(req.take_profit.limit_price) == pytest.approx(102.56)


@pytest.mark.parametrize("side,stop,target", [
    ("buy", 101.0, 102.0),     # stop above a long entry
    ("buy", 99.0, 99.5),       # target below a long entry
    ("sell", 99.0, 98.0),      # stop below a short entry
    ("sell", 101.0, 101.5),    # target above a short entry
])
def test_exits_on_the_wrong_side_are_refused_before_sending(monkeypatch, side,
                                                            stop, target):
    """An inverted bracket is a sign slip, and it inverts the whole trade.

    Caught here rather than read back off a 422 mid-session.
    """
    with pytest.raises(BrokerError, match="wrong side"):
        _send(_broker(monkeypatch), side=side, stop_loss=stop, take_profit=target)


def test_a_correct_short_bracket_is_allowed(monkeypatch):
    req = _send(_broker(monkeypatch), side="sell", stop_loss=101.0,
                take_profit=98.0)
    assert float(req.stop_loss.stop_price) == pytest.approx(101.0)


def test_a_bracket_needs_a_supported_time_in_force(monkeypatch):
    with pytest.raises(BrokerError, match="day or gtc"):
        _send(_broker(monkeypatch), time_in_force="ioc", stop_loss=99.0)


def test_a_limit_order_without_a_price_is_refused(monkeypatch):
    with pytest.raises(BrokerError, match="limit_price"):
        _send(_broker(monkeypatch), limit_price=None)


@pytest.mark.parametrize("qty", [0, -5])
def test_a_nonpositive_quantity_is_refused(monkeypatch, qty):
    with pytest.raises(BrokerError, match="positive"):
        _send(_broker(monkeypatch), qty=qty)
