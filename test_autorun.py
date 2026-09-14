"""When the background loop keeps going, and when it is allowed to stop.

The bug this file guards: a loop that stops when the trading window closes
leaves anything still open sitting there overnight. The window shuts at
15:30; the flatten is at 15:55; the market closes at 16:00. Stopping at any
of the first two with a position open is the failure.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from autorun import still_needed
from core.marketclock import MarketCalendar

DAY = date(2026, 9, 14)


def _cal():
    return MarketCalendar.from_rows([{"date": DAY, "open": "09:30", "close": "16:00"}])


class FakeBroker:
    def __init__(self, positions=None, raises=False):
        self._positions = positions or {}
        self._raises = raises

    def get_positions(self):
        if self._raises:
            from core.broker import BrokerError
            raise BrokerError("connection dropped")
        return self._positions


def _at(h, m):
    return datetime(2026, 9, 14, h, m)


# -- during the day -------------------------------------------------------

@pytest.mark.parametrize("h,m", [(9, 35), (10, 40), (13, 0), (15, 29)])
def test_it_keeps_running_through_the_session(h, m):
    keep, _ = still_needed(FakeBroker(), _cal(), _at(h, m))
    assert keep


def test_it_keeps_running_after_the_window_shuts_so_the_flatten_can_happen():
    """The window closes at 15:30 and the flatten is at 15:55. A loop that
    stops at the window is not there to do the thing it promised."""
    keep, _ = still_needed(FakeBroker(), _cal(), _at(15, 40))
    assert keep


def test_it_keeps_running_right_up_to_the_close_even_when_flat():
    keep, _ = still_needed(FakeBroker(), _cal(), _at(15, 59))
    assert keep


# -- after the close ------------------------------------------------------

def test_it_stops_once_the_session_is_over_and_nothing_is_open():
    keep, why = still_needed(FakeBroker(), _cal(), _at(16, 5))
    assert not keep
    assert "flat" in why


def test_it_does_not_stop_while_something_is_still_open():
    """The exact shape of the overnight bug: market shut, position there,
    loop walks away."""
    broker = FakeBroker({"QQQ": object()})
    keep, why = still_needed(broker, _cal(), _at(16, 5))
    assert keep
    assert "still open" in why


def test_a_broker_it_cannot_reach_counts_as_work_to_do():
    """Never resolve doubt in favour of going home. If we cannot tell whether
    anything is open, assume something is."""
    keep, _ = still_needed(FakeBroker(raises=True), _cal(), _at(16, 5))
    assert keep


# -- not a trading day ----------------------------------------------------

def test_it_stops_on_a_day_the_calendar_does_not_cover():
    keep, why = still_needed(FakeBroker(), _cal(), datetime(2026, 9, 19, 11, 0))
    assert not keep
    assert "not a trading day" in why


def test_a_closed_day_stops_even_with_a_position_showing():
    """Nothing can be done about it today, and a loop spinning on a Saturday
    is noise that hides a real one."""
    broker = FakeBroker({"QQQ": object()})
    keep, _ = still_needed(broker, _cal(), datetime(2026, 9, 19, 11, 0))
    assert not keep
