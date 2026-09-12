"""Session hours, half-days, and bar alignment.

The case that matters is the early close. A day-trading rule that flattens at
a hardcoded 15:55 holds overnight on a 13:00 day, which is the one thing the
strategy promised not to do.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time

import pytest

from core.marketclock import (
    CalendarError,
    MarketCalendar,
    Session,
    next_bar_close,
)

NORMAL = date(2026, 11, 25)        # Wednesday before Thanksgiving
HALF = date(2026, 11, 27)          # day after Thanksgiving -- 13:00 close
HOLIDAY = date(2026, 11, 26)       # Thanksgiving


def _cal() -> MarketCalendar:
    return MarketCalendar.from_rows([
        {"date": NORMAL, "open": "09:30", "close": "16:00"},
        {"date": HALF, "open": "09:30", "close": "13:00"},
    ])


# -- the half-day ---------------------------------------------------------

def test_half_day_is_recognised():
    assert _cal().session(HALF).is_half_day
    assert not _cal().session(NORMAL).is_half_day


def test_flatten_deadline_follows_the_real_close():
    """The whole point. 15:55 on a normal day, 12:55 on a half-day.

    A hardcoded 15:55 would fire two hours and fifty-five minutes after the
    half-day close.
    """
    cal = _cal()
    assert cal.session(NORMAL).flatten_deadline(5) == datetime(2026, 11, 25, 15, 55)
    assert cal.session(HALF).flatten_deadline(5) == datetime(2026, 11, 27, 12, 55)


def test_a_hardcoded_1555_would_be_after_the_half_day_close():
    """States the bug as an assertion so it cannot quietly come back."""
    half = _cal().session(HALF)
    assert datetime(2026, 11, 27, 15, 55) > half.close
    assert half.flatten_deadline(5) < half.close


def test_trading_window_is_clipped_to_a_short_session():
    """10:30-15:30 cannot run to 15:30 on a day that closes at 13:00."""
    lo, hi = _cal().session(HALF).window(time(10, 30), time(15, 30))
    assert lo == datetime(2026, 11, 27, 10, 30)
    assert hi == datetime(2026, 11, 27, 13, 0)


def test_trading_window_is_untouched_on_a_normal_session():
    lo, hi = _cal().session(NORMAL).window(time(10, 30), time(15, 30))
    assert (lo, hi) == (datetime(2026, 11, 25, 10, 30),
                        datetime(2026, 11, 25, 15, 30))


def test_window_starting_after_an_early_close_is_refused():
    """Better to raise than to hand back a window with no minutes in it."""
    with pytest.raises(CalendarError, match="does not overlap"):
        _cal().session(HALF).window(time(14, 0), time(15, 30))


def test_window_rejects_a_backwards_range():
    with pytest.raises(ValueError):
        _cal().session(NORMAL).window(time(15, 30), time(10, 30))


def test_window_starting_before_the_open_is_clipped_to_the_open():
    """Bars exist from 04:00, so a window can silently reach into pre-market.

    Without clipping, a 09:00 start would hand back 30 minutes of thin
    pre-market trading as if it were part of the session -- wide spreads, and
    exactly the hours the strategy's cost model was not measured on.
    """
    lo, hi = _cal().session(NORMAL).window(time(9, 0), time(15, 30))
    assert lo == datetime(2026, 11, 25, 9, 30)
    assert hi == datetime(2026, 11, 25, 15, 30)


def test_flatten_deadline_rejects_a_negative_offset():
    """A negative offset would put the deadline *after* the close.

    It reads like a small sign slip and produces the exact bug this module
    exists to prevent, so it is refused rather than computed.
    """
    with pytest.raises(ValueError):
        _cal().session(NORMAL).flatten_deadline(-5)


# -- refusing to guess ----------------------------------------------------

def test_a_holiday_is_not_a_trading_day():
    with pytest.raises(CalendarError, match="not a trading day"):
        _cal().session(HOLIDAY)
    assert not _cal().is_trading_day(HOLIDAY)


def test_a_date_outside_the_loaded_range_says_so_distinctly():
    """Different from a holiday, and the difference is the whole safety story.

    "Not a trading day" means stand down. "Outside the calendar" means the
    calendar cannot answer, and proceeding on an assumption of 16:00 is
    exactly the mistake this module exists to prevent.
    """
    with pytest.raises(CalendarError, match="outside the loaded calendar"):
        _cal().session(date(2030, 6, 3))


def test_an_empty_calendar_never_claims_a_day_is_tradeable():
    with pytest.raises(CalendarError):
        MarketCalendar().session(NORMAL)


def test_load_without_a_cache_refuses(tmp_path):
    with pytest.raises(CalendarError, match="no cached calendar"):
        MarketCalendar.load(tmp_path / "absent.json")


# -- containment ----------------------------------------------------------

@pytest.mark.parametrize("moment,inside", [
    (datetime(2026, 11, 25, 9, 29), False),
    (datetime(2026, 11, 25, 9, 30), True),    # the open counts
    (datetime(2026, 11, 25, 15, 59), True),
    (datetime(2026, 11, 25, 16, 0), False),   # at the close you are done
])
def test_session_containment_boundaries(moment, inside):
    assert _cal().session(NORMAL).contains(moment) is inside


def test_half_day_afternoon_is_outside_the_session():
    assert not _cal().session(HALF).contains(datetime(2026, 11, 27, 14, 0))


# -- bar alignment --------------------------------------------------------

@pytest.mark.parametrize("now,expected", [
    ("2026-11-25 10:31:00", "2026-11-25 10:35:00"),
    ("2026-11-25 10:34:59", "2026-11-25 10:35:00"),
    ("2026-11-25 10:35:00", "2026-11-25 10:40:00"),   # on a boundary -> the next one
])
def test_five_minute_alignment(now, expected):
    s = _cal().session(NORMAL)
    assert next_bar_close(datetime.fromisoformat(now), 5, s) == \
        datetime.fromisoformat(expected)


def test_hourly_bars_are_counted_from_the_open_not_midnight():
    """09:30 is not on the midnight hourly grid.

    Counting from midnight puts the first boundary at 10:00, which is 30
    minutes into a bar that has not closed -- the loop would act on a
    half-formed bar and the backtest would never show it.
    """
    s = _cal().session(NORMAL)
    assert next_bar_close(datetime(2026, 11, 25, 9, 45), 60, s) == \
        datetime(2026, 11, 25, 10, 30)


def test_alignment_before_the_open_lands_on_the_first_bar():
    s = _cal().session(NORMAL)
    assert next_bar_close(datetime(2026, 11, 25, 4, 0), 5, s) == \
        datetime(2026, 11, 25, 9, 35)


def test_alignment_rejects_a_nonpositive_interval():
    with pytest.raises(ValueError):
        next_bar_close(datetime(2026, 11, 25, 10, 0), 0)


# -- round trip -----------------------------------------------------------

def test_calendar_survives_save_and_load(tmp_path):
    path = tmp_path / "calendar.json"
    _cal().save(path)
    back = MarketCalendar.load(path)
    assert len(back) == 2
    assert back.session(HALF).close == datetime(2026, 11, 27, 13, 0)
    assert [s.day for s in back.half_days()] == [HALF]


def test_saved_calendar_is_plain_readable_json(tmp_path):
    path = tmp_path / "calendar.json"
    _cal().save(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert {"date", "open", "close"} <= set(rows[0])


def test_rows_accept_time_objects_as_well_as_strings():
    cal = MarketCalendar.from_rows([
        {"date": NORMAL, "open": time(9, 30), "close": time(13, 0)}])
    assert cal.session(NORMAL).is_half_day


def test_session_length_reflects_the_short_day():
    assert _cal().session(NORMAL).length.total_seconds() / 3600 == 6.5
    assert _cal().session(HALF).length.total_seconds() / 3600 == 3.5
