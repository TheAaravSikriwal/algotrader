"""The bar index must be Eastern wall-clock time, not UTC.

This is the bug the independent audit found: ``_normalise`` dropped the
timezone without converting, so a session filter written the obvious way
selected pre-market plus the first two hours and looked correct while doing it.

These tests build bars whose true session is known, so there is a right answer
to check against rather than a plausible-looking number.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.data import _normalise, session


def _bars(index) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {"open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
         "close": [1.0] * n, "volume": [100.0] * n},
        index=index,
    )


def test_utc_aware_bars_become_eastern():
    # 13:30 UTC in July is 09:30 in New York -- the opening bar.
    idx = pd.date_range("2026-07-01 13:30", periods=3, freq="1min", tz="UTC")
    out = _normalise(_bars(idx))
    assert out.index.tz is None, "the public contract is a tz-naive index"
    assert str(out.index[0]) == "2026-07-01 09:30:00"


def test_eastern_aware_bars_survive_the_round_trip():
    # yfinance hands back intraday bars already stamped America/New_York.
    # The old code converted these *to* UTC, which is how 09:30 became 13:30.
    idx = pd.date_range("2026-07-01 09:30", periods=3, freq="1min",
                        tz="America/New_York")
    out = _normalise(_bars(idx))
    assert str(out.index[0]) == "2026-07-01 09:30:00"


def test_the_offset_follows_daylight_saving():
    """A fixed four-hour shift is wrong for half the year.

    Same 14:30 UTC stamp: 10:30 in July (EDT, UTC-4) but 09:30 in January
    (EST, UTC-5). Code that subtracted a constant would put one of these in
    the wrong place, and the winter one would land exactly on the open --
    plausible enough to go unnoticed.
    """
    summer = _normalise(_bars(pd.DatetimeIndex(["2026-07-01 14:30"], tz="UTC")))
    winter = _normalise(_bars(pd.DatetimeIndex(["2026-01-02 14:30"], tz="UTC")))
    assert str(summer.index[0]) == "2026-07-01 10:30:00"
    assert str(winter.index[0]) == "2026-01-02 09:30:00"


def test_naive_daily_bars_are_not_shifted():
    """yfinance daily bars arrive tz-naive and must stay on their own date.

    Converting these would move every bar to 20:00 on the *previous* day,
    silently shifting the entire history back by one session.
    """
    idx = pd.DatetimeIndex(["2026-08-20", "2026-08-21", "2026-08-24"])
    out = _normalise(_bars(idx))
    assert list(out.index) == list(idx)


def test_alpaca_daily_bars_land_on_midnight():
    """Alpaca stamps daily bars 05:00 UTC in winter, 04:00 in summer.

    In Eastern both are midnight, which is what makes them joinable with
    yfinance dailies and with anything else keyed by date.
    """
    idx = pd.DatetimeIndex(["2026-01-02 05:00", "2026-07-01 04:00"], tz="UTC")
    out = _normalise(_bars(idx))
    assert (out.index.normalize() == out.index).all()
    assert [str(t.date()) for t in out.index] == ["2026-01-02", "2026-07-01"]


def test_session_keeps_the_regular_hours_and_drops_the_rest():
    """The real failure was quantitative: the wrong filter kept ~36% of volume.

    Here the whole extended day is one bar per minute, so "how many bars
    survived" is a direct count of whether the window is right. 09:30 through
    15:59 inclusive is 390 minutes.
    """
    idx = pd.date_range("2026-07-01 04:00", "2026-07-01 19:59", freq="1min",
                        tz="America/New_York")
    out = session(_normalise(_bars(idx)))
    assert len(out) == 390
    assert str(out.index[0]) == "2026-07-01 09:30:00"
    assert str(out.index[-1]) == "2026-07-01 15:59:00"


def test_session_keeps_78_five_minute_bars():
    idx = pd.date_range("2026-07-01 04:00", "2026-07-01 19:55", freq="5min",
                        tz="America/New_York")
    assert len(session(_normalise(_bars(idx)))) == 78


def test_session_leaves_daily_bars_alone():
    """Midnight-stamped bars fall outside any session window.

    Filtering them would return nothing, so `session` has to recognise daily
    data and pass it through instead.
    """
    idx = pd.DatetimeIndex(["2026-08-20", "2026-08-21", "2026-08-24"])
    out = session(_normalise(_bars(idx)))
    assert len(out) == 3


@pytest.mark.parametrize("stamp,keep", [
    ("2026-07-01 09:29", False),   # last pre-market minute
    ("2026-07-01 09:30", True),    # opening bar
    ("2026-07-01 15:59", True),    # last bar of continuous trading
    ("2026-07-01 16:00", False),   # closing-auction stub, not a traded minute
])
def test_session_boundaries(stamp, keep):
    idx = pd.DatetimeIndex([stamp], tz="America/New_York")
    out = session(_normalise(_bars(idx)))
    assert (len(out) == 1) is keep
