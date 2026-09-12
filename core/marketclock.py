"""When the market is actually open, and when to be flat.

A day-trading rule says things like "flat by 15:55". That is wrong roughly
nine days a year. The US equity market closes at 13:00 Eastern the day after
Thanksgiving, on Christmas Eve in some years, and on 3 July when it falls
beside the holiday -- and on those days a 15:55 flatten fires two hours after
the close, so the position is held overnight. For a strategy whose entire
premise is not holding overnight, that is the failure mode that matters.

So every time here is derived from the session's *actual* close rather than a
hardcoded 16:00, and an unknown date raises instead of guessing. Refusing to
trade a day you have no calendar for costs you one session. Assuming 16:00 on
a half-day costs you an unhedged overnight position in a strategy that never
agreed to take one.

Times are naive `America/New_York`, matching the bar index from `core.data`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data_cache" / "calendar.json"

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


class CalendarError(RuntimeError):
    """Raised rather than guessing a session's hours."""


@dataclass(frozen=True)
class Session:
    day: Date
    open: datetime
    close: datetime

    @property
    def is_half_day(self) -> bool:
        return self.close.time() < REGULAR_CLOSE

    @property
    def length(self) -> timedelta:
        return self.close - self.open

    def contains(self, ts: datetime) -> bool:
        """Open-inclusive, close-exclusive: at the close you are done, not trading."""
        return self.open <= ts < self.close

    def flatten_deadline(self, minutes_before_close: int = 5) -> datetime:
        """Latest moment to still be sending exit orders.

        Relative to this session's close, so it lands at 12:55 on a half-day
        and 15:55 on a normal one without the caller knowing which it is.
        """
        if minutes_before_close < 0:
            raise ValueError("minutes_before_close cannot be negative")
        return self.close - timedelta(minutes=minutes_before_close)

    def window(self, start: time, end: time) -> tuple[datetime, datetime]:
        """A wall-clock window, clipped to the session.

        The strategy's 10:30-15:30 window is 5.5 of a 6.5 hour day. Clipping
        matters on a half-day, where a naive 15:30 end is after the 13:00
        close -- the window would appear to be open for two hours that do not
        exist.
        """
        if start >= end:
            raise ValueError(f"window start {start} is not before end {end}")
        lo = max(datetime.combine(self.day, start), self.open)
        hi = min(datetime.combine(self.day, end), self.close)
        if lo >= hi:
            raise CalendarError(
                f"{self.day}: the {start:%H:%M}-{end:%H:%M} window does not "
                f"overlap a session that runs {self.open:%H:%M}-{self.close:%H:%M}")
        return lo, hi


class MarketCalendar:
    """Trading sessions, keyed by date.

    Built from Alpaca's calendar endpoint, which is the venue's own answer and
    therefore the one that counts. Cached to disk so a backtest does not need
    network access, and so a run is reproducible.
    """

    def __init__(self, sessions: dict[Date, Session] | None = None):
        self._sessions: dict[Date, Session] = dict(sessions or {})

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, day) -> bool:
        return _as_date(day) in self._sessions

    def is_trading_day(self, day) -> bool:
        return _as_date(day) in self._sessions

    def session(self, day) -> Session:
        """The session for `day`, or a refusal.

        Two different failures, deliberately distinguished: a weekend or
        holiday is a normal "no trading today", while a date outside the
        loaded range means the calendar cannot answer and the caller must not
        proceed on an assumption.
        """
        d = _as_date(day)
        found = self._sessions.get(d)
        if found is not None:
            return found
        if self._sessions:
            lo, hi = min(self._sessions), max(self._sessions)
            if not (lo <= d <= hi):
                raise CalendarError(
                    f"{d} is outside the loaded calendar ({lo} to {hi}); "
                    "load a wider range rather than assuming regular hours")
        raise CalendarError(f"{d} is not a trading day")

    def sessions(self) -> list[Session]:
        return [self._sessions[d] for d in sorted(self._sessions)]

    def half_days(self) -> list[Session]:
        return [s for s in self.sessions() if s.is_half_day]

    # -- construction -----------------------------------------------------

    @classmethod
    def from_rows(cls, rows) -> "MarketCalendar":
        out = {}
        for row in rows:
            day = _as_date(row["date"])
            out[day] = Session(
                day=day,
                open=_combine(day, row["open"]),
                close=_combine(day, row["close"]),
            )
        return cls(out)

    @classmethod
    def load(cls, path: Path | None = None) -> "MarketCalendar":
        path = Path(path or CACHE)
        if not path.exists():
            raise CalendarError(
                f"no cached calendar at {path}; run MarketCalendar.fetch() once "
                "while online, or pass sessions explicitly")
        return cls.from_rows(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or CACHE)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"date": s.day.isoformat(),
                 "open": s.open.strftime("%H:%M"),
                 "close": s.close.strftime("%H:%M")} for s in self.sessions()]
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return path

    @classmethod
    def fetch(cls, start, end, save: bool = True) -> "MarketCalendar":
        """Pull the venue's own calendar. Needs Alpaca credentials."""
        import os

        key = os.getenv("ALPACA_API_KEY_ID")
        secret = os.getenv("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise CalendarError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set")
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetCalendarRequest
        except ImportError as exc:
            raise CalendarError("alpaca-py is not installed") from exc

        client = TradingClient(key, secret, paper=True)
        days = client.get_calendar(GetCalendarRequest(
            start=pd.Timestamp(start).date(), end=pd.Timestamp(end).date()))
        cal = cls.from_rows([
            {"date": d.date, "open": d.open, "close": d.close} for d in days])
        if save:
            cal.save()
        return cal


# -- bar alignment --------------------------------------------------------

def next_bar_close(ts: datetime, minutes: int, session: Session | None = None) -> datetime:
    """The next bar boundary at or after `ts`.

    A loop that polls on its own schedule reads half-formed bars. Aligning to
    the boundary means the bar being acted on is closed and will not change.

    Boundaries are counted from the session open, not from midnight. For 5 and
    15 minute bars those agree because 09:30 is already on the grid; for a
    30 or 60 minute bar they do not, and counting from midnight would put the
    boundary at 10:00 when the first full hour actually ends at 10:30.
    """
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    anchor = session.open if session is not None else ts.replace(
        hour=0, minute=0, second=0, microsecond=0)
    if ts <= anchor:
        return anchor + timedelta(minutes=minutes)
    elapsed = (ts - anchor).total_seconds() / 60.0
    return anchor + timedelta(minutes=minutes * (int(elapsed // minutes) + 1))


def _as_date(day) -> Date:
    if isinstance(day, Date) and not isinstance(day, datetime):
        return day
    if isinstance(day, datetime):
        return day.date()
    return pd.Timestamp(day).date()


def _combine(day: Date, clock) -> datetime:
    if isinstance(clock, time):
        return datetime.combine(day, clock)
    if isinstance(clock, datetime):
        return datetime.combine(day, clock.time())
    hh, mm = str(clock).strip().split(":")[:2]
    return datetime.combine(day, time(int(hh), int(mm)))
