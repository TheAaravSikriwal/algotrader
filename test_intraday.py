"""Day-trading backtests of ordinary strategies.

The failure modes worth guarding are the ones that silently flatter a result:
acting on the bar the signal was derived from, carrying a position overnight,
trading the expensive first minutes, and charging one side of the cost instead
of two.
"""
from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from core.intraday import IntradayConfig, combine, run
from core.marketclock import MarketCalendar
from core.strategy import REGISTRY, Param, Strategy, register

DAY = date(2026, 9, 14)
HALF = date(2026, 11, 27)


def _cal():
    return MarketCalendar.from_rows([
        {"date": DAY, "open": "09:30", "close": "16:00"},
        {"date": date(2026, 9, 15), "open": "09:30", "close": "16:00"},
        {"date": HALF, "open": "09:30", "close": "13:00"},
    ])


def _session(day="2026-09-14", n=78, start=100.0, step=0.0, seed=1):
    """One full session of five-minute bars, 09:30 to 15:55."""
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min")
    rng = np.random.default_rng(seed)
    close = start + np.arange(n) * step + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "open": close - 0.01, "high": close + 0.05,
        "low": close - 0.05, "close": close,
        "volume": 10_000.0}, index=idx)


@register
class _AlwaysLong(Strategy):
    name = "ZZ always long"
    description = "test fixture"
    params: list[Param] = []

    def generate_signals(self, df):
        return pd.Series(1.0, index=df.index)


@register
class _CloseOracle(Strategy):
    """Knows the next bar's close. Should NOT profit: by the time the order
    fills at the next open, that information has already been consumed."""
    name = "ZZ close oracle"
    description = "test fixture"
    params: list[Param] = []

    def generate_signals(self, df):
        return (df["close"].shift(-1) > df["close"]).astype(float)


@register
class _FillOracle(Strategy):
    """Knows the two prices it will actually trade at. Should profit."""
    name = "ZZ fill oracle"
    description = "test fixture"
    params: list[Param] = []

    def generate_signals(self, df):
        return (df["open"].shift(-2) > df["open"].shift(-1)).astype(float)


# -- the session boundary -------------------------------------------------

def test_no_position_survives_the_close():
    """A day trade that sleeps overnight is not a day trade."""
    bars = pd.concat([_session("2026-09-14", step=0.02),
                      _session("2026-09-15", start=105.0, step=0.02)])
    r = run(bars, "ZZ always long", "T", cfg=IntradayConfig(), calendar=_cal())
    assert r.sessions == 2
    for _, t in r.trades.iterrows():
        assert t["entry_ts"].date() == t["exit_ts"].date()


def test_the_flatten_deadline_moves_on_a_half_day():
    bars = _session("2026-11-27", n=78)
    r = run(bars, "ZZ always long", "T", cfg=IntradayConfig(), calendar=_cal())
    if not r.trades.empty:
        assert r.trades["exit_ts"].max().time() <= time(13, 0)


def test_trading_is_confined_to_the_window():
    """The open is excluded because it is where the spread is widest."""
    bars = _session(step=0.02)
    r = run(bars, "ZZ always long", "T", cfg=IntradayConfig(), calendar=_cal())
    assert not r.trades.empty
    assert r.trades["entry_ts"].min().time() >= time(10, 30)
    assert r.trades["exit_ts"].max().time() <= time(15, 55)


def test_a_wider_window_trades_earlier():
    bars = _session(step=0.02)
    wide = IntradayConfig(window_start=time(9, 35), window_end=time(15, 55))
    r = run(bars, "ZZ always long", "T", cfg=wide, calendar=_cal())
    assert r.trades["entry_ts"].min().time() < time(10, 30)


# -- lookahead ------------------------------------------------------------

def test_a_signal_is_acted_on_at_the_next_bars_open():
    """Filling at the close of the signal bar uses a price the signal saw."""
    bars = _session(step=0.05)
    r = run(bars, "ZZ always long", "T", cfg=IntradayConfig(cost_bps=0.0),
            calendar=_cal())
    first = r.trades.iloc[0]
    day = bars[bars.index.normalize() == pd.Timestamp("2026-09-14")]
    assert first["entry_px"] == pytest.approx(
        float(day.loc[first["entry_ts"], "open"]))


def test_one_bar_lookahead_is_neutralised_by_the_execution_delay():
    """Knowing the next CLOSE is worthless when you fill at the next OPEN.

    This is the delay doing its job, and it is worth asserting directly: a
    strategy that peeks one bar ahead gains nothing, because the order does
    not fill until after that bar has already happened. The first draft of
    this test assumed such a rule would win and was wrong about the engine,
    not about the rule.
    """
    bars = _session(seed=5)
    cfg = IntradayConfig(cost_bps=0.0)
    peeker = run(bars, "ZZ close oracle", "T", cfg=cfg, calendar=_cal())
    assert peeker.trades["return_pct"].sum() <= 0.05, (
        "a one-bar peek should not turn into profit at execution prices")


def test_the_harness_can_still_detect_a_real_edge():
    """The other half of the check.

    A rule that knows the two prices it will actually trade at must come out
    ahead -- otherwise the harness could not detect an edge that exists, and
    the test above would pass for the wrong reason.
    """
    bars = _session(seed=5)
    cfg = IntradayConfig(cost_bps=0.0)
    cheat = run(bars, "ZZ fill oracle", "T", cfg=cfg, calendar=_cal())
    honest = run(bars, "ZZ always long", "T", cfg=cfg, calendar=_cal())
    assert cheat.trades["return_pct"].sum() > 0
    assert cheat.trades["return_pct"].sum() > honest.trades["return_pct"].sum()


# -- costs ----------------------------------------------------------------

def test_costs_are_charged_on_both_sides():
    """One side charged would halve the cost of a strategy that trades often,
    which is exactly where cost decides the answer."""
    bars = _session(step=0.02)
    free = run(bars, "ZZ always long", "T",
               cfg=IntradayConfig(cost_bps=0.0), calendar=_cal())
    costed = run(bars, "ZZ always long", "T",
                 cfg=IntradayConfig(cost_bps=10.0), calendar=_cal())
    gap = (free.trades["return_pct"].iloc[0]
           - costed.trades["return_pct"].iloc[0])
    assert gap == pytest.approx(0.10, abs=1e-9)


def test_gross_is_recorded_alongside_net():
    bars = _session(step=0.02)
    r = run(bars, "ZZ always long", "T",
            cfg=IntradayConfig(cost_bps=8.0), calendar=_cal())
    t = r.trades.iloc[0]
    assert t["gross_pct"] > t["return_pct"]


# -- shorting -------------------------------------------------------------

def test_short_signals_are_dropped_when_shorting_is_off():
    @register
    class _AlwaysShort(Strategy):
        name = "ZZ always short"
        description = "test fixture"
        params: list[Param] = []

        def generate_signals(self, df):
            return pd.Series(-1.0, index=df.index)

    bars = _session(step=0.02)
    off = run(bars, "ZZ always short", "T",
              cfg=IntradayConfig(allow_short=False), calendar=_cal())
    on = run(bars, "ZZ always short", "T",
             cfg=IntradayConfig(allow_short=True), calendar=_cal())
    assert off.trades.empty
    assert not on.trades.empty
    assert (on.trades["direction"] == -1).all()


# -- robustness -----------------------------------------------------------

def test_a_session_too_short_to_trade_is_skipped():
    bars = _session(n=6)
    assert run(bars, "ZZ always long", "T", calendar=_cal()).sessions == 0


def test_a_non_trading_day_is_skipped():
    bars = _session("2026-09-20")          # a Sunday
    assert run(bars, "ZZ always long", "T", calendar=_cal()).sessions == 0


def test_an_unknown_strategy_raises_rather_than_returning_nothing():
    with pytest.raises(KeyError):
        run(_session(), "no such rule", "T", calendar=_cal())


def test_a_strategy_that_throws_is_skipped_not_crashed():
    @register
    class _Broken(Strategy):
        name = "ZZ broken"
        description = "test fixture"
        params: list[Param] = []

        def generate_signals(self, df):
            raise ValueError("boom")

    assert run(_session(), "ZZ broken", "T", calendar=_cal()).trades.empty


def test_combining_symbols_keeps_them_labelled():
    bars = _session(step=0.02)
    a = run(bars, "ZZ always long", "SPY", cfg=IntradayConfig(), calendar=_cal())
    b = run(bars, "ZZ always long", "QQQ", cfg=IntradayConfig(), calendar=_cal())
    both = combine([a, b])
    assert set(both["symbol"]) == {"SPY", "QQQ"}
    assert len(both) == len(a.trades) + len(b.trades)


def test_combining_nothing_is_empty_not_an_error():
    assert combine([]).empty
