"""The day-trading rule.

Every test builds bars where the correct answer is known by construction, so a
wrong result is visible rather than merely plausible. The ones that matter most
are the lookahead guards: this rule's stop sits close to its entry, which is
exactly the shape where a one-bar off-by-one manufactures a large fake edge.
"""
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from core.daytrade import (
    DayTradeConfig,
    Setup,
    backtest,
    fair_value_gaps,
    find_setups,
    simulate,
    summarise,
    swing_levels,
)
from core.marketclock import MarketCalendar

DAY = date(2026, 9, 14)          # a Monday, regular hours
HALF = date(2026, 11, 27)        # 13:00 close


def _cal() -> MarketCalendar:
    return MarketCalendar.from_rows([
        {"date": DAY, "open": "09:30", "close": "16:00"},
        {"date": HALF, "open": "09:30", "close": "13:00"},
    ])


def _bars(rows, start="2026-09-14 09:30", freq="5min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx).assign(volume=1000.0)


# -- the gap definition ---------------------------------------------------

def test_bullish_gap_needs_candle_one_high_below_candle_three_low():
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)])
    g = fair_value_gaps(df)
    assert len(g) == 1 and g[0].direction == +1
    assert (g[0].lo, g[0].hi) == (11.0, 12.0)
    assert g[0].mid == pytest.approx(11.5)


def test_touching_wicks_are_not_a_gap():
    """`high[i-2] < low[i]` is strict. Equal prices mean the range was traded."""
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 11, 13)])
    assert fair_value_gaps(df) == []


def test_bearish_gap_is_the_mirror():
    df = _bars([(14, 14, 13, 13), (12, 12, 10, 10), (10, 11, 9, 10)])
    g = fair_value_gaps(df)
    assert len(g) == 1 and g[0].direction == -1
    assert (g[0].lo, g[0].hi) == (11.0, 13.0)


def test_gap_is_indexed_to_the_third_candle_not_the_first():
    """It cannot be known before candle three closes."""
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)])
    assert fair_value_gaps(df)[0].idx == 2


# -- lookahead ------------------------------------------------------------

def test_a_swing_is_not_published_before_it_is_confirmed():
    """A pivot at bar i needs k bars either side, so it is known at i+k.

    Publishing at bar i would place stops using bars that had not printed.
    """
    rows = [(10, 10.5, 9.5, 10)] * 3 + [(10, 10.2, 8.0, 9)] + [(9, 10.5, 9.0, 10)] * 4
    lo, _ = swing_levels(_bars(rows), k=3)
    assert pd.isna(lo[3]), "the pivot bar itself must not know it is a pivot"
    assert pd.isna(lo[5]), "still unconfirmed one bar before k bars have passed"
    assert lo[6] == pytest.approx(8.0)


def test_entry_bar_can_also_be_the_bar_that_stops_you_out():
    """The off-by-one that manufactures a fake edge.

    A limit that fills and is then blown through inside the same candle must
    record the loss. Skipping the fill bar would book the fill and no loss.
    """
    rows = [
        (10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13),   # gap 11..12, mid 11.5
        (12.5, 12.6, 9.0, 9.2),                                  # fills 11.5, then collapses
    ]
    df = _bars(rows)
    s = Setup(symbol="T", signal_ts=df.index[2], direction=+1, entry_px=11.5,
              stop_px=11.0, target_r=2.0, expires_after_bars=12)
    res = simulate(df, s, DayTradeConfig())
    assert res is not None, "the limit was touched, so it filled"
    assert res["reason"] == "stop"
    assert res["r"] < 0


def test_a_stop_gapped_through_fills_at_the_open_not_at_the_stop():
    """Otherwise the backtest caps a loss that was never capped in life."""
    rows = [
        (10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13),
        (12.0, 12.0, 11.4, 11.6),      # fills at 11.5, holds
        (8.0, 8.2, 7.5, 7.8),          # opens far below the 11.0 stop
    ]
    df = _bars(rows)
    s = Setup(symbol="T", signal_ts=df.index[2], direction=+1, entry_px=11.5,
              stop_px=11.0, target_r=2.0, expires_after_bars=12)
    res = simulate(df, s, DayTradeConfig())
    assert res["exit_px"] == pytest.approx(8.0), "filled at the open, not the stop"
    assert res["r"] < -1.0, "a gap loses more than one R"


def test_stop_wins_when_one_bar_covers_both_stop_and_target():
    """A single bar says nothing about the path inside it, so assume the worst."""
    rows = [
        (10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13),
        (12.0, 13.0, 10.5, 11.0),      # touches entry 11.5, stop 11.0 and target 12.5
    ]
    df = _bars(rows)
    s = Setup(symbol="T", signal_ts=df.index[2], direction=+1, entry_px=11.5,
              stop_px=11.0, target_r=2.0, expires_after_bars=12)
    assert simulate(df, s, DayTradeConfig())["reason"] == "stop"


# -- the limit that never fills ------------------------------------------

def test_an_untouched_limit_is_not_a_trade():
    rows = [(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)] + \
           [(13, 14, 12.5, 13)] * 5       # never trades down to 11.5
    df = _bars(rows)
    s = Setup(symbol="T", signal_ts=df.index[2], direction=+1, entry_px=11.5,
              stop_px=11.0, target_r=2.0, expires_after_bars=12)
    assert simulate(df, s, DayTradeConfig()) is None


def test_the_limit_expires_and_stops_being_eligible():
    """Cancel-after-N is what keeps a stale level from being traded hours later."""
    rows = [(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)] + \
           [(13, 14, 12.5, 13)] * 6 + [(13, 14, 10.0, 11)]   # dips only at bar 9
    df = _bars(rows)
    common = dict(symbol="T", signal_ts=df.index[2], direction=+1,
                  entry_px=11.5, stop_px=11.0, target_r=2.0)
    assert simulate(df, Setup(**common, expires_after_bars=3), DayTradeConfig()) is None
    assert simulate(df, Setup(**common, expires_after_bars=12), DayTradeConfig()) is not None


# -- the session window ---------------------------------------------------

def test_setups_before_1030_are_ignored():
    """Not a preference. SPY's spread is 3.78 bps in the first fifteen minutes
    against 1.59 mid-day, so the open alone eats the edge."""
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)],
               start="2026-09-14 09:30")
    assert find_setups(df, "T", DayTradeConfig()) == []


def test_setups_inside_the_window_are_kept():
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)],
               start="2026-09-14 11:00")
    out = find_setups(df, "T", DayTradeConfig())
    assert len(out) == 1
    assert out[0].entry_px == pytest.approx(11.5)


def test_setups_after_1530_are_ignored():
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)],
               start="2026-09-14 15:25")
    assert find_setups(df, "T", DayTradeConfig()) == []


def test_the_window_shrinks_on_a_half_day():
    """12:40 is inside 10:30-15:30 but the market shut at 13:00.

    Without the calendar the setup looks fine; with it, the window is clipped
    to the real close and a signal at 12:40 has no room to resolve.
    """
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)],
               start="2026-11-27 12:50")
    sess = _cal().session(HALF)
    assert find_setups(df, "T", DayTradeConfig(), sess) == []
    assert len(find_setups(df, "T", DayTradeConfig())) == 1, "unclipped it slips through"


def test_position_is_flattened_before_a_half_day_close():
    rows = [(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)] + \
           [(12.0, 12.2, 11.4, 11.6)] * 6
    df = _bars(rows, start="2026-11-27 12:20")
    s = Setup(symbol="T", signal_ts=df.index[2], direction=+1, entry_px=11.5,
              stop_px=11.0, target_r=2.0, expires_after_bars=12)
    res = simulate(df, s, DayTradeConfig(), _cal().session(HALF))
    assert res["reason"] == "flatten"
    assert res["exit_ts"].time() <= time(12, 55)


# -- sizing ---------------------------------------------------------------

def test_shares_are_sized_so_a_stop_costs_the_stated_risk():
    s = Setup(symbol="T", signal_ts=pd.Timestamp(DAY), direction=+1,
              entry_px=100.0, stop_px=99.5, target_r=2.0, expires_after_bars=12)
    qty = s.shares(equity=100_000, risk_frac=0.005)   # $500 risk / $0.50 = 1000
    assert qty == 1000
    assert qty * s.risk_per_share == pytest.approx(500.0)


def test_share_count_is_truncated_never_rounded_up():
    """Rounding up risks more than the rule promised."""
    s = Setup(symbol="T", signal_ts=pd.Timestamp(DAY), direction=+1,
              entry_px=100.0, stop_px=99.37, target_r=2.0, expires_after_bars=12)
    assert s.shares(100_000, 0.005) == 793      # 500/0.63 = 793.65


def test_a_zero_width_stop_sizes_to_nothing_rather_than_dividing_by_zero():
    s = Setup(symbol="T", signal_ts=pd.Timestamp(DAY), direction=+1,
              entry_px=100.0, stop_px=100.0, target_r=2.0, expires_after_bars=12)
    assert s.shares(100_000, 0.005) == 0


def test_target_sits_at_the_stated_multiple_of_risk():
    long = Setup(symbol="T", signal_ts=pd.Timestamp(DAY), direction=+1,
                 entry_px=100.0, stop_px=99.0, target_r=2.0, expires_after_bars=12)
    short = Setup(symbol="T", signal_ts=pd.Timestamp(DAY), direction=-1,
                  entry_px=100.0, stop_px=101.0, target_r=2.0, expires_after_bars=12)
    assert long.target_px == pytest.approx(102.0)
    assert short.target_px == pytest.approx(98.0)


def test_sides_are_broker_language_not_position_language():
    long = Setup(symbol="T", signal_ts=pd.Timestamp(DAY), direction=+1,
                 entry_px=100.0, stop_px=99.0, target_r=2.0, expires_after_bars=12)
    assert long.side == "buy"


# -- caps -----------------------------------------------------------------

def test_one_trade_per_symbol_per_day():
    block = [(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)]
    df = _bars(block * 4, start="2026-09-14 11:00")
    assert len(find_setups(df, "T", DayTradeConfig())) == 1
    cfg = DayTradeConfig(max_trades_per_symbol_per_day=3)
    assert len(find_setups(df, "T", cfg)) == 3


def test_an_unrealistically_tight_stop_is_rejected():
    """A stop a fraction of a basis point wide sizes into an enormous position."""
    df = _bars([(100, 100.01, 100, 100.01),
                (100.01, 100.02, 100.01, 100.02),
                (100.02, 100.03, 100.011, 100.02)], start="2026-09-14 11:00")
    assert find_setups(df, "T", DayTradeConfig(min_stop_frac=0.0005)) == []


# -- aggregation ----------------------------------------------------------

def test_backtest_skips_non_trading_days():
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13),
                (12.0, 12.2, 11.4, 11.6)], start="2026-09-20 11:00")   # a Sunday inside the covered range
    assert backtest({"T": df}, DayTradeConfig(), _cal()).empty


def test_costs_are_subtracted_from_the_return():
    rows = [(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13),
            (12.0, 12.2, 11.4, 11.6), (11.6, 14.0, 11.5, 13.0)]
    df = _bars(rows, start="2026-09-14 11:00")
    free = backtest({"T": df}, DayTradeConfig(), _cal(), cost_bps=0.0)
    costed = backtest({"T": df}, DayTradeConfig(), _cal(), cost_bps=10.0)
    assert len(free) == len(costed) == 1
    assert costed["ret_pct"].iloc[0] == pytest.approx(free["ret_pct"].iloc[0] - 0.10)


def test_breakeven_is_the_cost_that_would_erase_the_edge():
    """The number that decides everything, against a 1.59 bps measured spread."""
    trades = pd.DataFrame({
        "r": [1.0, 1.0], "ret_pct": [0.05, 0.05], "symbol": ["SPY", "QQQ"],
        "reason": ["target", "target"], "risk_per_share": [0.5, 0.5],
        "entry_px": [100.0, 100.0]})
    assert summarise(trades)["breakeven_bps"] == pytest.approx(5.0)


def test_summarise_of_nothing_is_not_an_error():
    assert summarise(pd.DataFrame()) == {"trades": 0}


def test_backtest_refuses_a_calendar_that_does_not_span_the_bars():
    """The bug this guard exists for.

    A calendar fetched for 2026 silently skipped every session from 2021 to
    2025, and the run still reported a confident summary built from the
    fraction that happened to be covered.
    """
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13),
                (12.0, 12.2, 11.4, 11.6)], start="2021-03-01 11:00")
    with pytest.raises(Exception, match="fetch a wider range"):
        backtest({"T": df}, DayTradeConfig(), _cal())


def test_a_window_that_misses_a_half_day_yields_no_setups_not_an_error():
    """A 15:00 window on a 13:00 close means "nothing today", not a crash.

    Raising would abort a five-year backtest because of one short Friday.
    """
    from datetime import time as _t
    df = _bars([(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)],
               start="2026-11-27 11:00")
    cfg = DayTradeConfig(window_start=_t(15, 0), window_end=_t(15, 55))
    assert find_setups(df, "T", cfg, _cal().session(HALF)) == []
