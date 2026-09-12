"""Risk and reward arithmetic.

Every case here has an answer that can be worked out by hand, so a wrong
result is visible rather than merely plausible. The ones that matter most are
the sizing tests: they encode the fact that past a certain bet size, more risk
reliably makes less money, and that no size rescues a losing rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.expectancy import (
    Expectancy,
    project,
    risk_of_ruin,
    sizing_curve,
    summarise,
)


def _coin(win_rate: float, win: float, loss: float, n: int = 1000):
    """Exactly `win_rate` of n trades win. No sampling noise to argue with."""
    wins = int(round(n * win_rate))
    return [win] * wins + [-loss] * (n - wins)


# -- the basics -----------------------------------------------------------

def test_expectancy_is_the_probability_weighted_average():
    """60% at +2%, 40% at -1% -> 0.6*2 - 0.4*1 = +0.8% a trade."""
    e = summarise(_coin(0.6, 2.0, 1.0))
    assert e.expectancy_pct == pytest.approx(0.8)
    assert e.win_rate == pytest.approx(0.6)
    assert e.avg_win == pytest.approx(2.0)
    assert e.avg_loss == pytest.approx(1.0), "reported positive, not signed"
    assert e.payoff_ratio == pytest.approx(2.0)


def test_a_low_win_rate_can_beat_a_high_one():
    """The headline number people quote is the one that decides least.

    30% at 4:1 makes money; 70% at 1:3 loses it. Win rate alone is not an
    edge and this is the case that proves it.
    """
    sharp = summarise(_coin(0.30, 4.0, 1.0))
    frequent = summarise(_coin(0.70, 1.0, 3.0))
    assert sharp.win_rate < frequent.win_rate
    assert sharp.expectancy_pct > 0 > frequent.expectancy_pct


def test_breakeven_win_rate_matches_the_payoff():
    """At 3:1 you need to be right one time in four just to stand still."""
    e = summarise(_coin(0.5, 3.0, 1.0))
    assert e.breakeven_win_rate == pytest.approx(0.25)


def test_profit_factor_is_gross_wins_over_gross_losses():
    e = summarise(_coin(0.5, 2.0, 1.0, n=100))
    assert e.profit_factor == pytest.approx(2.0)


def test_a_rule_with_no_losses_has_no_profit_factor_rather_than_infinity():
    """Reporting inf would put a three-trade sample top of any ranking."""
    e = summarise([1.0, 2.0, 3.0])
    assert np.isnan(e.profit_factor)


def test_no_trades_is_not_an_error():
    e = summarise([])
    assert e.trades == 0 and e.expectancy_pct == 0.0


# -- Kelly ----------------------------------------------------------------

def test_kelly_matches_the_closed_form():
    """W - (1-W)/R: 0.6 - 0.4/2 = 0.4."""
    e = summarise(_coin(0.6, 2.0, 1.0))
    assert e.kelly_fraction == pytest.approx(0.4)


def test_kelly_on_a_losing_rule_is_zero_not_negative():
    """The growth-maximising bet on a losing rule is nothing at all."""
    e = summarise(_coin(0.3, 1.0, 1.0))
    assert e.expectancy_pct < 0
    assert e.kelly_fraction == 0.0


def test_the_suggested_size_is_a_quarter_of_kelly():
    """Full Kelly assumes the edge is known exactly. It is estimated."""
    e = summarise(_coin(0.6, 2.0, 1.0))
    assert e.suggested_fraction == pytest.approx(e.kelly_fraction / 4)
    assert e.suggested_fraction < e.kelly_fraction


# -- the sizing trap ------------------------------------------------------

def test_growth_peaks_at_kelly_and_falls_after_it():
    """More risk stops meaning more money, and starts meaning less.

    This is the single most useful thing the module knows, and the reason
    "big risk for big profit" needs a number attached.
    """
    e = summarise(_coin(0.6, 2.0, 1.0))
    curve = sizing_curve(e, trades=200)
    peak = curve.loc[curve["log_growth"].idxmax(), "risk_frac"]
    assert peak == pytest.approx(e.kelly_fraction, abs=0.02)

    past = curve[curve["risk_frac"] > e.kelly_fraction + 0.1]
    assert (past["log_growth"] < curve["log_growth"].max()).all()


def test_betting_far_past_kelly_loses_money_despite_a_real_edge():
    """Every trade has positive expected value and the account still shrinks."""
    e = summarise(_coin(0.6, 2.0, 1.0))
    assert e.expectancy_pct > 0
    reckless = project(e, risk_frac=0.90, trades=200)
    assert reckless["median_growth"] < 0


def test_no_size_rescues_a_losing_rule():
    """The case the module exists to make impossible to miss."""
    e = summarise(_coin(0.35, 1.0, 1.0))
    for f in (0.005, 0.02, 0.10, 0.30):
        assert project(e, f, trades=200)["median_growth"] < 0


def test_a_size_that_can_wipe_the_account_in_one_trade_is_flagged():
    e = summarise(_coin(0.6, 2.0, 1.0))
    out = project(e, risk_frac=1.5, trades=50)
    assert out["median_growth"] == -1.0
    assert "note" in out


def test_mean_and_median_growth_diverge_as_size_rises():
    """The average is carried by a few lucky paths. The median is your life."""
    e = summarise(_coin(0.55, 2.0, 1.0))
    small = project(e, 0.02, 300)
    large = project(e, 0.40, 300)
    assert small["mean_growth"] - small["median_growth"] < \
        large["mean_growth"] - large["median_growth"]


# -- ruin -----------------------------------------------------------------

def test_bigger_bets_raise_the_chance_of_ruin():
    e = summarise(_coin(0.55, 2.0, 1.0))
    low = risk_of_ruin(e, 0.01, trades=200, runs=800, seed=1)
    high = risk_of_ruin(e, 0.25, trades=200, runs=800, seed=1)
    assert high["risk_of_ruin"] > low["risk_of_ruin"]
    assert high["worst_drawdown"] > low["worst_drawdown"]


def test_a_losing_rule_at_size_is_near_certain_ruin():
    e = summarise(_coin(0.40, 1.0, 1.0))
    out = risk_of_ruin(e, 0.20, trades=300, runs=800, seed=2)
    assert out["risk_of_ruin"] > 0.8


def test_a_tiny_bet_on_a_good_rule_rarely_ruins():
    e = summarise(_coin(0.60, 2.0, 1.0))
    out = risk_of_ruin(e, 0.005, trades=300, runs=800, seed=3)
    assert out["risk_of_ruin"] < 0.05


# -- sample size ----------------------------------------------------------

def test_a_smaller_edge_needs_more_trades_to_confirm():
    big = summarise(_coin(0.60, 2.0, 1.0))
    small = summarise(_coin(0.52, 1.05, 1.0))
    assert small.trades_needed() > big.trades_needed()


def test_a_losing_rule_has_no_confirming_sample_size():
    """There is no number of trades that turns a negative edge positive."""
    assert summarise(_coin(0.3, 1.0, 1.0)).trades_needed() is None
