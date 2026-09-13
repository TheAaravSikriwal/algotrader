"""The recommender that decides which tested rule to run.

The tests that matter here are the refusals. A recommender that always names
a winner is worse than none at all, because it launders "least bad of a
losing set" into "best choice" -- and the evidence in this repo says the
losing set is the normal case.
"""
from __future__ import annotations

import pytest

from core.recommend import (
    MIN_TRADES,
    SWITCH_MARGIN,
    Candidate,
    best_for,
    load_candidates,
    rank,
    review,
)


def _c(strategy, expectancy, t=3.0, trades=500.0, horizon="short",
       symbol="SPY", hold=4.0):
    return Candidate(strategy=strategy, symbol=symbol,
                     expectancy_pct=expectancy, t_stat=t, trades=trades,
                     exposure=0.4, avg_hold_bars=hold, horizon=horizon)


# -- refusing to recommend ------------------------------------------------

def test_a_losing_field_gets_stand_aside_not_a_winner():
    """The case this module exists for.

    Every candidate loses. Naming the least bad one as a recommendation is
    how a ranking of ways to lose becomes a trade.
    """
    rec = best_for(candidates=[_c("A", -2.0), _c("B", -5.0), _c("C", -9.0)])
    assert rec.action == "stand_aside"
    assert "ways to lose" in rec.reason
    assert rec.best is None


def test_an_empty_field_refuses_rather_than_inventing_one():
    rec = best_for(candidates=[])
    assert rec.action == "stand_aside"
    assert "evidence" in rec.reason


def test_a_positive_result_on_too_few_trades_is_not_credible():
    """A big average on a small sample is noise wearing a decimal point."""
    thin = _c("Thin", +40.0, t=2.5, trades=MIN_TRADES - 1)
    assert not thin.credible
    assert best_for(candidates=[thin]).action == "stand_aside"


# -- ranking --------------------------------------------------------------

def test_evidence_outranks_raw_return():
    """Twelve trades at +40% must not outrank eight hundred at +3%."""
    ordered = rank([_c("Thin", +40.0, trades=12), _c("Solid", +3.0, trades=800)])
    assert ordered[0].strategy == "Solid"


def test_a_significant_result_leads_an_insignificant_larger_one():
    ordered = rank([_c("Loud", +6.0, t=0.4), _c("Real", +4.0, t=3.5)])
    assert ordered[0].strategy == "Real"


def test_the_also_rans_are_still_returned():
    """The caller shows them; hiding them would misrepresent the field."""
    assert len(rank([_c("A", +3.0), _c("B", -1.0), _c("C", +1.0)])) == 3


# -- confidence is reported, never assumed --------------------------------

def test_an_insignificant_winner_is_flagged_as_such():
    rec = best_for(candidates=[_c("Weak", +2.0, t=0.9)])
    assert rec.action == "switch"
    assert rec.confident is False
    assert "inside the noise" in rec.reason


def test_a_significant_winner_is_marked_confident():
    rec = best_for(candidates=[_c("Strong", +5.0, t=4.2)])
    assert rec.confident is True


# -- switching ------------------------------------------------------------

def test_a_small_improvement_does_not_justify_switching():
    """Switching on noise converts it into turnover, and turnover into spread."""
    pool = [_c("Running", +4.0), _c("Slightly better", +4.4)]
    rec = review("Running", candidates=pool)
    assert rec.action == "hold"
    assert "turnover" in rec.reason


def test_a_clear_improvement_does_justify_switching():
    pool = [_c("Running", +2.0), _c("Much better", +8.0)]
    rec = review("Running", candidates=pool)
    assert rec.action == "switch"
    assert rec.best.strategy == "Much better"
    assert rec.incumbent.strategy == "Running"


def test_the_switch_margin_is_the_line():
    """Just under the margin holds; just over it switches."""
    under = review("Running", candidates=[
        _c("Running", +4.0), _c("Rival", 4.0 * SWITCH_MARGIN - 0.01)])
    over = review("Running", candidates=[
        _c("Running", +4.0), _c("Rival", 4.0 * SWITCH_MARGIN + 0.5)])
    assert under.action == "hold"
    assert over.action == "switch"


def test_a_losing_incumbent_in_a_losing_field_means_stand_aside():
    rec = review("Running", candidates=[_c("Running", -3.0), _c("Other", -1.0)])
    assert rec.action == "stand_aside"
    assert "flat is the evidenced position" in rec.reason


def test_an_unknown_incumbent_is_said_to_be_unknown():
    """Silently treating an unevaluated rule as a loser would be a lie."""
    rec = review("Never tested", candidates=[_c("Known", +5.0)])
    assert rec.action == "switch"
    assert "no evaluation on record" in rec.reason


def test_holding_a_credible_incumbent_reports_its_own_confidence():
    rec = review("Running", candidates=[_c("Running", +4.0, t=0.5),
                                        _c("Rival", +4.1, t=5.0)])
    assert rec.action == "hold"
    assert rec.confident is False, "the incumbent's own t-stat decides this"


# -- horizon --------------------------------------------------------------

def test_short_horizon_only_returns_short_horizon_rules():
    pool = [_c("Fast", +3.0, horizon="short"), _c("Slow", +9.0, horizon="long")]
    rec = best_for(horizon="short", candidates=[c for c in pool
                                                if c.horizon == "short"])
    assert rec.best.strategy == "Fast"


def test_loading_from_a_missing_file_returns_nothing_rather_than_raising(tmp_path):
    assert load_candidates(path=tmp_path / "absent.csv") == []


# -- the real evaluation --------------------------------------------------

def test_against_the_real_evaluation_the_answer_is_stand_aside():
    """The honest end-to-end check.

    348 measured strategy-symbol pairs, not one of which clears the corrected
    bar. If this ever starts returning a confident recommendation, something
    either improved or broke, and both are worth noticing.
    """
    short = load_candidates(horizon="short")
    if not short:
        pytest.skip("evaluate_all.py has not been run")
    rec = best_for(horizon="short", candidates=short)
    assert rec.action == "stand_aside" or not rec.confident
