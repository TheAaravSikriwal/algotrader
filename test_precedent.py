"""Precedent checks: outcomes outlive the positions that made them."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from core.overlay import Overlay, record
from core.precedent import (Outcome, confidence_note, find_precedents,
                            load_outcomes, precedents_for_text, record_outcomes,
                            score_overlay, score_pending, summarise_precedents)


def bars_for(symbols, n=400, drift=0.0, seed=1):
    """Flat market, with `drift` applied only to the first symbol."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(date.today() - timedelta(days=int(n * 1.5)), periods=n)
    out = {}
    for i, s in enumerate(symbols):
        r = rng.normal(drift if i == 0 else 0.0, 0.004, n)
        px = 100 * np.exp(np.cumsum(r))
        out[s] = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                               "volume": 1e6}, index=idx)
    flat = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
    return out, pd.Series(flat, index=idx)


def overlay_issued(days_ago: int, symbol="XLE", weight=0.10, expires=21,
                   evidence="OPEC announces production cut") -> Overlay:
    return Overlay(
        rationale="test", bucket="macro",
        issued=(date.today() - timedelta(days=days_ago)).isoformat(),
        adjustments=[{"symbol": symbol, "action": "tilt", "weight": weight,
                      "confidence": 1.0, "expires_days": expires,
                      "reason": "supply cut lifts crude, flows to sector revenue",
                      "evidence": evidence}])


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_live_adjustments_are_not_scored():
    """Nothing to learn from a position that has not run its course."""
    bars, bench = bars_for(["XLE"])
    assert score_overlay(overlay_issued(3, expires=21), bars, bench) == []


def test_expired_adjustments_are_scored():
    bars, bench = bars_for(["XLE"])
    outcomes = score_overlay(overlay_issued(60, expires=21), bars, bench)
    assert len(outcomes) == 1
    assert outcomes[0].bars_held > 0
    assert outcomes[0].symbol == "XLE"


def test_abnormal_return_strips_the_market():
    """A tilt that merely rode a rising market earns no credit."""
    bars, bench = bars_for(["XLE"], drift=0.003)
    outcome = score_overlay(overlay_issued(60), bars, bench)[0]
    assert outcome.raw_return > outcome.abnormal_return, "market move was not removed"


def test_direction_correct_tracks_the_sign():
    bars, bench = bars_for(["XLE"], drift=0.004)
    up = score_overlay(overlay_issued(60, weight=0.10), bars, bench)[0]
    down = score_overlay(overlay_issued(60, weight=-0.10), bars, bench)[0]
    assert up.direction_correct != down.direction_correct
    assert up.contribution == -down.contribution


def test_contribution_is_signed_by_the_tilt():
    bars, bench = bars_for(["XLE"], drift=-0.004)
    short = score_overlay(overlay_issued(60, weight=-0.10), bars, bench)[0]
    assert short.abnormal_return < 0
    assert short.contribution > 0, "a correct negative tilt should score positive"


def test_missing_price_history_is_skipped_not_fatal():
    bars, bench = bars_for(["XLE"])
    assert score_overlay(overlay_issued(60, symbol="UNKNOWN"), bars, bench) == []


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
def test_outcomes_round_trip():
    path = Path(mkdtemp()) / "outcomes.jsonl"
    bars, bench = bars_for(["XLE"])
    outcomes = score_overlay(overlay_issued(60), bars, bench)
    assert record_outcomes(outcomes, path) == 1
    loaded = load_outcomes(path)
    assert len(loaded) == 1
    assert loaded[0].symbol == "XLE"


def test_scoring_is_idempotent():
    """Re-scoring must not duplicate an outcome and inflate the sample."""
    path = Path(mkdtemp()) / "outcomes.jsonl"
    bars, bench = bars_for(["XLE"])
    outcomes = score_overlay(overlay_issued(60), bars, bench)
    record_outcomes(outcomes, path)
    assert record_outcomes(outcomes, path) == 0
    assert len(load_outcomes(path)) == 1


def test_score_pending_walks_the_whole_ledger():
    tmp = Path(mkdtemp())
    ledger, outcomes_path = tmp / "o.jsonl", tmp / "out.jsonl"
    record(overlay_issued(60, symbol="XLE"), ledger)
    record(overlay_issued(60, symbol="TLT", evidence="Fed signals rate hike"), ledger)
    record(overlay_issued(2, symbol="GLD"), ledger)          # still live

    bars, bench = bars_for(["XLE", "TLT", "GLD"])
    fresh = score_pending(bars, bench, ledger, outcomes_path)
    assert len(fresh) == 2, "scored a live overlay, or missed an expired one"
    assert score_pending(bars, bench, ledger, outcomes_path) == []


# ---------------------------------------------------------------------------
# retrieval -- the part that makes knowledge outlive the position
# ---------------------------------------------------------------------------
def outcome(categories, symbol="XLE", contribution=0.02, issued="2026-03-01"):
    return Outcome(
        overlay_id=f"id{issued}{symbol}", symbol=symbol, action="tilt",
        effective=0.1, issued=issued, expired=issued, reason="r",
        bucket="macro", categories=categories,
        abnormal_return=contribution, raw_return=contribution, bars_held=15)


def test_precedents_match_on_category_overlap():
    """A different war is still a precedent for a war."""
    path = Path(mkdtemp()) / "out.jsonl"
    record_outcomes([
        outcome(["geopolitical", "energy"], issued="2026-03-01"),
        outcome(["geopolitical"], issued="2026-05-01"),
        outcome(["earnings"], issued="2026-06-01"),
    ], path)

    frame = find_precedents(["geopolitical"], path=path)
    assert len(frame) == 2, "missed a precedent that shared a category"
    assert "earnings" not in " ".join(frame["categories"])


def test_precedents_can_be_narrowed_to_a_symbol():
    path = Path(mkdtemp()) / "out.jsonl"
    record_outcomes([outcome(["energy"], symbol="XLE"),
                     outcome(["energy"], symbol="XLK", issued="2026-04-01")], path)
    assert len(find_precedents(["energy"], symbol="XLE", path=path)) == 1


def test_summary_reports_hit_rate_and_sample_size():
    path = Path(mkdtemp()) / "out.jsonl"
    record_outcomes([outcome(["energy"], contribution=0.03, issued=f"2026-0{i}-01")
                     for i in range(1, 5)], path)
    stats = summarise_precedents(find_precedents(["energy"], path=path))
    assert stats["n"] == 4
    assert stats["hit_rate"] == 1.0
    assert stats["mean_contribution_%"] > 0


def test_confidence_note_scales_with_evidence():
    assert "first observation" in confidence_note(0)
    assert "anecdote" in confidence_note(1)
    assert "luck" in confidence_note(3)
    assert "distribution" in confidence_note(30)


def test_summary_of_nothing_says_so():
    stats = summarise_precedents(pd.DataFrame())
    assert stats["n"] == 0
    assert "first observation" in stats["note"]


def test_precedents_for_text_classifies_and_matches():
    path = Path(mkdtemp()) / "out.jsonl"
    record_outcomes([outcome(["energy"], issued="2026-03-01")], path)
    frame, stats = precedents_for_text(
        "OPEC output cut sends crude above $95 per barrel", path=path)
    assert stats["n"] == 1
    assert "anecdote" in stats["note"]
    assert frame["matched_on"].iloc[0] == "energy"


def test_losing_precedents_are_kept():
    """A tilt that was wrong is the most informative row in the table."""
    path = Path(mkdtemp()) / "out.jsonl"
    record_outcomes([outcome(["energy"], contribution=0.02),
                     outcome(["energy"], contribution=-0.04, issued="2026-04-01")],
                    path)
    frame = find_precedents(["energy"], path=path)
    assert len(frame) == 2
    assert (frame["correct"] == False).any(), "dropped an unfavourable outcome"  # noqa: E712
    assert summarise_precedents(frame)["hit_rate"] == 0.5


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc or 'assertion failed'}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print("\nall green" if not failures else f"\n{failures} failing")
    raise SystemExit(1 if failures else 0)
