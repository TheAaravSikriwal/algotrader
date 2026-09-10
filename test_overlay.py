"""Reasoning-overlay checks: the contract and its safety rails. No network."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from core.briefing import (build_briefing, is_market_reaction, render_markdown)
from core.overlay import (DEFAULT_MAX_TILT, MAX_SINGLE_TILT, Adjustment,
                          Overlay, OverlayError, apply_overlay, ledger_frame,
                          load_overlays, parse_overlay, record)

BASE = pd.Series({"XLE": 0.25, "XLK": 0.25, "XLF": 0.25, "XLV": 0.25})


def overlay(**kwargs) -> Overlay:
    defaults = dict(rationale="test", bucket="macro", adjustments=[
        {"symbol": "XLE", "action": "tilt", "weight": 0.10,
         "reason": "supply shock raises crude, flows to sector revenue"}])
    defaults.update(kwargs)
    return Overlay(**defaults)


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------
def test_adjustment_requires_a_reason():
    """An untraceable tilt cannot be reviewed or scored later."""
    try:
        Adjustment(symbol="XLE", weight=0.1, reason="  ")
        raise AssertionError("accepted an adjustment with no reason")
    except OverlayError as exc:
        assert "reason" in str(exc)


def test_unknown_action_is_rejected():
    try:
        Adjustment(symbol="XLE", action="yolo", reason="x")
        raise AssertionError("accepted an unknown action")
    except OverlayError:
        pass


def test_single_tilt_is_clipped():
    a = Adjustment(symbol="XLE", weight=0.9, reason="very confident")
    assert a.weight == MAX_SINGLE_TILT
    assert a.clipped is True


def test_confidence_scales_the_tilt():
    strong = Adjustment(symbol="XLE", weight=0.10, confidence=1.0, reason="x")
    weak = Adjustment(symbol="XLE", weight=0.10, confidence=0.2, reason="x")
    assert abs(strong.effective - 0.10) < 1e-9
    assert abs(weak.effective - 0.02) < 1e-9


def test_empty_overlay_is_rejected():
    try:
        Overlay(rationale="nothing", adjustments=[])
        raise AssertionError("accepted an overlay that does nothing")
    except OverlayError:
        pass


def test_parses_json_and_fenced_json():
    payload = {"schema": "algotrader.overlay.v1", "rationale": "r",
               "adjustments": [{"symbol": "XLE", "weight": 0.05, "reason": "why"}]}
    plain = parse_overlay(json.dumps(payload))
    fenced = parse_overlay("```json\n" + json.dumps(payload) + "\n```")
    assert plain.adjustments[0].symbol == "XLE"
    assert fenced.adjustments[0].symbol == "XLE"


def test_bad_json_explains_itself():
    try:
        parse_overlay("{not json at all")
        raise AssertionError("accepted malformed json")
    except OverlayError as exc:
        assert "json" in str(exc).lower()


# ---------------------------------------------------------------------------
# the safety rails
# ---------------------------------------------------------------------------
def test_tilt_moves_the_named_symbol():
    adjusted, report = apply_overlay(BASE, overlay())
    assert adjusted["XLE"] > BASE["XLE"]
    assert len(report["applied"]) == 1


def test_gross_exposure_is_preserved():
    """The overlay changes what is held, never how much leverage is taken."""
    adjusted, report = apply_overlay(BASE, overlay())
    assert abs(adjusted.abs().sum() - BASE.abs().sum()) < 1e-9
    assert abs(report["gross_after"] - report["gross_before"]) < 1e-9


def test_total_tilt_is_capped():
    """One confident wrong call costs a slice, not the book."""
    big = overlay(adjustments=[
        {"symbol": s, "action": "tilt", "weight": MAX_SINGLE_TILT,
         "confidence": 1.0, "reason": "conviction"} for s in BASE.index])
    adjusted, report = apply_overlay(BASE, big, max_tilt=DEFAULT_MAX_TILT)
    assert report["scaled_by"] < 1.0, "did not cap an oversized overlay"
    # the cap binds on the sum, not on any single name
    assert (adjusted - BASE).abs().sum() <= DEFAULT_MAX_TILT + 1e-6


def test_expired_adjustments_are_ignored():
    """A March narrative must not still be in the book in September."""
    stale = overlay(issued=(date.today() - timedelta(days=90)).isoformat(),
                    adjustments=[{"symbol": "XLE", "weight": 0.1,
                                  "reason": "old news", "expires_days": 21}])
    adjusted, report = apply_overlay(BASE, stale)
    assert report["expired"] == 1
    assert report["applied"] == []
    assert adjusted.equals(BASE)


def test_adjustment_active_before_expiry():
    fresh = overlay(issued=(date.today() - timedelta(days=5)).isoformat(),
                    adjustments=[{"symbol": "XLE", "weight": 0.1,
                                  "reason": "recent", "expires_days": 21}])
    _, report = apply_overlay(BASE, fresh)
    assert len(report["applied"]) == 1


def test_block_zeroes_a_position():
    blocked = overlay(adjustments=[
        {"symbol": "XLK", "action": "block", "reason": "regulatory overhang"}])
    adjusted, _ = apply_overlay(BASE, blocked)
    assert adjusted["XLK"] < BASE["XLK"]


def test_scale_multiplies_a_position():
    halved = overlay(adjustments=[
        {"symbol": "XLK", "action": "scale", "weight": 0.5,
         "reason": "reduce exposure"}])
    adjusted, _ = apply_overlay(BASE, halved)
    assert adjusted["XLK"] < BASE["XLK"]


def test_unknown_symbols_are_reported_not_silently_dropped():
    stray = overlay(adjustments=[
        {"symbol": "NOTHELD", "weight": 0.1, "reason": "not in the book"}])
    _, report = apply_overlay(BASE, stray)
    assert report["unknown_symbols"] == ["NOTHELD"]


def test_no_shorts_unless_allowed():
    heavy = overlay(adjustments=[
        {"symbol": "XLE", "action": "tilt", "weight": -MAX_SINGLE_TILT,
         "reason": "negative view"}])
    adjusted, _ = apply_overlay(BASE, heavy, allow_short=False)
    assert (adjusted >= -1e-9).all()


def test_base_is_never_mutated():
    before = BASE.copy()
    apply_overlay(BASE, overlay())
    assert BASE.equals(before), "apply_overlay modified the base strategy in place"


def test_contribution_is_measurable():
    """The overlay's effect is exactly overlaid minus base, which is testable."""
    adjusted, report = apply_overlay(BASE, overlay())
    assert report["turnover"] > 0
    assert abs((adjusted - BASE).abs().sum() - report["turnover"]) < 1e-9


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------
def test_overlay_round_trips_through_the_ledger():
    path = Path(mkdtemp()) / "overlays.jsonl"
    record(overlay(), path)
    record(overlay(adjustments=[{"symbol": "XLF", "weight": -0.05,
                                 "reason": "rate pressure"}]), path)
    loaded = load_overlays(path)
    assert len(loaded) == 2
    assert {a.symbol for o in loaded for a in o.adjustments} == {"XLE", "XLF"}


def test_ledger_frame_flags_active_rows():
    path = Path(mkdtemp()) / "overlays.jsonl"
    record(overlay(), path)
    record(overlay(issued=(date.today() - timedelta(days=90)).isoformat(),
                   adjustments=[{"symbol": "XLV", "weight": 0.05,
                                 "reason": "stale", "expires_days": 21}]), path)
    frame = ledger_frame(path)
    assert len(frame) == 2
    assert frame["active"].sum() == 1


# ---------------------------------------------------------------------------
# the briefing
# ---------------------------------------------------------------------------
def articles():
    now = pd.Timestamp.now("UTC").tz_convert(None)
    return pd.DataFrame({
        "timestamp": [now - pd.Timedelta(days=1)] * 4,
        "title": [
            "OPEC announces production cut of 2 million barrels",
            "Energy stocks rallied 4% after the supply news",
            "Apple unveils new product line at annual event",
            "Fed signals a further interest rate decision in November",
        ],
        "text": [
            "OPEC announces production cut of 2 million barrels",
            "Energy stocks rallied 4% after the supply news",
            "Apple unveils new product line at annual event",
            "Fed signals a further interest rate decision in November",
        ],
        "symbols": ["XLE", "XLE", "AAPL", "SPY"],
        "source_count": [8, 5, 6, 9],
    })


def test_market_reactions_are_detected():
    assert is_market_reaction("Energy stocks rallied 4% after the supply news")
    assert is_market_reaction("Shares fell 3% on the announcement")
    assert not is_market_reaction("OPEC announces production cut")


def test_briefing_drops_market_reactions():
    """A headline reporting the move is an answer key, not an input."""
    b = build_briefing(articles(), BASE, bucket="macro", since_days=7)
    titles = " ".join(b.articles["title"])
    assert "OPEC announces" in titles
    assert "rallied 4%" not in titles, "handed the model the outcome"


def test_briefing_filters_to_the_requested_bucket():
    b = build_briefing(articles(), BASE, bucket="macro", since_days=7)
    assert not b.articles.empty
    assert "Apple unveils new product" not in " ".join(b.articles["title"])


def test_briefing_excludes_old_news():
    old = articles()
    old["timestamp"] = pd.Timestamp.now("UTC").tz_convert(None) - pd.Timedelta(days=60)
    b = build_briefing(old, BASE, bucket="macro", since_days=7)
    assert b.articles.empty


def test_markdown_contains_the_schema_and_the_book():
    b = build_briefing(articles(), BASE, bucket="macro", since_days=7)
    text = render_markdown(b)
    assert "algotrader.overlay.v1" in text
    assert "XLE" in text
    assert "expires_days" in text
    assert "empty list is a valid" in text, "did not tell the model it may decline"


def test_markdown_handles_no_events():
    empty = pd.DataFrame(columns=["timestamp", "title", "text", "symbols"])
    text = render_markdown(build_briefing(empty, BASE, bucket="macro"))
    assert "No qualifying events" in text


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
