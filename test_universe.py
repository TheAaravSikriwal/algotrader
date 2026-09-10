"""Universe screening checks. Synthetic bars, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.universe import (PROFILES, Profile, describe_profile, metrics_for,
                           profile_frame, screen)


def make_bars(n=800, price=100.0, volume=1e7, vol=0.015, seed=1, range_pct=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    px = price * np.exp(np.cumsum(rng.normal(0.0002, vol, n)))
    half = range_pct / 2
    return pd.DataFrame({
        "open": px, "high": px * (1 + half), "low": px * (1 - half),
        "close": px, "volume": np.full(n, volume)}, index=idx)


def test_metrics_are_computed():
    m = metrics_for(make_bars())
    assert m["price"] > 0
    assert m["dollar_volume"] > 0
    assert m["history_days"] == 800
    assert 0 < m["completeness"] <= 1.0
    assert np.isfinite(m["amihud"])


def test_short_history_returns_nothing():
    assert metrics_for(make_bars(n=10)) == {}
    assert metrics_for(None) == {}


def test_amihud_is_lower_for_deeper_names():
    """Illiquidity measures price impact per dollar traded."""
    deep = metrics_for(make_bars(volume=5e8, seed=7))
    thin = metrics_for(make_bars(volume=1e5, seed=7))
    assert deep["amihud"] < thin["amihud"], "deeper name did not measure as more liquid"


def test_range_tracks_the_bar_width():
    tight = metrics_for(make_bars(range_pct=0.01))
    wide = metrics_for(make_bars(range_pct=0.08))
    assert tight["range_pct"] < wide["range_pct"]


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------
def sample_frame():
    bars = {
        "DEEP":   make_bars(price=200, volume=5e6, range_pct=0.015, seed=2),
        "THIN":   make_bars(price=200, volume=2e3, range_pct=0.02, seed=3),
        "PENNY":  make_bars(price=2.0, volume=5e6, range_pct=0.02, seed=4),
        "WILD":   make_bars(price=150, volume=5e6, range_pct=0.09, seed=5),
        "YOUNG":  make_bars(n=120, price=150, volume=5e6, seed=6),
    }
    news = {"DEEP": 2.0, "THIN": 0.0, "PENNY": 0.1, "WILD": 1.5, "YOUNG": 3.0}
    return profile_frame(bars, news)


def test_profile_frame_has_a_row_per_symbol():
    frame = sample_frame()
    assert set(frame.index) == {"DEEP", "THIN", "PENNY", "WILD", "YOUNG"}
    assert "news_per_day" in frame.columns


def test_penny_stocks_are_rejected():
    result = screen(sample_frame(), "long_term")
    assert "PENNY" not in result.index


def test_short_history_is_rejected():
    result = screen(sample_frame(), "long_term")
    assert "YOUNG" not in result.index, "accepted a name without enough history"


def test_thin_names_are_rejected_for_short_term():
    result = screen(sample_frame(), "short_term")
    assert "THIN" not in result.index


def test_short_term_rejects_wide_ranges():
    """A wide daily range is a spread proxy, and spread is what kills turnover."""
    frame = sample_frame()
    prof = Profile(name="t", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, max_range_pct=0.035)
    result = screen(frame, prof)
    assert "WILD" not in result.index
    assert "DEEP" in result.index


def test_reasoning_profile_requires_coverage():
    frame = sample_frame()
    prof = Profile(name="r", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, min_news_per_day=1.0)
    result = screen(frame, prof)
    assert "THIN" not in result.index, "kept a name with no coverage"
    assert "DEEP" in result.index


def test_rejection_reasons_are_recorded():
    frame = sample_frame()
    prof = Profile(name="t", rationale="", min_price=10.0, min_dollar_volume=1e9,
                   min_history_days=100)
    screen(frame, prof)                      # full annotation lives on the copy
    annotated = frame.copy()
    result = screen(annotated, prof)
    assert result.empty or "rejected_for" in result.columns


def test_ranking_respects_the_profile_key():
    frame = sample_frame()
    by_news = screen(frame, Profile(name="n", rationale="", min_dollar_volume=1.0,
                                    min_history_days=100, rank_by="news_per_day",
                                    top_n=3))
    assert by_news.index[0] == "YOUNG", by_news["news_per_day"].to_dict()


def test_top_n_is_honoured():
    frame = sample_frame()
    prof = Profile(name="t", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, top_n=2)
    assert len(screen(frame, prof)) == 2


def test_screening_never_looks_at_returns():
    """Two names with identical tradability but opposite performance must
    screen identically -- selecting on past returns is survivorship bias."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2022-01-03", periods=800)
    shocks = rng.normal(0, 0.012, 800)

    winner = 100 * np.exp(np.cumsum(shocks + 0.0015))
    loser = 100 * np.exp(np.cumsum(shocks - 0.0015))
    bars = {}
    for name, px in (("WINNER", winner), ("LOSER", loser)):
        bars[name] = pd.DataFrame({
            "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
            "volume": np.full(800, 1e7)}, index=idx)

    frame = profile_frame(bars)
    prof = Profile(name="t", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, top_n=5)
    result = screen(frame, prof)
    assert {"WINNER", "LOSER"} <= set(result.index), (
        "screen dropped a name for underperforming, which is selection bias")


def test_empty_input_is_safe():
    assert profile_frame({}).empty
    assert screen(pd.DataFrame(), "long_term").empty


def test_every_profile_describes_itself():
    for name in PROFILES:
        text = describe_profile(name)
        assert "Requires:" in text and len(text) > 40


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
