"""Company vs macro classification checks. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.taxonomy import (COMPANY_CATEGORIES, MACRO_CATEGORIES, asset_type,
                           bucket_of, classify, profile_texts, top_categories)
from core.universe import Profile, profile_frame, screen


def test_company_events_land_in_the_company_bucket():
    tags = classify("Apple reports Q3 earnings, beats revenue estimates")
    assert "earnings" in tags["company"]
    assert bucket_of("Apple reports Q3 earnings, beats estimates") == "company"


def test_macro_events_land_in_the_macro_bucket():
    tags = classify("Federal Reserve holds interest rates steady at FOMC meeting")
    assert "monetary" in tags["macro"]
    assert not tags["company"]
    assert bucket_of("Fed holds interest rates steady") == "macro"


def test_a_story_can_be_both():
    """Forcing one label would discard half the information."""
    text = "Fed rate hike squeezes bank earnings across the sector"
    tags = classify(text)
    assert "monetary" in tags["macro"]
    assert "earnings" in tags["company"]
    assert bucket_of(text) == "both"


def test_unclassified_is_its_own_answer():
    assert bucket_of("The company will hold a meeting on Tuesday") == "unclassified"
    assert bucket_of("") == "unclassified"


def test_new_company_categories_match():
    assert "labor" in classify("Company announces layoffs of 3,000 staff")["company"]
    assert "supply_chain" in classify("Chip shortage forces production halt")["company"]


def test_macro_subcategories_are_distinguished():
    cases = {
        "OPEC cuts output, crude jumps above $90 per barrel": "energy",
        "CPI inflation cools to 2.1% in latest consumer price report": "inflation",
        "Missile strike prompts new sanctions and troops on the border": "geopolitical",
        "New tariff on imports escalates the trade war": "trade",
        "Nonfarm payrolls beat as unemployment falls": "employment",
        "Debt ceiling standoff raises government shutdown risk": "fiscal",
    }
    for text, expected in cases.items():
        assert expected in classify(text)["macro"], f"{expected} missed in: {text}"


def test_etfs_are_recognised():
    assert asset_type("SPY") == "etf"
    assert asset_type("xlk") == "etf"
    assert asset_type("NVDA") == "company"
    assert asset_type("SOFI") == "company"


# ---------------------------------------------------------------------------
# per-stock profiles
# ---------------------------------------------------------------------------
def test_profile_shares_reflect_the_mix():
    texts = ["Apple beats earnings estimates",          # company
             "Microsoft reports quarterly results",      # company
             "Fed holds interest rates steady",          # macro
             "Weather is pleasant today"]                # unclassified
    p = profile_texts(texts)
    assert p["articles"] == 4.0
    assert abs(p["company_share"] - 0.5) < 1e-9
    assert abs(p["macro_share"] - 0.25) < 1e-9
    assert abs(p["unclassified_share"] - 0.25) < 1e-9
    assert abs(p["company_earnings"] - 0.5) < 1e-9


def test_empty_profile_is_zeroed_not_missing():
    p = profile_texts([])
    assert p["articles"] == 0.0
    assert p["company_share"] == 0.0
    assert all(f"company_{k}" in p for k in COMPANY_CATEGORIES)
    assert all(f"macro_{k}" in p for k in MACRO_CATEGORIES)


def test_top_categories_ranks_by_share():
    p = profile_texts([
        "Fed raises interest rates", "Fed cuts interest rates",
        "Fed signals rate path", "Oil prices jump as OPEC meets"])
    top = top_categories(p, "macro", 2)
    assert top[0][0] == "monetary"
    assert top[0][1] > top[1][1]


def test_both_counts_toward_each_bucket():
    p = profile_texts(["Fed rate hike squeezes bank earnings"])
    assert p["both_share"] == 1.0
    assert p["company_earnings"] == 1.0
    assert p["macro_monetary"] == 1.0


# ---------------------------------------------------------------------------
# screening integration
# ---------------------------------------------------------------------------
def bars_for(symbols, n=800):
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2022-01-03", periods=n)
    out = {}
    for s in symbols:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.014, n)))
        out[s] = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                               "close": px, "volume": np.full(n, 1e7)}, index=idx)
    return out


def test_company_profile_excludes_etfs():
    """News 'about SPY' is market commentary, not a corporate event."""
    bars = bars_for(["NVDA", "SPY", "XLK"])
    counts = {"NVDA": 5.0, "SPY": 20.0, "XLK": 3.0}
    profiles = {s: profile_texts(["Company beats earnings estimates"] * 10)
                for s in bars}
    frame = profile_frame(bars, counts, 252, profiles)

    prof = Profile(name="c", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, asset_types=("company",))
    result = screen(frame, prof)
    assert "NVDA" in result.index
    assert "SPY" not in result.index and "XLK" not in result.index


def test_macro_profile_keeps_only_baskets():
    bars = bars_for(["NVDA", "SPY", "XLK"])
    frame = profile_frame(bars, {s: 5.0 for s in bars}, 252, None)
    prof = Profile(name="m", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, asset_types=("etf",))
    result = screen(frame, prof)
    assert set(result.index) == {"SPY", "XLK"}


def test_bucket_share_threshold_filters_mismatched_coverage():
    """A name whose coverage is all macro should not pass a company screen."""
    bars = bars_for(["AAA", "BBB"])
    profiles = {
        "AAA": profile_texts(["Company beats earnings estimates"] * 10),
        "BBB": profile_texts(["Fed holds interest rates steady"] * 10),
    }
    frame = profile_frame(bars, {"AAA": 5.0, "BBB": 5.0}, 252, profiles)
    prof = Profile(name="c", rationale="", min_dollar_volume=1.0,
                   min_history_days=100, bucket="company", min_bucket_share=0.5)
    result = screen(frame, prof)
    assert "AAA" in result.index
    assert "BBB" not in result.index, "kept a name covered only for macro news"


def test_bucket_news_per_day_is_derived():
    bars = bars_for(["AAA"])
    profiles = {"AAA": profile_texts(
        ["Company beats earnings"] * 5 + ["Fed holds rates"] * 5)}
    frame = profile_frame(bars, {"AAA": 10.0}, 252, profiles)
    assert abs(frame.loc["AAA", "company_news_per_day"] - 5.0) < 1e-6
    assert abs(frame.loc["AAA", "macro_news_per_day"] - 5.0) < 1e-6


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
