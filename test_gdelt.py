"""GDELT feature checks. Synthetic rows throughout -- no network."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.gdelt import (AVAILABILITY_LAG_DAYS, COL_TONE, GdeltError,
                        add_regime_features, attach_to_bars, _aggregate)


def row(event_date="20240315", quad=1, root="01", goldstein=0.0, tone=0.0,
        articles=1, actor1="USA", actor2="RUS"):
    """One GDELT export row with the fields the aggregator reads."""
    cells = [""] * (COL_TONE + 1)
    cells[0] = "1"
    cells[1] = event_date
    cells[7] = actor1
    cells[17] = actor2
    cells[28] = root
    cells[29] = str(quad)
    cells[30] = str(goldstein)
    cells[31] = "1"
    cells[32] = "1"
    cells[33] = str(articles)
    cells[34] = str(tone)
    return cells


def test_quad_class_shares_sum_to_one():
    rows = [row(quad=1), row(quad=2), row(quad=3), row(quad=4)]
    f = _aggregate(rows, date(2024, 3, 15))
    shares = sum(f[f"gdelt_{k}"] for k in
                 ("verbal_cooperation", "material_cooperation",
                  "verbal_conflict", "material_conflict"))
    assert abs(shares - 1.0) < 1e-9
    assert f["gdelt_events"] == 4.0


def test_tone_is_weighted_by_article_count():
    """A story carried by 100 outlets should outweigh one carried by 1."""
    quiet = _aggregate([row(tone=10.0, articles=1), row(tone=-10.0, articles=1)],
                       date(2024, 3, 15))
    loud = _aggregate([row(tone=10.0, articles=1), row(tone=-10.0, articles=99)],
                      date(2024, 3, 15))
    assert abs(quiet["gdelt_tone"]) < 1e-9
    assert loud["gdelt_tone"] < -8.0


def test_publication_lag_counts_backdated_events():
    rows = [row(event_date="20240315"), row(event_date="20240315"),
            row(event_date="20240101"), row(event_date="20230101")]
    f = _aggregate(rows, date(2024, 3, 15))
    assert abs(f["gdelt_publication_lag"] - 0.5) < 1e-9


def test_country_conflict_counts_material_conflict_only():
    rows = [row(quad=4, actor1="UKR", actor2="RUS"),
            row(quad=1, actor1="UKR", actor2="RUS"),   # verbal, must not count
            row(quad=4, actor1="CHN", actor2="TWN")]
    f = _aggregate(rows, date(2024, 3, 15), countries=("UKR", "RUS", "CHN"))
    assert f["gdelt_conflict_ukr"] == 1.0
    assert f["gdelt_conflict_rus"] == 1.0
    assert f["gdelt_conflict_chn"] == 1.0


def test_root_event_codes_are_tracked():
    rows = [row(root="19"), row(root="19"), row(root="14"), row(root="01")]
    f = _aggregate(rows, date(2024, 3, 15))
    assert abs(f["gdelt_fight"] - 0.5) < 1e-9
    assert abs(f["gdelt_protest"] - 0.25) < 1e-9


def test_malformed_rows_are_skipped_not_fatal():
    rows = [row(), ["too", "short"], row(tone="not-a-number")]
    f = _aggregate(rows, date(2024, 3, 15))
    assert f["gdelt_events"] == 1.0


def test_empty_file_raises_rather_than_returning_zeros():
    try:
        _aggregate([["short"]], date(2024, 3, 15))
        raise AssertionError("accepted a file with no usable rows")
    except GdeltError:
        pass


# ---------------------------------------------------------------------------
# the parts that could leak the future
# ---------------------------------------------------------------------------
def synthetic_features(n=60, start="2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "gdelt_material_conflict": rng.uniform(0.1, 0.2, n),
        "gdelt_tone": rng.uniform(-4, -1, n),
        "gdelt_goldstein": rng.uniform(-1, 1, n),
    }, index=idx)


def test_regime_features_are_causal():
    """Changing the last day must not move any earlier row."""
    base = synthetic_features()
    bumped = base.copy()
    bumped.iloc[-1, 0] = 5.0

    a = add_regime_features(base)
    b = add_regime_features(bumped)
    for column in ("gdelt_material_conflict_ma7", "gdelt_material_conflict_z30"):
        assert np.allclose(a[column].iloc[:-1], b[column].iloc[:-1],
                           equal_nan=True), f"{column} leaked backwards"


def test_attach_respects_publication_lag():
    """A day's file is not public until the next day, so it must not land on
    the bar for its own date."""
    features = pd.DataFrame({"gdelt_tone": [-9.0]},
                            index=pd.DatetimeIndex(["2024-03-14"]))
    bars = pd.DataFrame(index=pd.DatetimeIndex(
        ["2024-03-13", "2024-03-14", "2024-03-15", "2024-03-18"]))

    aligned = attach_to_bars(features, bars, carry_forward=0)
    assert aligned.loc["2024-03-14", "gdelt_tone"] == 0.0, "used a file that did not exist yet"
    assert aligned.loc["2024-03-15", "gdelt_tone"] == -9.0


def test_lag_constant_is_at_least_one_day():
    assert AVAILABILITY_LAG_DAYS >= 1


def test_weekend_events_roll_to_the_next_session():
    features = pd.DataFrame({"gdelt_tone": [-7.0]},
                            index=pd.DatetimeIndex(["2024-03-16"]))   # Saturday
    bars = pd.DataFrame(index=pd.DatetimeIndex(
        ["2024-03-15", "2024-03-18", "2024-03-19"]))
    aligned = attach_to_bars(features, bars, carry_forward=0)
    assert aligned.loc["2024-03-18", "gdelt_tone"] == -7.0
    assert aligned.loc["2024-03-15", "gdelt_tone"] == 0.0, "leaked backwards"


def test_carry_forward_holds_the_last_reading():
    features = pd.DataFrame({"gdelt_tone": [-5.0]},
                            index=pd.DatetimeIndex(["2024-03-14"]))
    bars = pd.DataFrame(index=pd.bdate_range("2024-03-15", periods=4))
    aligned = attach_to_bars(features, bars, carry_forward=2)
    assert aligned["gdelt_tone"].iloc[0] == -5.0
    assert aligned["gdelt_tone"].iloc[2] == -5.0
    assert aligned["gdelt_tone"].iloc[3] == 0.0, "carried further than allowed"


def test_a_genuine_zero_is_not_treated_as_missing_data():
    """Tone, Goldstein and publication lag can legitimately be exactly zero.
    Treating that as 'no file today' and forward-filling yesterday's number
    silently replaces real data with stale data."""
    features = pd.DataFrame(
        {"gdelt_tone": [-5.0, 0.0, -3.0]},
        index=pd.DatetimeIndex(["2024-03-11", "2024-03-12", "2024-03-13"]))
    bars = pd.DataFrame(index=pd.bdate_range("2024-03-12", periods=3))

    aligned = attach_to_bars(features, bars, carry_forward=5)
    assert aligned["gdelt_tone"].iloc[1] == 0.0, (
        f"a real zero became {aligned['gdelt_tone'].iloc[1]} -- yesterday's value")


def test_sessions_landing_on_one_bar_are_not_summed():
    """Friday, Saturday and Sunday all reach the Monday bar. Adding their tone
    together makes every Monday read three times the true level."""
    features = pd.DataFrame(
        {"gdelt_tone": [-2.0] * 3},
        index=pd.DatetimeIndex(["2024-03-15", "2024-03-16", "2024-03-17"]))
    bars = pd.DataFrame(index=pd.DatetimeIndex(["2024-03-18", "2024-03-19"]))

    aligned = attach_to_bars(features, bars, carry_forward=0)
    monday = aligned.loc["2024-03-18", "gdelt_tone"]
    assert abs(monday - (-2.0)) < 1e-9, (
        f"Monday reads {monday} -- the sessions were summed instead of combined")


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
