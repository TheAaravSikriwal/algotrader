"""Research journal checks -- the statistics and the ledger."""
from __future__ import annotations

import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

from core.journal import (Experiment, ForwardTest, Journal, benjamini_hochberg,
                          bonferroni_bar, critical_t, normal_cdf, two_sided_p)


def temp_journal():
    tmp = Path(tempfile.mkdtemp())
    return Journal(tmp / "exp.jsonl", tmp / "fwd.jsonl"), tmp


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_normal_cdf_matches_known_values():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-9
    assert abs(normal_cdf(1.96) - 0.975) < 1e-3
    assert abs(normal_cdf(-1.96) - 0.025) < 1e-3


def test_critical_t_is_the_familiar_196():
    assert abs(critical_t(0.05) - 1.96) < 0.01
    assert abs(critical_t(0.01) - 2.576) < 0.01


def test_two_sided_p_round_trips():
    assert abs(two_sided_p(1.96) - 0.05) < 1e-3
    assert two_sided_p(0.0) == 1.0
    assert two_sided_p(5.0) < 1e-5


def test_bar_rises_with_the_number_of_tests():
    """The core anti-p-hacking property: more looks, higher bar."""
    one = bonferroni_bar(1)
    twenty = bonferroni_bar(20)
    hundred = bonferroni_bar(100)
    assert abs(one - 1.96) < 0.01
    assert twenty > one and hundred > twenty
    assert twenty > 3.0, f"20 tests should demand |t| > 3, got {twenty:.2f}"


def test_benjamini_hochberg_keeps_the_strong_and_drops_the_weak():
    pvals = [0.001, 0.002, 0.60, 0.80, 0.95]
    keep = benjamini_hochberg(pvals, 0.05)
    assert keep[0] and keep[1]
    assert not any(keep[2:])


def test_benjamini_hochberg_rejects_all_noise():
    assert not any(benjamini_hochberg([0.4, 0.5, 0.6, 0.7, 0.9], 0.05))


def test_benjamini_hochberg_handles_empty():
    assert benjamini_hochberg([], 0.05) == []


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------
def make_experiment(strategy="X", tstat=1.0, obs=1000, **kw):
    return Experiment(hypothesis="h", strategy=strategy, symbols=["SPY"],
                      start="2015-01-01", end="2025-01-01",
                      metrics={"return_%": 10.0, "sharpe": 0.5},
                      tstat=tstat, observations=obs, **kw)


def test_record_and_read_back():
    journal, tmp = temp_journal()
    journal.record(make_experiment())
    assert journal.count() == 1
    assert journal.experiments()[0].strategy == "X"
    shutil.rmtree(tmp, ignore_errors=True)


def test_identical_specs_count_once():
    """Re-running the same test is not a new look at the data."""
    journal, tmp = temp_journal()
    journal.record(make_experiment(strategy="Same"))
    journal.record(make_experiment(strategy="Same"))
    assert journal.count() == 1, "re-running inflated the test count"
    shutil.rmtree(tmp, ignore_errors=True)


def test_different_specs_each_raise_the_bar():
    journal, tmp = temp_journal()
    first = journal.bar()
    for i in range(10):
        journal.record(make_experiment(strategy=f"S{i}"))
    assert journal.bar() > first
    shutil.rmtree(tmp, ignore_errors=True)


def test_a_strong_result_survives_a_short_journal():
    journal, tmp = temp_journal()
    journal.record(make_experiment(strategy="Strong", tstat=4.5))
    assert journal.frame().iloc[0]["verdict"] == "significant"
    shutil.rmtree(tmp, ignore_errors=True)


def test_the_same_result_fails_once_enough_has_been_tested():
    """t = 2.3 is a finding after one test and noise after forty."""
    journal, tmp = temp_journal()
    journal.record(make_experiment(strategy="Borderline", tstat=2.3))
    assert journal.frame().iloc[0]["verdict"] == "significant"

    for i in range(40):
        journal.record(make_experiment(strategy=f"Other{i}", tstat=0.4))

    frame = journal.frame()
    row = frame[frame["strategy"] == "Borderline"].iloc[0]
    assert row["verdict"] == "null", "the bar did not rise with the test count"
    shutil.rmtree(tmp, ignore_errors=True)


def test_significant_negatives_are_labelled_separately():
    journal, tmp = temp_journal()
    journal.record(make_experiment(strategy="Bad", tstat=-4.0))
    assert journal.frame().iloc[0]["verdict"] == "significant-negative"
    shutil.rmtree(tmp, ignore_errors=True)


def test_too_few_observations_is_underpowered_not_null():
    journal, tmp = temp_journal()
    journal.record(make_experiment(strategy="Thin", tstat=3.0, obs=40))
    assert journal.frame().iloc[0]["verdict"] == "underpowered"
    shutil.rmtree(tmp, ignore_errors=True)


def test_summary_counts_expected_false_positives():
    journal, tmp = temp_journal()
    for i in range(20):
        journal.record(make_experiment(strategy=f"S{i}", tstat=0.5))
    summary = journal.summary(0.05)
    assert summary["tests_run"] == 20
    assert abs(summary["expected_false_positives_uncorrected"] - 1.0) < 1e-9
    assert summary["corrected_bar"] > summary["uncorrected_bar"]
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# forward tests
# ---------------------------------------------------------------------------
def test_forward_test_registers_and_is_not_ready_immediately():
    journal, tmp = temp_journal()
    test = journal.register(ForwardTest(hypothesis="h", strategy="X",
                                        symbols=["SPY"], min_days=180))
    assert test.days_elapsed == 0
    assert not test.ready
    assert len(journal.open_forward_tests()) == 1
    assert journal.ready_forward_tests() == []
    shutil.rmtree(tmp, ignore_errors=True)


def test_forward_test_becomes_ready_after_enough_time():
    journal, tmp = temp_journal()
    old = (date.today() - timedelta(days=200)).isoformat()
    journal.register(ForwardTest(hypothesis="h", strategy="X", symbols=["SPY"],
                                 min_days=180, registered=old))
    assert len(journal.ready_forward_tests()) == 1
    shutil.rmtree(tmp, ignore_errors=True)


def test_duplicate_registration_is_rejected():
    journal, tmp = temp_journal()
    spec = dict(hypothesis="h", strategy="X", symbols=["SPY"])
    journal.register(ForwardTest(**spec))
    try:
        journal.register(ForwardTest(**spec))
        raise AssertionError("registered the same spec twice")
    except ValueError:
        pass
    shutil.rmtree(tmp, ignore_errors=True)


def test_closing_a_forward_test_removes_it_from_open():
    journal, tmp = temp_journal()
    test = journal.register(ForwardTest(hypothesis="h", strategy="X",
                                        symbols=["SPY"]))
    journal.close_forward_test(test.id)
    assert journal.open_forward_tests() == []
    assert len(journal.forward_tests()) == 1, "the record itself was lost"
    assert journal.forward_tests()[0].status == "scored"
    shutil.rmtree(tmp, ignore_errors=True)


def test_registration_date_is_locked_at_creation():
    """Scoring must only ever use data after this date."""
    test = ForwardTest(hypothesis="h", strategy="X", symbols=["SPY"])
    assert test.registered == date.today().isoformat()
    assert test.id, "a forward test must be identifiable"


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
