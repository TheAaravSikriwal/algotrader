"""A research ledger, so testing accumulates instead of evaporating.

Every backtest in this project has been a one-off: run it, read it, forget it.
That loses two things that matter more than any single result.

**How many times you have looked.** Test one strategy at the 5% level and a
t-stat past 2 means something. Test forty and roughly two will clear that bar
on noise alone -- you have not found an edge, you have found the best of forty
coin flips. The bar has to rise with the count, and only a ledger knows the
count.

**Whether an edge survived contact with new data.** Every backtest is run on
history someone has already picked over, including you. The one genuinely
unsearched dataset is the future. Registering a locked specification today and
scoring it in six months on bars that did not exist when you registered it is
the only test here that cannot be gamed -- not by optimisation, not by
survivorship, not by hindsight in choosing the universe.

Nothing in this module fits a model or predicts a price. It keeps score.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

JOURNAL_DIR = Path(__file__).resolve().parent.parent / "research"
JOURNAL_DIR.mkdir(exist_ok=True)

EXPERIMENTS = JOURNAL_DIR / "experiments.jsonl"
FORWARD = JOURNAL_DIR / "forward_tests.jsonl"

SIGNIFICANT = "significant"
SIGNIFICANT_NEGATIVE = "significant-negative"
NULL = "null"
UNDERPOWERED = "underpowered"


# ---------------------------------------------------------------------------
# statistics without a scipy dependency
# ---------------------------------------------------------------------------
def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def two_sided_p(tstat: float) -> float:
    """Large-sample p-value for a t-stat. Daily samples run to thousands of
    observations, where the t distribution is indistinguishable from normal."""
    return 2.0 * (1.0 - normal_cdf(abs(float(tstat))))


def critical_t(alpha: float) -> float:
    """Two-sided critical value for a given alpha, by bisection on the CDF."""
    if alpha <= 0:
        return float("inf")
    if alpha >= 1:
        return 0.0
    target = 1.0 - alpha / 2.0
    low, high = 0.0, 12.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if normal_cdf(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def bonferroni_bar(n_tests: int, alpha: float = 0.05) -> float:
    """The |t| a result must clear once you account for how often you looked.

    Controls the chance of *any* false positive across the whole family of
    tests. Strict by design: it is the honest bar when you intend to act on
    whichever strategy comes out on top.
    """
    return critical_t(alpha / max(int(n_tests), 1))


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Which results survive at a given false-discovery rate.

    Less brutal than Bonferroni: it accepts that some share of the discoveries
    will be false, and controls that share. Appropriate when the output is a
    shortlist to investigate rather than one strategy to fund.
    """
    n = len(pvalues)
    if n == 0:
        return []
    ordered = sorted(range(n), key=lambda i: pvalues[i])
    keep = [False] * n
    cutoff = -1
    for rank, idx in enumerate(ordered, start=1):
        if pvalues[idx] <= alpha * rank / n:
            cutoff = rank
    for rank, idx in enumerate(ordered, start=1):
        if rank <= cutoff:
            keep[idx] = True
    return keep


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    """One evaluated hypothesis."""
    hypothesis: str
    strategy: str
    params: dict = field(default_factory=dict)
    symbols: list = field(default_factory=list)
    universe: str = ""
    start: str = ""
    end: str = ""
    benchmark: str = "equal_weight"
    window: str = "holdout"          # in-sample | holdout | forward
    metrics: dict = field(default_factory=dict)
    tstat: float | None = None
    observations: int = 0
    notes: str = ""
    id: str = ""
    recorded: str = ""

    def __post_init__(self):
        if not self.recorded:
            self.recorded = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.id:
            seed = f"{self.strategy}|{sorted(self.params.items())}|{self.symbols}|{self.start}|{self.end}|{self.window}"
            self.id = hashlib.sha1(seed.encode()).hexdigest()[:12]

    @property
    def pvalue(self) -> float | None:
        return None if self.tstat is None else two_sided_p(self.tstat)

    def verdict(self, bar: float) -> str:
        """Judged against a bar that already accounts for the test count."""
        if self.tstat is None:
            return UNDERPOWERED
        if self.observations and self.observations < 100:
            return UNDERPOWERED
        if self.tstat >= bar:
            return SIGNIFICANT
        if self.tstat <= -bar:
            return SIGNIFICANT_NEGATIVE
        return NULL


@dataclass
class ForwardTest:
    """A specification locked today, to be scored on data that does not exist yet.

    The `registered` date is the whole point. Scoring only ever uses bars after
    it, so no amount of searching before registration can contaminate the
    result.
    """
    hypothesis: str
    strategy: str
    params: dict = field(default_factory=dict)
    symbols: list = field(default_factory=list)
    kind: str = "single"             # single | panel
    benchmark: str = "SPY"
    min_days: int = 180
    notes: str = ""
    id: str = ""
    registered: str = ""
    status: str = "open"             # open | scored | abandoned

    def __post_init__(self):
        if not self.registered:
            self.registered = date.today().isoformat()
        if not self.id:
            # The date is deliberately NOT in the seed. With it, the same spec
            # registered on two different days produced two different ids and
            # the duplicate guard never fired.
            seed = f"{self.strategy}|{sorted(self.params.items())}|{sorted(self.symbols)}"
            self.id = hashlib.sha1(seed.encode()).hexdigest()[:12]

    @property
    def days_elapsed(self) -> int:
        return (date.today() - date.fromisoformat(self.registered)).days

    @property
    def ready(self) -> bool:
        return self.days_elapsed >= self.min_days


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------
def _append(path: Path, payload: dict):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class Journal:
    def __init__(self, experiments: Path = EXPERIMENTS, forward: Path = FORWARD):
        self.experiments_path = experiments
        self.forward_path = forward

    # -- experiments ------------------------------------------------------
    def record(self, experiment: Experiment) -> Experiment:
        _append(self.experiments_path, asdict(experiment))
        return experiment

    def experiments(self) -> list[Experiment]:
        return [Experiment(**row) for row in _read(self.experiments_path)]

    def count(self) -> int:
        """Distinct hypotheses tested -- re-running the same spec is not a new look."""
        return len({e.id for e in self.experiments()})

    def searches(self) -> int:
        """Parameter combinations evaluated but never recorded as hypotheses.

        A tournament sweeps hundreds of settings per strategy per symbol and
        reports the best. Each of those is a look at the same data, and a bar
        computed from the handful of *recorded* hypotheses is calibrated on a
        fraction of the real search -- which makes everything that "failed"
        fail against a threshold that was far too lenient.
        """
        return sum(int(e.metrics.get("combinations_searched", 0) or 0)
                   for e in self.experiments())

    def total_looks(self) -> int:
        """Every look at the data, recorded hypotheses and sweeps alike."""
        return max(self.count(), 1) + self.searches()

    def bar(self, alpha: float = 0.05) -> float:
        """The |t| a new result must clear, given everything tried so far."""
        return bonferroni_bar(self.total_looks(), alpha)

    def frame(self, alpha: float = 0.05) -> pd.DataFrame:
        # Deduplicate by id, as `count` does. Feeding repeated rows to
        # Benjamini-Hochberg inflates its discovery count: identical small
        # p-values raise the rank cutoff, so logging one real finding five
        # times reported five discoveries.
        seen, records = set(), []
        for e in self.experiments():
            if e.id in seen:
                continue
            seen.add(e.id)
            records.append(e)
        if not records:
            return pd.DataFrame()

        bar = self.bar(alpha)
        rows = []
        for e in records:
            rows.append({
                "id": e.id, "recorded": e.recorded[:10], "strategy": e.strategy,
                "universe": e.universe or ",".join(e.symbols[:4]),
                "window": e.window, "start": e.start, "end": e.end,
                "return_%": e.metrics.get("return_%"),
                "sharpe": e.metrics.get("sharpe"),
                "max_dd_%": e.metrics.get("max_dd_%"),
                "t": e.tstat, "p": e.pvalue,
                "verdict": e.verdict(bar), "hypothesis": e.hypothesis,
            })
        frame = pd.DataFrame(rows)

        pvals = [r if r is not None else 1.0 for r in frame["p"]]
        frame["survives_fdr"] = benjamini_hochberg(pvals, alpha)
        return frame

    def summary(self, alpha: float = 0.05) -> dict:
        frame = self.frame(alpha)
        n = self.count()
        expected_false = n * alpha
        return {
            "tests_run": n,
            "uncorrected_bar": round(critical_t(alpha), 2),
            "corrected_bar": round(self.bar(alpha), 2),
            "expected_false_positives_uncorrected": round(expected_false, 1),
            "significant_positive": int((frame["verdict"] == SIGNIFICANT).sum())
                                    if not frame.empty else 0,
            "significant_negative": int((frame["verdict"] == SIGNIFICANT_NEGATIVE).sum())
                                    if not frame.empty else 0,
            "null": int((frame["verdict"] == NULL).sum()) if not frame.empty else 0,
            "survives_fdr": int(frame["survives_fdr"].sum()) if not frame.empty else 0,
        }

    # -- forward tests ----------------------------------------------------
    def register(self, test: ForwardTest) -> ForwardTest:
        # The whole value of a forward test is that its date was fixed before
        # the outcome existed. `registered` is an ordinary field, so a caller
        # could set it to 2015 and have the test score immediately against
        # "genuinely unseen data". The check lives here rather than in the
        # constructor because loading a historical record from the ledger
        # legitimately reconstructs one with a past date.
        today = date.today().isoformat()
        if test.registered < today:
            raise ValueError(
                f"cannot register a forward test dated {test.registered}: that "
                "is in the past, and a forward test is only worth anything if "
                "the outcome did not exist when it was written.")

        existing = {t.id for t in self.forward_tests()}
        if test.id in existing:
            raise ValueError(f"forward test {test.id} is already registered")
        _append(self.forward_path, asdict(test))
        return test

    def forward_tests(self) -> list[ForwardTest]:
        latest: dict[str, dict] = {}
        for row in _read(self.forward_path):
            latest[row["id"]] = row          # later lines supersede earlier ones
        return [ForwardTest(**row) for row in latest.values()]

    def open_forward_tests(self) -> list[ForwardTest]:
        return [t for t in self.forward_tests() if t.status == "open"]

    def ready_forward_tests(self) -> list[ForwardTest]:
        return [t for t in self.open_forward_tests() if t.ready]

    def close_forward_test(self, test_id: str, status: str = "scored"):
        tests = {t.id: t for t in self.forward_tests()}
        if test_id not in tests:
            raise KeyError(f"no forward test {test_id}")
        record = asdict(tests[test_id])
        record["status"] = status
        _append(self.forward_path, record)
