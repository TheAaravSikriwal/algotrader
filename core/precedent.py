"""What happened last time: outcomes that outlive the positions that made them.

Positions expire. Knowledge must not. A war in March is over as a trade within
weeks, but as a *precedent* it never stops being informative -- if the same
parties escalate again in September, what happened in March is evidence about
what happens next. An overlay that expires and vanishes teaches nothing; an
overlay that expires and leaves a scored outcome behind is one observation
toward an event study.

That is the whole learning mechanism, and it is deliberately not the model
learning. It is an accumulating empirical record fed back into the next
briefing. Each entry is a real measured outcome on a real instrument, and --
this is the part that makes it worth anything -- the reasoning was timestamped
before the outcome existed, so no precedent here can have been written with
hindsight.

The honest caveat, enforced in the summary rather than left to the reader: one
precedent is an anecdote. Two are a coincidence. The count is reported with
every match, and `confidence_note` says plainly which of those you are looking
at.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .overlay import LEDGER, Overlay, load_overlays
from .taxonomy import classify

OUTCOMES = Path(__file__).resolve().parent.parent / "research" / "outcomes.jsonl"


@dataclass
class Outcome:
    """One adjustment, and what the market actually did while it was live."""
    overlay_id: str
    symbol: str
    action: str
    effective: float
    issued: str
    expired: str
    reason: str
    evidence: str = ""
    bucket: str = ""
    categories: list = field(default_factory=list)
    abnormal_return: float = 0.0     # vs benchmark, over the live window
    raw_return: float = 0.0
    bars_held: int = 0
    scored_on: str = ""

    @property
    def direction_correct(self) -> bool:
        """Did the market move the way the tilt was pointing?"""
        if abs(self.effective) < 1e-9:
            return False
        return (self.effective > 0) == (self.abnormal_return > 0)

    @property
    def contribution(self) -> float:
        """Return the tilt actually earned, sign included."""
        return float(np.sign(self.effective) * self.abnormal_return)


def _window_return(series: pd.Series, start, end) -> tuple[float, int]:
    """Simple return between the first bar on/after start and on/before end."""
    if series is None or series.empty:
        return 0.0, 0
    index = series.index
    left = int(index.searchsorted(pd.Timestamp(start), side="left"))
    right = int(index.searchsorted(pd.Timestamp(end), side="right")) - 1
    if left >= len(index) or right <= left:
        return 0.0, 0
    first, last = float(series.iloc[left]), float(series.iloc[right])
    if first <= 0:
        return 0.0, 0
    return last / first - 1.0, right - left


def score_overlay(overlay: Overlay, bars: dict[str, pd.DataFrame],
                  benchmark: pd.Series | None = None,
                  as_of=None) -> list[Outcome]:
    """Measure what happened over each adjustment's live window.

    Returns are abnormal -- measured against the benchmark -- so a tilt that
    merely rode a rising market is not credited with foresight.
    """
    as_of = pd.Timestamp(as_of or date.today())
    issued = date.fromisoformat(overlay.issued)
    outcomes = []

    for item in overlay.adjustments:
        expiry = issued + timedelta(days=item.expires_days)
        if pd.Timestamp(expiry) > as_of:
            continue                      # still live, nothing to score yet

        df = bars.get(item.symbol)
        if df is None or df.empty:
            continue

        raw, held = _window_return(df["close"], issued, expiry)
        if held == 0:
            continue
        bench, _ = ((_window_return(benchmark, issued, expiry))
                    if benchmark is not None and not benchmark.empty else (0.0, 0))

        text = f"{item.evidence} {item.reason}".strip()
        tags = classify(text)
        outcomes.append(Outcome(
            overlay_id=overlay.id, symbol=item.symbol, action=item.action,
            effective=item.effective, issued=overlay.issued,
            expired=expiry.isoformat(), reason=item.reason,
            evidence=item.evidence, bucket=overlay.bucket,
            categories=sorted(set(tags["macro"] + tags["company"])),
            abnormal_return=raw - bench, raw_return=raw, bars_held=held,
            scored_on=date.today().isoformat()))
    return outcomes


def record_outcomes(outcomes: list[Outcome], path: Path = OUTCOMES) -> int:
    """Append outcomes, skipping any already scored."""
    existing = {(o["overlay_id"], o["symbol"]) for o in _read(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for outcome in outcomes:
            if (outcome.overlay_id, outcome.symbol) in existing:
                continue
            fh.write(json.dumps(asdict(outcome)) + "\n")
            written += 1
    return written


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


def load_outcomes(path: Path = OUTCOMES) -> list[Outcome]:
    known = set(Outcome.__dataclass_fields__)
    return [Outcome(**{k: v for k, v in row.items() if k in known})
            for row in _read(path)]


def score_pending(bars: dict[str, pd.DataFrame], benchmark: pd.Series | None = None,
                  ledger: Path = LEDGER, outcomes_path: Path = OUTCOMES,
                  as_of=None) -> list[Outcome]:
    """Score every expired adjustment that has not been scored yet."""
    already = {(o.overlay_id, o.symbol) for o in load_outcomes(outcomes_path)}
    fresh = []
    for overlay in load_overlays(ledger):
        for outcome in score_overlay(overlay, bars, benchmark, as_of):
            if (outcome.overlay_id, outcome.symbol) not in already:
                fresh.append(outcome)
    record_outcomes(fresh, outcomes_path)
    return fresh


# ---------------------------------------------------------------------------
# retrieval: what happened last time something like this came up
# ---------------------------------------------------------------------------
def find_precedents(categories: list[str] | None = None, symbol: str | None = None,
                    bucket: str | None = None,
                    outcomes: list[Outcome] | None = None,
                    path: Path = OUTCOMES) -> pd.DataFrame:
    """Past scored adjustments matching an event's shape.

    Matching is on category overlap and optionally the instrument. Deliberately
    loose: a Russia-Ukraine escalation and an Israel-Iran escalation are both
    geopolitical shocks to energy, and treating them as unrelated would throw
    away the only precedents a rare event ever has.
    """
    records = outcomes if outcomes is not None else load_outcomes(path)
    if not records:
        return pd.DataFrame()

    wanted = set(categories or [])
    rows = []
    for outcome in records:
        overlap = wanted & set(outcome.categories)
        if wanted and not overlap:
            continue
        if symbol and outcome.symbol != symbol.upper():
            continue
        if bucket and outcome.bucket != bucket:
            continue
        rows.append({
            "issued": outcome.issued, "symbol": outcome.symbol,
            "categories": ",".join(outcome.categories),
            "matched_on": ",".join(sorted(overlap)) if wanted else "",
            "tilt": outcome.effective,
            "abnormal_return": outcome.abnormal_return,
            "contribution": outcome.contribution,
            "correct": outcome.direction_correct,
            "bars_held": outcome.bars_held,
            "reason": outcome.reason,
        })

    frame = pd.DataFrame(rows)
    return frame.sort_values("issued", ascending=False) if not frame.empty else frame


def confidence_note(n: int) -> str:
    """Say plainly what this many precedents is worth."""
    if n == 0:
        return "No precedent on record. This is the first observation."
    if n == 1:
        return "One precedent. That is an anecdote, not evidence -- it tells you what happened once."
    if n < 5:
        return f"{n} precedents. Still closer to anecdote than evidence; a run of three can easily be luck."
    if n < 15:
        return f"{n} precedents. Suggestive, but well short of the sample an event study needs."
    return f"{n} precedents. Enough to look at the distribution rather than the individual cases."


def summarise_precedents(frame: pd.DataFrame) -> dict:
    """Hit rate and average outcome, with the sample size front and centre."""
    if frame is None or frame.empty:
        return {"n": 0, "note": confidence_note(0)}

    n = len(frame)
    contributions = frame["contribution"].astype(float)
    sd = contributions.std(ddof=1) if n > 1 else 0.0
    return {
        "n": n,
        "hit_rate": float(frame["correct"].mean()),
        "mean_contribution_%": float(contributions.mean() * 100),
        "mean_abnormal_%": float(frame["abnormal_return"].astype(float).mean() * 100),
        "tstat": float(contributions.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
        "symbols": int(frame["symbol"].nunique()),
        "note": confidence_note(n),
    }


def precedents_for_text(text: str, symbol: str | None = None,
                        path: Path = OUTCOMES) -> tuple[pd.DataFrame, dict]:
    """Everything on record resembling this headline."""
    tags = classify(text)
    categories = sorted(set(tags["macro"] + tags["company"]))
    frame = find_precedents(categories, symbol, path=path)
    return frame, summarise_precedents(frame)
