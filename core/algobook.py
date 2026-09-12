"""Your algorithms: what you own, which quadrant it lives in, and why.

The app is organised around four quadrants, split on two questions:

                    | fixed rule            | everchanging
    ----------------|-----------------------|---------------------------
    short term      | Short term            | Short term, live
    long term       | Long term             | Long term, live

**Fixed** means the rule was backtested once and then left alone. Its
parameters do not move. You can trust its backtest because the rule that was
tested is the rule that runs.

**Everchanging** ("live") means the rule is reviewed as conditions change --
news arrives, the position moves, the market regime shifts -- and may be
swapped for a better-evidenced one or rebuilt for the moment. That review is
deliberately a human step, not an automatic refit: an algorithm that quietly
rewrites itself has no backtest at all, because whatever was tested is not
what is running.

The horizon split is measured, not declared. Each strategy's average holding
period comes out of the evaluation in `research/results/evaluate_all.csv`
(exposure x bars / trades), and anything held under ten days on average is
short term. That keeps the classification honest -- a rule is short term
because it actually trades quickly, not because someone labelled it so.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

BOOK = Path(__file__).resolve().parent.parent / "research" / "algobook.json"
EVAL = Path(__file__).resolve().parent.parent / "research" / "results" / "evaluate_all.csv"

SHORT_TERM = "short_term"
LONG_TERM = "long_term"
SHORT_LIVE = "short_live"
LONG_LIVE = "long_live"

#: Average holding period, in bars, at or above which a rule counts as long term.
LONG_TERM_BARS = 10.0


@dataclass(frozen=True)
class Quadrant:
    key: str
    title: str
    blurb: str
    detail: str
    horizon: str          # short | long
    live: bool


QUADRANTS: dict[str, Quadrant] = {
    SHORT_TERM: Quadrant(
        key=SHORT_TERM,
        title="Short term",
        blurb="In and out within days. Fixed rules.",
        detail=("Rules that hold a position for hours to about a week, then "
                "close it. The rule never changes, so the backtest you see is "
                "the rule that runs. Higher turnover means costs matter more "
                "here than anywhere else."),
        horizon="short", live=False),
    LONG_TERM: Quadrant(
        key=LONG_TERM,
        title="Long term",
        blurb="Held for weeks to years. Fixed rules.",
        detail=("Rules that hold for ten days or longer -- often months. Fewer "
                "trades, so costs bite less, but you wait far longer to find "
                "out whether the rule works."),
        horizon="long", live=False),
    SHORT_LIVE: Quadrant(
        key=SHORT_LIVE,
        title="Short term, everchanging",
        blurb="Reviewed as the market moves. Day trading.",
        detail=("The same short-horizon rules, but reviewed while you are in "
                "the position. When the stock moves or news lands, the app "
                "re-checks whether the rule you are running is still the "
                "best-evidenced one, and names a better tested rule if it is "
                "not. Changing it is your decision, not the app's."),
        horizon="short", live=True),
    LONG_LIVE: Quadrant(
        key=LONG_LIVE,
        title="Long term, everchanging",
        blurb="Held long, adjusted as the world changes.",
        detail=("Long-horizon holdings whose weights are tilted by news and "
                "macro context rather than left untouched. The base rule sets "
                "the positions; the review adjusts how much of each."),
        horizon="long", live=True),
}


@dataclass
class SavedAlgo:
    """One algorithm you have added, plus the evidence behind it."""
    name: str
    quadrant: str
    strategy: str                          # a name from the strategy registry
    params: dict = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    stop_loss_pct: float = 0.0             # 0 means the rule exits on its own signal
    take_profit_pct: float = 0.0
    rationale: str = ""                    # why this rule is supposed to work
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: str = field(default_factory=lambda: date.today().isoformat())
    backtest: dict = field(default_factory=dict)
    active: bool = True

    def __post_init__(self):
        if self.quadrant not in QUADRANTS:
            raise ValueError(
                f"unknown quadrant {self.quadrant!r}; expected one of "
                f"{sorted(QUADRANTS)}")
        if not str(self.name).strip():
            raise ValueError("an algorithm needs a name")
        if self.stop_loss_pct < 0 or self.take_profit_pct < 0:
            raise ValueError("stop loss and take profit cannot be negative")
        self.symbols = [s.strip().upper() for s in self.symbols if str(s).strip()]

    @property
    def quadrant_title(self) -> str:
        return QUADRANTS[self.quadrant].title

    @property
    def is_live(self) -> bool:
        return QUADRANTS[self.quadrant].live

    @property
    def tested(self) -> bool:
        return bool(self.backtest) and "total_return_pct" in self.backtest

    def summary(self) -> str:
        """One line for a list view."""
        if not self.tested:
            return "not tested yet"
        b = self.backtest
        return (f"{b['total_return_pct']:+.1f}% over the test, "
                f"{b.get('vs_hold_pct', 0):+.1f}% vs holding")


class AlgoBook:
    """Everything you have saved. A plain JSON file, readable without the app."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or BOOK)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return raw if isinstance(raw, list) else []

    def all(self) -> list[SavedAlgo]:
        out = []
        for row in self._read():
            try:
                out.append(SavedAlgo(**row))
            except (TypeError, ValueError):
                continue          # a row from an older shape, skipped not guessed
        return out

    def by_quadrant(self, quadrant: str) -> list[SavedAlgo]:
        return [a for a in self.all() if a.quadrant == quadrant]

    def get(self, algo_id: str) -> SavedAlgo | None:
        return next((a for a in self.all() if a.id == algo_id), None)

    def add(self, algo: SavedAlgo) -> SavedAlgo:
        rows = self._read()
        rows.append(asdict(algo))
        self._write(rows)
        return algo

    def update(self, algo: SavedAlgo) -> SavedAlgo:
        rows = [asdict(algo) if r.get("id") == algo.id else r
                for r in self._read()]
        self._write(rows)
        return algo

    def remove(self, algo_id: str) -> bool:
        rows = self._read()
        kept = [r for r in rows if r.get("id") != algo_id]
        self._write(kept)
        return len(kept) != len(rows)

    def _write(self, rows: list[dict]):
        self.path.write_text(json.dumps(rows, indent=1, default=str),
                             encoding="utf-8")


def quadrant_totals(book: AlgoBook) -> dict[str, dict]:
    """Holistic gain or loss per quadrant, for the landing page.

    Averages the saved backtests across the algorithms in each quadrant. This
    is *backtested* performance, not money you made -- the distinction is
    stated wherever the number is shown, because a landing page full of green
    percentages that were never traded is exactly how people talk themselves
    into funding something.
    """
    out = {}
    for key, q in QUADRANTS.items():
        algos = [a for a in book.by_quadrant(key) if a.tested]
        if not algos:
            out[key] = {"algos": len(book.by_quadrant(key)), "tested": 0,
                        "total_return_pct": None, "vs_hold_pct": None,
                        "last_month_pct": None, "beat_hold": 0}
            continue
        rets = [a.backtest["total_return_pct"] for a in algos]
        vs = [a.backtest.get("vs_hold_pct", 0.0) for a in algos]
        last = [a.backtest.get("last_month_pct") for a in algos
                if a.backtest.get("last_month_pct") is not None]
        out[key] = {
            "algos": len(book.by_quadrant(key)),
            "tested": len(algos),
            "total_return_pct": float(sum(rets) / len(rets)),
            "vs_hold_pct": float(sum(vs) / len(vs)),
            "last_month_pct": float(sum(last) / len(last)) if last else None,
            "beat_hold": sum(1 for v in vs if v > 0),
        }
    return out


def measured_horizons(path: Path | None = None) -> dict[str, float]:
    """Average holding period in bars, per strategy, from the evaluation run.

    Returns an empty mapping when the evaluation has not been run -- callers
    fall back to asking the user rather than inventing a classification.
    """
    path = Path(path or EVAL)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if not {"strategy", "exposure", "trades"} <= set(df.columns):
        return {}

    df = df[df["trades"] > 0].copy()
    # 1,676 daily bars span the evaluation window (2010-2026). Holding period
    # is the time actually in a position divided by the number of round trips.
    df["avg_hold"] = df["exposure"] * 1676.0 / df["trades"]
    return df.groupby("strategy")["avg_hold"].median().to_dict()


def suggest_quadrant(strategy: str, live: bool = False,
                     horizons: dict[str, float] | None = None) -> str:
    """Which quadrant a strategy belongs in, from its measured holding period.

    Falls back to long term when the strategy has never been evaluated, which
    is the conservative side of the error: mislabelling a fast rule as slow
    understates how much its costs matter, and the app says so either way.
    """
    horizons = measured_horizons() if horizons is None else horizons
    hold = horizons.get(strategy)
    short = hold is not None and hold < LONG_TERM_BARS
    if short:
        return SHORT_LIVE if live else SHORT_TERM
    return LONG_LIVE if live else LONG_TERM
