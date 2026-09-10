"""The reasoning layer's output: bounded, expiring tilts on a base strategy.

The design constraint that shapes everything here: **the base strategy stays
deterministic and the overlay only tilts it.** A reasoning layer that replaces
the strategy is unfalsifiable -- you can never tell whether the rules or the
narrative produced the result. Keeping the base fixed means the overlay's
contribution is exactly `overlaid minus base`, which is a number you can test.

Three safety properties, each because the failure mode is real:

  * **Everything expires.** A war narrative that made sense in March should not
    still be sitting in the book in September because nobody remembered to
    remove it. No expiry, no adjustment.
  * **Total tilt is capped.** One confident, wrong call should cost a slice of
    the portfolio, not the portfolio.
  * **Every overlay is journaled at issue time**, before any outcome is known.
    That timestamp is what makes the reasoning layer forward-testable rather
    than a story told afterwards -- see the note in core/journal.py.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SCHEMA = "algotrader.overlay.v1"
LEDGER = Path(__file__).resolve().parent.parent / "research" / "overlays.jsonl"

ACTIONS = ("tilt", "block", "scale")
DEFAULT_EXPIRY_DAYS = 21
DEFAULT_MAX_TILT = 0.30           # total absolute adjustment across the book
MAX_SINGLE_TILT = 0.15            # any one name


class OverlayError(ValueError):
    pass


@dataclass
class Adjustment:
    symbol: str
    action: str = "tilt"
    weight: float = 0.0            # tilt: additive. scale: multiplier.
    reason: str = ""
    confidence: float = 0.5        # 0..1, scales the tilt
    expires_days: int = DEFAULT_EXPIRY_DAYS
    evidence: str = ""             # headline or url the claim rests on

    def __post_init__(self):
        self.symbol = str(self.symbol).upper().strip()
        self.action = str(self.action).lower().strip()
        if self.action not in ACTIONS:
            raise OverlayError(
                f"{self.symbol}: unknown action {self.action!r}. "
                f"Use one of {', '.join(ACTIONS)}.")
        if not self.symbol:
            raise OverlayError("an adjustment needs a symbol")
        if not self.reason.strip():
            raise OverlayError(
                f"{self.symbol}: every adjustment needs a reason. An untraceable "
                "tilt cannot be reviewed or scored later.")

        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        self.expires_days = max(int(self.expires_days), 1)

        if self.action == "tilt":
            capped = min(max(float(self.weight), -MAX_SINGLE_TILT), MAX_SINGLE_TILT)
            self.clipped = abs(capped - float(self.weight)) > 1e-9
            self.weight = capped
        elif self.action == "scale":
            self.weight = float(min(max(self.weight, 0.0), 2.0))
            self.clipped = False
        else:                       # block
            self.weight = 0.0
            self.clipped = False

    @property
    def effective(self) -> float:
        """Tilt after confidence weighting. Low conviction moves less."""
        return self.weight * self.confidence if self.action == "tilt" else self.weight


@dataclass
class Overlay:
    rationale: str
    adjustments: list = field(default_factory=list)
    bucket: str = "company"        # company | macro
    issued: str = ""
    author: str = "claude-code"
    id: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.issued:
            self.issued = date.today().isoformat()
        self.adjustments = [a if isinstance(a, Adjustment) else Adjustment(**a)
                            for a in self.adjustments]
        if not self.adjustments:
            raise OverlayError("an overlay with no adjustments does nothing")
        if not self.id:
            seed = f"{self.issued}|{self.bucket}|" + "|".join(
                f"{a.symbol}:{a.action}:{a.weight}" for a in self.adjustments)
            self.id = hashlib.sha1(seed.encode()).hexdigest()[:12]

    def active_on(self, as_of) -> list[Adjustment]:
        """Adjustments that have not yet expired."""
        as_of = pd.Timestamp(as_of).date()
        issued = date.fromisoformat(self.issued)
        return [a for a in self.adjustments
                if as_of <= issued + timedelta(days=a.expires_days)]

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "id": self.id, "issued": self.issued,
            "bucket": self.bucket, "author": self.author,
            "rationale": self.rationale, "notes": self.notes,
            "adjustments": [
                {k: v for k, v in asdict(a).items() if k != "clipped"}
                for a in self.adjustments],
        }


def parse_overlay(payload) -> Overlay:
    """Read an overlay from JSON text, a dict, or a file path."""
    if isinstance(payload, Overlay):
        return payload
    if isinstance(payload, (str, bytes)):
        text = payload.decode() if isinstance(payload, bytes) else payload
        candidate = Path(text) if len(text) < 260 and not text.lstrip().startswith("{") else None
        if candidate is not None and candidate.exists():
            text = candidate.read_text(encoding="utf-8")
        # tolerate a fenced code block, since this arrives pasted from chat
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```")[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise OverlayError(f"not valid JSON: {exc}") from exc
    elif isinstance(payload, dict):
        data = payload
    else:
        raise OverlayError(f"cannot read an overlay from {type(payload).__name__}")

    if data.get("schema") not in (SCHEMA, None):
        raise OverlayError(f"unexpected schema {data.get('schema')!r}, want {SCHEMA}")
    known = {"rationale", "adjustments", "bucket", "issued", "author", "id", "notes"}
    return Overlay(**{k: v for k, v in data.items() if k in known})


def apply_overlay(base: pd.Series, overlay: Overlay, as_of=None,
                  max_tilt: float = DEFAULT_MAX_TILT,
                  allow_short: bool = False) -> tuple[pd.Series, dict]:
    """Tilt a base weight vector. Returns (adjusted weights, report).

    Gross exposure is preserved: the overlay changes *what* is held, not how
    much leverage is taken. A reasoning layer that could quietly raise total
    exposure would conflate a view with a bet on size.
    """
    as_of = as_of or date.today()
    active = overlay.active_on(as_of)
    report = {
        "overlay_id": overlay.id, "as_of": str(as_of),
        "applied": [], "expired": len(overlay.adjustments) - len(active),
        "unknown_symbols": [], "scaled_by": 1.0, "clipped": [],
    }
    if base.empty or not active:
        return base.copy(), report

    adjusted = base.astype(float).copy()
    tilts = pd.Series(0.0, index=adjusted.index)

    for item in active:
        if item.symbol not in adjusted.index:
            report["unknown_symbols"].append(item.symbol)
            continue
        if getattr(item, "clipped", False):
            report["clipped"].append(item.symbol)

        if item.action == "tilt":
            tilts[item.symbol] += item.effective
        elif item.action == "block":
            tilts[item.symbol] -= adjusted[item.symbol]
        elif item.action == "scale":
            tilts[item.symbol] += adjusted[item.symbol] * (item.weight - 1.0)
        report["applied"].append(
            {"symbol": item.symbol, "action": item.action,
             "effective": round(item.effective, 4), "reason": item.reason})

    # one confident wrong call should cost a slice, not the book
    total = float(tilts.abs().sum())
    if total > max_tilt > 0:
        scale = max_tilt / total
        tilts *= scale
        report["scaled_by"] = round(scale, 4)

    adjusted = adjusted + tilts
    if not allow_short:
        adjusted = adjusted.clip(lower=0.0)

    gross_before = float(base.abs().sum())
    gross_after = float(adjusted.abs().sum())
    if gross_after > 1e-12 and gross_before > 0:
        adjusted *= gross_before / gross_after

    report["gross_before"] = gross_before
    report["gross_after"] = float(adjusted.abs().sum())
    report["turnover"] = float((adjusted - base).abs().sum())
    return adjusted, report


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------
def record(overlay: Overlay, path: Path | None = None) -> Overlay:
    """Journal an overlay at issue time, before any outcome is known."""
    path = path or LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(overlay.to_dict()) + "\n")
    return overlay


def load_overlays(path: Path | None = None) -> list[Overlay]:
    # resolved at call time, not bound as a default -- a module constant used
    # as a default argument freezes at import and cannot be redirected
    path = path or LEDGER
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(parse_overlay(json.loads(line)))
        except (json.JSONDecodeError, OverlayError):
            continue
    return out


def active_overlays(as_of=None, bucket: str | None = None,
                    path: Path | None = None) -> list[Overlay]:
    as_of = as_of or date.today()
    return [o for o in load_overlays(path)
            if o.active_on(as_of) and (bucket is None or o.bucket == bucket)]


def ledger_frame(path: Path | None = None) -> pd.DataFrame:
    """One row per adjustment, for review and later scoring."""
    rows = []
    for overlay in load_overlays(path):
        issued = date.fromisoformat(overlay.issued)
        for item in overlay.adjustments:
            expiry = issued + timedelta(days=item.expires_days)
            rows.append({
                "overlay": overlay.id, "issued": overlay.issued,
                "bucket": overlay.bucket, "symbol": item.symbol,
                "action": item.action, "weight": item.weight,
                "confidence": item.confidence, "effective": item.effective,
                "expires": expiry.isoformat(),
                "active": date.today() <= expiry,
                "reason": item.reason, "evidence": item.evidence,
            })
    columns = ["overlay", "issued", "bucket", "symbol", "action", "weight",
               "confidence", "effective", "expires", "active", "reason", "evidence"]
    return pd.DataFrame(rows, columns=columns)
