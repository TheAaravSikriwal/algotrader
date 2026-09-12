"""Fill quality: what you actually paid versus what the backtest assumed.

Every backtest in this repo makes an execution assumption. `core.engine` fills
at the next bar's open. A limit strategy assumes the limit fills whenever the
bar's range covers it. Those assumptions are invisible until something records
what really happened, and an edge of a few basis points is exactly the size
that a wrong assumption erases.

So the loop writes a row per order attempt, and `summarise` turns the rows into
the three numbers that decide whether a strategy is tradeable:

  * **fill rate** -- how often orders the backtest counted as fills really fill.
    A resting limit is the case that bites: the backtest fills it whenever the
    bar trades through, but in life it fills when someone wants the other side,
    which is disproportionately when they are right and you are wrong.
  * **slippage** -- signed so that positive always means *worse for you*,
    whichever way you were going. Averaging raw price differences across buys
    and sells hides the cost by letting the two sides cancel.
  * **effective spread** -- distance from the mid at the moment of the order,
    doubled. Comparable to the quoted-spread numbers a cost model uses.

Nothing here places orders or decides anything. It only records and measures.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BPS = 10_000.0


@dataclass
class FillRecord:
    """One order attempt, from decision to outcome.

    `reference_price` is the price the decision was made against -- the mid,
    or the last trade if no quote was available. Slippage is measured from it,
    so it must be captured *before* the order goes out. Recording it afterwards
    measures nothing: the order has already moved the price it would be
    compared against.
    """

    symbol: str
    side: str                         # buy | sell
    qty: float                        # shares intended
    reference_price: float            # mid/last at decision time
    order_type: str = "market"        # market | limit
    limit_price: float | None = None
    filled_qty: float = 0.0
    filled_price: float | None = None
    status: str = "unfilled"          # filled | partial | unfilled | cancelled | rejected
    strategy: str = ""
    order_id: str = ""
    bid: float | None = None          # quote at decision time, when available
    ask: float | None = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        self.side = str(self.side).lower()
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be buy or sell, got {self.side!r}")
        if self.reference_price <= 0:
            raise ValueError("reference_price must be positive")
        if self.qty <= 0:
            raise ValueError("qty must be positive")

    @property
    def slippage_bps(self) -> float | None:
        """Signed so positive is always a cost, on either side of the market.

        A buy filled above the reference is adverse; a sell filled below it is
        equally adverse. Without the flip, a portfolio of buys and sells looks
        cost-free because the signs cancel.
        """
        if self.filled_price is None or self.filled_qty <= 0:
            return None
        diff = self.filled_price - self.reference_price
        if self.side == "sell":
            diff = -diff
        return diff / self.reference_price * BPS

    @property
    def effective_spread_bps(self) -> float | None:
        """Twice the distance from the mid -- the usual round-trip convention.

        Comparable to a quoted spread, so a measured 3 bps here can be set
        against the 1.59 bps mid-day SPY figure a cost model assumes.
        """
        slip = self.slippage_bps
        return None if slip is None else 2.0 * slip

    @property
    def fill_ratio(self) -> float:
        """Shares filled over shares wanted. Partial fills are not free."""
        return min(self.filled_qty / self.qty, 1.0) if self.qty else 0.0

    @property
    def notional(self) -> float:
        px = self.filled_price if self.filled_price is not None else self.reference_price
        return self.filled_qty * px


def mid(bid: float | None, ask: float | None) -> float | None:
    """Midpoint, or None when either side of the quote is missing or crossed."""
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return (bid + ask) / 2.0


class FillLog:
    """Append-only JSONL of order attempts.

    Append-only on purpose: a fill log you can rewrite is a fill log that will
    eventually be rewritten to agree with the backtest.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: FillRecord) -> FillRecord:
        row = asdict(rec)
        row["slippage_bps"] = rec.slippage_bps
        row["fill_ratio"] = rec.fill_ratio
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return rec

    def frame(self) -> pd.DataFrame:
        """Every recorded attempt, with the derived columns recomputed.

        Recomputed rather than read back, so a hand-edited slippage column in
        the file cannot influence the analysis.
        """
        if not self.path.exists():
            return pd.DataFrame()
        rows = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # a half-written line from a killed process
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        keep = {f for f in FillRecord.__dataclass_fields__}
        recs = []
        for row in df.to_dict("records"):
            try:
                recs.append(FillRecord(**{k: v for k, v in row.items()
                                          if k in keep and pd.notna(v)}))
            except (ValueError, TypeError):
                continue              # malformed row, not a silent zero
        df = pd.DataFrame([asdict(r) for r in recs])
        df["slippage_bps"] = [r.slippage_bps for r in recs]
        df["effective_spread_bps"] = [r.effective_spread_bps for r in recs]
        df["fill_ratio"] = [r.fill_ratio for r in recs]
        df["notional"] = [r.notional for r in recs]
        return df


def summarise(df: pd.DataFrame) -> dict:
    """Headline fill-quality numbers.

    Slippage is reported both equal-weighted and notional-weighted. They come
    apart when the big orders are the ones that fill badly, which is the normal
    case -- and the equal-weighted number is the flattering one.
    """
    if df is None or df.empty:
        return {"orders": 0}

    attempted = len(df)
    filled = df[df["filled_qty"] > 0]
    slips = filled["slippage_bps"].dropna()

    out = {
        "orders": attempted,
        "filled_orders": int(len(filled)),
        "fill_rate": len(filled) / attempted,
        "share_fill_rate": float(df["fill_ratio"].mean()),
        "notional": float(filled["notional"].sum()),
    }
    if len(slips):
        weights = filled.loc[slips.index, "notional"]
        out.update({
            "slippage_bps_mean": float(slips.mean()),
            "slippage_bps_median": float(slips.median()),
            "slippage_bps_worst": float(slips.max()),
            "effective_spread_bps": float(2.0 * slips.mean()),
            "slippage_bps_notional_weighted":
                float((slips * weights).sum() / weights.sum()) if weights.sum() else float(slips.mean()),
        })
    return out


def verdict(df: pd.DataFrame, assumed_cost_bps: float,
            min_fill_rate: float = 0.60) -> dict:
    """Hold the live numbers against what the backtest assumed.

    `assumed_cost_bps` is the round-trip cost the backtest charged. Measured
    effective spread is a one-way figure doubled, so it is already on the same
    footing.

    Returns a verdict rather than a number because the honest answer is usually
    "not enough orders yet" -- and a strategy stopped for a bad twenty-order
    sample is as costly a mistake as one that runs on a broken assumption.
    """
    s = summarise(df)
    if not s.get("orders"):
        return {"verdict": "no data", "orders": 0}

    measured = s.get("effective_spread_bps")
    reasons, failed = [], False

    if s["fill_rate"] < min_fill_rate:
        failed = True
        reasons.append(
            f"fill rate {s['fill_rate']:.0%} is below the {min_fill_rate:.0%} floor -- "
            "the backtest counted fills that are not happening")

    if measured is not None and measured > assumed_cost_bps:
        failed = True
        reasons.append(
            f"measured cost {measured:.2f} bps exceeds the {assumed_cost_bps:.2f} bps "
            "the backtest charged, so its net edge was overstated")

    if s["orders"] < 30:
        return {"verdict": "insufficient data", "orders": s["orders"],
                "note": f"{s['orders']} orders is too few to judge; provisional signs: "
                        + ("; ".join(reasons) if reasons else "nothing adverse yet"),
                **s}

    return {"verdict": "fails" if failed else "holds",
            "reasons": reasons,
            "measured_cost_bps": measured,
            "assumed_cost_bps": assumed_cost_bps,
            **s}
