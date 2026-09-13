"""Which tested algorithm is the best one to run right now.

This is the engine behind the everchanging quadrants. It ranks the strategies
that have actually been evaluated and answers two questions:

  * **Is the rule I am running still the best-evidenced choice?**
  * **If not, which tested rule should replace it, and by how much?**

Three rules govern what it is allowed to say, and they exist because a
recommender that always names a winner is worse than no recommender at all:

1. **It may return nothing.** If no candidate has positive expectancy after
   costs, the recommendation is to stand aside. "The least bad of a losing
   set" is not a recommendation, it is a ranking of ways to lose.
2. **Evidence outranks return.** A rule is ranked on expectancy *and* on how
   many trades that expectancy rests on. Twelve trades at +2% loses to eight
   hundred at +0.3%, because the first number is noise wearing a decimal point.
3. **It never claims significance it does not have.** Every recommendation
   carries the t-statistic and the count, and says out loud when the margin is
   inside the noise.

The switching rule deserves a note. Swapping strategies on small differences
is itself a losing strategy -- it converts noise into turnover, and turnover
into spread. So a replacement has to beat the incumbent by a *margin*, not by
any amount at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.algobook import LONG_TERM_BARS, measured_horizons

EVAL = Path(__file__).resolve().parent.parent / "research" / "results" / "evaluate_all.csv"

#: A challenger must beat the incumbent's expectancy by this multiple before
#: switching is worth the spread it costs to switch.
SWITCH_MARGIN = 1.25

#: Below this many observations a result is reported but never recommended.
MIN_TRADES = 100


@dataclass
class Candidate:
    strategy: str
    symbol: str
    expectancy_pct: float          # per bar held, excess over buy-and-hold
    t_stat: float
    trades: float
    exposure: float
    avg_hold_bars: float
    horizon: str                   # short | long
    sharpe_gap: float = 0.0
    note: str = ""

    @property
    def credible(self) -> bool:
        """Enough observations and a positive result. Not the same as proven."""
        return self.trades >= MIN_TRADES and self.expectancy_pct > 0

    @property
    def significant(self) -> bool:
        return abs(self.t_stat) >= 1.96 and self.t_stat > 0


@dataclass
class Recommendation:
    """What to do, and how much to trust it."""
    action: str                    # hold | switch | stand_aside
    reason: str
    best: Candidate | None = None
    incumbent: Candidate | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    confident: bool = False

    @property
    def headline(self) -> str:
        if self.action == "stand_aside":
            return "Stand aside"
        if self.action == "hold":
            return "Keep what you are running"
        return f"Consider switching to {self.best.strategy}" if self.best else "No change"


def load_candidates(horizon: str | None = None,
                    symbol: str | None = None,
                    path: Path | None = None) -> list[Candidate]:
    """Every evaluated strategy-symbol pair, as ranked candidates.

    Reads the out-of-sample half only. In-sample numbers are where a strategy
    is allowed to look good; recommending on them would be recommending on the
    data the strategy was chosen from.
    """
    path = Path(path or EVAL)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    needed = {"strategy", "symbol", "oos_excess_ann_pct", "oos_t",
              "trades", "exposure"}
    if not needed <= set(df.columns):
        return []

    df = df[df["strategy"] != "Buy and hold"].copy()
    df = df[df["trades"] > 0]
    horizons = measured_horizons()

    out = []
    for _, r in df.iterrows():
        hold = horizons.get(r["strategy"], float(r["exposure"]) * 1676.0
                            / max(float(r["trades"]), 1.0))
        h = "short" if hold < LONG_TERM_BARS else "long"
        if horizon and h != horizon:
            continue
        if symbol and str(r["symbol"]).upper() != symbol.upper():
            continue
        out.append(Candidate(
            strategy=str(r["strategy"]),
            symbol=str(r["symbol"]),
            expectancy_pct=float(r["oos_excess_ann_pct"]),
            t_stat=float(r["oos_t"]) if pd.notna(r["oos_t"]) else 0.0,
            trades=float(r["trades"]),
            exposure=float(r["exposure"]),
            avg_hold_bars=float(hold),
            horizon=h,
            sharpe_gap=float(r.get("oos_sharpe_gap", 0.0) or 0.0),
        ))
    return out


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Best first, on evidence rather than on raw return.

    Sorted by expectancy, but only credible candidates can lead: a result on
    forty trades does not outrank one on nine hundred just because its
    average is higher. Everything is still returned, so the caller can show
    the also-rans honestly.
    """
    credible = sorted([c for c in candidates if c.credible],
                      key=lambda c: (c.significant, c.expectancy_pct),
                      reverse=True)
    rest = sorted([c for c in candidates if not c.credible],
                  key=lambda c: c.expectancy_pct, reverse=True)
    return credible + rest


def best_for(horizon: str = "short", symbol: str | None = None,
             candidates: list[Candidate] | None = None) -> Recommendation:
    """The best-evidenced algorithm to run, or an honest refusal."""
    pool = candidates if candidates is not None else load_candidates(horizon, symbol)
    if not pool:
        return Recommendation(
            action="stand_aside",
            reason=("Nothing has been evaluated for this horizon yet. Run "
                    "research/evaluate_all.py first -- recommending without "
                    "evidence is just guessing with extra steps."))

    ordered = rank(pool)
    winners = [c for c in ordered if c.credible]

    if not winners:
        top = ordered[0]
        return Recommendation(
            action="stand_aside",
            reason=(f"No tested rule for this horizon has positive expectancy "
                    f"on enough trades. The least bad is {top.strategy} at "
                    f"{top.expectancy_pct:+.2f}%/yr against buy-and-hold, which "
                    f"is a ranking of ways to lose, not a recommendation."),
            alternatives=ordered[:5])

    best = winners[0]
    return Recommendation(
        action="switch",
        reason=(f"{best.strategy} on {best.symbol} is the best-evidenced choice: "
                f"{best.expectancy_pct:+.2f}%/yr over buy-and-hold on "
                f"{best.trades:.0f} trades (t={best.t_stat:.2f})."
                + ("" if best.significant else
                   "  The margin is inside the noise -- treat it as the least "
                   "bad option rather than a proven edge.")),
        best=best,
        alternatives=winners[1:6],
        confident=best.significant)


def review(running: str, horizon: str = "short", symbol: str | None = None,
           candidates: list[Candidate] | None = None) -> Recommendation:
    """Is the rule you are running still the right one?

    Called each time the position or the news moves. Answers hold, switch or
    stand aside -- and prefers "hold" on anything short of a clear margin,
    because switching costs the spread every time and small differences
    between rules are mostly noise.
    """
    pool = candidates if candidates is not None else load_candidates(horizon, symbol)
    incumbent = next((c for c in pool if c.strategy == running), None)
    ordered = rank(pool)
    winners = [c for c in ordered if c.credible]

    if not winners:
        return Recommendation(
            action="stand_aside",
            reason=("Nothing tested for this horizon currently has positive "
                    "expectancy on a usable sample, including what you are "
                    "running. Being flat is the evidenced position."),
            incumbent=incumbent,
            alternatives=ordered[:5])

    best = winners[0]
    if incumbent is None:
        return Recommendation(
            action="switch", best=best, alternatives=winners[1:6],
            confident=best.significant,
            reason=(f"{running!r} has no evaluation on record for this "
                    f"horizon, so there is nothing to compare. "
                    f"{best.strategy} is the best-evidenced alternative."))

    if incumbent.credible and incumbent.expectancy_pct * SWITCH_MARGIN >= best.expectancy_pct:
        return Recommendation(
            action="hold", best=incumbent, incumbent=incumbent,
            alternatives=winners[:5], confident=incumbent.significant,
            reason=(f"{running} is still within {int((SWITCH_MARGIN - 1) * 100)}% "
                    f"of the best tested rule. Switching on a margin this small "
                    f"turns noise into turnover, and turnover into spread."))

    gap = best.expectancy_pct - incumbent.expectancy_pct
    return Recommendation(
        action="switch", best=best, incumbent=incumbent,
        alternatives=winners[1:6], confident=best.significant,
        reason=(f"{best.strategy} beats {running} by {gap:+.2f}%/yr on the "
                f"out-of-sample half ({best.trades:.0f} trades, "
                f"t={best.t_stat:.2f})."
                + ("" if best.significant else
                   "  Neither clears the significance bar, so this is the "
                   "better guess rather than a better rule.")))
