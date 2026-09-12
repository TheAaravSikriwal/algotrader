"""What a rule is worth per trade, and what size turns that into growth.

This is the arithmetic behind "big profits, big risks". The governing quantity
is **expectancy** -- the average result of one trade:

    E = win_rate * average_win  -  loss_rate * average_loss

Everything else follows from its sign.

  * **E > 0.** Position size raises long-run growth up to a point (the Kelly
    fraction) and destroys it past that point. Betting more than about twice
    Kelly has *negative* expected growth even though every individual trade
    has positive expected value -- because losses compound against a smaller
    base than gains compound on.
  * **E < 0.** No size is safe and no size is profitable. Larger bets simply
    reach zero faster. This is the case the module exists to make impossible
    to miss, because a losing rule traded small looks like patience and a
    losing rule traded large looks like conviction, and both are the same rule.

A high win rate is not the goal and neither is a high payoff ratio. A rule
winning 30% of the time at 4:1 beats a rule winning 70% of the time at 1:3.
`summarise` reports both alongside the only number that combines them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Expectancy:
    """The per-trade economics of a rule, plus what they imply for sizing."""

    trades: int
    win_rate: float
    avg_win: float                 # mean return of winning trades, in %
    avg_loss: float                # mean LOSS of losing trades, positive, in %
    expectancy_pct: float          # average % result of one trade
    payoff_ratio: float            # avg_win / avg_loss
    profit_factor: float           # gross wins / gross losses
    stdev_pct: float
    t_stat: float

    @property
    def edge_is_positive(self) -> bool:
        return self.expectancy_pct > 0

    @property
    def kelly_fraction(self) -> float:
        """Fraction of capital that maximises long-run growth.

        Bet-sized form: f = W - (1-W)/R, with W the win rate and R the payoff
        ratio. Negative when the rule loses money, which is the answer: the
        growth-maximising bet on a losing rule is nothing.
        """
        if self.payoff_ratio <= 0:
            return 0.0
        f = self.win_rate - (1.0 - self.win_rate) / self.payoff_ratio
        return float(max(f, 0.0))

    @property
    def suggested_fraction(self) -> float:
        """Kelly is the ceiling, not the target.

        Full Kelly is the loudest number in trading and almost nobody should
        use it: it assumes your win rate and payoff are known exactly, and
        they are estimated from a finite sample. Overestimating the edge even
        slightly puts you past the growth-optimal point, where more risk means
        less money. A quarter of Kelly gives up about 25% of the growth for
        roughly a sixteenth of the variance.
        """
        return self.kelly_fraction / 4.0

    @property
    def breakeven_win_rate(self) -> float:
        """The win rate this payoff ratio needs just to break even."""
        if self.payoff_ratio <= 0:
            return 1.0
        return float(1.0 / (1.0 + self.payoff_ratio))

    def trades_needed(self, power: float = 0.8) -> int | None:
        """How many trades before this edge could be told apart from zero.

        Uses the measured per-trade dispersion. Returns None when the rule
        loses money -- there is no sample size that establishes a negative
        edge as a positive one.
        """
        if not self.edge_is_positive or self.stdev_pct <= 0:
            return None
        z_alpha, z_beta = 1.96, 0.84 if power >= 0.8 else 0.52
        n = ((z_alpha + z_beta) * self.stdev_pct / self.expectancy_pct) ** 2
        return int(np.ceil(n))


def summarise(returns) -> Expectancy:
    """Per-trade economics from a series of trade returns, in percent.

    `returns` is one number per *trade*, not per bar. Feeding bar returns in
    produces a fluent-looking answer to a different question.
    """
    r = pd.Series(list(returns), dtype=float).dropna()
    n = len(r)
    if n == 0:
        return Expectancy(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"))

    wins, losses = r[r > 0], r[r < 0]
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    # A rule with no losses has an undefined profit factor, not an infinite
    # edge. Reporting inf would put it top of any ranking on a sample of three.
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("nan")

    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    t = float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")

    return Expectancy(
        trades=n,
        win_rate=float(win_rate),
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_pct=float(r.mean()),
        payoff_ratio=float(avg_win / avg_loss) if avg_loss > 0 else 0.0,
        profit_factor=profit_factor,
        stdev_pct=sd,
        t_stat=t,
    )


def risk_of_ruin(exp: Expectancy, risk_frac: float, trades: int = 250,
                 ruin_frac: float = 0.5, runs: int = 4000,
                 seed: int = 0) -> dict:
    """Chance of losing `ruin_frac` of the account within `trades` trades.

    Simulated rather than solved, because the closed forms assume a fixed
    win/loss size and real rules do not have one. Each trade risks
    `risk_frac` of *current* equity, so losses shrink the bet as the account
    falls -- the same compounding that makes recovery hard.
    """
    if exp.trades == 0 or exp.avg_loss <= 0:
        return {"risk_of_ruin": float("nan"), "median_end": float("nan"),
                "worst_drawdown": float("nan")}

    rng = np.random.default_rng(seed)
    # Express each outcome as a multiple of the amount risked, so position
    # size scales it. A rule whose average loss is 2% and average win 6% is a
    # 3R winner regardless of how much capital is behind it.
    win_r = exp.avg_win / exp.avg_loss
    ruined = 0
    ends, worst = [], []

    for _ in range(runs):
        equity, peak, low = 1.0, 1.0, 1.0
        hit = False
        draws = rng.random(trades)
        for d in draws:
            stake = equity * risk_frac
            equity += stake * win_r if d < exp.win_rate else -stake
            peak = max(peak, equity)
            low = min(low, equity / peak)
            if equity <= ruin_frac:
                hit = True
                break
        ruined += hit
        ends.append(equity)
        worst.append(1.0 - low)

    return {
        "risk_of_ruin": ruined / runs,
        "median_end": float(np.median(ends)),
        "worst_drawdown": float(np.median(worst)),
        "risk_frac": risk_frac,
        "trades": trades,
    }


def project(exp: Expectancy, risk_frac: float, trades: int) -> dict:
    """Expected growth over `trades` trades at a given size.

    Reports the *median* path as well as the mean. They diverge sharply as
    size rises: a strategy can have a rising average outcome and a falling
    typical one, because the average is carried by a few paths that got lucky
    early and compounded. The median is what you are more likely to live.
    """
    if exp.trades == 0 or exp.avg_loss <= 0:
        return {"mean_growth": float("nan"), "median_growth": float("nan")}

    win_r = exp.avg_win / exp.avg_loss
    w = exp.win_rate
    # Log growth per trade -- the quantity Kelly maximises.
    up, down = 1.0 + risk_frac * win_r, 1.0 - risk_frac
    if down <= 0:
        return {"mean_growth": float("-inf"), "median_growth": -1.0,
                "note": "this size can lose more than the account on one trade"}

    g = w * np.log(up) + (1 - w) * np.log(down)
    arith = w * (risk_frac * win_r) - (1 - w) * risk_frac
    return {
        "log_growth_per_trade": float(g),
        "median_growth": float(np.exp(g * trades) - 1.0),
        "mean_growth": float((1 + arith) ** trades - 1.0),
        "risk_frac": risk_frac,
        "trades": trades,
    }


def sizing_curve(exp: Expectancy, trades: int = 250,
                 fractions=None) -> pd.DataFrame:
    """Growth against position size -- the picture that makes the trap obvious.

    For a rule with an edge this rises, peaks at Kelly, and then falls through
    zero. Seeing the far side of the peak is the point: past it, taking more
    risk reliably makes less money.
    """
    if fractions is None:
        fractions = np.arange(0.005, 0.51, 0.005)
    rows = []
    for f in fractions:
        p = project(exp, float(f), trades)
        rows.append({"risk_frac": float(f),
                     "median_growth_pct": p["median_growth"] * 100
                     if np.isfinite(p["median_growth"]) else np.nan,
                     "log_growth": p.get("log_growth_per_trade", np.nan)})
    return pd.DataFrame(rows)
