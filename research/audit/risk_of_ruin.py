"""What source 05's own position-sizing example does to a small account.

Source 05, verbatim: "1,000 shares, 20 cents a share = $200." And on accuracy:
"even if you're only right 50% of the time hopefully you can still walk away
with a little bit of profit."

Those two numbers are stated pages apart and are never multiplied together. The
PDT repeal on 2026-06-04 dropped the day-trading minimum from $25,000 to $2,000,
so the reader this material now reaches may well be sizing a $200 risk against a
$2,000 account -- 10% of equity on one trade.

This simulates the account path implied by the source's own numbers, so the
consequence is a number rather than an adjective. It is a MONTE CARLO ON THE
SOURCE'S ASSUMPTIONS, not a claim about what gap-and-go actually returns; the
edge is set to exactly what the source claims, and the point is that even the
claimed edge blows up at that size.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402


def simulate(equity0: float, risk_frac: float, win_rate: float, rr: float,
             trades: int, sims: int = 20_000, cost_R: float = 0.0,
             seed: int = 4) -> dict:
    """Fixed-fractional sizing, one trade a day, `trades` trading days."""
    rng = np.random.default_rng(seed)
    wins = rng.random((sims, trades)) < win_rate
    R = np.where(wins, rr, -1.0) - cost_R
    eq = np.full(sims, equity0, dtype=float)
    ruined = np.zeros(sims, dtype=bool)
    dd_floor = np.full(sims, equity0, dtype=float)
    peak = np.full(sims, equity0, dtype=float)
    for t in range(trades):
        eq = eq * (1.0 + risk_frac * R[:, t])
        eq = np.maximum(eq, 0.0)
        peak = np.maximum(peak, eq)
        dd_floor = np.minimum(dd_floor, eq / peak * equity0)
        ruined |= eq <= equity0 * 0.5      # "ruin" = down 50%, unrecoverable in practice
    return {
        "equity0": equity0, "risk_pct": risk_frac * 100, "win_rate": win_rate,
        "rr": rr, "trades": trades, "cost_R": cost_R,
        "p_down_50pct": float(ruined.mean()),
        "median_final": float(np.median(eq)),
        "p05_final": float(np.percentile(eq, 5)),
        "p95_final": float(np.percentile(eq, 95)),
        "p_below_start": float((eq < equity0).mean()),
        "mean_worst_dd_pct": float((1 - dd_floor / equity0).mean() * 100),
    }


def main():
    rows = []
    # The source's own stated numbers: 50% accuracy, and a reward:risk it implies
    # is better than 1:1 ("still walk away with a little bit of profit").
    for equity0, risk_dollars in [(2_000, 200), (5_000, 200), (10_000, 200),
                                  (25_000, 200), (30_000, 200)]:
        rows.append(simulate(equity0, risk_dollars / equity0, 0.50, 1.5, 252))
    # sizing discipline, same claimed edge, on a $5,000 account
    for rf in (0.10, 0.05, 0.02, 0.01, 0.005):
        rows.append(simulate(5_000, rf, 0.50, 1.5, 252))
    # what happens if the claimed edge is not there (a coin flip at 1:1, minus cost)
    for rf in (0.10, 0.02, 0.01):
        rows.append(simulate(5_000, rf, 0.50, 1.0, 252, cost_R=0.05))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    A.save(out, "risk_of_ruin.csv")

    print("\n--- the source's example against the post-repeal minimum ---")
    r = simulate(2_000, 0.10, 0.50, 1.5, 252)
    print(f"$2,000 account, $200 risk per trade (10%), 50% win rate at 1.5:1 "
          f"-- i.e. a GENUINE edge of +0.25R per trade:")
    print(f"  probability of being down 50% within one year : {r['p_down_50pct']:.1%}")
    print(f"  probability of ending below where you started : {r['p_below_start']:.1%}")
    print(f"  average worst drawdown                        : {r['mean_worst_dd_pct']:.1f}%")
    print("\nThe edge is real in this simulation and the account still fails that often.")
    print("That is a sizing failure, not a strategy failure.")


if __name__ == "__main__":
    main()
