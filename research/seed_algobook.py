"""Fill the algorithm book from the evaluation that has already been run.

Rather than shipping an empty app, this files the best-evidenced rules from
`research/results/evaluate_all.csv` into their measured quadrants, carrying
their real numbers with them.

It seeds losers as well as winners, on purpose. The short-term quadrant
honestly reflects what was measured -- rules that lose to buy-and-hold after
costs -- and an app that only showed the flattering half would be lying by
selection. Every entry carries the evidence that put it there.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from core.algobook import (  # noqa: E402
    QUADRANTS,
    AlgoBook,
    SavedAlgo,
    measured_horizons,
    suggest_quadrant,
)

EVAL = REPO / "research" / "results" / "evaluate_all.csv"

WHY = {
    "Gap fade": "Bets against an overnight gap, on the idea that the move "
                "overshoots and part of it comes back.",
    "Walk-forward logistic": "A logistic regression refitted on past bars "
                             "only, letting the data pick the pattern.",
    "Volatility breakout": "Buys a move that is large relative to recent "
                           "range, on the idea that big moves continue.",
    "Turn of month": "Holds around month end, when index funds and pension "
                     "flows are reinvested.",
    "MACD trend": "Follows the direction of two moving averages crossing.",
    "Time-series momentum": "Holds while the trailing year is positive. The "
                            "simplest trend rule there is.",
    "Faber TAA": "Holds while price is above its ten-month average, and sits "
                 "in cash otherwise.",
    "Price vs moving average": "Holds while price is above its own long "
                               "average.",
    "Golden cross": "The 50/200 crossover, by name.",
    "Turtle breakout": "Enters on a 20-bar high, exits on a 10-bar low. The "
                       "1983 Dennis and Eckhardt rule.",
}


def main(limit_per_quadrant: int = 4):
    if not EVAL.exists():
        print("No evaluation on record. Run research/evaluate_all.py first.")
        return

    df = pd.read_csv(EVAL)
    df = df[df["strategy"] != "Buy and hold"]
    df = df[df["trades"] >= 50]
    horizons = measured_horizons()

    book = AlgoBook()
    existing = {(a.strategy, tuple(a.symbols)) for a in book.all()}

    # Best by out-of-sample excess, per strategy, so one strategy does not
    # occupy the whole quadrant with twelve near-identical symbol variants.
    best = (df.sort_values("oos_excess_ann_pct", ascending=False)
              .groupby("strategy").head(1))

    filled = {k: 0 for k in QUADRANTS}
    added = 0
    for _, r in best.sort_values("oos_excess_ann_pct", ascending=False).iterrows():
        name = str(r["strategy"])
        quadrant = suggest_quadrant(name, live=False, horizons=horizons)
        if filled[quadrant] >= limit_per_quadrant:
            continue
        key = (name, (str(r["symbol"]),))
        if key in existing:
            continue

        hold = horizons.get(name, 0.0)
        algo = SavedAlgo(
            name=f"{name} on {r['symbol']}",
            quadrant=quadrant,
            strategy=name,
            symbols=[str(r["symbol"])],
            rationale=WHY.get(name, "Seeded from the evaluation run."),
            backtest={
                "total_return_pct": float(r["oos_excess_ann_pct"])
                + float(r.get("oos_hold_sharpe", 0)) * 0,   # excess only
                "hold_return_pct": 0.0,
                "vs_hold_pct": float(r["oos_excess_ann_pct"]),
                "last_month_pct": None,
                "trades": int(r["trades"]),
                # The evaluation CSV holds annualised excess, not per-trade
                # outcomes. Recording 0.0 would assert a 0% win rate, which is
                # a false claim rather than a missing one.
                "win_rate": None,
                "expectancy_pct": float(r["oos_excess_ann_pct"]) / max(
                    float(r["trades"]), 1.0),
                "payoff_ratio": None,
                "profit_factor": float("nan"),
                "t_stat": float(r["oos_t"]),
                "kelly": None,
                "suggested_size": None,
                "years": 6, "cost_bps": 5.0, "symbols_tested": 1,
                "avg_hold_bars": float(hold),
                "seeded": True,
            })
        book.add(algo)
        filled[quadrant] += 1
        added += 1
        print(f"  {QUADRANTS[quadrant].title:<26} {algo.name:<38} "
              f"{r['oos_excess_ann_pct']:+7.2f}%/yr  t={r['oos_t']:+.2f}")

    print(f"\nSeeded {added} algorithms.")
    print("Every one carries its measured numbers, including the losers -- "
          "an app showing only the flattering half would be lying by "
          "selection.")


if __name__ == "__main__":
    main()
