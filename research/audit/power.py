"""What can a paper-trading window actually detect?

The owner wants to start paper trading and needs a stopping rule. A stopping
rule is only meaningful if the window has the statistical power to distinguish
the edge from zero. This computes the minimum detectable edge for realistic
windows, given the per-trade dispersion actually measured on 5-minute intraday
brackets (sd approximately 59 bps of notional).

The conclusion this produces is uncomfortable and is the point: for edges of the
size anything in this material plausibly has, a paper-trading window of weeks or
months has essentially no power. Paper trading can validate INFRASTRUCTURE --
that orders fire, fill where expected, and flatten on time -- but it cannot
validate a 3 bps edge. Treating "I was up over 20 paper days" as evidence is how
a trader talks themselves into funding a coin flip.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402

Z_A, Z_B = 1.959964, 0.841621     # two-sided 0.05, 80% power


def min_detectable(n: int, sd_bps: float) -> float:
    return (Z_A + Z_B) * sd_bps / np.sqrt(n)


def main(sd_bps: float = 58.5, trades_per_symbol_per_day: float = 0.42):
    rows = []
    for days in (20, 40, 60, 90, 126, 252, 504, 1260):
        for symbols in (1, 5, 10, 20):
            n = days * symbols * trades_per_symbol_per_day
            if n < 5:
                continue
            rows.append({
                "paper_days": days, "symbols": symbols, "trades": round(n),
                "se_bps": sd_bps / np.sqrt(n),
                "min_detectable_edge_bps": min_detectable(n, sd_bps),
            })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)

    print(f"per-trade sd assumed = {sd_bps:.1f} bps of notional "
          f"(measured on 5,917 source-10 bracket trades)\n")
    print("Minimum edge detectable at 80% power, two-sided 0.05:\n")
    print(df.pivot(index="paper_days", columns="symbols",
                   values="min_detectable_edge_bps").round(1).to_string())
    print("\nnumber of trades behind each cell:\n")
    print(df.pivot(index="paper_days", columns="symbols",
                   values="trades").to_string())

    print("\nFor scale: the best measured gross edge anywhere in this audit is "
          "about 5.4 bps per trade,\nand realistic round-trip cost on these names "
          "is 1-3 bps, leaving roughly 2.4-4.4 bps net.")
    print("Cells above ~4 bps cannot distinguish that edge from zero.")
    A.save(df, "power_paper_trading.csv")

    # the flip side: how likely is a 60-day paper test to LOOK good by luck?
    rng = np.random.default_rng(1)
    for days, symbols, true_edge in [(60, 5, 0.0), (60, 5, 3.4), (252, 10, 3.4)]:
        n = int(days * symbols * trades_per_symbol_per_day)
        sims = rng.normal(true_edge, sd_bps, size=(20000, n)).mean(axis=1)
        print(f"\n{days} paper days x {symbols} symbols = {n} trades, "
              f"true edge {true_edge:.1f} bps:")
        print(f"   P(observed mean > 0)        = {(sims > 0).mean():.1%}")
        print(f"   P(observed mean > 5 bps)    = {(sims > 5).mean():.1%}")
        print(f"   P(statistically significant)= "
              f"{(np.abs(sims) > Z_A * sd_bps / np.sqrt(n)).mean():.1%}")


if __name__ == "__main__":
    main()
