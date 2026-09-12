"""Source 01 audit -- the adjudications the grid implies.

Momentum reading vs its mirror, head to head; how many cells clear which bar;
what the base rate does to each; and the tax arithmetic.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s01_core as C  # noqa: E402
from core.journal import bonferroni_bar, benjamini_hochberg  # noqa: E402

PAIRS = [("a1_ret5_topdec", "a1_ret5_topdec_INV"),
         ("a2_ret10_topdec", "a2_ret10_topdec_INV"),
         ("a3_ret20_topdec", "a3_ret20_topdec_INV"),
         ("b1_ma20_up", "b1_ma20_up_INV"),
         ("b2_ma50_up", "b2_ma50_up_INV"),
         ("c1_rsi_gt60", "c1_rsi_gt60_INV"),
         ("c2_rsi_gt70", "c2_rsi_gt70_INV"),
         ("d_macd_cross_up", "d_macd_cross_up_INV"),
         ("e_20d_high", "e_20d_high_INV"),
         ("f_relvol_up", "f_relvol_up_INV"),
         ("g_adx_bull", "g_adx_bull_INV")]

MOMENTUM_ONLY = [p[0] for p in PAIRS]


def main():
    g = pd.read_csv(A.RESULTS / "s01_grid_entry_by_hold.csv")
    full = g[(g["sample"] == "full")]
    f = full[full.entry != "BASE_RATE"]
    bar = bonferroni_bar(132)

    # ---- momentum reading vs its mirror, holds 2-5 only -------------------
    rows = []
    for mom, inv in PAIRS:
        for hold in (2, 3, 4, 5):
            a = f[(f.entry == mom) & (f.hold == hold)].iloc[0]
            b = f[(f.entry == inv) & (f.hold == hold)].iloc[0]
            rows.append({"reading": C.LABELS[mom], "hold": hold,
                         "momentum_mean_pct": a.mean_pct,
                         "mirror_mean_pct": b.mean_pct,
                         "base_pct": a.base_rate_pct,
                         "momentum_excess": a.excess_pct,
                         "mirror_excess": b.excess_pct,
                         "mirror_wins": b.mean_pct > a.mean_pct})
    head = pd.DataFrame(rows)
    A.save(head, "s01_momentum_vs_mirror.csv")
    print("=== MOMENTUM READING vs ITS MIRROR, holds 2-5 (percent per trade, gross) ===")
    print(head.round(4).to_string(index=False))
    print(f"\nmirror beats the momentum reading in {int(head.mirror_wins.sum())} "
          f"of {len(head)} head-to-head cells")

    # ---- how many momentum cells beat simply being long -------------------
    m25 = f[(f.entry.isin(MOMENTUM_ONLY)) & (f.hold.between(2, 5))]
    print(f"\nmomentum cells (11 readings x holds 2-5) = {len(m25)}")
    print(f"  beat the unconditional base rate at all:  "
          f"{int((m25.excess_pct > 0).sum())}")
    print(f"  excess p<0.05 uncorrected:                "
          f"{int((m25.p_excess_dateclust < 0.05).sum())}")
    print(f"  excess |t| >= bonferroni_bar(132)={bar:.3f}: "
          f"{int((m25.t_excess_dateclust.abs() >= bar).sum())}")
    print(f"  breakeven round-trip cost below the base rate's own breakeven "
          f"({full[full.entry=='BASE_RATE'].set_index('hold').loc[3,'mean_pct']*100:.2f} bps @3d): "
          f"{int(sum(m25[m25.hold==3].breakeven_bps < 19.893))} of 11 at hold 3")

    # ---- Benjamini-Hochberg as the softer alternative ---------------------
    p = f.p_excess_dateclust.tolist()
    keep = benjamini_hochberg(p, 0.05)
    print(f"\nBenjamini-Hochberg FDR 5% on the 132 excess p-values: "
          f"{sum(keep)} discoveries")
    keep10 = benjamini_hochberg(p, 0.10)
    print(f"Benjamini-Hochberg FDR 10%: {sum(keep10)} discoveries")

    # ---- the base rate itself vs the Bonferroni bar -----------------------
    br = full[full.entry == "BASE_RATE"]
    print("\n=== the control clears the bar on its own ===")
    print(br[["hold", "mean_pct", "t_pooled", "t_dateclust"]].round(4).to_string(index=False))
    print(f"the unconditional long position has |t_dateclust| up to "
          f"{br.t_dateclust.max():.3f} > {bar:.3f}. Any long-only rule inherits this; "
          "it is not evidence about the rule.")

    # ---- tax arithmetic ---------------------------------------------------
    print("\n=== TAX ===")
    ca = pd.read_csv(A.RESULTS / "s01_cost_arithmetic.csv")
    for h in (3, 5):
        r = ca[ca.hold_days == h].iloc[0]
        print(f"hold {h}d: {r.round_trips_per_year:.0f} round trips/yr, "
              f"gross {r.ann_gross_pct:.2f}%/yr, net of 5bps {r['ann_net_pct@5bps']:.2f}%/yr")
    top_st, top_lt = 0.408, 0.238          # fed 37%/20% + 3.8% NIIT, before state
    for label, g_ in (("hold-5d strategy", 16.68), ("buy & hold equal-weight", 16.21)):
        print(f"  {label}: gross {g_:.2f}%")
    req = (1 - top_lt) / (1 - top_st)
    print(f"short-term rate {top_st:.1%} vs long-term {top_lt:.1%} (top federal + NIIT, "
          f"state on top). A 2-5 day hold realises everything short-term every year, so it "
          f"needs {req:.2f}x the pre-tax return of a held position to leave the same "
          f"after-tax amount -- i.e. about {(req - 1) * 100:.0f}% more, on top of costs.")


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    main()
