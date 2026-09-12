"""Source 01 audit -- the full (entry x hold) grid, costs, OOS split, self-tests.

Everything writes to research/audit/results/ with an s01_ prefix.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

import s01_core as C  # noqa: E402
import s01_fetch  # noqa: E402
import s01_universe as U  # noqa: E402

from core.journal import bonferroni_bar  # noqa: E402
from core.panel import Panel  # noqa: E402

BT_START = pd.Timestamp("2015-01-01")
IS_END = pd.Timestamp("2020-12-31")
OOS_START = pd.Timestamp("2021-01-01")
BT_END = pd.Timestamp("2026-09-11")

COST_GRID = (0, 2, 5, 10, 20)
HEADLINE_COST = 5.0          # bps round trip, the pessimistic end of 1-3bps


def build_world():
    bars = s01_fetch.load_all(verbose=False)
    frame, picked = U.build(bars)
    symbols = sorted(picked.index.tolist())
    panel = Panel.from_bars({s: bars[s] for s in symbols})
    # indicators need warm-up; trim the *measurement* window to 2015-01-01+
    return bars, frame, picked, panel


def window_mask(index: pd.DatetimeIndex, lo, hi) -> np.ndarray:
    return (index >= lo) & (index <= hi)


def run_grid(panel: Panel, sigs: dict, lo, hi, tag: str) -> pd.DataFrame:
    idx = panel.index
    keep = window_mask(idx, lo, hi)
    rows = []
    for hold in C.HOLDS:
        fwd_full = C.forward_open_returns(panel, hold)
        fwd = fwd_full[keep]
        base = C.base_for(fwd)
        rows.append({"entry": "BASE_RATE", "label": "unconditional (any symbol, any day)",
                     "hold": hold, "n": base["n"], "mean_pct": base["mu"],
                     "median_pct": base["median"], "win_rate": base["win_rate"],
                     "std_pct": base["std"],
                     "t_pooled": C._t(fwd.to_numpy(dtype=float)[np.isfinite(fwd.to_numpy(dtype=float))]),
                     "t_dateclust": C.newey_west_t(base["daily"].dropna(), hold),
                     "base_rate_pct": base["mu"], "excess_pct": 0.0})
        for name in C.ALL_NAMES:
            mask = sigs[name][keep]
            rows.append(C.cell_stats(fwd, mask, hold, name=name,
                                     cost_bps=HEADLINE_COST, base=base))
    df = pd.DataFrame(rows)
    df.insert(0, "sample", tag)
    return df


def main():
    bars, frame, picked, panel = build_world()
    symbols = panel.symbols
    print(f"universe: {len(symbols)} single names screened on tradability as of {U.AS_OF}")
    for note in panel.survivorship_warning():
        print("  survivorship:", note)

    sigs = C.build_signals(panel, bars)
    print(f"signals built: {len(sigs)} readings")

    full = run_grid(panel, sigs, BT_START, BT_END, "full")
    ins = run_grid(panel, sigs, BT_START, IS_END, "in_sample")
    oos = run_grid(panel, sigs, OOS_START, BT_END, "out_of_sample")
    grid = pd.concat([full, ins, oos], ignore_index=True)
    A.save(grid, "s01_grid_entry_by_hold.csv")

    n_tests = len(C.ALL_NAMES) * len(C.HOLDS)
    bar = bonferroni_bar(n_tests)
    print(f"\ncells tested = {len(C.ALL_NAMES)} readings x {len(C.HOLDS)} holds = {n_tests}")
    print(f"bonferroni_bar({n_tests}) = |t| >= {bar:.3f}   (uncorrected 0.05 bar = 1.960)")

    # ---------------- full-sample table ------------------------------------
    f = full[full.entry != "BASE_RATE"].copy()
    print("\n=== FULL SAMPLE 2015-01-01..2026-09-11, gross ===")
    show = ["label", "hold", "n", "mean_pct", "median_pct", "win_rate",
            "t_pooled", "p_pooled", "t_dateclust", "p_dateclust",
            "base_rate_pct", "excess_pct", "t_excess_dateclust",
            "p_excess_dateclust", "mean_net5bps", "breakeven_bps"]
    print(f[show].round(5).to_string(index=False))

    print("\n=== BASE RATES (unconditional) ===")
    print(full[full.entry == "BASE_RATE"][
        ["hold", "n", "mean_pct", "median_pct", "win_rate", "t_pooled", "t_dateclust"]
    ].round(5).to_string(index=False))

    # ---------------- who clears what --------------------------------------
    f["clears_raw_05"] = f.p_dateclust < 0.05
    f["clears_bonf"] = f.t_dateclust.abs() >= bar
    f["exc_clears_raw_05"] = f.p_excess_dateclust < 0.05
    f["exc_clears_bonf"] = f.t_excess_dateclust.abs() >= bar
    print("\ncells with |t_dateclust| over the Bonferroni bar (raw return):",
          int(f.clears_bonf.sum()), "of", len(f))
    print("cells with |t_excess_dateclust| over the Bonferroni bar (vs base rate):",
          int(f.exc_clears_bonf.sum()), "of", len(f))
    print("cells with excess p<0.05 uncorrected:", int(f.exc_clears_raw_05.sum()))
    if f.exc_clears_raw_05.any():
        print(f[f.exc_clears_raw_05][
            ["label", "hold", "n", "excess_pct", "t_excess_dateclust",
             "p_excess_dateclust", "t_excess_welch"]].round(5).to_string(index=False))

    # ---------------- in-sample pick, out-of-sample score -------------------
    isf = ins[ins.entry != "BASE_RATE"].copy()
    isf = isf[isf.n >= 200]
    # pick on the honest criterion: excess over base rate, date-clustered t
    best = isf.sort_values("t_excess_dateclust", ascending=False).iloc[0]
    best2 = isf.sort_values("mean_pct", ascending=False).iloc[0]
    # the source specifies 2-5 days, so also pick the best cell inside its own range
    hold25 = isf[isf.hold.between(2, 5)]
    best3 = hold25.sort_values("t_excess_dateclust", ascending=False).iloc[0]
    best4 = hold25.sort_values("mean_pct", ascending=False).iloc[0]
    best5 = hold25.sort_values("t_dateclust", ascending=False).iloc[0]
    picks = []
    for crit, row in (("max t_excess_dateclust, any hold", best),
                      ("max mean_pct, any hold", best2),
                      ("max t_excess_dateclust, holds 2-5", best3),
                      ("max mean_pct, holds 2-5", best4),
                      ("max t_dateclust (raw, no base-rate control), holds 2-5", best5)):
        o = oos[(oos.entry == row.entry) & (oos.hold == row.hold)].iloc[0]
        picks.append({"criterion": crit, "entry": row.entry, "label": row.label,
                      "hold": int(row.hold),
                      "IS_n": row.n, "IS_mean_pct": row.mean_pct,
                      "IS_excess_pct": row.excess_pct,
                      "IS_t_excess": row.t_excess_dateclust,
                      "IS_t_dateclust": row.t_dateclust,
                      "IS_mean_net5": row.mean_net5bps,
                      "OOS_n": o.n, "OOS_mean_pct": o.mean_pct,
                      "OOS_base_pct": o.base_rate_pct,
                      "OOS_excess_pct": o.excess_pct,
                      "OOS_t_excess": o.t_excess_dateclust,
                      "OOS_p_excess": o.p_excess_dateclust,
                      "OOS_t_dateclust": o.t_dateclust,
                      "OOS_mean_net5": o.mean_net5bps,
                      "OOS_breakeven_bps": o.breakeven_bps})
    pk = pd.DataFrame(picks)
    A.save(pk, "s01_oos_pick.csv")
    print("\n=== IN-SAMPLE PICK -> OUT-OF-SAMPLE SCORE ===")
    print(pk.round(5).to_string(index=False))

    # rank stability: how the in-sample top 10 ranks out of sample
    isr = isf.sort_values("t_excess_dateclust", ascending=False).head(10)
    merged = isr[["entry", "label", "hold", "excess_pct", "t_excess_dateclust"]].merge(
        oos[["entry", "hold", "excess_pct", "t_excess_dateclust", "mean_pct",
             "base_rate_pct", "mean_net5bps"]],
        on=["entry", "hold"], suffixes=("_IS", "_OOS"))
    A.save(merged, "s01_is_top10_oos.csv")
    print("\n=== IN-SAMPLE TOP 10 BY EXCESS t, SCORED OUT OF SAMPLE ===")
    print(merged.round(5).to_string(index=False))

    isj = isf[["entry", "hold", "t_excess_dateclust", "excess_pct"]].merge(
        oos[["entry", "hold", "t_excess_dateclust", "excess_pct"]],
        on=["entry", "hold"], suffixes=("_IS", "_OOS"))
    sp = isj.excess_pct_IS.rank().corr(isj.excess_pct_OOS.rank())
    spt = isj.t_excess_dateclust_IS.rank().corr(isj.t_excess_dateclust_OOS.rank())
    print("\nIS-vs-OOS correlation across all cells: "
          f"spearman(excess)={sp:.3f}  pearson(excess)="
          f"{isj.excess_pct_IS.corr(isj.excess_pct_OOS):.3f}  "
          f"spearman(t_excess)={spt:.3f}")
    # same thing restricted to the holds the source actually specifies
    j25 = isj[isj.hold.between(2, 5)]
    print("  restricted to holds 2-5 only: "
          f"spearman(excess)={j25.excess_pct_IS.rank().corr(j25.excess_pct_OOS.rank()):.3f}"
          f"  n_cells={len(j25)}")

    # ---------------- cost sweep on the headline cells ---------------------
    sweeps = []
    # the source says 2-5 days, so sweep costs on cells inside its own range
    f25 = f[f.hold.between(2, 5)]
    highlight = list(f25.sort_values("mean_pct", ascending=False).head(4)[["entry", "hold"]]
                     .itertuples(index=False, name=None))
    highlight += [(best3.entry, int(best3.hold)), (best4.entry, int(best4.hold))]
    # the momentum readings the source most plausibly means, at a 3-day hold
    highlight += [("e_20d_high", 3), ("c1_rsi_gt60", 3), ("f_relvol_up", 3),
                  ("g_adx_bull", 3), ("b1_ma20_up", 3), ("a3_ret20_topdec", 3),
                  ("d_macd_cross_up", 3)]
    for name, hold in dict.fromkeys(highlight):
        fwd = C.forward_open_returns(panel, hold)
        keep = window_mask(panel.index, BT_START, BT_END)
        fwd = fwd[keep]
        F = fwd.to_numpy(dtype=float)
        M = sigs[name][keep].to_numpy(dtype=bool)
        sel = F[M & np.isfinite(F)]
        sw = A.cost_sweep(sel, COST_GRID)
        sw.insert(0, "hold", hold)
        sw.insert(0, "entry", name)
        sw["breakeven_bps"] = A.breakeven_bps(sel)
        sweeps.append(sw)
    sweep = pd.concat(sweeps, ignore_index=True)
    A.save(sweep, "s01_cost_sweep.csv")
    print("\n=== COST SWEEP (round-trip bps of notional) ===")
    print(sweep.round(5).to_string(index=False))

    # base-rate cost sweep: what a coin flip costs
    fwd3 = C.forward_open_returns(panel, 3)[window_mask(panel.index, BT_START, BT_END)]
    F3 = fwd3.to_numpy(dtype=float)
    bsw = A.cost_sweep(F3[np.isfinite(F3)], COST_GRID)
    print("\nunconditional 3-day hold under the same cost grid:")
    print(bsw.round(5).to_string(index=False))

    # ---------------- self-tests -------------------------------------------
    print("\n=== SELF-TESTS ===")
    st = self_tests(panel, sigs)
    A.save(st, "s01_selftests.csv")
    print(st.round(5).to_string(index=False))

    return grid, pk, sweep, st


def self_tests(panel: Panel, sigs: dict) -> pd.DataFrame:
    """Two deliberate breakages that must behave in known ways.

    1. INVERT. Excess returns sum to zero over the whole cross-section by
       construction, so the complement of any signal must carry excess of the
       opposite sign. If masking or alignment were wrong this identity breaks.
    2. LOOKAHEAD. Take the signal from bar t+1 but still fill at the open of
       t+1. That is one day of peeking. If results do not jump, the pipeline
       is insensitive to lookahead and therefore cannot be trusted to have
       excluded it -- the test has to FAIL loudly to prove the real run passes.
    """
    keep = window_mask(panel.index, BT_START, BT_END)
    rows = []
    for name in ("a1_ret5_topdec", "e_20d_high", "c1_rsi_gt60", "b1_ma20_up"):
        for hold in (3, 5):
            fwd = C.forward_open_returns(panel, hold)[keep]
            base = C.base_for(fwd)
            mask = sigs[name][keep]
            honest = C.cell_stats(fwd, mask, hold, name, HEADLINE_COST, base)

            inv = C.cell_stats(fwd, ~mask, hold, name + "_COMPLEMENT",
                               HEADLINE_COST, base)

            # one day of lookahead: signal known at t+1, still filled at t+1 open
            peek = sigs[name].shift(-1).fillna(False).astype(bool)[keep]
            look = C.cell_stats(fwd, peek, hold, name + "_LOOKAHEAD1",
                                HEADLINE_COST, base)

            rows.append({
                "entry": name, "hold": hold,
                "honest_excess_pct": honest["excess_pct"],
                "honest_t_excess": honest["t_excess_dateclust"],
                "complement_excess_pct": inv["excess_pct"],
                "sign_flipped": np.sign(honest["excess_pct"]) != np.sign(inv["excess_pct"]),
                "lookahead_excess_pct": look["excess_pct"],
                "lookahead_t_excess": look["t_excess_dateclust"],
                "lookahead_uplift_x": (look["excess_pct"] / honest["excess_pct"]
                                       if honest["excess_pct"] != 0 else np.nan),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    main()
