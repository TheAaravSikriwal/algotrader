"""Source 03 audit -- (5) sabotage self-tests + cross-sectional spread analysis.

Two jobs:
  1. Prove the ranking / allocation machinery CAN detect an effect, by feeding
     it an oracle signal, and that inverting the oracle flips the sign. The
     real-data inversion self-test came back ~unchanged, which is only
     informative once we know inversion flips a signal that is really there.
  2. The model's actual claim is cross-sectional (archetype X beats the others
     in quadrant Q), so test the long/short SPREADS, which strip out market
     beta and the 2009-2021 bull-market confound.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s03_core as S  # noqa: E402
import s03_analysis as AN  # noqa: E402
from core.journal import two_sided_p  # noqa: E402
from core.metrics import equity_stats  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

QUADS = AN.QUADS
PRIMARY4 = AN.PRIMARY4


def banner(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# ---------------------------------------------------------------------------
def causality_assert(macro, quads, mep):
    """Hard check: can the quadrant at t be computed from data dated <= t?"""
    banner("SELF-TEST 1: no-lookahead assertion on the signal")
    walcl = macro["WALCL"].dropna()
    ff = macro["FEDFUNDS"].dropna()
    bad = 0
    for t in quads.index:
        row = quads.loc[t]
        if pd.isna(row["walcl"]):
            continue
        # the WALCL level used must exist at an index date <= t - 7d
        hits = walcl[(walcl.index <= t - pd.Timedelta(days=7))
                     & (np.isclose(walcl.values, row["walcl"]))]
        if len(hits) == 0:
            bad += 1
        if not pd.isna(row["ff"]):
            fh = ff[(ff.index <= (t - pd.DateOffset(months=1)).replace(day=1))
                    & (np.isclose(ff.values, row["ff"]))]
            if len(fh) == 0:
                bad += 1
    print(f"  rows whose inputs were NOT available by the observation date: {bad}")
    print("  PASS" if bad == 0 else "  *** FAIL ***")

    # forward return alignment spot-check
    banner("SELF-TEST 2: forward-return alignment spot check")
    f3 = S.forward_returns(mep, 3)
    i = 100
    t0, t3 = mep.index[i], mep.index[i + 3]
    manual = mep["SPY"].iloc[i + 3] / mep["SPY"].iloc[i] - 1.0
    print(f"  SPY {t0.date()} -> {t3.date()}: manual {manual:+.6f}  "
          f"table {f3['SPY'].iloc[i]:+.6f}  diff {abs(manual - f3['SPY'].iloc[i]):.2e}")
    print("  PASS" if abs(manual - f3["SPY"].iloc[i]) < 1e-12 else "  *** FAIL ***")


def oracle_test(mep, macro, fwd):
    """Build a cheating signal, confirm the tests light up, then invert it."""
    banner("SELF-TEST 3: ORACLE signal (deliberate lookahead) -- machinery validation")
    f3 = fwd[3]
    # label each month by which archetype actually wins over the next 3 months,
    # mapped to the quadrant whose CLAIMED preference is that archetype.
    arch_of = {"D": "IWO", "C": "QQQ", "B": "SPY", "A": "IWD"}
    inv_pref = {v: k for k, v in S.PREFERRED.items()}   # archetype -> quadrant
    labels = []
    for t in mep.index:
        row = f3.loc[t, PRIMARY4]
        if row.isna().any():
            labels.append(None)
            continue
        best_sym = row.idxmax()
        labels.append(inv_pref[AN.SYM2ARCH[best_sym]])
    oracle = pd.DataFrame({"quad": labels}, index=mep.index)

    qd = AN.by_quadrant(oracle, fwd, symbols=PRIMARY4, verbose=False)
    h3, h12 = AN.rank_hits(qd, 3), AN.rank_hits(qd, 12)
    c, e, _ = AN.allocation_rule(oracle, mep, macro)
    st = equity_stats(c, periods_per_year=12)
    print(f"  oracle ranking hits: 3m={h3}/4  12m={h12}/4   (expect 4/4 at 3m)")
    print(f"  oracle allocation: CAGR={st['CAGR']:.4f} Sharpe={st['Sharpe']:.4f} "
          f"avg exposure={e:.3f}")
    print("  PASS (machinery detects a real effect)" if h3 == 4
          else "  *** FAIL: machinery cannot even see a planted effect ***")

    # invert the oracle's preference map -> must collapse
    inv_map = {"Q1": "IWD", "Q2": "QQQ", "Q3": "SPY", "Q4": "IWO"}
    c_inv, _, _ = AN.allocation_rule(oracle, mep, macro, prefer_map=inv_map,
                                     weights={q: 1.0 for q in QUADS})
    c_ok, _, _ = AN.allocation_rule(oracle, mep, macro,
                                    weights={q: 1.0 for q in QUADS})
    s_ok = equity_stats(c_ok, periods_per_year=12)
    s_inv = equity_stats(c_inv, periods_per_year=12)
    print(f"  oracle @100%   : CAGR={s_ok['CAGR']:.4f} Sharpe={s_ok['Sharpe']:.4f}")
    print(f"  oracle INVERTED: CAGR={s_inv['CAGR']:.4f} Sharpe={s_inv['Sharpe']:.4f}")
    print("  PASS (inversion flips a genuine signal)" if s_inv["CAGR"] < s_ok["CAGR"]
          else "  *** FAIL ***")
    return s_ok, s_inv


def spreads(quads, mep, fwd):
    """Cross-sectional claim, market beta removed."""
    banner("CROSS-SECTIONAL SPREADS: preferred archetype MINUS the other three (equal wt)")
    rows = []
    for h in [3, 12]:
        f = fwd[h]
        for q in QUADS:
            dates = quads.index[quads["quad"] == q]
            pref_sym = S.PRIMARY[S.PREFERRED[q]]
            others = [s for s in PRIMARY4 if s != pref_sym]
            sub = f.loc[f.index.intersection(dates), PRIMARY4].dropna()
            if len(sub) < 5:
                continue
            sp = sub[pref_sym] - sub[others].mean(axis=1)
            r = S.stats_row(sp, h, label=f"{q}: {S.PREFERRED[q]} - rest")
            r.update({"quad": q, "horizon_m": h, "pref": S.PREFERRED[q]})
            rows.append(r)
    df = pd.DataFrame(rows)
    print(df[["horizon_m", "quad", "pref", "n", "mean_pct", "t_naive", "p_naive",
              "t_nw", "p_nw", "n_eff", "hit_rate"]].to_string(index=False))
    df.to_csv(A.RESULTS / "s03_spreads.csv", index=False)

    banner("EXCESS OVER UNCONDITIONAL: quadrant mean MINUS same proxy's full-sample mean")
    out = []
    for h in [3, 12]:
        f = fwd[h]
        uncond = {s: f[s].dropna().mean() for s in PRIMARY4}
        for q in QUADS:
            dates = quads.index[quads["quad"] == q]
            for s in PRIMARY4:
                x = f.loc[f.index.intersection(dates), s].dropna()
                if len(x) < 5:
                    continue
                ex = x - uncond[s]
                mu, se, t = S.newey_west_mean_t(ex.values, lag=max(h - 1, 1))
                out.append({"horizon_m": h, "quad": q, "label": s,
                            "archetype": AN.SYM2ARCH[s],
                            "preferred": AN.SYM2ARCH[s] == S.PREFERRED[q],
                            "n": len(x), "excess_pct": 100 * mu, "t_nw": t,
                            "p_nw": two_sided_p(t) if np.isfinite(t) else np.nan})
    ex_df = pd.DataFrame(out)
    print(ex_df.to_string(index=False))
    ex_df.to_csv(A.RESULTS / "s03_excess_vs_uncond.csv", index=False)
    return df, ex_df


def q1_decomposition(quads, mep, fwd):
    """Is Q1 just 2009-2014 + 2020-2021?"""
    banner("Q1 CONCENTRATION: which calendar windows supply the Q1 observations")
    ep = S.episodes(quads["quad"])
    q1 = ep[ep["quad"] == "Q1"].sort_values("months", ascending=False)
    tot = q1["months"].sum()
    q1 = q1.assign(share=lambda d: d["months"] / tot)
    print(q1.to_string(index=False))
    print(f"\n  top 3 Q1 episodes supply {q1['months'].head(3).sum()}/{tot} = "
          f"{q1['months'].head(3).sum()/tot:.1%} of all Q1 months")
    print(f"  top 5 Q1 episodes supply {q1['months'].head(5).sum()/tot:.1%}")
    # drop the single biggest episode and redo the 12m ranking
    for drop_n in [1, 2, 3]:
        big = set(q1.head(drop_n).index)
        keep = quads.copy()
        for i in big:
            r = ep.loc[i]
            m = (keep.index >= r["start"]) & (keep.index <= r["end"])
            keep.loc[m, "quad"] = None
        d = AN.by_quadrant(keep, fwd, symbols=PRIMARY4, verbose=False)
        sub = d[(d["horizon_m"] == 12) & (d["quad"] == "Q1")].sort_values(
            "mean_pct", ascending=False)
        if sub.empty:
            continue
        print(f"\n  Q1 12m ranking after dropping the {drop_n} largest Q1 episode(s) "
              f"(n={int(sub['n'].iloc[0])}):")
        print("    " + " > ".join(f"{a}({m:+.2f}%)" for a, m in
                                  zip(sub["archetype"], sub["mean_pct"])))


def main():
    macro = S.load_macro()
    px = S.load_prices()
    mep = S.month_end_prices(px)
    fwd = {h: S.forward_returns(mep, h) for h in [3, 12]}
    quads = S.build_quadrants(macro, mep.index, walcl_lag_days=7, ff_lag_months=1)

    causality_assert(macro, quads, mep)
    oracle_test(mep, macro, fwd)
    spreads(quads, mep, fwd)
    q1_decomposition(quads, mep, fwd)


if __name__ == "__main__":
    main()
