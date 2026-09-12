"""Block-shuffle null for the cross-sectional spreads + final multiplicity bar.

The only result in the whole audit that clears any significance bar is the Q4
'A minus rest' spread -- and it is NEGATIVE, i.e. the claimed preferred
archetype underperforms. Check whether even that survives the honest test, in
which the unit of independent information is a macro EPISODE, not a month.
Uses a max-|t| statistic across all 4 quadrants so multiplicity is handled by
construction.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s03_core as S  # noqa: E402
import s03_analysis as AN  # noqa: E402
from core.journal import bonferroni_bar  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

QUADS, PRIMARY4 = AN.QUADS, AN.PRIMARY4


def spread_ts(quads, fwd, h):
    """t_nw for each quadrant's 'preferred minus the other three' spread."""
    out = {}
    f = fwd[h]
    for q in QUADS:
        dates = quads.index[quads["quad"] == q]
        pref = S.PRIMARY[S.PREFERRED[q]]
        others = [s for s in PRIMARY4 if s != pref]
        sub = f.loc[f.index.intersection(dates), PRIMARY4].dropna()
        if len(sub) < 5:
            out[q] = np.nan
            continue
        sp = sub[pref] - sub[others].mean(axis=1)
        _, _, t = S.newey_west_mean_t(sp.values, lag=max(h - 1, 1))
        out[q] = t
    return out


def main():
    macro = S.load_macro()
    mep = S.month_end_prices(S.load_prices())
    fwd = {h: S.forward_returns(mep, h) for h in (3, 12)}
    quads = S.build_quadrants(macro, mep.index, walcl_lag_days=7, ff_lag_months=1)

    rng = np.random.default_rng(11)
    rows = []
    for h in (3, 12):
        real = spread_ts(quads, fwd, h)
        real_max = max(abs(v) for v in real.values() if np.isfinite(v))
        # one-sided in the CLAIMED direction: preferred should be positive
        real_best_pos = max(v for v in real.values() if np.isfinite(v))
        null_max, null_pos, null_q4 = [], [], []
        for _ in range(1000):
            sq = AN.block_shuffle_labels(quads, rng)
            t = spread_ts(sq, fwd, h)
            vals = [v for v in t.values() if np.isfinite(v)]
            if not vals:
                continue
            null_max.append(max(abs(v) for v in vals))
            null_pos.append(max(vals))
            if np.isfinite(t.get("Q4", np.nan)):
                null_q4.append(t["Q4"])
        nm, npos, nq4 = map(lambda a: np.asarray(a, float), (null_max, null_pos, null_q4))
        rows.append({
            "horizon_m": h,
            "real_t_Q1": real["Q1"], "real_t_Q2": real["Q2"],
            "real_t_Q3": real["Q3"], "real_t_Q4": real["Q4"],
            "real_max_abs_t": real_max,
            "null_max_abs_t_p95": np.percentile(nm, 95),
            "p_max_abs": float((nm >= real_max).mean()),
            "real_best_signed_t": real_best_pos,
            "null_best_signed_p95": np.percentile(npos, 95),
            "p_best_signed": float((npos >= real_best_pos).mean()),
            "p_Q4_two_sided": float((np.abs(nq4) >= abs(real["Q4"])).mean()),
        })
    df = pd.DataFrame(rows)
    print("\nBLOCK-SHUFFLE NULL FOR THE CROSS-SECTIONAL SPREADS (1000 draws)")
    print(df.T.to_string())
    df.to_csv(A.RESULTS / "s03_spread_null.csv", index=False)

    print("\nFINAL MULTIPLICITY")
    n = 111   # 68 (s03_analysis) + 8 spreads + 32 excess + 3 Q1-drop tests
    print(f"  distinct specifications evaluated across the whole audit: {n}")
    print(f"  Bonferroni |t| bar at alpha=0.05: {bonferroni_bar(n):.3f}")
    print(f"  unadjusted 5% bar: 1.960")


if __name__ == "__main__":
    main()
