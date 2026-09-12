"""TEST 4b -- the two controls that actually adjudicate Source 11 vs Source 10.

(1) DRIFT ADJUSTMENT. Every Fib level shows a hugely significant positive
    forward return, because every random day in 2015-2026 does. The only
    meaningful statistic is the EXCESS over the same symbol's unconditional
    forward return at the same horizon.

(2) NON-FIBONACCI CONTROL LEVELS. If 0.382 and 0.618 are special, P(resume |
    touched f) and the excess return must show a BUMP at those f. Arbitrary
    levels (0.30, 0.45, 0.55, 0.70) interleaved between them test exactly that.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s04_test4_fib as T4  # noqa: E402
from core.journal import bonferroni_bar, two_sided_p  # noqa: E402

# Fibonacci levels marked True, arbitrary controls marked False
GRID = [(0.20, False), (0.236, True), (0.30, False), (0.382, True),
        (0.45, False), (0.50, True), (0.55, False), (0.618, True),
        (0.70, False), (0.786, True), (0.85, False)]
T4.LEVELS = [f for f, _ in GRID]
IS_FIB = dict(GRID)


def main():
    syms = T4.universe()
    print(f"universe: {len(syms)} symbols, daily 2015-01-02 .. 2026-08-31, k={T4.K}")

    eps, fwd = [], []
    uncond = {}
    for sym in syms:
        df = T4.load(sym)
        c = df["close"]
        uncond[sym] = {hz: float((c.shift(-hz) / c - 1.0).dropna().mean() * 100)
                       for hz in (5, 10)}
        recs, cl, hi, lo = T4.episodes(sym, df)
        n = len(df)
        for r in recs:
            eps.append(r)
            for f in T4.LEVELS:
                j = r[f"touch_{f}"]
                if j is None:
                    continue
                for hz in (5, 10):
                    e = min(n - 1, j + hz)
                    raw = (cl[e] / cl[j] - 1.0) * 100.0
                    fwd.append({"symbol": sym, "level": f, "horizon": hz,
                                "raw_pct": raw,
                                "excess_pct": raw - uncond[sym][hz],
                                "resume": r["outcome"] == "resume"})
    E = pd.DataFrame(eps)
    F = pd.DataFrame(fwd)
    N = len(E)
    p_res = float((E.outcome == "resume").mean())
    print(f"episodes n={N}, unconditional P(resume)={p_res:.4f}, "
          f"P(fail)={float((E.outcome=='fail').mean()):.4f}\n")

    rows = []
    for f in T4.LEVELS:
        touched = E[f"touch_{f}"].notna()
        nt = int(touched.sum())
        pr = float((E.loc[touched, "outcome"] == "resume").mean())
        # binomial SE for P(resume|touch) vs the unconditional base rate
        se = np.sqrt(p_res * (1 - p_res) / nt)
        z = (pr - p_res) / se
        row = {"level": f, "fibonacci": IS_FIB[f], "n_touch": nt,
               "P_touch": nt / N, "P_resume_given_touch": pr,
               "z_vs_baserate": float(z)}
        for hz in (5, 10):
            s = F[(F.level == f) & (F.horizon == hz)]
            per_sym = s.groupby("symbol")["excess_pct"].mean()
            t = A.tstat(per_sym)
            row[f"excess{hz}d_pct"] = float(s["excess_pct"].mean())
            row[f"t{hz}d_bysym"] = t
            row[f"p{hz}d_bysym"] = two_sided_p(t)
        rows.append(row)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("=== every level, Fibonacci and arbitrary, side by side ===")
    print(R.to_string(index=False))
    A.save(R, "s04_t4b_levels_with_controls.csv")

    print("\n=== is there a BUMP at the Fibonacci levels? ===")
    for hz in (5, 10):
        fib = R.loc[R.fibonacci, f"excess{hz}d_pct"]
        ctl = R.loc[~R.fibonacci, f"excess{hz}d_pct"]
        print(f" fwd{hz}d excess: Fibonacci levels mean {fib.mean():+.4f}% "
              f"(n_levels={len(fib)}), arbitrary controls {ctl.mean():+.4f}% "
              f"(n_levels={len(ctl)})")
    # smoothness: fit P(resume|touch) on level depth and look at residuals
    x = R["level"].to_numpy(float)
    y = R["P_resume_given_touch"].to_numpy(float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    R2 = 1 - resid.var() / y.var()
    print(f"\n P(resume|touch) is linear in retracement depth: slope {b:+.4f}/unit, "
          f"R^2 = {R2:.4f}")
    print(" residual at each level (a 'special' level would stick out):")
    for f, rr in zip(R["level"], resid):
        print(f"   {f:.3f} {'FIB' if IS_FIB[f] else '   '}  {rr:+.5f}")
    print(f" largest |residual| = {np.abs(resid).max():.5f} at level "
          f"{R['level'][np.argmax(np.abs(resid))]:.3f}; "
          f"residual sd = {resid.std(ddof=1):.5f}")

    # ---- placebo null: same episodes, level replaced by a random offset ----
    print("\n=== PLACEBO: enter at a random bar inside the episode window ===")
    rng = np.random.default_rng(41)
    pl = []
    for sym in syms:
        df = T4.load(sym)
        cl = df["close"].to_numpy()
        n = len(cl)
        sub = E[E.symbol == sym]
        for _, r in sub.iterrows():
            j = int(rng.integers(r["start"], max(r["start"] + 1, r["out_i"] + 1)))
            if j + 5 < n:
                pl.append({"symbol": sym,
                           "excess_pct": (cl[j + 5] / cl[j] - 1.0) * 100 - uncond[sym][5]})
    P = pd.DataFrame(pl)
    ps = P.groupby("symbol")["excess_pct"].mean()
    print(f" placebo fwd5d excess: n={len(P)} mean={P['excess_pct'].mean():+.4f}% "
          f"t(by symbol)={A.tstat(ps):+.3f}")
    r382 = R.loc[R.level == 0.382].iloc[0]
    r618 = R.loc[R.level == 0.618].iloc[0]
    print(f" 0.382  fwd5d excess: mean={r382['excess5d_pct']:+.4f}% "
          f"t={r382['t5d_bysym']:+.3f}")
    print(f" 0.618  fwd5d excess: mean={r618['excess5d_pct']:+.4f}% "
          f"t={r618['t5d_bysym']:+.3f}")

    # ---- the base-rate inversion, stated explicitly -----------------------
    print("\n=== the inverted conditional both sources use ===")
    inv = []
    res_mask = E.outcome == "resume"
    for f in T4.LEVELS:
        inv.append({"level": f, "fibonacci": IS_FIB[f],
                    "P_touch_given_resume": float(E.loc[res_mask, f"touch_{f}"].notna().mean()),
                    "P_touch": float(E[f"touch_{f}"].notna().mean()),
                    "P_resume_given_touch": float(
                        (E.loc[E[f"touch_{f}"].notna(), "outcome"] == "resume").mean())})
    I = pd.DataFrame(inv)
    print(I.to_string(index=False))
    A.save(I, "s04_t4b_conditional_inversion.csv")

    specs = len(T4.LEVELS) * 2 + len(T4.LEVELS)
    print(f"\nspecifications in TEST 4+4b: {specs} (11 levels x 2 horizons + 11 resume rates)")
    print(f"Bonferroni |t| bar at alpha=0.05: {bonferroni_bar(specs):.3f}")
    mx = R[["t5d_bysym", "t10d_bysym"]].abs().to_numpy().max()
    print(f"max |t| on DRIFT-ADJUSTED forward returns across all 22: {mx:.3f} "
          f"-> {'CLEARS' if mx > bonferroni_bar(specs) else 'FAILS'} the bar")


if __name__ == "__main__":
    main()
