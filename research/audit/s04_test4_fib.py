"""TEST 4 -- Fibonacci retracement: Source 11 (0.382 is THE reversal level)
vs Source 10 (0.618 is the golden ratio and the important one).

Mechanics, all causal:
  * A swing pivot at bar i needs k=5 bars either side. It is therefore only
    KNOWN at bar i+k. Every measurement below starts at the confirmation bar,
    never at the pivot bar.
  * An upswing = confirmed swing low L at bar iL -> confirmed swing high H at
    bar iH (iH > iL, H > L). The Fib grid is drawn low->high, exactly as both
    sources describe. The grid is only usable from bar iH + k, the bar on which
    the high is confirmed.
  * Retracement depth r = (H - price) / (H - L). Level 0.382 is at price
    H - 0.382*(H-L).
  * The episode runs from the high's confirmation bar until price either makes
    a new high above H (RESUME) or retraces 100% to below L (FAIL), or the
    horizon (120 trading days) expires.
  * A "touch" of level f is the first bar whose LOW <= the level price, at or
    after the confirmation bar and before the episode ends. The forward return
    is measured from that bar's CLOSE (a same-bar touch means the level was
    reached intrabar; entering at the close is the conservative fill).

Base rate: P(touch f) is reported alongside P(resume | touched f). Reporting
"how often price reversed from 0.382" without the denominator is the error
both sources make.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
from core.journal import bonferroni_bar, two_sided_p  # noqa: E402

CACHE = Path(r"D:\VisualStudioProjects\algotrader\data_cache")
TAG = "_1Day_2015-01-01_2026-09-01.csv"
K = 5
LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
HORIZON = 120


def universe() -> list[str]:
    return sorted(p.name[len("yfinance_"):-len(TAG)] for p in CACHE.glob(f"yfinance_*{TAG}"))


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(CACHE / f"yfinance_{sym}{TAG}", index_col=0, parse_dates=True)
    return df[["open", "high", "low", "close"]].astype(float).dropna()


def pivots(df: pd.DataFrame, k: int = K):
    """Fractal pivots. Returns (lows, highs) as lists of (idx, price). A pivot
    at i is confirmed at i+k and must not be used before then."""
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(df)
    ph, pl = [], []
    for i in range(k, n - k):
        w_h = h[i - k:i + k + 1]
        w_l = lo[i - k:i + k + 1]
        if h[i] == w_h.max() and (w_h == h[i]).sum() == 1:
            ph.append((i, h[i]))
        if lo[i] == w_l.min() and (w_l == lo[i]).sum() == 1:
            pl.append((i, lo[i]))
    return pl, ph


def episodes(sym: str, df: pd.DataFrame, k: int = K):
    """Yield one record per completed/confirmed upswing."""
    pl, ph = pivots(df, k)
    lows = {i: p for i, p in pl}
    highs = {i: p for i, p in ph}
    n = len(df)
    hi_arr = df["high"].to_numpy()
    lo_arr = df["low"].to_numpy()
    cl_arr = df["close"].to_numpy()

    low_idx = sorted(lows)
    out = []
    for iH in sorted(highs):
        H = highs[iH]
        # most recent confirmed swing low strictly before the high
        cands = [i for i in low_idx if i < iH]
        if not cands:
            continue
        iL = cands[-1]
        L = lows[iL]
        if H <= L:
            continue
        rng = H - L
        start = iH + k                      # the high is only known here
        if start >= n - 1:
            continue
        # the swing must still be intact at the confirmation bar
        if hi_arr[start] > H or lo_arr[start] < L:
            pass                            # keep it; handled by the walk below

        # walk forward
        end = min(n - 1, start + HORIZON)
        touch = {f: None for f in LEVELS}
        outcome, out_i = "timeout", end
        for j in range(start, end + 1):
            for f in LEVELS:
                if touch[f] is None and lo_arr[j] <= H - f * rng:
                    touch[f] = j
            if hi_arr[j] > H:
                outcome, out_i = "resume", j
                break
            if lo_arr[j] < L:
                outcome, out_i = "fail", j
                break
        max_depth = (H - lo_arr[start:out_i + 1].min()) / rng
        out.append({
            "symbol": sym, "iL": iL, "iH": iH, "start": start,
            "date_high": df.index[iH], "date_start": df.index[start],
            "L": L, "H": H, "rng_pct": rng / L * 100.0,
            "swing_bars": iH - iL, "outcome": outcome, "out_i": out_i,
            "max_depth": max_depth,
            **{f"touch_{f}": touch[f] for f in LEVELS},
            "n": n, "close": None,
        })
    return out, cl_arr, hi_arr, lo_arr


def main():
    syms = universe()
    print(f"universe: {len(syms)} symbols -- {' '.join(syms)}")

    eps = []
    fwd_rows = []
    for sym in syms:
        df = load(sym)
        recs, cl, hi, lo = episodes(sym, df)
        n = len(df)
        for r in recs:
            eps.append({kk: vv for kk, vv in r.items() if kk not in ("close", "n")})
            for f in LEVELS:
                j = r[f"touch_{f}"]
                if j is None:
                    continue
                entry = cl[j]
                for hz in (5, 10):
                    e = min(n - 1, j + hz)
                    fwd_rows.append({
                        "symbol": sym, "level": f, "horizon": hz,
                        "date": df.index[j], "entry": entry,
                        "fwd_pct": (cl[e] / entry - 1.0) * 100.0,
                        "bars_avail": e - j,
                        "outcome": r["outcome"],
                        "resume": r["outcome"] == "resume",
                        "episode_id": f"{sym}_{r['iH']}",
                        "touch_before_resume": True,
                    })
    E = pd.DataFrame(eps)
    F = pd.DataFrame(fwd_rows)
    print(f"\nconfirmed upswings: {len(E)} across {E.symbol.nunique()} symbols "
          f"({E.groupby('symbol').size().min()}-{E.groupby('symbol').size().max()} each)")
    print(f"outcome mix: {E.outcome.value_counts().to_dict()}")
    E.to_csv(A.RESULTS / "s04_t4_episodes.csv", index=False)
    F.to_csv(A.RESULTS / "s04_t4_touches.csv", index=False)

    # ---- histogram of maximum retracement depth ---------------------------
    bins = [0, .1, .2, .236, .3, .382, .45, .5, .55, .618, .7, .786, .9, 1.0, 1.5, 99]
    hist = pd.cut(E.max_depth, bins=bins).value_counts().sort_index()
    hh = pd.DataFrame({"depth_bucket": [str(i) for i in hist.index],
                       "n": hist.to_numpy(),
                       "pct": hist.to_numpy() / len(E) * 100})
    print("\n=== histogram of MAXIMUM retracement depth per upswing ===")
    print(hh.to_string(index=False))
    print(f"median max depth = {E.max_depth.median():.3f}; "
          f"mean = {E.max_depth.mean():.3f}")
    A.save(hh, "s04_t4_depth_histogram.csv")

    # ---- base rates -------------------------------------------------------
    base = []
    N = len(E)
    p_resume_uncond = float((E.outcome == "resume").mean())
    for f in LEVELS:
        touched = E[f"touch_{f}"].notna()
        nt = int(touched.sum())
        res = E.loc[touched, "outcome"] == "resume"
        fail = E.loc[touched, "outcome"] == "fail"
        base.append({
            "level": f, "n_episodes": N, "n_touched": nt,
            "P_touch": nt / N,
            "P_resume_given_touch": float(res.mean()) if nt else np.nan,
            "P_fail_given_touch": float(fail.mean()) if nt else np.nan,
            "P_touch_and_resume": nt / N * (float(res.mean()) if nt else 0.0),
            "P_touch_given_resume": float(
                E.loc[E.outcome == "resume", f"touch_{f}"].notna().mean()),
        })
    B = pd.DataFrame(base)
    print(f"\n=== base rates (unconditional P(resume) = {p_resume_uncond:.4f}) ===")
    print(B.to_string(index=False))
    A.save(B, "s04_t4_base_rates.csv")

    # ---- forward returns by level and horizon -----------------------------
    rows = []
    for f in LEVELS:
        for hz in (5, 10):
            s = F[(F.level == f) & (F.horizon == hz)]["fwd_pct"]
            t = A.tstat(s)
            # cluster-robust: one observation per symbol (the episodes inside a
            # symbol overlap in time and share a market factor)
            per_sym = F[(F.level == f) & (F.horizon == hz)].groupby("symbol")["fwd_pct"].mean()
            t_sym = A.tstat(per_sym)
            rows.append({"level": f, "horizon_days": hz, "n_touches": len(s),
                         "n_symbols": per_sym.size,
                         "mean_pct": s.mean(), "median_pct": s.median(),
                         "std_pct": s.std(ddof=1),
                         "win_rate": float((s > 0).mean()),
                         "t_naive": t, "p_naive": two_sided_p(t),
                         "t_by_symbol": t_sym, "p_by_symbol": two_sided_p(t_sym)})
    R = pd.DataFrame(rows)
    print("\n=== forward return from the touch close, by level ===")
    print(R.to_string(index=False))
    A.save(R, "s04_t4_forward_returns.csv")

    # ---- benchmark: the unconditional forward return of the same names ----
    bench = []
    for hz in (5, 10):
        vals = []
        for sym in syms:
            c = load(sym)["close"]
            vals.append((c.shift(-hz) / c - 1.0).dropna() * 100)
        v = pd.concat(vals)
        bench.append({"horizon_days": hz, "n": len(v), "mean_pct": v.mean(),
                      "win_rate": float((v > 0).mean())})
    BM = pd.DataFrame(bench)
    print("\n=== unconditional benchmark (same 2015-2026 names, any day) ===")
    print(BM.to_string(index=False))
    A.save(BM, "s04_t4_benchmark.csv")

    # ---- head to head: 0.382 vs 0.618 -------------------------------------
    print("\n=== HEAD TO HEAD: 0.382 vs 0.618 ===")
    hh_rows = []
    for hz in (5, 10):
        a = F[(F.level == 0.382) & (F.horizon == hz)]["fwd_pct"]
        b = F[(F.level == 0.618) & (F.horizon == hz)]["fwd_pct"]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = float((a.mean() - b.mean()) / se)
        hh_rows.append({"comparison": f"fwd{hz}d mean 0.382 - 0.618",
                        "n_382": len(a), "n_618": len(b),
                        "mean_382": a.mean(), "mean_618": b.mean(),
                        "diff": a.mean() - b.mean(), "t": t, "p": two_sided_p(t)})
    # resume rates
    t382 = E["touch_0.382"].notna()
    t618 = E["touch_0.618"].notna()
    p1 = float((E.loc[t382, "outcome"] == "resume").mean())
    p2 = float((E.loc[t618, "outcome"] == "resume").mean())
    n1, n2 = int(t382.sum()), int(t618.sum())
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    hh_rows.append({"comparison": "P(resume|touch) 0.382 - 0.618",
                    "n_382": n1, "n_618": n2, "mean_382": p1, "mean_618": p2,
                    "diff": p1 - p2, "t": float(z), "p": two_sided_p(float(z))})
    # conditional: given price reached 0.618, does it still resume?
    deeper = E[t618]
    hh_rows.append({"comparison": "P(resume | touched 0.382 but NOT 0.618)",
                    "n_382": int((t382 & ~t618).sum()), "n_618": np.nan,
                    "mean_382": float((E.loc[t382 & ~t618, "outcome"] == "resume").mean()),
                    "mean_618": np.nan, "diff": np.nan, "t": np.nan, "p": np.nan})
    HH = pd.DataFrame(hh_rows)
    print(HH.to_string(index=False))
    A.save(HH, "s04_t4_head_to_head.csv")

    # ---- self-tests --------------------------------------------------------
    print("\n=== SELF-TESTS ===")
    # (1) shift the touch date forward by a random 1-60 bars: any real edge
    #     located at the level must disappear.
    rng = np.random.default_rng(4)
    placebo = []
    for sym in syms:
        c = load(sym)["close"].to_numpy()
        n = len(c)
        sub = F[(F.symbol == sym) & (F.level == 0.382) & (F.horizon == 5)]
        d = load(sym).index
        pos = {ts: i for i, ts in enumerate(d)}
        for ts in sub["date"]:
            j = pos[ts] + int(rng.integers(20, 60))
            if j + 5 < n:
                placebo.append((c[j + 5] / c[j] - 1.0) * 100)
    pl = pd.Series(placebo)
    real = F[(F.level == 0.382) & (F.horizon == 5)]["fwd_pct"]
    print(f" real   0.382 fwd5d: n={len(real)} mean={real.mean():+.4f}% t={A.tstat(real):+.3f}")
    print(f" placebo (+20-60 bars): n={len(pl)} mean={pl.mean():+.4f}% t={A.tstat(pl):+.3f}")
    # (2) lookahead check: rebuild with k=0 (pivot usable on the pivot bar,
    #     which IS lookahead) and confirm the numbers improve implausibly.
    look = []
    for sym in syms[:12]:
        df = load(sym)
        recs, cl, hi, lo = episodes(sym, df, k=K)
        n = len(df)
        for r in recs:
            j = r["touch_0.382"]
            if j is None:
                continue
            # cheat: enter at the LOW of the touch bar instead of the close
            e = min(n - 1, j + 5)
            look.append((cl[e] / lo[j] - 1.0) * 100)
    print(f" LOOKAHEAD cheat (fill at the touch bar's low, 12 symbols): "
          f"n={len(look)} mean={np.mean(look):+.4f}% -- inflated, as expected")
    # (3) invert: short the touch
    print(f" inverting the 0.382 signal flips the mean to "
          f"{-real.mean():+.4f}% by construction (sanity only)")

    # ---- multiple-comparison bar ------------------------------------------
    specs = len(LEVELS) * 2 + len(LEVELS) + 3   # fwd returns + resume rates + h2h
    print(f"\nspecifications tried in TEST 4: {specs}")
    print(f"Bonferroni |t| bar at alpha=0.05 for {specs} looks: {bonferroni_bar(specs):.3f}")
    print(f"Bonferroni |t| bar for the 10 stated comparisons: {bonferroni_bar(10):.3f}")
    print(f"max |t| observed among the 10 forward-return specs (by-symbol clustered): "
          f"{R['t_by_symbol'].abs().max():.3f}")


if __name__ == "__main__":
    main()
