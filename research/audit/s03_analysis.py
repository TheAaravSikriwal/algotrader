"""Source 03 audit -- tests (a) through (f) plus self-tests.

Run: .venv/Scripts/python.exe research/audit/s03_analysis.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s03_core as S  # noqa: E402
from core.journal import bonferroni_bar, two_sided_p  # noqa: E402
from core.metrics import equity_stats  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

PRIMARY4 = ["IWO", "QQQ", "SPY", "IWD"]   # D, C, B, A
SYM2ARCH = {"IWO": "D", "QQQ": "C", "SPY": "B", "IWD": "A"}
QUADS = ["Q1", "Q2", "Q3", "Q4"]
HORIZONS = [3, 12]

SPEC_COUNT = []          # every distinct specification we evaluate


def note_spec(name):
    SPEC_COUNT.append(name)


def banner(txt):
    print("\n" + "=" * 100)
    print(txt)
    print("=" * 100)


# ---------------------------------------------------------------------------
def unconditional(mep, fwd):
    banner("(a) UNCONDITIONAL forward returns -- the baseline the quadrant must beat")
    rows = []
    for h in HORIZONS:
        for sym in S.ALL_PROXIES:
            if sym not in mep.columns:
                continue
            x = fwd[h][sym].dropna()
            if len(x) < 12:
                continue
            r = S.stats_row(x, h, label=sym)
            r["horizon_m"] = h
            r["start"] = mep[sym].dropna().index.min().date()
            r["n_months_price"] = int(mep[sym].dropna().shape[0])
            rows.append(r)
            note_spec(f"uncond_{sym}_{h}m")
    df = pd.DataFrame(rows)
    print(df[["horizon_m", "label", "start", "n", "mean_pct", "t_naive", "t_nw",
              "n_eff", "hit_rate"]].to_string(index=False))
    df.to_csv(A.RESULTS / "s03_unconditional.csv", index=False)
    return df


# ---------------------------------------------------------------------------
def by_quadrant(quads, fwd, tag="lagged", symbols=None, verbose=True):
    symbols = symbols or S.ALL_PROXIES
    rows = []
    for h in HORIZONS:
        for q in QUADS:
            mask = quads["quad"] == q
            dates = quads.index[mask]
            for sym in symbols:
                if sym not in fwd[h].columns:
                    continue
                x = fwd[h].loc[fwd[h].index.intersection(dates), sym].dropna()
                if len(x) < 4:
                    continue
                r = S.stats_row(x, h, label=sym)
                r.update({"quad": q, "horizon_m": h, "spec": tag,
                          "archetype": SYM2ARCH.get(sym, "-"),
                          "preferred": SYM2ARCH.get(sym, "-") == S.PREFERRED[q]})
                rows.append(r)
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        banner(f"(b) FORWARD RETURNS BY QUADRANT [{tag}]  (overlapping monthly obs)")
        for h in HORIZONS:
            print(f"\n--- horizon {h} months ---")
            sub = df[(df["horizon_m"] == h) & (df["label"].isin(PRIMARY4))]
            print(sub[["quad", "label", "archetype", "preferred", "n", "mean_pct",
                       "t_naive", "p_naive", "t_nw", "p_nw", "n_eff", "hit_rate"]]
                  .to_string(index=False))
    return df


def nonoverlap_by_quadrant(quads, mep, h):
    """Non-overlapping forward returns: stride-h subsample, all h phase offsets."""
    idx = mep.index
    rows = []
    for off in range(h):
        sel = idx[off::h]
        f = (mep.shift(-h) / mep - 1.0).loc[sel]
        for q in QUADS:
            dates = quads.index[quads["quad"] == q]
            for sym in PRIMARY4:
                x = f.loc[f.index.intersection(dates), sym].dropna()
                if len(x) < 3:
                    continue
                t = A.tstat(x)
                rows.append({"offset": off, "quad": q, "label": sym,
                             "archetype": SYM2ARCH[sym], "n": len(x),
                             "mean_pct": 100 * float(x.mean()), "t": t,
                             "p": two_sided_p(t)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def ranking_test(qdf, horizon, label=""):
    """(c) does the preferred archetype rank #1 among the four proxies?"""
    out = []
    sub = qdf[(qdf["horizon_m"] == horizon) & (qdf["label"].isin(PRIMARY4))]
    for q in QUADS:
        s = sub[sub["quad"] == q].sort_values("mean_pct", ascending=False)
        if s.empty:
            continue
        order = list(s["archetype"])
        claimed = S.PREFERRED[q]
        out.append({
            "quad": q, "claim": claimed,
            "realised_rank": " > ".join(f"{a}({m:+.2f}%)" for a, m in
                                        zip(s["archetype"], s["mean_pct"])),
            "top": order[0], "claim_rank": order.index(claimed) + 1,
            "hit": order[0] == claimed, "n": int(s["n"].iloc[0]),
        })
    df = pd.DataFrame(out)
    if label:
        banner(f"(c) RANKING TEST [{label}] horizon={horizon}m")
        print(df.to_string(index=False))
        hits = int(df["hit"].sum())
        print(f"\n  preferred archetype was top-of-four in {hits}/4 quadrants "
              f"(random expectation 1.0; P(all 4)=1/256=0.0039)")
    return df


def rank_hits(qdf, horizon):
    sub = qdf[(qdf["horizon_m"] == horizon) & (qdf["label"].isin(PRIMARY4))]
    hits = 0
    for q in QUADS:
        s = sub[sub["quad"] == q].sort_values("mean_pct", ascending=False)
        if s.empty:
            continue
        if s["archetype"].iloc[0] == S.PREFERRED[q]:
            hits += 1
    return hits


# ---------------------------------------------------------------------------
def cash_series(macro, dates):
    dtb3 = macro["DTB3"].dropna()
    vals = []
    for t in dates:
        h = dtb3[dtb3.index <= t]
        vals.append(float(h.iloc[-1]) if len(h) else 0.0)
    return pd.Series(vals, index=dates) / 100.0


def allocation_rule(quads, mep, macro, prefer_map=None, weights=None, label="source rule"):
    """(d) monthly-rebalanced: preferred proxy at stated max weight, rest in T-bills."""
    prefer_map = prefer_map or {q: S.PRIMARY[S.PREFERRED[q]] for q in QUADS}
    weights = weights or S.MAX_WEIGHT
    dates = mep.index
    cash_ann = cash_series(macro, dates)
    mret = mep.pct_change().shift(-1)          # return over the NEXT month
    eq, exposures, held = [1.0], [], []
    for i, t in enumerate(dates[:-1]):
        q = quads["quad"].get(t)
        if q in weights:
            sym, w = prefer_map[q], weights[q]
            r = mret[sym].iloc[i]
            if not np.isfinite(r):
                sym, w, r = "CASH", 0.0, 0.0
        else:
            sym, w, r = "CASH", 0.0, 0.0
        c = cash_ann.iloc[i] / 12.0
        step = w * r + (1.0 - w) * c
        eq.append(eq[-1] * (1.0 + step))
        exposures.append(w)
        held.append(sym)
    curve = pd.Series(eq, index=dates)
    avg_exp = float(np.mean(exposures))
    return curve, avg_exp, pd.Series(held, index=dates[:-1])


def benchmark_curves(mep, macro, avg_exp):
    dates = mep.index
    cash_ann = cash_series(macro, dates)
    mret = mep.pct_change().shift(-1)
    spy, mix = [1.0], [1.0]
    for i in range(len(dates) - 1):
        r = mret["SPY"].iloc[i]
        r = 0.0 if not np.isfinite(r) else r
        c = cash_ann.iloc[i] / 12.0
        spy.append(spy[-1] * (1.0 + r))
        mix.append(mix[-1] * (1.0 + avg_exp * r + (1 - avg_exp) * c))
    return (pd.Series(spy, index=dates), pd.Series(mix, index=dates))


# ---------------------------------------------------------------------------
def block_shuffle_labels(quads, rng):
    """Permute contiguous quadrant episodes, preserving run-length structure."""
    q = quads["quad"]
    valid = q.dropna()
    grp = (valid != valid.shift()).cumsum()
    blocks = [seg for _, seg in valid.groupby(grp)]
    order = rng.permutation(len(blocks))
    labels, lens = [blocks[i].iloc[0] for i in order], [len(b) for b in blocks]
    new = []
    for lab, n in zip(labels, lens):
        new.extend([lab] * n)
    out = q.copy()
    out.loc[valid.index] = new[:len(valid)]
    res = quads.copy()
    res["quad"] = out
    return res


def shuffle_test(quads, mep, macro, fwd, n_draws=1000, seed=7):
    banner(f"(f) NEGATIVE CONTROL: block-shuffled quadrant labels, {n_draws} draws")
    rng = np.random.default_rng(seed)

    real_q = by_quadrant(quads, fwd, symbols=PRIMARY4, verbose=False)
    real_hits = {h: rank_hits(real_q, h) for h in HORIZONS}
    real_curve, real_exp, _ = allocation_rule(quads, mep, macro)
    real_stats = equity_stats(real_curve, periods_per_year=12)
    real_sharpe, real_cagr = real_stats["Sharpe"], real_stats["CAGR"]

    # sharpest directional claim: Q1 D minus A spread at 12m
    def q1_spread(qd, h=12):
        s = qd[(qd["horizon_m"] == h) & (qd["quad"] == "Q1")]
        d = s[s["label"] == "IWO"]["mean_pct"]
        a = s[s["label"] == "IWD"]["mean_pct"]
        return float(d.iloc[0] - a.iloc[0]) if len(d) and len(a) else np.nan
    real_spread = q1_spread(real_q)

    null = {"hits3": [], "hits12": [], "sharpe": [], "cagr": [], "spread": []}
    for _ in range(n_draws):
        sq = block_shuffle_labels(quads, rng)
        qd = by_quadrant(sq, fwd, symbols=PRIMARY4, verbose=False)
        null["hits3"].append(rank_hits(qd, 3))
        null["hits12"].append(rank_hits(qd, 12))
        null["spread"].append(q1_spread(qd))
        c, _, _ = allocation_rule(sq, mep, macro)
        st = equity_stats(c, periods_per_year=12)
        null["sharpe"].append(st["Sharpe"])
        null["cagr"].append(st["CAGR"])

    rows = []
    for key, real in [("hits3", real_hits[3]), ("hits12", real_hits[12]),
                      ("sharpe", real_sharpe), ("cagr", real_cagr),
                      ("spread", real_spread)]:
        arr = np.asarray(null[key], dtype=float)
        arr = arr[np.isfinite(arr)]
        pct = float((arr <= real).mean())
        pval = float((arr >= real).mean())
        rows.append({"statistic": key, "real": real, "null_mean": arr.mean(),
                     "null_sd": arr.std(ddof=1), "null_p05": np.percentile(arr, 5),
                     "null_p95": np.percentile(arr, 95),
                     "pctile_of_real": pct, "p_one_sided_ge": pval})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(A.RESULTS / "s03_shuffle_null.csv", index=False)
    note_spec("shuffle_test")
    return df


# ---------------------------------------------------------------------------
def main():
    macro = S.load_macro()
    px = S.load_prices()
    mep = S.month_end_prices(px)
    fwd = {h: S.forward_returns(mep, h) for h in HORIZONS}

    quads = S.build_quadrants(macro, mep.index, walcl_lag_days=7, ff_lag_months=1)
    quads_nolag = S.build_quadrants(macro, mep.index, walcl_lag_days=0, ff_lag_months=0)
    note_spec("quad_lagged")
    note_spec("quad_nolag")

    banner("QUADRANT DEFINITION AND TIMELINE (primary spec = lagged)")
    print("rate_dir: sign of FEDFUNDS[m-1] - FEDFUNDS[m-4], band +/-0.10pp, "
          "else carry forward previous state")
    print("bs_dir  : sign of WALCL(latest obs dated <= t-7d) - WALCL(13 weekly obs earlier)")
    print(quads["quad"].value_counts(dropna=False).to_string())
    ep = S.episodes(quads["quad"])
    print("\nepisodes per quadrant (contiguous runs):")
    print(ep.groupby("quad")["months"].agg(["count", "sum"]).to_string())
    print("\nepisodes >= 6 months (the ones that actually carry the sample):")
    print(ep[ep["months"] >= 6].to_string(index=False))

    # ---------------- (a) ----------------
    unconditional(mep, fwd)

    # ---------------- (b) ----------------
    qd = by_quadrant(quads, fwd, tag="lagged")
    qd.to_csv(A.RESULTS / "s03_by_quadrant.csv", index=False)
    for h in HORIZONS:
        for q in QUADS:
            for sym in PRIMARY4:
                note_spec(f"quad_{q}_{sym}_{h}m")

    qd_nolag = by_quadrant(quads_nolag, fwd, tag="nolag", symbols=PRIMARY4, verbose=False)
    banner("(b2) LAG SENSITIVITY: lagged vs no-lag mean forward returns (primary 4)")
    m = qd[qd["label"].isin(PRIMARY4)].merge(
        qd_nolag, on=["quad", "horizon_m", "label"], suffixes=("_lag", "_nolag"))
    print(m[["horizon_m", "quad", "label", "n_lag", "mean_pct_lag", "t_nw_lag",
             "n_nolag", "mean_pct_nolag", "t_nw_nolag"]].to_string(index=False))
    print(f"\n  mean |difference| in mean_pct: "
          f"{(m['mean_pct_lag'] - m['mean_pct_nolag']).abs().mean():.3f} pp")
    m.to_csv(A.RESULTS / "s03_lag_sensitivity.csv", index=False)

    banner("(b3) NON-OVERLAPPING forward returns by quadrant (all phase offsets)")
    for h in HORIZONS:
        no = nonoverlap_by_quadrant(quads, mep, h)
        agg = no.groupby(["quad", "label", "archetype"]).agg(
            n_mean=("n", "mean"), mean_pct=("mean_pct", "mean"),
            t_min=("t", "min"), t_med=("t", "median"), t_max=("t", "max")).reset_index()
        print(f"\n--- horizon {h}m, non-overlapping ---")
        print(agg.to_string(index=False))
        no.to_csv(A.RESULTS / f"s03_nonoverlap_{h}m.csv", index=False)
        note_spec(f"nonoverlap_{h}m")

    # ---------------- (c) ----------------
    r3 = ranking_test(qd, 3, label="lagged")
    r12 = ranking_test(qd, 12, label="lagged")
    pd.concat([r3.assign(h=3), r12.assign(h=12)]).to_csv(
        A.RESULTS / "s03_ranking.csv", index=False)

    banner("(c2) RANKING with alternate D/C/A proxies (robustness)")
    for alt, syms in [("ARKK/IWF/VTV", ["ARKK", "IWF", "SPY", "VTV"]),
                      ("XBI/QQQ/VTV", ["XBI", "QQQ", "SPY", "VTV"])]:
        mp = dict(zip(syms, ["D", "C", "B", "A"]))
        sub = by_quadrant(quads, fwd, symbols=syms, verbose=False)
        sub["archetype"] = sub["label"].map(mp)
        sub["preferred"] = sub["archetype"] == sub["quad"].map(S.PREFERRED)
        for h in HORIZONS:
            hits, detail = 0, []
            for q in QUADS:
                s = sub[(sub["quad"] == q) & (sub["horizon_m"] == h)].sort_values(
                    "mean_pct", ascending=False)
                if s.empty:
                    detail.append(f"{q}: no data")
                    continue
                top = s["archetype"].iloc[0]
                hits += top == S.PREFERRED[q]
                detail.append(f"{q}: claim {S.PREFERRED[q]}, top {top} "
                              f"(n={int(s['n'].iloc[0])}, " +
                              ", ".join(f"{a}{v:+.1f}" for a, v in
                                        zip(s['archetype'], s['mean_pct'])) + ")")
            print(f"  [{alt}] h={h}m  hits={hits}/4")
            for d in detail:
                print("      " + d)
            note_spec(f"rank_alt_{alt}_{h}m")

    # ---------------- (d) ----------------
    banner("(d) ALLOCATION RULE vs benchmarks")
    curve, avg_exp, held = allocation_rule(quads, mep, macro)
    spy_c, mix_c = benchmark_curves(mep, macro, avg_exp)
    note_spec("alloc_source_rule")
    print(f"strategy average equity exposure: {avg_exp:.4f}")
    print("held-symbol month counts:\n" + held.value_counts().to_string())
    tbl = pd.DataFrame({
        "source quadrant rule": equity_stats(curve, periods_per_year=12),
        "100% SPY buy&hold": equity_stats(spy_c, periods_per_year=12),
        f"{avg_exp:.0%} SPY + cash": equity_stats(mix_c, periods_per_year=12),
    })
    print(tbl.to_string())
    tbl.to_csv(A.RESULTS / "s03_allocation_stats.csv")
    pd.DataFrame({"strategy": curve, "spy": spy_c, "mix": mix_c}).to_csv(
        A.RESULTS / "s03_equity_curves.csv")

    # variant: preferred archetype at FULL weight (isolate picking skill from timing)
    curve_full, exp_full, _ = allocation_rule(
        quads, mep, macro, weights={q: 1.0 for q in QUADS})
    note_spec("alloc_full_weight")
    # variant: always SPY at the stated weights (isolate timing from picking)
    curve_timing, exp_t, _ = allocation_rule(
        quads, mep, macro, prefer_map={q: "SPY" for q in QUADS})
    note_spec("alloc_timing_only")
    tbl2 = pd.DataFrame({
        "picking only (pref @100%)": equity_stats(curve_full, periods_per_year=12),
        "timing only (SPY @ stated w)": equity_stats(curve_timing, periods_per_year=12),
    })
    print("\ndecomposition:")
    print(tbl2.to_string())
    tbl2.to_csv(A.RESULTS / "s03_allocation_decomp.csv")

    # ---------------- (e) ----------------
    banner("(e) WALK FORWARD: sample split in half by time")
    mid = mep.index[len(mep.index) // 2]
    print(f"split at {mid.date()}")
    for name, sl in [("first half", mep.index <= mid), ("second half", mep.index > mid)]:
        sub_q = quads[sl]
        sub_f = {h: fwd[h][sl] for h in HORIZONS}
        d = by_quadrant(sub_q, sub_f, symbols=PRIMARY4, verbose=False)
        print(f"\n--- {name} ({sub_q.index.min().date()} .. {sub_q.index.max().date()}) ---")
        print("quadrant months: " + sub_q["quad"].value_counts().to_dict().__str__())
        for h in HORIZONS:
            rk = ranking_test(d, h)
            print(f"  h={h}m hits={int(rk['hit'].sum())}/{len(rk)}")
            print(rk[["quad", "claim", "top", "claim_rank", "n", "realised_rank"]]
                  .to_string(index=False))
            note_spec(f"halfsample_{name}_{h}m")
        c, e, _ = allocation_rule(sub_q, mep[sl], macro)
        s_c, m_c = benchmark_curves(mep[sl], macro, e)
        print(f"  allocation rule: " + str({k: round(v, 4) for k, v in
              equity_stats(c, periods_per_year=12).items()
              if k in ("CAGR", "Sharpe", "Max drawdown")}))
        print(f"  SPY buy&hold  : " + str({k: round(v, 4) for k, v in
              equity_stats(s_c, periods_per_year=12).items()
              if k in ("CAGR", "Sharpe", "Max drawdown")}))

    # ---------------- (f) ----------------
    shuffle_test(quads, mep, macro, fwd, n_draws=1000)

    # ---------------- (5) self tests ----------------
    banner("(5) SELF-TEST / DELIBERATE SABOTAGE")
    inv = {"Q1": "A", "Q2": "C", "Q3": "B", "Q4": "D"}     # inverted preference map
    inv_sym = {q: S.PRIMARY[inv[q]] for q in QUADS}
    c_inv, e_inv, _ = allocation_rule(quads, mep, macro, prefer_map=inv_sym)
    print("inverted-preference allocation rule:")
    print({k: round(v, 4) for k, v in equity_stats(c_inv, periods_per_year=12).items()
           if k in ("CAGR", "Sharpe", "Max drawdown", "Total return")})
    print("real rule:")
    print({k: round(v, 4) for k, v in equity_stats(curve, periods_per_year=12).items()
           if k in ("CAGR", "Sharpe", "Max drawdown", "Total return")})

    look = S.build_quadrants(macro, mep.index, walcl_lag_days=7, ff_lag_months=1,
                             signal_shift_months=6)
    qd_look = by_quadrant(look, fwd, symbols=PRIMARY4, verbose=False)
    c_look, e_look, _ = allocation_rule(look, mep, macro)
    print("\nLOOKAHEAD (+6m signal shift) allocation rule:")
    print({k: round(v, 4) for k, v in equity_stats(c_look, periods_per_year=12).items()
           if k in ("CAGR", "Sharpe", "Max drawdown")})
    print(f"lookahead ranking hits: 3m={rank_hits(qd_look,3)}/4  "
          f"12m={rank_hits(qd_look,12)}/4   vs real 3m={rank_hits(qd,3)}/4 "
          f"12m={rank_hits(qd,12)}/4")

    lab_agree = (look["quad"] == quads["quad"]).mean()
    print(f"lookahead vs real label agreement: {lab_agree:.3f} (sanity: should be <1)")

    # constant-label control: every month Q1 -> must equal buy&hold IWO
    const = quads.copy()
    const["quad"] = "Q1"
    c_const, _, _ = allocation_rule(const, mep, macro)
    iwo_bh = (mep["IWO"] / mep["IWO"].iloc[0])
    err = float((c_const / c_const.iloc[0] - iwo_bh).abs().max())
    print(f"\nplumbing check: all-Q1 rule (100% IWO) vs IWO buy&hold, max abs diff = {err:.2e}")

    # ---------------- (6) multiplicity ----------------
    banner("(6) MULTIPLICITY")
    n_tests = len(set(SPEC_COUNT))
    print(f"distinct specifications evaluated: {n_tests}")
    print(f"Bonferroni |t| bar at alpha=0.05: {bonferroni_bar(n_tests):.3f}")
    print(f"(unadjusted 5% bar is 1.960)")


if __name__ == "__main__":
    main()
