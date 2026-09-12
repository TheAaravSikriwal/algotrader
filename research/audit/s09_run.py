"""Run the source-09 SLC model, its controls and its self-tests. Writes results/s09_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402
import s09_slc as M  # noqa: E402
import structure as S  # noqa: E402
from core.journal import bonferroni_bar  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN"]
START, END = "2021-01-01", "2026-09-01"
OOS_FROM = pd.Timestamp("2024-06-01")


def load(tf="5Min"):
    full, rth = {}, {}
    for s in SYMBOLS:
        f = A.bars_chunked(s, START, END, tf, months=12)
        full[s] = f
        rth[s] = A.rth(f)
    return full, rth


def summarise(df: pd.DataFrame, label: str) -> dict:
    if df is None or df.empty:
        return {"label": label, "n": 0}
    R, ret = df["R"], df["ret_pct"]
    d = {"label": label, "n": len(R), "mean_R": R.mean(), "median_R": R.median(),
         "win_rate": (R > 0).mean(), "t_R": A.tstat(R), "p_R": A.two_sided_p(A.tstat(R)),
         "mean_ret_pct": ret.mean(), "t_ret": A.tstat(ret),
         "p_ret": A.two_sided_p(A.tstat(ret)), "breakeven_bps": A.breakeven_bps(ret),
         "median_risk_pct": df["risk_pct_of_px"].median(),
         "mean_bars_held": df["bars_held"].mean(),
         "target_rate": (df["exit_reason"] == "target").mean(),
         "stop_rate": (df["exit_reason"] == "stop").mean(),
         "eod_rate": df["exit_reason"].isin(["eod", "eod_rollover"]).mean(),
         "n_symbols": df["symbol"].nunique()}
    d["t_R_symclust"] = A.tstat(df.groupby("symbol")["R"].mean())
    return d


def main(tf="5Min"):
    print(f"loading {tf} bars ...", flush=True)
    full, rth = load(tf)
    print(f"{len(rth)} symbols, "
          f"{sum(len(set(pd.DatetimeIndex(v.index).normalize())) for v in rth.values())} "
          f"symbol-sessions\n", flush=True)

    specs_tried = 0
    rows = []

    # ---- the stochastic sweep the source forces on us ----------------------
    # "custom settings shown on screen only" -- so every plausible setting must
    # be tried and the multiplicity paid for. This is the source's doing.
    grid = []
    for sk in (5, 9, 14, 21):
        for band in (70.0, 80.0, 90.0):
            for imp in (1.5, 2.0, 3.0):
                grid.append(M.Spec(name=f"K{sk}_band{band:g}_imp{imp:g}",
                                   stoch_k=sk, stoch_upper=band, stoch_lower=100 - band,
                                   min_impulse_atr=imp))
    results = {}
    for spec in grid:
        specs_tried += 1
        d = M.run(rth, full, spec, "model")
        results[spec.name] = (spec, d)
        rows.append(summarise(d, spec.name))
        print(f"  {spec.name:24s} n={len(d):5d} "
              f"meanR={d['R'].mean() if len(d) else float('nan'):+.4f} "
              f"win={(d['R'] > 0).mean() if len(d) else float('nan'):.3f}", flush=True)

    grid_df = pd.DataFrame(rows)
    A.save(grid_df, f"s09_grid_{tf}.csv")

    # ---- headline: the mid-of-grid default ---------------------------------
    base = M.Spec(name="slc_as_written")
    real = M.run(rth, full, base, "model")
    print(f"\nSLC as written: n={len(real)} trades, "
          f"{real['symbol'].nunique() if len(real) else 0} symbols", flush=True)

    ctrl_rows = [summarise(real, "SLC full (structure + level + confirmation)")]
    for mode, lbl in [("no_structure", "drop S: level + confirmation only"),
                      ("no_confirmation", "drop C: structure + level only"),
                      ("random_zone", "placebo: zones relocated at random"),
                      ("inverted", "self-test: trade AGAINST the level"),
                      ("coinflip", "control: same levels, random side")]:
        specs_tried += 1
        d = M.run(rth, full, base, mode, seed=7)
        ctrl_rows.append(summarise(d, lbl))
        print(f"  {lbl:44s} n={len(d):5d} "
              f"meanR={d['R'].mean() if len(d) else float('nan'):+.4f}", flush=True)

    ctrl = pd.DataFrame(ctrl_rows)
    A.save(ctrl, f"s09_controls_{tf}.csv")

    # ---- random-entry null --------------------------------------------------
    print("\nrandom-entry null (200 draws) ...", flush=True)
    sessions = {}
    for sym, df in rth.items():
        for day, sub in df.groupby(pd.DatetimeIndex(df.index).normalize()):
            sessions[(sym, pd.Timestamp(day))] = sub
    setups = []
    for r in real.itertuples():
        sub = sessions.get((r.symbol, pd.Timestamp(r.date)))
        if sub is not None and len(sub) >= 30:
            setups.append((sub, r.direction, r.risk_px))
    rng = np.random.default_rng(23)
    means = []
    for _ in range(200):
        rs = []
        for sub, direction, risk in setups:
            i = int(rng.integers(5, len(sub) - 6))
            entry = float(sub["close"].iloc[i])
            res = S.simulate_bracket(sub, i, entry, entry - direction * risk,
                                     direction, base.r_target, max_bars=len(sub))
            if res:
                rs.append(res["R"])
        if rs:
            means.append(float(np.mean(rs)))
    means = np.array(means)
    A.save(pd.DataFrame({"draw_mean_R": means}), f"s09_random_entry_null_{tf}.csv")
    real_mean = real["R"].mean()
    print(f"  real mean R = {real_mean:+.4f}; null mean {means.mean():+.4f} "
          f"sd {means.std():.4f}; real at the "
          f"{float((means < real_mean).mean() * 100):.1f}th percentile", flush=True)

    # ---- costs ---------------------------------------------------------------
    cs = A.cost_sweep(real["ret_pct"], (0, 2, 5, 10, 20, 40))
    A.save(cs, f"s09_cost_sweep_{tf}.csv")
    print("\ncost sweep:")
    print(cs.to_string(index=False))

    # ---- out of sample --------------------------------------------------------
    is_scores = []
    for name, (spec, d) in results.items():
        d_is = d[pd.to_datetime(d["date"]) < OOS_FROM] if len(d) else d
        is_scores.append({"spec": name, "n_is": len(d_is),
                          "mean_R_is": d_is["R"].mean() if len(d_is) else np.nan,
                          "t_is": A.tstat(d_is["R"]) if len(d_is) else np.nan})
    is_scores = pd.DataFrame(is_scores).sort_values("t_is", ascending=False)
    A.save(is_scores, f"s09_is_ranking_{tf}.csv")
    pick = is_scores.iloc[0]["spec"]
    d_pick = results[pick][1]
    oos = pd.DataFrame([
        summarise(real[pd.to_datetime(real["date"]) < OOS_FROM], "SLC as written IN SAMPLE"),
        summarise(real[pd.to_datetime(real["date"]) >= OOS_FROM], "SLC as written OUT OF SAMPLE"),
        summarise(d_pick[pd.to_datetime(d_pick["date"]) < OOS_FROM],
                  f"best-in-sample [{pick}] IN SAMPLE"),
        summarise(d_pick[pd.to_datetime(d_pick["date"]) >= OOS_FROM],
                  f"best-in-sample [{pick}] OUT OF SAMPLE"),
    ])
    A.save(oos, f"s09_oos_{tf}.csv")
    print("\nout of sample:")
    print(oos[["label", "n", "mean_R", "win_rate", "t_R", "p_R"]].to_string(index=False))

    print(f"\nspecifications tried: {specs_tried}; "
          f"Bonferroni |t| bar = {bonferroni_bar(specs_tried):.3f}")
    A.save(real, f"s09_trades_{tf}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5Min")
