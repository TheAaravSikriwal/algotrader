"""Run the source-10 model, its controls and its self-tests. Writes results/s10_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402
import s10_fvg_choch as M  # noqa: E402
from core.journal import bonferroni_bar  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN"]
START, END = "2021-01-01", "2026-09-01"
OOS_FROM = pd.Timestamp("2024-06-01")     # last ~40% of the span held out


def load(tf="5Min"):
    bars = {}
    for s in SYMBOLS:
        bars[s] = A.rth(A.bars_chunked(s, START, END, tf, months=12))
    return bars


def summarise(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0}
    R = df["R"]
    ret = df["ret_pct"]
    d = {
        "label": label, "n": len(R),
        "mean_R": R.mean(), "median_R": R.median(),
        "win_rate": (R > 0).mean(),
        "t_R": A.tstat(R), "p_R": A.two_sided_p(A.tstat(R)),
        "mean_ret_pct": ret.mean(),
        "t_ret": A.tstat(ret), "p_ret": A.two_sided_p(A.tstat(ret)),
        "breakeven_bps": A.breakeven_bps(ret),
        "median_risk_pct": df["risk_pct_of_px"].median(),
        "mean_bars_held": df["bars_held"].mean(),
        "target_rate": (df["exit_reason"] == "target").mean(),
        "stop_rate": (df["exit_reason"] == "stop").mean(),
        "eod_rate": df["exit_reason"].isin(["eod", "eod_rollover"]).mean(),
    }
    # cluster by symbol: sessions within a name are not independent draws
    per_sym = df.groupby("symbol")["R"].mean()
    d["t_R_symclust"] = A.tstat(per_sym)
    d["n_symbols"] = len(per_sym)
    return d


def main(tf="5Min"):
    print(f"loading {tf} bars ...", flush=True)
    bars = load(tf)
    sessions = M.split_sessions(bars)
    print(f"{len(bars)} symbols, {len(sessions)} symbol-sessions, "
          f"{sum(len(v) for v in bars.values())} RTH bars\n", flush=True)

    rows, specs_tried = [], 0

    # ---- the model as written, and the knobs the source leaves open --------
    grid = []
    for r_target in (2.0, 3.0, 4.0):
        for k in (2, 3, 5):
            for be in (None, 1.0):
                grid.append(M.Spec(name=f"full_R{r_target:g}_k{k}_be{be}",
                                   k=k, r_target=r_target, breakeven_at_r=be))
    results = {}
    for spec in grid:
        specs_tried += 1
        d = M.run(bars, spec, "model", sessions=sessions)
        results[spec.name] = (spec, d)
        rows.append(summarise(d, spec.name))
        print(f"  {spec.name:26s} n={len(d):5d} "
              f"meanR={d['R'].mean() if len(d) else float('nan'):+.4f} "
              f"win={(d['R'] > 0).mean() if len(d) else float('nan'):.3f}", flush=True)

    grid_df = pd.DataFrame(rows)
    A.save(grid_df, f"s10_grid_{tf}.csv")

    # ---- headline spec: exactly what the source says (4R, BE on new BOS) ---
    base = M.Spec(name="source_as_written", k=3, r_target=4.0, breakeven_at_r=None)
    real = M.run(bars, base, "model", sessions=sessions)
    print(f"\nsource as written: n={len(real)} trades on "
          f"{real['date'].nunique() if len(real) else 0} distinct dates, "
          f"{real['symbol'].nunique() if len(real) else 0} symbols")

    ctrl_rows = [summarise(real, "model (source as written)")]
    for mode in ("fvg_only", "inverted", "coinflip", "skipfill"):
        d = M.run(bars, base, mode, seed=5, sessions=sessions)
        lbl = ("BUG DEMO: resolve from the bar after the intrabar fill"
               if mode == "skipfill" else f"control: {mode}")
        ctrl_rows.append(summarise(d, lbl))
        specs_tried += 1

    # no-structure baseline: buy the open, exit 15:55, same sessions
    hold_rows = []
    for r in real.itertuples():
        sub = sessions.get((r.symbol, pd.Timestamp(r.date)))
        if sub is None or len(sub) < 5:
            continue
        e, x = float(sub["open"].iloc[0]), float(sub["close"].iloc[-1])
        hold_rows.append({"R": np.nan, "ret_pct": (x - e) / e * 100 * r.direction,
                          "risk_pct_of_px": np.nan, "bars_held": len(sub),
                          "exit_reason": "eod", "symbol": r.symbol})
    hold = pd.DataFrame(hold_rows)
    if not hold.empty:
        ctrl_rows.append({"label": "control: open->close, same session+side",
                          "n": len(hold), "mean_ret_pct": hold["ret_pct"].mean(),
                          "t_ret": A.tstat(hold["ret_pct"]),
                          "p_ret": A.two_sided_p(A.tstat(hold["ret_pct"])),
                          "win_rate": (hold["ret_pct"] > 0).mean(),
                          "breakeven_bps": A.breakeven_bps(hold["ret_pct"])})

    ctrl = pd.DataFrame(ctrl_rows)
    A.save(ctrl, f"s10_controls_{tf}.csv")
    print("\ncontrols:")
    print(ctrl[["label", "n", "mean_R", "win_rate", "mean_ret_pct", "t_ret"]].to_string(index=False))

    # ---- the random-entry null --------------------------------------------
    print("\nrandom-entry null (200 draws) ...")
    means, wins = M.random_entry_null(sessions, base, real, draws=200)
    real_mean = real["R"].mean()
    pct = float((means < real_mean).mean() * 100)
    null = pd.DataFrame({"draw_mean_R": means, "draw_win_rate": wins})
    A.save(null, f"s10_random_entry_null_{tf}.csv")
    print(f"  real mean R = {real_mean:+.4f}")
    print(f"  null mean   = {means.mean():+.4f}  sd={means.std():.4f}  "
          f"p5={np.percentile(means, 5):+.4f} p95={np.percentile(means, 95):+.4f}")
    print(f"  real sits at the {pct:.1f}th percentile of the random-entry null")

    # ---- costs -------------------------------------------------------------
    cs = A.cost_sweep(real["ret_pct"], (0, 2, 5, 10, 20, 40))
    A.save(cs, f"s10_cost_sweep_{tf}.csv")
    print("\ncost sweep (round-trip bps of notional):")
    print(cs.to_string(index=False))

    # ---- out of sample ------------------------------------------------------
    is_df = real[pd.to_datetime(real["date"]) < OOS_FROM]
    oos_df = real[pd.to_datetime(real["date"]) >= OOS_FROM]
    best_is = grid_df.copy()
    # pick the best grid cell IN SAMPLE only, then score it out of sample
    is_scores = []
    for name, (spec, d) in results.items():
        d_is = d[pd.to_datetime(d["date"]) < OOS_FROM] if len(d) else d
        is_scores.append({"spec": name, "n_is": len(d_is),
                          "mean_R_is": d_is["R"].mean() if len(d_is) else np.nan,
                          "t_is": A.tstat(d_is["R"]) if len(d_is) else np.nan})
    is_scores = pd.DataFrame(is_scores).sort_values("t_is", ascending=False)
    pick = is_scores.iloc[0]["spec"]
    d_pick = results[pick][1]
    d_pick_oos = d_pick[pd.to_datetime(d_pick["date"]) >= OOS_FROM]
    oos = pd.DataFrame([
        summarise(is_df, "source-as-written IN SAMPLE (<2024-06)"),
        summarise(oos_df, "source-as-written OUT OF SAMPLE (>=2024-06)"),
        summarise(d_pick[pd.to_datetime(d_pick["date"]) < OOS_FROM],
                  f"best-in-sample cell [{pick}] IN SAMPLE"),
        summarise(d_pick_oos, f"best-in-sample cell [{pick}] OUT OF SAMPLE"),
    ])
    A.save(oos, f"s10_oos_{tf}.csv")
    print("\nout of sample:")
    print(oos[["label", "n", "mean_R", "win_rate", "t_R", "p_R"]].to_string(index=False))

    print(f"\nspecifications tried: {specs_tried}; "
          f"Bonferroni |t| bar = {bonferroni_bar(specs_tried):.3f}")
    A.save(real, f"s10_trades_{tf}.csv")
    return real, grid_df, ctrl


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5Min")
