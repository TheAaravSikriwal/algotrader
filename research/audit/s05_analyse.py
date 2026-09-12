"""Source 05 audit -- STEPS 3/4/5a/6/7: grid summary, costs, control, OOS.

Writes results/s05_grid_summary.csv, s05_cost_sensitivity.csv,
       s05_oos.csv, s05_inverted_control.csv, s05_equity.csv,
       s05_trades_headline.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
from core.journal import bonferroni_bar  # noqa: E402

pd.set_option("display.width", 200)
KEYS = ["ma", "vwap", "fill", "stop", "target"]
SPREAD_BPS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
SLIP_BPS = 10.0   # extra adverse fill on a stop-buy into a fast gapper, each side


def summarise(g: pd.DataFrame, label: str) -> dict:
    d = A.describe_trades(g["pnl_pct"], label)
    if d["n"]:
        d["avg_R"] = float(g["R_mult"].mean())
        d["median_R"] = float(g["R_mult"].median())
        d["expectancy_R"] = float(g["R_mult"].mean())
        d["pct_stopped"] = float(g["exit_reason"].str.startswith("stop").mean())
        d["pct_target"] = float(g["exit_reason"].str.startswith("target").mean())
        d["pct_eod"] = float((g["exit_reason"] == "eod").mean())
        d["mean_R_dollars"] = float(g["R_dollars"].mean())
    return d


def main():
    td = pd.read_parquet(A.CACHE / "s05_trades.parquet")
    td["date"] = pd.to_datetime(td["date"])
    real = td[~td["inverted"]].copy()
    inv = td[td["inverted"]].copy()
    print(f"trade rows: real={len(real):,} inverted={len(inv):,}", flush=True)

    n_tests = real.groupby(KEYS).ngroups
    bar = bonferroni_bar(n_tests)
    print(f"\nSTEP 3: {n_tests} parameter combinations tested. "
          f"Bonferroni |t| bar at alpha=0.05: {bar:.3f}", flush=True)

    rows = []
    for k, g in real.groupby(KEYS):
        d = summarise(g, "|".join(k))
        d.update(dict(zip(KEYS, k)))
        d["breakeven_bps"] = A.breakeven_bps(g["pnl_pct"])
        d["mean_net_1x"] = g["pnl_pct"].mean() - (SPREAD_BPS + 2 * SLIP_BPS) / 100.0
        d["mean_net_2x"] = g["pnl_pct"].mean() - 2 * (SPREAD_BPS + 2 * SLIP_BPS) / 100.0
        d["passes_bonferroni"] = abs(d["t"]) > bar
        rows.append(d)
    grid = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    A.save(grid, "s05_grid_summary.csv")

    cols = ["ma", "vwap", "fill", "stop", "target", "n", "mean_pct", "median_pct",
            "win_rate", "avg_R", "t", "p", "pct_stopped", "pct_target",
            "breakeven_bps", "mean_net_1x", "mean_net_2x", "passes_bonferroni"]
    print("\n=== FULL GRID, sorted by gross mean per-trade % ===")
    print(grid[cols].round(4).to_string(index=False), flush=True)

    # ---------------- STEP 4: cost sensitivity on the top variants ---------
    print(f"\n=== STEP 4: cost sensitivity (spread estimate {SPREAD_BPS:.0f} bps "
          f"round trip + {2 * SLIP_BPS:.0f} bps slippage) ===")
    best = grid.iloc[0]
    bk = tuple(best[k] for k in KEYS)
    bg = real.groupby(KEYS).get_group(bk)
    rt = SPREAD_BPS + 2 * SLIP_BPS
    cs = A.cost_sweep(bg["pnl_pct"], [0, 20, rt, 2 * rt, 3 * rt])
    cs.insert(0, "variant", "|".join(map(str, bk)))
    print(cs.round(4).to_string(index=False), flush=True)
    print(f"break-even round-trip cost for best variant: "
          f"{A.breakeven_bps(bg['pnl_pct']):.1f} bps", flush=True)

    # widest, most-traded variants too
    frames = [cs]
    for label, mask in [("most_trades", grid["n"].idxmax()),
                        ("best_t", grid["t"].idxmax())]:
        row = grid.loc[mask]
        k = tuple(row[c] for c in KEYS)
        g = real.groupby(KEYS).get_group(k)
        c2 = A.cost_sweep(g["pnl_pct"], [0, 20, rt, 2 * rt, 3 * rt])
        c2.insert(0, "variant", f"{label}:" + "|".join(map(str, k)))
        frames.append(c2)
        print(f"\n{label}: " + "|".join(map(str, k)))
        print(c2.round(4).to_string(index=False), flush=True)
    A.save(pd.concat(frames, ignore_index=True), "s05_cost_sensitivity.csv")

    # ---------------- STEP 5a: inverted control ---------------------------
    print("\n=== STEP 5a: DELIBERATELY BROKEN CONTROL (enter on first new LOW) ===")
    irows = []
    for k, g in inv.groupby(KEYS):
        d = summarise(g, "|".join(k))
        d.update(dict(zip(KEYS, k)))
        irows.append(d)
    igrid = pd.DataFrame(irows)
    A.save(igrid, "s05_inverted_control.csv")
    merged = grid[KEYS + ["n", "mean_pct", "win_rate", "avg_R", "t"]].merge(
        igrid[KEYS + ["n", "mean_pct", "win_rate", "avg_R", "t"]],
        on=KEYS, suffixes=("_real", "_inv"))
    print(merged.sort_values("mean_pct_real", ascending=False).head(12)
          .round(4).to_string(index=False), flush=True)
    print("\naggregate over all 120 combos:")
    print(f"  real     mean={grid['mean_pct'].mean():.4f}%  "
          f"median t={grid['t'].median():.3f}")
    print(f"  inverted mean={igrid['mean_pct'].mean():.4f}%  "
          f"median t={igrid['t'].median():.3f}", flush=True)

    # ---------------- STEP 6: out of sample -------------------------------
    dates = np.sort(real["date"].unique())
    cut = dates[int(len(dates) * 0.60)]
    ins = real[real["date"] < cut]
    oos = real[real["date"] >= cut]
    print(f"\n=== STEP 6: OOS split at {pd.Timestamp(cut).date()} "
          f"(IS {pd.Timestamp(dates[0]).date()}..{pd.Timestamp(cut).date()}, "
          f"OOS ..{pd.Timestamp(dates[-1]).date()}) ===")
    orows = []
    for k, g in ins.groupby(KEYS):
        if len(g) < 30:
            continue
        go = oos.groupby(KEYS).get_group(k) if k in oos.groupby(KEYS).groups else None
        orows.append({
            **dict(zip(KEYS, k)),
            "is_n": len(g), "is_mean_pct": g["pnl_pct"].mean(),
            "is_win": (g["pnl_pct"] > 0).mean(), "is_avg_R": g["R_mult"].mean(),
            "is_t": A.tstat(g["pnl_pct"]),
            "oos_n": 0 if go is None else len(go),
            "oos_mean_pct": np.nan if go is None else go["pnl_pct"].mean(),
            "oos_win": np.nan if go is None else (go["pnl_pct"] > 0).mean(),
            "oos_avg_R": np.nan if go is None else go["R_mult"].mean(),
            "oos_t": np.nan if go is None else A.tstat(go["pnl_pct"]),
        })
    od = pd.DataFrame(orows).sort_values("is_mean_pct", ascending=False)
    A.save(od, "s05_oos.csv")
    print(od.head(15).round(4).to_string(index=False), flush=True)
    ch = od.iloc[0]
    print(f"\nvariant CHOSEN on in-sample only: "
          f"{ch['ma']}|{ch['vwap']}|{ch['fill']}|{ch['stop']}|{ch['target']}")
    print(f"  IS : n={ch['is_n']:.0f} mean={ch['is_mean_pct']:.4f}% "
          f"win={ch['is_win']:.3f} avgR={ch['is_avg_R']:.3f} t={ch['is_t']:.2f}")
    print(f"  OOS: n={ch['oos_n']:.0f} mean={ch['oos_mean_pct']:.4f}% "
          f"win={ch['oos_win']:.3f} avgR={ch['oos_avg_R']:.3f} t={ch['oos_t']:.2f}")
    print(f"  OOS net of {rt:.0f} bps: {ch['oos_mean_pct'] - rt / 100:.4f}%",
          flush=True)
    print(f"\ncorrelation IS mean vs OOS mean across {len(od)} variants: "
          f"{od['is_mean_pct'].corr(od['oos_mean_pct']):.3f}", flush=True)

    # ---------------- benchmark -------------------------------------------
    bm = pd.read_csv(A.RESULTS / "s05_benchmarks.csv", keep_default_na=False)
    print("\n=== STEP 5c: TRIVIAL BENCHMARK, buy 09:30 open sell 15:55 close ===")
    d = A.describe_trades(bm["pnl_pct"], "open_to_1555")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()},
          flush=True)
    print("by gap decile:")
    bm["gap_dec"] = pd.qcut(bm["gap"], 10, duplicates="drop")
    print(bm.groupby("gap_dec", observed=True)["pnl_pct"]
          .agg(["count", "mean", "median", lambda s: (s > 0).mean()])
          .rename(columns={"<lambda_0>": "win"}).round(3).to_string(), flush=True)

    # ---------------- equity curve, one trade per day ----------------------
    hb = real.groupby(KEYS).get_group(bk).copy()
    A.save(hb.assign(date=hb["date"].dt.strftime("%Y-%m-%d")), "s05_trades_headline.csv")
    one = hb.sort_values(["date", "rank"]).groupby("date").first().reset_index()
    eq_rows = []
    for cost in (0.0, rt, 2 * rt):
        r = one["R_mult"].to_numpy() * 1.0
        # 1% of equity risked per trade; cost charged in % of notional, converted
        # to R via the per-trade R/entry ratio
        rr = (one["R_dollars"] / one["entry_px"]).to_numpy()
        cost_R = (cost / 10000.0) / np.where(rr > 0, rr, np.nan)
        net_R = r - np.nan_to_num(cost_R, nan=0.0)
        eq = np.cumprod(1.0 + 0.01 * net_R)
        yrs = (one["date"].iloc[-1] - one["date"].iloc[0]).days / 365.25
        cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 and yrs > 0 else np.nan
        eq_rows.append({"cost_bps": cost, "n_days": len(one), "final_mult": eq[-1],
                        "cagr": cagr, "mean_net_R": float(np.nanmean(net_R)),
                        "max_dd": float((1 - eq / np.maximum.accumulate(eq)).max())})
        if cost == 0:
            one["equity_frictionless"] = eq
    ed = pd.DataFrame(eq_rows)
    print("\n=== equity, ONE trade/day (best-ranked gapper with a signal), "
          "1% of equity risked per trade ===")
    print(ed.round(4).to_string(index=False), flush=True)
    A.save(one.assign(date=one["date"].astype(str)), "s05_equity.csv")


if __name__ == "__main__":
    main()
