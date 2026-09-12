"""Source 01 audit -- engine cross-check, panel equity curves, cost arithmetic.

Three things the event-study grid cannot say on its own:

1. **Is the event-study arithmetic the same thing the engine does?**
   `core.engine.run_backtest` is the repo's no-lookahead contract. Running one
   (entry, hold) cell through it per symbol and comparing its trade log to the
   open-to-open forward returns is the check that the grid is measuring what a
   backtest would have earned, not an idealisation of it.

2. **What does the cross-sectional reading look like as a funded strategy?**
   `core.panel.run_panel_backtest` holds the top decile in overlapping
   tranches, pays slippage on every rebalance, and is scored against the
   equal-weight universe by `core.metrics.equity_stats`.

3. **What does a 2-5 day hold cost per year?** Per-trade bps are easy to wave
   away. Annualised, they are not.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

import s01_core as C  # noqa: E402
import s01_run as R  # noqa: E402

from core.engine import BacktestConfig, run_backtest  # noqa: E402
from core.metrics import equity_stats  # noqa: E402
from core.panel import PanelConfig, equal_weight_benchmark, run_panel_backtest  # noqa: E402


# ---------------------------------------------------------------------------
# 1. engine cross-check
# ---------------------------------------------------------------------------
def engine_crosscheck(panel, bars, sigs, name: str, hold: int, n_symbols: int = 12):
    """Run the same cell through core.engine.run_backtest, symbol by symbol.

    The signal is held for exactly `hold` bars after it fires, so the engine
    enters at the open of t+1 and exits at the open of t+1+hold. With zero
    slippage and zero commission the engine's `return_pct` per trade must equal
    the event study's open-to-open return.
    """
    fwd = C.forward_open_returns(panel, hold)
    keep = R.window_mask(panel.index, R.BT_START, R.BT_END)
    rows = []
    for sym in panel.symbols[:n_symbols]:
        df = bars[sym].loc[R.BT_START:R.BT_END]
        sig = sigs[name][sym].reindex(df.index).fillna(False).astype(float)
        # hold for `hold` bars: exposure is 1 if a signal fired in the last
        # `hold` bars inclusive of today
        held = (sig.rolling(hold, min_periods=1).max() > 0).astype(float)
        res = run_backtest(df, held, BacktestConfig(
            slippage_bps=0.0, commission_pct=0.0, position_size=1.0))
        tr = res.trades
        if tr.empty:
            continue

        # every entry the engine took, matched to the event study
        ev = []
        for _, t in tr.iterrows():
            i = df.index.get_loc(t.entry_time)
            j = df.index.get_loc(t.exit_time)
            if j >= len(df) - 1 and t.exit_reason == "end of data":
                continue
            o_in, o_out = df["open"].iloc[i], df["open"].iloc[j]
            ev.append((t.return_pct, (o_out / o_in - 1.0) * 100.0))
        if not ev:
            continue
        e = np.array(ev)
        rows.append({"symbol": sym, "trades": len(e),
                     "engine_mean_pct": e[:, 0].mean(),
                     "openopen_mean_pct": e[:, 1].mean(),
                     "max_abs_diff_bps": float(np.abs(e[:, 0] - e[:, 1]).max() * 100)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. panel equity curves for the cross-sectional readings
# ---------------------------------------------------------------------------
def overlapping_weights(mask: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Hold each day's picks for `hold` bars, in `hold` overlapping tranches.

    Each day 1/hold of the book is refreshed, which is how a fixed-hold rule is
    actually run. Weights are normalised to sum to 1 across whatever is held.
    """
    m = mask.astype(float)
    held = m.rolling(hold, min_periods=1).sum()
    tot = held.sum(axis=1)
    w = held.div(tot.replace(0, np.nan), axis=0).fillna(0.0)
    return w


def panel_curves(panel, sigs, cells, slippages=(0.0, 1.0, 2.5, 5.0, 10.0)):
    keep = R.window_mask(panel.index, R.BT_START, R.BT_END)
    sub = panel.__class__(opens=panel.opens[keep], highs=panel.highs[keep],
                          lows=panel.lows[keep], closes=panel.closes[keep],
                          volumes=panel.volumes[keep])
    bench = equal_weight_benchmark(sub)
    bstats = equity_stats(bench)
    rows = [{"entry": "EQUAL-WEIGHT UNIVERSE (buy & hold)", "hold": "-",
             "slippage_bps_per_side": 0.0, "annual_turnover": 0.0,
             **{k: bstats[k] for k in ("CAGR", "Sharpe", "Max drawdown", "Total return")}}]
    for name, hold in cells:
        w = overlapping_weights(sigs[name][keep], hold)
        for slip in slippages:
            res = run_panel_backtest(sub, w, PanelConfig(slippage_bps=slip,
                                                         min_weight_change=0.0005))
            st = equity_stats(res.equity)
            rows.append({"entry": C.LABELS.get(name, name), "hold": hold,
                         "slippage_bps_per_side": slip,
                         "annual_turnover": res.annual_turnover,
                         **{k: st[k] for k in ("CAGR", "Sharpe", "Max drawdown",
                                               "Total return")}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. cost arithmetic and the monotonicity question
# ---------------------------------------------------------------------------
def cost_arithmetic(grid: pd.DataFrame) -> pd.DataFrame:
    """What the base rate is per day held, and what costs are per YEAR."""
    b = grid[(grid["sample"] == "full") & (grid.entry == "BASE_RATE")]
    rows = []
    for _, r in b.iterrows():
        h = int(r.hold)
        trips = 252.0 / h
        rows.append({
            "hold_days": h,
            "base_mean_pct": r.mean_pct,
            "base_pct_per_day_held": r.mean_pct / h,
            "round_trips_per_year": trips,
            "ann_gross_pct": r.mean_pct * trips,
            **{f"ann_cost_pct@{c}bps": trips * c / 100.0 for c in R.COST_GRID},
            **{f"ann_net_pct@{c}bps": (r.mean_pct - c / 100.0) * trips
               for c in R.COST_GRID},
        })
    return pd.DataFrame(rows)


def monotonicity(grid: pd.DataFrame) -> pd.DataFrame:
    """Mean return PER DAY HELD for every reading. If 2-5 days were special
    this column would bulge there; if it is flat, the hold length is doing
    nothing and the source's '2 to 5 days' is arbitrary."""
    g = grid[(grid["sample"] == "full")].copy()
    g["pct_per_day_held"] = g.mean_pct / g.hold
    piv = g.pivot_table(index="label", columns="hold", values="pct_per_day_held")
    return piv


def main():
    bars, frame, picked, panel = R.build_world()
    sigs = C.build_signals(panel, bars)
    grid = pd.read_csv(A.RESULTS / "s01_grid_entry_by_hold.csv")

    print("=== 1. ENGINE CROSS-CHECK (core.engine.run_backtest vs the event study) ===")
    for name, hold in (("e_20d_high", 3), ("a3_ret20_topdec", 5)):
        cc = engine_crosscheck(panel, bars, sigs, name, hold)
        print(f"\n{C.LABELS[name]}, hold {hold}:")
        print(cc.round(6).to_string(index=False))
        print(f"  worst per-trade disagreement across all symbols: "
              f"{cc.max_abs_diff_bps.max():.6f} bps")
        A.save(cc, f"s01_engine_crosscheck_{name}_h{hold}.csv")

    print("\n=== 2. PANEL EQUITY CURVES (core.panel.run_panel_backtest) ===")
    cells = [("a3_ret20_topdec", 3), ("a3_ret20_topdec", 5),
             ("a1_ret5_topdec", 3), ("a1_ret5_topdec_INV", 3),
             ("e_20d_high", 3), ("c1_rsi_gt60", 3)]
    pc = panel_curves(panel, sigs, cells)
    A.save(pc, "s01_panel_curves.csv")
    print(pc.round(4).to_string(index=False))

    print("\n=== 3. COST ARITHMETIC, ANNUALISED ===")
    ca = cost_arithmetic(grid)
    A.save(ca, "s01_cost_arithmetic.csv")
    print(ca.round(4).to_string(index=False))

    print("\n=== 4. RETURN PER DAY HELD (is 2-5 days special?) ===")
    mono = monotonicity(grid)
    A.save(mono.reset_index(), "s01_per_day_held.csv")
    print(mono.round(5).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    main()
