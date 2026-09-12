"""Where do the source-10 model's wins actually come from?

Allowing the intrabar fill bar to resolve the trade made results BETTER, not
worse, which is the opposite of what a pessimism fix should do. That is a
signal to look harder, not to bank the number.

The suspicion: a resting buy limit at the FVG midpoint is being filled at the
exact LOW of a bar that then runs to the target inside the same candle. Two
separate optimistic assumptions stack there:

  1. FILL AT THE EXTREME. `low <= mid` counts as a fill even when `mid` is the
     bar's low tick. In reality a limit at the extreme of a move is usually not
     filled -- price has to trade THROUGH your price, or you need queue
     priority at it. Requiring `low < mid` strictly is the honest version.
  2. FAVOURABLE INTRABAR PATH. On the fill bar we know only the range, not the
     path. simulate_bracket already resolves stop-before-target, but a bar that
     never touches the stop and does touch the target books a full win from a
     fill at its own extreme.

This script quantifies both.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402
import s10_fvg_choch as M  # noqa: E402
import s10_run as R  # noqa: E402


def main(tf="5Min"):
    bars = R.load(tf)
    sessions = M.split_sessions(bars)
    base = M.Spec(name="source_as_written", k=3, r_target=4.0)

    honest = M.run(bars, base, "model", sessions=sessions)
    print(f"trades: {len(honest)}")

    same = honest[honest["bars_held"] == 0]
    later = honest[honest["bars_held"] > 0]
    print("\n--- resolved ON the fill bar vs later ---")
    for lbl, d in [("same-bar (bars_held==0)", same), ("later bars", later)]:
        if len(d):
            print(f"{lbl:26s} n={len(d):5d} ({len(d)/len(honest):5.1%})  "
                  f"meanR={d['R'].mean():+.4f}  win={(d['R']>0).mean():.3f}  "
                  f"contribution to total R = {d['R'].sum():+.1f}")
    print(f"\ntotal R = {honest['R'].sum():+.1f}; "
          f"same-bar share of total R = "
          f"{same['R'].sum()/honest['R'].sum() if honest['R'].sum() else float('nan'):.1%}")
    print("\nexit reasons, same-bar trades:")
    print(same["exit_reason"].value_counts().to_string())

    rows = []
    for lbl, d in [("ALL trades", honest), ("drop same-bar resolutions", later)]:
        rows.append({"variant": lbl, "n": len(d), "mean_R": d["R"].mean(),
                     "win": (d["R"] > 0).mean(), "t_R": A.tstat(d["R"]),
                     "mean_ret_pct": d["ret_pct"].mean(),
                     "breakeven_bps": A.breakeven_bps(d["ret_pct"])})
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    A.save(out, f"s10_diagnose_samebar_{tf}.csv")
    A.save(honest.assign(same_bar=honest["bars_held"] == 0), f"s10_trades_diag_{tf}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5Min")
