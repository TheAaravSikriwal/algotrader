"""The controls that decide whether source 10 has an edge or an execution artefact.

The model showed a positive mean R. Before that can be called an edge, three
alternative explanations have to be killed, and each needs its own control:

  A. LIQUIDITY PROVISION, NOT GEOMETRY. The entry is a resting BUY limit placed
     BELOW the market (or a sell limit above it). Buying dips with limit orders
     earns the spread and harvests short-horizon intraday mean reversion. That
     would produce a positive mean R with the FVG doing no work at all.
     Control `random_level`: same session, same direction, same stop distance,
     same time-to-live -- but the limit is placed at a RANDOM offset drawn from
     the model's own distribution of (price - FVG mid) offsets. If this matches
     the model, the fair value gap is decoration on a dip-buying rule.

  B. FILL AT THE EXTREME. `low <= mid` books a fill even when mid is the bar's
     low tick, where in practice you are at the back of the queue and usually
     do not trade. Control `strict_fill`: require `low < mid` strictly.

  C. LIMIT VS MARKET EXECUTION. Control `market_entry`: same signal, but enter
     at the OPEN of the bar after the FVG completes, paying the spread like a
     market order instead of earning it. If the edge vanishes, the edge was
     execution, and a retail trader resting 5-minute limits in SPY is competing
     with market makers for it.
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
import structure as S  # noqa: E402
from core.indicators import atr as atr_ind  # noqa: E402


def variant_trades(sub: pd.DataFrame, spec: M.Spec, rng, variant: str) -> list[dict]:
    if len(sub) < 25:
        return []
    a = atr_ind(sub, 14)
    sw = S.swings(sub, spec.k)
    events = S.structure_events(sub, spec.k)
    gaps = S.fair_value_gaps(sub, a, spec.min_fvg_atr)
    ev_by_bar, gap_by_bar = {}, {}
    for e in events:
        ev_by_bar.setdefault(e.idx, []).append(e)
    for g in gaps:
        gap_by_bar.setdefault(g.idx, []).append(g)

    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)
    opens = sub["open"].to_numpy(float)
    closes = sub["close"].to_numpy(float)
    n = len(sub)

    bos_run, bos_dir, choch_dir, choch_bar = 0, 0, 0, -10_000
    out, taken = [], False

    for i in range(n):
        for e in ev_by_bar.get(i, []):
            if e.kind == "BOS_up":
                bos_run = bos_run + 1 if bos_dir >= 0 else 1
                bos_dir = 1
            elif e.kind == "BOS_down":
                bos_run = bos_run + 1 if bos_dir <= 0 else 1
                bos_dir = -1
            elif e.kind == "CHOCH_up":
                if bos_dir < 0 and bos_run >= spec.min_bos:
                    choch_dir, choch_bar = 1, i
                bos_run, bos_dir = 1, 1
            elif e.kind == "CHOCH_down":
                if bos_dir > 0 and bos_run >= spec.min_bos:
                    choch_dir, choch_bar = -1, i
                bos_run, bos_dir = 1, -1
        if taken:
            continue

        for g in gap_by_bar.get(i, []):
            if choch_dir == 0 or g.direction != choch_dir or i - choch_bar > spec.max_choch_age:
                continue
            direction = g.direction

            if variant == "market_entry":
                if i + 1 >= n:
                    continue
                fill_i, entry_px = i + 1, opens[i + 1]
                resolve = i + 1
            else:
                if variant == "random_level":
                    # same OFFSET DISTRIBUTION from the current price, random draw
                    off = abs(closes[i] - g.mid) / max(closes[i], 1e-9)
                    off = off * float(rng.uniform(0.5, 1.5))
                    entry_px = closes[i] * (1 - direction * off)
                else:
                    entry_px = g.mid
                fill_i = None
                for j in range(i + 1, min(i + 1 + spec.ttl, n)):
                    hit = (lows[j] < entry_px < highs[j]) if variant == "strict_fill" \
                        else (lows[j] <= entry_px <= highs[j])
                    if hit:
                        fill_i = j
                        break
                if fill_i is None:
                    continue
                resolve = fill_i

            if direction > 0:
                cands = [s.price for s in sw
                         if s.kind == "low" and s.known_at <= fill_i and s.price < entry_px]
                stop_px = min(cands[-1], g.lo) if cands else g.lo
            else:
                cands = [s.price for s in sw
                         if s.kind == "high" and s.known_at <= fill_i and s.price > entry_px]
                stop_px = max(cands[-1], g.hi) if cands else g.hi
            if (entry_px - stop_px) * direction <= 0:
                continue

            r = S.simulate_bracket(sub, fill_i, entry_px, stop_px, direction,
                                   spec.r_target, max_bars=n, resolve_from=resolve)
            if r:
                r["variant"] = variant
                out.append(r)
                taken = True
            break
    return out


def run(sessions, spec, variant, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (sym, day), sub in sessions.items():
        for t in variant_trades(sub, spec, rng, variant):
            t["symbol"], t["date"] = sym, day
            rows.append(t)
    return pd.DataFrame(rows)


def main(tf="5Min"):
    bars = R.load(tf)
    sessions = M.split_sessions(bars)
    spec = M.Spec(name="source_as_written", k=3, r_target=4.0)

    rows = []
    base = M.run(bars, spec, "model", sessions=sessions)
    for lbl, d in [("model: limit at FVG midpoint", base)]:
        rows.append(_row(lbl, d))
        print(f"{lbl:42s} n={len(d):5d} meanR={d['R'].mean():+.4f}", flush=True)

    for variant, lbl in [("strict_fill", "B: strict fill (low < mid, not <=)"),
                         ("market_entry", "C: market entry at next bar open"),
                         ("random_level", "A: limit at a RANDOM level, same offsets")]:
        d = run(sessions, spec, variant, seed=13)
        rows.append(_row(lbl, d))
        print(f"{lbl:42s} n={len(d):5d} "
              f"meanR={d['R'].mean() if len(d) else float('nan'):+.4f}", flush=True)

    # random_level repeated, to get a null distribution rather than one draw
    means = []
    for s in range(30):
        d = run(sessions, spec, "random_level", seed=100 + s)
        if len(d):
            means.append(d["R"].mean())
    means = np.array(means)
    print(f"\nrandom-level null over 30 draws: mean {means.mean():+.4f} "
          f"sd {means.std():.4f} p5 {np.percentile(means,5):+.4f} "
          f"p95 {np.percentile(means,95):+.4f}")
    print(f"model mean R = {base['R'].mean():+.4f}; percentile in null = "
          f"{float((means < base['R'].mean()).mean()*100):.1f}")

    out = pd.DataFrame(rows)
    A.save(out, f"s10_controls2_{tf}.csv")
    A.save(pd.DataFrame({"draw_mean_R": means}), f"s10_randomlevel_null_{tf}.csv")
    print("\n" + out.to_string(index=False))


def _row(lbl, d):
    if d is None or d.empty:
        return {"variant": lbl, "n": 0}
    return {"variant": lbl, "n": len(d), "mean_R": d["R"].mean(),
            "win": (d["R"] > 0).mean(), "t_R": A.tstat(d["R"]),
            "p_R": A.two_sided_p(A.tstat(d["R"])),
            "mean_ret_pct": d["ret_pct"].mean(),
            "t_ret": A.tstat(d["ret_pct"]),
            "breakeven_bps": A.breakeven_bps(d["ret_pct"]),
            "median_risk_pct": d["risk_pct_of_px"].median(),
            "same_bar_share": (d["bars_held"] == 0).mean()}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5Min")
