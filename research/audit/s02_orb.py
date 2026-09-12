"""s02 STEP 3 -- Opening Range Breakout, done properly.

ORB is NOT one of the eleven audited sources. It is included because it is the
best-documented retail intraday rule in the public literature, so it is the
benchmark the sources should be judged against: if the strongest published
intraday rule does not clear costs, no source that is vaguer than it does either.

DATA AND TIMESTAMPS -- what every signal knows, and when
--------------------------------------------------------
* 5Min Alpaca consolidated (SIP) bars, 2021-01-01..2026-09-01, converted to
  Eastern with auditlib.to_eastern and filtered to 09:30-15:55 with auditlib.rth.
* Opening range = high/low of the first N minutes. For N=5 that is the single
  09:30 bar; it is complete, and therefore known, at 09:35:00 ET. N=15 -> known
  at 09:45:00. N=30 -> known at 10:00:00. The first bar that can trigger is the
  one that OPENS at or after that instant. No bar contributes to a decision
  before its own close except through its open, which is known at its open.
* ATR(14) is built from daily high/low/close aggregated from these same intraday
  bars and then SHIFTED ONE DAY, so the ATR used on session d is computed from
  sessions d-14..d-1 and is known at 09:30 on d.
* Entry fill: a stop order resting at the OR boundary fills at the boundary, or
  at the bar open if the bar opened through it -- max(open, or_high) for a long.
* The bracket is resolved from the bar AFTER entry by structure.simulate_bracket,
  which gives an ambiguous bar to the stop and fills gaps at the open.

Writes results/s02_orb_grid.csv, s02_orb_oos.csv, s02_orb_null.csv,
       s02_orb_selftest.csv, s02_orb_costsweep.csv, s02_orb_trades.parquet
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import structure as S  # noqa: E402
from core.journal import bonferroni_bar  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "TSLA", "NVDA"]
START, END = "2021-01-01", "2026-09-01"
IS_END = "2024-06-01"          # in-sample ends, out-of-sample begins
N_MINUTES = [5, 15, 30]
STOPS = ["or", "atr10"]
TARGETS = {"1R": 1.0, "2R": 2.0, "3R": 3.0, "eod": 1e9}
ENTRIES = ["touch", "close"]
DIRECTIONS = ["long_only", "short_only", "long_short"]
SEED = 20260911
N_DRAWS = 200


# ---------------------------------------------------------------------------
def sessions_of(sym: str):
    bars = A.rth(A.bars_chunked(sym, START, END, "5Min"))
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    day = pd.DatetimeIndex(bars.index).normalize()
    # daily aggregation from the SAME bars, then ATR(14) shifted one day
    g = bars.groupby(day)
    daily = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last()})
    pc = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"],
                    (daily["high"] - pc).abs(),
                    (daily["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean().shift(1)   # known at 09:30 of day d
    out = []
    for d, sl in bars.groupby(day):
        if len(sl) < 12:
            continue
        out.append((d, sl, float(atr.get(d, np.nan))))
    return out


def triggers(sl: pd.DataFrame, n_min: int, entry_rule: str):
    """Return (j, entry_px, direction) for the first breakout, or None."""
    k = n_min // 5
    if len(sl) < k + 3:
        return None
    o = sl["open"].to_numpy(float); h = sl["high"].to_numpy(float)
    l = sl["low"].to_numpy(float);  c = sl["close"].to_numpy(float)
    or_hi = float(h[:k].max()); or_lo = float(l[:k].min())
    if not np.isfinite(or_hi) or or_hi <= or_lo:
        return None
    for j in range(k, len(sl) - 1):
        if entry_rule == "touch":
            up, dn = h[j] > or_hi, l[j] < or_lo
            if not (up or dn):
                continue
            if up and dn:
                # ambiguous bar. Decide from the OPEN only -- the level nearer the
                # open is the one price reaches first. Uses no future information.
                if o[j] >= or_hi:
                    up, dn = True, False
                elif o[j] <= or_lo:
                    up, dn = False, True
                elif (or_hi - o[j]) <= (o[j] - or_lo):
                    up, dn = True, False
                else:
                    up, dn = False, True
            if up:
                return j, max(o[j], or_hi), +1, or_hi, or_lo
            return j, min(o[j], or_lo), -1, or_hi, or_lo
        else:   # "close": the bar must CLOSE beyond the range; fill at that close
            if c[j] > or_hi:
                return j, float(c[j]), +1, or_hi, or_lo
            if c[j] < or_lo:
                return j, float(c[j]), -1, or_hi, or_lo
    return None


def stop_px(entry_px, direction, or_hi, or_lo, atr, stop_kind):
    if stop_kind == "or":
        return or_lo if direction > 0 else or_hi
    if not np.isfinite(atr) or atr <= 0:
        return np.nan
    return entry_px - direction * 0.10 * atr


def build_trades(sym: str, sess) -> pd.DataFrame:
    rows = []
    for n_min in N_MINUTES:
        for entry_rule in ENTRIES:
            trig = {}
            for d, sl, atr in sess:
                t = triggers(sl, n_min, entry_rule)
                if t:
                    trig[d] = (sl, atr, t)
            for stop_kind in STOPS:
                for tname, rmul in TARGETS.items():
                    for d, (sl, atr, (j, epx, direction, or_hi, or_lo)) in trig.items():
                        spx = stop_px(epx, direction, or_hi, or_lo, atr, stop_kind)
                        if not np.isfinite(spx):
                            continue
                        r = S.simulate_bracket(sl, j, epx, spx, direction, rmul,
                                               max_bars=len(sl), eod_exit=True)
                        if r is None:
                            continue
                        rows.append({
                            "symbol": sym, "date": d, "n_min": n_min,
                            "entry_rule": entry_rule, "stop": stop_kind,
                            "target": tname, "direction": direction,
                            "entry_time": r["entry_time"], "exit_time": r["exit_time"],
                            "entry_i": j, "bars_held": r["bars_held"],
                            "ret_pct": r["ret_pct"], "R": r["R"],
                            "risk_pct": r["risk_pct_of_px"],
                            "exit_reason": r["exit_reason"],
                        })
    return pd.DataFrame(rows)


def spec_stats(df: pd.DataFrame, direction_mode: str) -> dict | None:
    if direction_mode == "long_only":
        d = df[df["direction"] > 0]
        sign = 1
    elif direction_mode == "short_only":
        d = df[df["direction"] < 0]
        sign = 1
    else:
        d = df
        sign = 1
    if len(d) < 30:
        return None
    r = d["ret_pct"] * sign
    t = A.tstat(r)
    return {"n": len(d), "mean_pct": float(r.mean()), "std_pct": float(r.std(ddof=1)),
            "median_pct": float(r.median()), "win_rate": float((r > 0).mean()),
            "mean_R": float(d["R"].mean()), "t": t, "p": A.two_sided_p(t),
            "breakeven_bps": float(r.mean() * 100.0),
            "mean_risk_pct": float(d["risk_pct"].mean()),
            "trades_per_yr": len(d) / ((pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25)}


def grid(trades: pd.DataFrame, lo=None, hi=None, tag="full") -> pd.DataFrame:
    t = trades
    if lo is not None:
        t = t[(t["date"] >= lo) & (t["date"] < hi)]
    rows = []
    keys = ["symbol", "n_min", "entry_rule", "stop", "target"]
    for k, g in t.groupby(keys):
        for dm in DIRECTIONS:
            st = spec_stats(g, dm)
            if st:
                st.update(dict(zip(keys, k)))
                st["direction_mode"] = dm
                st["window"] = tag
                rows.append(st)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def random_entry_null(sym, sess, n_min, entry_rule, stop_kind, tname, direction_mode,
                      n_draws=N_DRAWS, seed=SEED):
    """Same sessions, same direction, same stop DISTANCE, same target -- but the
    entry bar is drawn uniformly from the session instead of being the breakout.
    Entry fills at that bar's open."""
    rmul = TARGETS[tname]
    rng = np.random.default_rng(seed)
    base = []
    for d, sl, atr in sess:
        t = triggers(sl, n_min, entry_rule)
        if not t:
            continue
        j, epx, direction, or_hi, or_lo = t
        spx = stop_px(epx, direction, or_hi, or_lo, atr, stop_kind)
        if not np.isfinite(spx):
            continue
        risk = abs(epx - spx)
        if risk <= 0:
            continue
        if direction_mode == "long_only" and direction < 0:
            continue
        if direction_mode == "short_only" and direction > 0:
            continue
        base.append((sl, direction, risk))
    means = []
    o_cache = [(sl["open"].to_numpy(float), sl, direction, risk) for sl, direction, risk in base]
    for _ in range(n_draws):
        vals = []
        for o, sl, direction, risk in o_cache:
            j = int(rng.integers(0, len(sl) - 1))
            epx = float(o[j])
            spx = epx - direction * risk
            r = S.simulate_bracket(sl, j, epx, spx, direction, rmul,
                                   max_bars=len(sl), eod_exit=True)
            if r:
                vals.append(r["ret_pct"])
        if vals:
            means.append(float(np.mean(vals)))
    return np.array(means), len(base)


def selftests(sym, sess):
    """(a) pure no-stop inversion must flip sign EXACTLY.
       (b) bracketed inversion must flip sign approximately.
       (c) a deliberate 1-bar lookahead must improve results dramatically."""
    out = []
    for n_min in N_MINUTES:
        real, inv, look, look_n = [], [], [], []
        real_br, inv_br = [], []
        for d, sl, atr in sess:
            t = triggers(sl, n_min, "touch")
            if not t:
                continue
            j, epx, direction, or_hi, or_lo = t
            c = sl["close"].to_numpy(float)
            # (a) pure: enter at epx, exit at last bar's close. No stop.
            pure = (c[-1] - epx) / epx * 100.0 * direction
            real.append(pure)
            inv.append((c[-1] - epx) / epx * 100.0 * (-direction))
            # (c) LOOKAHEAD: keep the trade only if bar j+1's close confirms it.
            #     close[j+1] is NOT knowable at the entry on bar j. Deliberate leak.
            if j + 1 < len(sl) and (c[j + 1] - epx) * direction > 0:
                look.append(pure)
                look_n.append(1)
            # (b) bracketed 2R, OR stop
            spx = or_lo if direction > 0 else or_hi
            r = S.simulate_bracket(sl, j, epx, spx, direction, 2.0,
                                   max_bars=len(sl), eod_exit=True)
            if r:
                real_br.append(r["ret_pct"])
            risk = abs(epx - spx)
            spx2 = epx + direction * risk          # mirrored stop, same distance
            r2 = S.simulate_bracket(sl, j, epx, spx2, -direction, 2.0,
                                    max_bars=len(sl), eod_exit=True)
            if r2:
                inv_br.append(r2["ret_pct"])
        out.append({"symbol": sym, "n_min": n_min, "test": "a_pure_inversion",
                    "n": len(real), "real_mean_pct": float(np.mean(real)),
                    "variant_mean_pct": float(np.mean(inv)),
                    "real_t": A.tstat(real), "variant_t": A.tstat(inv),
                    "note": "no stop, exit at 15:55; inversion must be exactly -1x"})
        out.append({"symbol": sym, "n_min": n_min, "test": "b_bracket_inversion",
                    "n": len(real_br), "real_mean_pct": float(np.mean(real_br)),
                    "variant_mean_pct": float(np.mean(inv_br)),
                    "real_t": A.tstat(real_br), "variant_t": A.tstat(inv_br),
                    "note": "2R target, OR stop; approx flip, residual = stop-first drag"})
        out.append({"symbol": sym, "n_min": n_min, "test": "c_lookahead_1bar",
                    "n": len(look), "real_mean_pct": float(np.mean(real)),
                    "variant_mean_pct": float(np.mean(look)) if look else np.nan,
                    "real_t": A.tstat(real), "variant_t": A.tstat(look) if look else np.nan,
                    "note": "keep trade only if close[j+1] confirms -- reads the future"})
    return out


def main():
    t0 = time.time()
    all_tr, all_self, sess_cache = [], [], {}
    for sym in SYMBOLS:
        sess = sessions_of(sym)
        sess_cache[sym] = sess
        print(f"{sym}: {len(sess)} sessions  ({time.time()-t0:.0f}s)", flush=True)
        tr = build_trades(sym, sess)
        print(f"  {len(tr)} simulated trades across the grid ({time.time()-t0:.0f}s)", flush=True)
        all_tr.append(tr)
        all_self.extend(selftests(sym, sess))
    trades = pd.concat(all_tr, ignore_index=True)
    trades.to_parquet(A.CACHE / "s02_orb_trades.parquet")

    n_specs_per_sym = len(N_MINUTES) * len(ENTRIES) * len(STOPS) * len(TARGETS) * len(DIRECTIONS)
    n_specs_total = n_specs_per_sym * len(SYMBOLS)
    print(f"\nSPECIFICATIONS: {n_specs_per_sym} per symbol, {n_specs_total} in total")
    print(f"  bonferroni_bar({n_specs_per_sym}) = {bonferroni_bar(n_specs_per_sym):.3f}")
    print(f"  bonferroni_bar({n_specs_total}) = {bonferroni_bar(n_specs_total):.3f}")

    g_full = grid(trades, tag="full")
    g_is = grid(trades, pd.Timestamp(START), pd.Timestamp(IS_END), "is_2021_2024M05")
    g_oos = grid(trades, pd.Timestamp(IS_END), pd.Timestamp(END), "oos_2024M06_2026M09")
    allg = pd.concat([g_full, g_is, g_oos], ignore_index=True)
    allg["bonf_bar_per_symbol"] = bonferroni_bar(n_specs_per_sym)
    allg["bonf_bar_total"] = bonferroni_bar(n_specs_total)
    A.save(allg, "s02_orb_grid.csv")

    cols = ["symbol", "n_min", "entry_rule", "stop", "target", "direction_mode",
            "n", "mean_pct", "std_pct", "win_rate", "mean_R", "t", "p",
            "breakeven_bps", "mean_risk_pct", "trades_per_yr"]
    print("\n" + "=" * 120)
    print("FULL SAMPLE 2021-01..2026-09 -- top 20 specifications by t-stat (GROSS of cost)")
    print("=" * 120)
    print(g_full.sort_values("t", ascending=False).head(20)[cols].round(4).to_string(index=False))
    print("\nbottom 8:")
    print(g_full.sort_values("t").head(8)[cols].round(4).to_string(index=False))
    print(f"\nof {len(g_full)} specs: {(g_full['t'] > 1.96).sum()} have t>1.96 GROSS, "
          f"{(g_full['t'] > bonferroni_bar(n_specs_total)).sum()} clear the Bonferroni bar "
          f"{bonferroni_bar(n_specs_total):.2f}")
    print(f"median breakeven_bps across all specs = {g_full['breakeven_bps'].median():.2f}; "
          f"max = {g_full['breakeven_bps'].max():.2f}")

    # ---- cost sweep on the headline specs -------------------------------
    sweeps = []
    head = g_full.sort_values("t", ascending=False).head(10)
    for _, row in head.iterrows():
        sel = trades[(trades["symbol"] == row["symbol"]) & (trades["n_min"] == row["n_min"]) &
                     (trades["entry_rule"] == row["entry_rule"]) & (trades["stop"] == row["stop"]) &
                     (trades["target"] == row["target"])]
        if row["direction_mode"] == "long_only":
            sel = sel[sel["direction"] > 0]
        elif row["direction_mode"] == "short_only":
            sel = sel[sel["direction"] < 0]
        sw = A.cost_sweep(sel["ret_pct"], (0, 1, 2, 3, 5, 8, 10, 20))
        sw["spec"] = (f"{row['symbol']}|OR{row['n_min']}|{row['entry_rule']}|"
                      f"{row['stop']}|{row['target']}|{row['direction_mode']}")
        sweeps.append(sw)
    sweep = pd.concat(sweeps, ignore_index=True)
    A.save(sweep, "s02_orb_costsweep.csv")
    print("\n" + "=" * 120)
    print("COST SWEEP on the 10 best gross specs (round-trip bps of notional)")
    print("=" * 120)
    print(sweep[["spec", "round_trip_bps", "n", "mean_pct", "total_pct", "win_rate", "t", "p"]]
          .round(4).to_string(index=False))

    # ---- out of sample ---------------------------------------------------
    keys = ["symbol", "n_min", "entry_rule", "stop", "target", "direction_mode"]
    m = g_is.merge(g_oos, on=keys, suffixes=("_is", "_oos"))
    oos_rows = []
    for sym in SYMBOLS + ["ANY"]:
        sub = m if sym == "ANY" else m[m["symbol"] == sym]
        if not len(sub):
            continue
        best = sub.sort_values("t_is", ascending=False).iloc[0]
        oos_rows.append({"picked_for": sym,
                         **{k: best[k] for k in keys},
                         "n_is": best["n_is"], "mean_pct_is": best["mean_pct_is"],
                         "t_is": best["t_is"], "breakeven_bps_is": best["breakeven_bps_is"],
                         "n_oos": best["n_oos"], "mean_pct_oos": best["mean_pct_oos"],
                         "t_oos": best["t_oos"], "p_oos": best["p_oos"],
                         "breakeven_bps_oos": best["breakeven_bps_oos"],
                         "win_rate_oos": best["win_rate_oos"]})
    oos = pd.DataFrame(oos_rows)
    A.save(oos, "s02_orb_oos.csv")
    print("\n" + "=" * 120)
    print(f"OUT OF SAMPLE: best IS spec (2021-01..{IS_END}) evaluated on {IS_END}..{END}")
    print("=" * 120)
    print(oos.round(4).to_string(index=False))
    print("\nIS->OOS correlation of spec mean returns: "
          f"{m['mean_pct_is'].corr(m['mean_pct_oos']):.3f} over {len(m)} specs; "
          f"share of IS-positive specs still positive OOS: "
          f"{(m[m['mean_pct_is'] > 0]['mean_pct_oos'] > 0).mean():.3f}")
    A.save(m, "s02_orb_is_vs_oos.csv")

    # ---- random-entry null ----------------------------------------------
    print("\n" + "=" * 120)
    print(f"RANDOM-ENTRY NULL: {N_DRAWS} draws, entry bar uniform in the session, "
          "same direction / stop distance / target")
    print("=" * 120)
    null_rows = []
    probes = [(sym, 15, "touch", "or", "eod", dm) for sym in SYMBOLS for dm in DIRECTIONS]
    probes += [(sym, 5, "touch", "or", "eod", "long_short") for sym in SYMBOLS]
    probes += [(sym, 30, "touch", "or", "2R", "long_short") for sym in SYMBOLS]
    for sym, n_min, er, sk, tn, dm in probes:
        real = g_full[(g_full["symbol"] == sym) & (g_full["n_min"] == n_min) &
                      (g_full["entry_rule"] == er) & (g_full["stop"] == sk) &
                      (g_full["target"] == tn) & (g_full["direction_mode"] == dm)]
        if not len(real):
            continue
        rm = float(real["mean_pct"].iloc[0])
        means, nb = random_entry_null(sym, sess_cache[sym], n_min, er, sk, tn, dm)
        if not len(means):
            continue
        pct = float((means < rm).mean())
        null_rows.append({"symbol": sym, "spec": f"OR{n_min}|{er}|{sk}|{tn}|{dm}",
                          "n_trades": nb, "real_mean_pct": rm,
                          "null_mean_pct": float(means.mean()),
                          "null_sd_pct": float(means.std(ddof=1)),
                          "null_p05": float(np.percentile(means, 5)),
                          "null_p95": float(np.percentile(means, 95)),
                          "real_percentile_in_null": pct,
                          "z_vs_null": (rm - means.mean()) / means.std(ddof=1)
                          if means.std(ddof=1) else np.nan})
        print(f"  {null_rows[-1]['symbol']:5s} {null_rows[-1]['spec']:32s} "
              f"real={rm:+.4f}%  null={means.mean():+.4f}% "
              f"[{np.percentile(means,5):+.4f},{np.percentile(means,95):+.4f}]  "
              f"pctile={pct:.3f}", flush=True)
    A.save(pd.DataFrame(null_rows), "s02_orb_null.csv")

    # ---- self tests ------------------------------------------------------
    st = pd.DataFrame(all_self)
    st["ratio"] = st["variant_mean_pct"] / st["real_mean_pct"]
    A.save(st, "s02_orb_selftest.csv")
    print("\n" + "=" * 120)
    print("NEGATIVE CONTROL AND LOOKAHEAD SELF-TEST")
    print("=" * 120)
    print(st.round(5).to_string(index=False))
    print(f"\ntotal runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
