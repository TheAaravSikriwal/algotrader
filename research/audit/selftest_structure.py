"""Self-tests for structure.py, on synthetic bars where the answer is known.

Deliberately NOT named test_*.py: the repo's root pytest run must keep its
existing 279 tests and nothing else. Run this directly.

The point of each test is to catch a specific way the primitive could be wrong
in a direction that would FLATTER a backtest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import structure as S  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def frame(rows, start="2024-01-02 09:30", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


# ---------------------------------------------------------------------------
def t_swings():
    print("\nswings()")
    # a clean single peak at position 5
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    rows = [[h - 0.5, h, h - 1, h - 0.5, 100] for h in highs]
    df = frame(rows)
    sw = S.swings(df, k=3)
    hs = [s for s in sw if s.kind == "high"]
    check("finds the peak", any(s.idx == 5 for s in hs), f"got {[s.idx for s in hs]}")
    peak = [s for s in hs if s.idx == 5][0]
    check("peak price correct", peak.price == 20)
    check("confirmed k bars later, not at the pivot", peak.known_at == 8,
          f"known_at={peak.known_at}")

    # THE important one: no swing may be knowable before its confirmation bar
    check("known_at is always >= idx + k", all(s.known_at >= s.idx + 3 for s in sw))

    # a plateau (equal highs) must not register -- it is not a strict fractal
    rows2 = [[h - 0.5, h, h - 1, h - 0.5, 100] for h in [10, 11, 12, 20, 20, 12, 11, 10, 9]]
    sw2 = [s for s in S.swings(frame(rows2), k=2) if s.kind == "high"]
    check("equal-high plateau is not a pivot", len(sw2) == 0, f"got {[s.idx for s in sw2]}")


def t_fvg():
    print("\nfair_value_gaps()")
    # bullish: candle 1 high = 10, candle 3 low = 12 -> gap 10..12, mid 11
    rows = [
        [9, 10, 8, 9.5, 100],      # 0
        [9.5, 13, 9.4, 12.8, 500],  # 1 -- the impulsive middle candle
        [12.9, 14, 12, 13.5, 300],  # 2 -- low 12 > high 10 of candle 0
    ]
    g = S.fair_value_gaps(frame(rows))
    check("detects the bullish gap", len(g) == 1 and g[0].direction == 1, f"got {g}")
    check("gap bounds are wick-to-wick", g[0].lo == 10 and g[0].hi == 12,
          f"lo={g[0].lo} hi={g[0].hi}")
    check("midpoint is the 50% consequent encroachment", g[0].mid == 11.0)
    check("known at candle 3, not candle 1", g[0].idx == 2)

    # overlap by a cent kills it -- the definition is strict non-overlap
    rows2 = [r[:] for r in rows]
    rows2[2][2] = 9.99                      # third candle low now below first high
    check("overlapping wicks produce no gap", len(S.fair_value_gaps(frame(rows2))) == 0)

    # bearish mirror
    rows3 = [
        [13, 14, 12, 12.5, 100],
        [12.4, 12.5, 9, 9.2, 500],
        [9.1, 11.5, 9, 10, 300],            # high 11.5 < low 12 of candle 0
    ]
    g3 = S.fair_value_gaps(frame(rows3))
    check("detects the bearish gap", len(g3) == 1 and g3[0].direction == -1)
    check("bearish bounds", g3[0].lo == 11.5 and g3[0].hi == 12.0,
          f"lo={g3[0].lo} hi={g3[0].hi}")


def t_structure_events():
    print("\nstructure_events()")
    # A clean downtrend -- lower highs and lower lows -- then a reversal up.
    # Every value distinct: equal highs are not strict fractals and would
    # silently fail to register as pivots.
    seq = [100, 97.5, 95, 93, 96.5, 94.5, 91.5, 89, 87, 89.5, 88.5, 85.5,
           83.5, 81.5, 84.5, 87.5, 91.5, 94.5, 97.5, 99.5]
    rows = [[p, p + 0.6, p - 0.6, p, 100] for p in seq]
    df = frame(rows)
    ev = S.structure_events(df, k=2)
    kinds = [e.kind for e in ev]
    check("produces events", len(ev) > 0, f"got {kinds}")
    check("downtrend produces BOS_down", any(k == "BOS_down" for k in kinds), f"{kinds}")
    check("reversal produces a CHOCH_up after the downtrend",
          any(k == "CHOCH_up" for k in kinds), f"{kinds}")
    downs = [i for i, k in enumerate(kinds) if k == "BOS_down"]
    ups = [i for i, k in enumerate(kinds) if k == "CHOCH_up"]
    if downs and ups:
        check("CHoCH comes after the BOS sequence", max(ups) > min(downs))

    # confirmation must be by CLOSE, not by wick
    rows2 = [[p, p + 0.6, p - 0.6, p, 100] for p in [100, 98, 96, 94, 92, 94, 93]]
    rows2.append([93, 101, 92.5, 94, 100])          # wick pokes above, close does not
    ev2 = S.structure_events(frame(rows2), k=2)
    wick_bos = [e for e in ev2 if e.idx == len(rows2) - 1 and e.kind.endswith("_up")]
    check("a wick above a swing high is not a break", len(wick_bos) == 0, f"{ev2}")


def t_bracket():
    print("\nsimulate_bracket()")
    # long from 100, stop 99 (risk 1), target 2R = 102. Price rises to 102.
    rows = [[100, 100.2, 99.8, 100, 100]]
    rows += [[100.5, 101, 100.2, 100.8, 100], [100.8, 102.5, 100.7, 102.2, 100]]
    df = frame(rows)
    r = S.simulate_bracket(df, 0, 100.0, 99.0, +1, 2.0)
    check("target hit gives +2R", r and abs(r["R"] - 2.0) < 1e-9, f"{r}")
    check("exit_reason is target", r["exit_reason"] == "target")

    # stop hit gives exactly -1R
    rows2 = [[100, 100.2, 99.8, 100, 100], [99.9, 100, 98.5, 98.8, 100]]
    r2 = S.simulate_bracket(frame(rows2), 0, 100.0, 99.0, +1, 2.0)
    check("stop hit gives -1R", r2 and abs(r2["R"] + 1.0) < 1e-9, f"{r2}")

    # a GAP through the stop must fill at the open, i.e. worse than -1R.
    # Filling at the stop level here is the classic backtest flatterer.
    rows3 = [[100, 100.2, 99.8, 100, 100], [95, 95.5, 94, 94.5, 100]]
    r3 = S.simulate_bracket(frame(rows3), 0, 100.0, 99.0, +1, 2.0)
    check("gap through the stop is worse than -1R", r3 and r3["R"] < -1.0,
          f"R={r3['R'] if r3 else None}")
    check("gap fill is the open, not the stop", r3 and abs(r3["exit_px"] - 95.0) < 1e-9)

    # when one bar covers stop AND target, the stop must win (pessimistic)
    rows4 = [[100, 100.2, 99.8, 100, 100], [100, 103, 98, 101, 100]]
    r4 = S.simulate_bracket(frame(rows4), 0, 100.0, 99.0, +1, 2.0)
    check("ambiguous bar resolves to the stop", r4 and r4["exit_reason"] == "stop", f"{r4}")

    # short mirror
    rows5 = [[100, 100.2, 99.8, 100, 100], [99.5, 99.6, 97.5, 97.8, 100]]
    r5 = S.simulate_bracket(frame(rows5), 0, 100.0, 101.0, -1, 2.0)
    check("short target gives +2R", r5 and abs(r5["R"] - 2.0) < 1e-9, f"{r5}")

    # never carries overnight
    idx = list(pd.date_range("2024-01-02 15:45", periods=3, freq="5min"))
    idx += list(pd.date_range("2024-01-03 09:30", periods=3, freq="5min"))
    df6 = pd.DataFrame([[100, 100.2, 99.9, 100, 1]] * 6,
                       columns=["open", "high", "low", "close", "volume"], index=idx)
    r6 = S.simulate_bracket(df6, 0, 100.0, 99.0, +1, 5.0)
    check("flat by end of session", r6 and r6["exit_time"].normalize() ==
          pd.Timestamp("2024-01-02"), f"{r6['exit_time'] if r6 else None}")

    # ret_pct must be in % of notional so bps costs map directly
    check("ret_pct is % of notional", abs(r["ret_pct"] - (102 - 100) / 100 * 100) < 1e-9)

    # THE OFF-BY-ONE THAT MANUFACTURES AN EDGE.
    # A resting limit at 100 is filled intrabar on bar 1, and that same bar
    # crashes through the stop at 99. Resolving from bar 2 books a fill and no
    # loss; resolving from bar 1 books the -1R that actually happened.
    # Bar 1 fills the limit at 100, wicks down through the stop at 99, then
    # closes back up. Bar 2 rallies to the 4R target. Skipping bar 1 turns a
    # real -1R into a +4R winner -- a 5R swing on a single off-by-one.
    rows7 = [[101, 101.2, 100.8, 101, 100],
             [100.9, 101.0, 97.0, 100.5, 100],    # fills the limit AND wicks the stop
             [100.5, 105.0, 100.4, 104.8, 100]]
    df7 = frame(rows7)
    bad = S.simulate_bracket(df7, 1, 100.0, 99.0, +1, 4.0, resolve_from=2)
    good = S.simulate_bracket(df7, 1, 100.0, 99.0, +1, 4.0, resolve_from=1)
    check("resolving from the bar AFTER the fill hides the loss and books a win",
          bad is not None and bad["R"] > 0, f"R={bad['R'] if bad else None}")
    check("resolving from the fill bar books the -1R that happened",
          good is not None and abs(good["R"] + 1.0) < 1e-9, f"R={good['R'] if good else None}")
    check("the off-by-one is worth ~5R on this trade",
          abs(good["R"] - bad["R"]) > 4.0, f"{good['R']} vs {bad['R']}")

    # entry at the open of bar t must let bar t resolve the trade
    rows8 = [[100, 100.2, 99.9, 100, 100], [100, 100.5, 98.0, 98.2, 100]]
    r8 = S.simulate_bracket(frame(rows8), 1, 100.0, 99.0, +1, 2.0, resolve_from=1)
    check("open-fill: the entry bar can stop you out",
          r8 is not None and r8["R"] <= -1.0, f"{r8}")


def t_vwap():
    print("\nsession_vwap()")
    idx = list(pd.date_range("2024-01-02 09:30", periods=3, freq="5min"))
    idx += list(pd.date_range("2024-01-03 09:30", periods=3, freq="5min"))
    df = pd.DataFrame({"open": [10] * 6, "high": [10] * 6, "low": [10] * 6,
                       "close": [10, 20, 30, 100, 100, 100],
                       "volume": [1, 1, 1, 1, 1, 1]}, index=idx)
    v = S.session_vwap(df)
    # VWAP is built from the TYPICAL price (h+l+c)/3, not the close. With
    # h=l=10 the typical prices are 10, 13.333, 16.667, so the running mean at
    # bar 2 is 13.333 -- NOT the mean of the closes. Getting this expectation
    # wrong is how a "VWAP" that is really a close-average slips into a model.
    check("first bar vwap equals its own typical price", abs(v.iloc[0] - 10) < 1e-9)
    check("accumulates typical price within the session",
          abs(v.iloc[2] - 40.0 / 3.0) < 1e-9, f"{v.iloc[2]}")
    check("resets at the session boundary", abs(v.iloc[3] - 40.0) < 1e-9, f"{v.iloc[3]}")
    check("reset is a real reset, not a blend",
          v.iloc[3] > v.iloc[2] * 2, f"{v.iloc[3]} vs {v.iloc[2]}")
    check("vwap uses no future bars",
          all(v.iloc[i] == S.session_vwap(df.iloc[:i + 1]).iloc[i] for i in range(3)))


def t_zones():
    print("\nzones()")
    # Quiet green drift, one red candle at index 5, then a sharp 3-bar rally.
    # The red candle is the demand base. Earlier bars are GREEN (not flat), so
    # no spurious earlier impulse can fire with an empty base search.
    rows = [[99.8 + 0.02 * i, 100.1 + 0.02 * i, 99.6 + 0.02 * i, 100.0 + 0.02 * i, 100]
            for i in range(5)]
    rows.append([100.0, 100.2, 99.0, 99.2, 100])       # red base candle, index 5
    rows.append([99.3, 102.0, 99.2, 101.8, 500])
    rows.append([101.8, 104.0, 101.7, 103.8, 500])
    rows.append([103.8, 106.0, 103.7, 105.8, 500])
    df = frame(rows)
    atr = pd.Series(0.6, index=df.index)

    z = S.zones(df, atr, impulse_bars=3, min_impulse_atr=2.0, base_rule="last_opposite")
    check("finds a demand zone", len(z) >= 1 and z[0].direction == 1, f"{z}")
    if z:
        check("base is the last red candle before the impulse", z[0].idx == 5,
              f"idx={z[0].idx}")
        check("zone is that candle's high/low", z[0].lo == 99.0 and z[0].hi == 100.2)
        check("not knowable before the impulse completes", z[0].known_at >= 8,
              f"known_at={z[0].known_at}")

    # The literal reading of the source takes the immediately preceding candle,
    # whatever its colour. On this fixture it therefore fires on an EARLIER,
    # weaker impulse than last_opposite does and marks a different bar. That
    # divergence is the reason both readings have to be carried through the
    # backtest as separate specifications rather than one being assumed.
    z2 = S.zones(df, atr, impulse_bars=3, min_impulse_atr=2.0, base_rule="last")
    check("base_rule='last' yields a zone", len(z2) >= 1, f"{z2}")
    check("the two base rules genuinely differ on this fixture",
          len(z2) >= 1 and z2[0].idx != z[0].idx,
          f"last={z2[0].idx if z2 else None} last_opposite={z[0].idx if z else None}")
    # the invariant: base sits exactly impulse_bars before the confirming bar
    check("base_rule='last' base == known_at - impulse_bars",
          all(x.idx == x.known_at - 3 for x in z2),
          f"{[(x.idx, x.known_at) for x in z2]}")

    # and on random data the literal rule can only ever produce MORE zones,
    # since last_opposite is the same scan with an extra filter
    rng = np.random.default_rng(3)
    p = 100 + np.cumsum(rng.normal(0, 0.4, 400))
    rr = []
    for x in p:
        o_ = x
        c_ = x + rng.normal(0, 0.3)
        rr.append([o_, max(o_, c_) + abs(rng.normal(0, .1)),
                   min(o_, c_) - abs(rng.normal(0, .1)), c_, 1000])
    dfr = frame(rr)
    atr_r = (dfr["high"] - dfr["low"]).rolling(14).mean()
    n_last = len(S.zones(dfr, atr_r, base_rule="last"))
    n_opp = len(S.zones(dfr, atr_r, base_rule="last_opposite"))
    check("literal rule is a superset in count", n_last >= n_opp, f"{n_last} vs {n_opp}")

    # THE BUG THE FIRST RUN CAUGHT: with no opposite-colour candle available the
    # old code silently fell back to an arbitrary bar and manufactured a zone.
    flat = [[100, 100.5, 99.5, 100, 100] for _ in range(5)]
    flat += [[100, 103, 99.9, 102.8, 500], [102.8, 105, 102.7, 104.8, 500],
             [104.8, 107, 104.7, 106.8, 500]]
    zf = S.zones(frame(flat), pd.Series(0.6, index=frame(flat).index),
                 impulse_bars=3, min_impulse_atr=2.0, base_rule="last_opposite")
    check("no opposite-colour base -> zone is skipped, not invented", len(zf) == 0,
          f"invented {[(x.idx, x.lo, x.hi) for x in zf]}")


def t_no_lookahead_global():
    print("\nglobal no-lookahead: truncation invariance")
    # Anything computed on bars[0:m] must be identical when computed on the full
    # series and then truncated. This is the single strongest structural check.
    rng = np.random.default_rng(7)
    p = 100 + np.cumsum(rng.normal(0, 0.3, 300))
    rows = [[x, x + abs(rng.normal(0, .2)), x - abs(rng.normal(0, .2)), x + rng.normal(0, .1), 1000]
            for x in p]
    rows = [[o, max(o, h, c), min(o, l, c), c, v] for o, h, l, c, v in rows]
    df = frame(rows)
    m = 200

    full_sw = [s for s in S.swings(df, 3) if s.known_at < m]
    part_sw = [s for s in S.swings(df.iloc[:m], 3) if s.known_at < m]
    check("swings known before bar m are unchanged by later bars",
          [(s.idx, s.kind, s.price) for s in full_sw] == [(s.idx, s.kind, s.price) for s in part_sw],
          f"{len(full_sw)} vs {len(part_sw)}")

    full_ev = [e for e in S.structure_events(df, 3) if e.idx < m - 3]
    part_ev = [e for e in S.structure_events(df.iloc[:m], 3) if e.idx < m - 3]
    check("structure events before bar m are unchanged by later bars",
          [(e.idx, e.kind) for e in full_ev] == [(e.idx, e.kind) for e in part_ev],
          f"{[(e.idx, e.kind) for e in full_ev][-3:]} vs {[(e.idx, e.kind) for e in part_ev][-3:]}")

    full_g = [g for g in S.fair_value_gaps(df) if g.idx < m]
    part_g = [g for g in S.fair_value_gaps(df.iloc[:m]) if g.idx < m]
    check("FVGs before bar m are unchanged by later bars",
          [(g.idx, g.lo, g.hi) for g in full_g] == [(g.idx, g.lo, g.hi) for g in part_g])


if __name__ == "__main__":
    t_swings()
    t_fvg()
    t_structure_events()
    t_bracket()
    t_vwap()
    t_zones()
    t_no_lookahead_global()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
