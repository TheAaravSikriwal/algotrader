"""Source 05 audit -- the Gap-and-Go "first pullback to VWAP/9MA" rule engine.

NO-LOOKAHEAD LEDGER (every input, and the timestamp at which it existed)
------------------------------------------------------------------------
  gap%                 : today 09:30 open / prior session close. Known 09:30:00.
                         Earliest entry produced below is 09:40. OK.
  prev_close, prev $vol: prior session. Known before the open. OK.
  5Min VWAP[i]         : cumulative over bars 0..i inclusive. Bar i's own
                         typical price/volume are final only at bar i's CLOSE,
                         so VWAP[i] is used only to judge bars strictly after i
                         -- except for the pullback touch test, which asks
                         whether bar i's LOW reached VWAP[i]; that is a
                         statement about bar i evaluated at bar i's close, and
                         the resulting trade is placed for bar i+1 onwards. OK.
  MA9[i]               : same -- final at bar i's close, acted on from i+1.
  entry trigger        : a resting stop-buy at high[m-1] + 1 tick, placed at
                         bar m-1's CLOSE. Uses nothing from bar m except
                         whether it traded through the price. OK.
  stop / target        : derived from entry and from bars at or before entry.
  exit simulation      : 1Min bars strictly at/after the fill bar, walked
                         forward one bar at a time, stop assumed to fill before
                         target when a single minute touches both.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

MIN_DIR = A.CACHE / "s05_min"
TICK = 0.01
PB_BAND = 0.001          # "within a small band": 10 bps above max(VWAP, MA9)
ENTRY_CUTOFF = dt.time(11, 0)   # pullback entry must trigger by 11:00 ET
EOD = dt.time(15, 55)


# ---------------------------------------------------------------- data
def load_minute(date: str) -> dict[str, pd.DataFrame]:
    path = MIN_DIR / f"{date}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if len(df) == 0:
        return {}
    out = {}
    for sym, sub in df.groupby("symbol"):
        sub = sub.set_index("timestamp").sort_index()
        sub = A.to_eastern(sub[["open", "high", "low", "close", "volume"]])
        sub = sub[~sub.index.duplicated(keep="last")]
        out[str(sym)] = sub
    return out


def to5(m1: pd.DataFrame) -> pd.DataFrame:
    """1Min -> 5Min, right-open bins labelled by their left edge (09:30, 09:35...)."""
    agg = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg.dropna(subset=["open", "high", "low", "close"])


def indicators(m5: pd.DataFrame, vwap_premarket: bool, ma_kind: str) -> pd.DataFrame:
    """m5 must be the FULL 04:00-16:00 series so the 9MA is warm at 09:30."""
    d = m5.copy()
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    if vwap_premarket:
        src = d
    else:
        src = d[d.index.time >= dt.time(9, 30)]
    pv = (tp.loc[src.index] * src["volume"]).cumsum()
    vv = src["volume"].cumsum().replace(0, np.nan)
    d["vwap"] = (pv / vv).reindex(d.index)
    if ma_kind == "ema":
        d["ma9"] = d["close"].ewm(span=9, adjust=False).mean()
    else:
        d["ma9"] = d["close"].rolling(9).mean()
    return d


# ------------------------------------------------------- the entry rule
def find_entry(d5: pd.DataFrame, invert: bool = False) -> dict | None:
    """First pullback to max(VWAP, MA9) after an initial push, then first bar
    whose high exceeds the prior bar's high (or, if invert, whose low breaks
    the prior bar's low -- the deliberately-broken control)."""
    r = d5[(d5.index.time >= dt.time(9, 30)) & (d5.index.time < dt.time(16, 0))]
    if len(r) < 6:
        return None
    h = r["high"].to_numpy(); lo = r["low"].to_numpy(); c = r["close"].to_numpy()
    vw = r["vwap"].to_numpy(); ma = r["ma9"].to_numpy()
    ts = r.index

    ref = np.maximum(vw, ma)
    # 1. initial push: first bar after 09:30 that makes a new session high
    push = None
    run = h[0]
    for i in range(1, len(r)):
        if h[i] > run:
            push = i
            break
        run = max(run, h[i])
    if push is None:
        return None
    if not np.isfinite(ref[push]) or c[push] <= ref[push]:
        return None   # must be trading ABOVE the VWAP/9MA band to pull back to it

    # 2. first pullback: first bar after the push whose LOW touches the band
    pb = None
    for j in range(push + 1, len(r)):
        if ts[j].time() > ENTRY_CUTOFF:
            return None
        if not np.isfinite(ref[j]):
            continue
        if lo[j] <= ref[j] * (1.0 + PB_BAND):
            pb = j
            break
    if pb is None or pb >= len(r) - 2:
        return None

    # 3. trigger: first later bar taking out the prior bar's high (or low)
    for m in range(pb + 1, len(r)):
        if ts[m].time() > ENTRY_CUTOFF:
            return None
        if not invert and h[m] > h[m - 1]:
            trig = h[m - 1] + TICK
            break
        if invert and lo[m] < lo[m - 1]:
            trig = lo[m - 1] - TICK
            break
    else:
        return None

    pull_low = float(lo[pb:m].min())
    return {"push_i": push, "pb_i": pb, "trig_i": m,
            "pb_ts": ts[pb], "trig_ts": ts[m], "prev_ts": ts[m - 1],
            "trigger_px": float(trig), "pullback_low": pull_low,
            "vwap_at_pb": float(vw[pb]), "ma9_at_pb": float(ma[pb])}


def atr5(d5: pd.DataFrame, upto: pd.Timestamp, n: int = 14) -> float:
    d = d5[d5.index <= upto]
    if len(d) < n + 1:
        return float("nan")
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


# ------------------------------------------------------------- exits
class Path:
    """Numpy view of one session's RTH 1Min bars, built once per event."""

    __slots__ = ("ts", "o", "h", "l", "c", "eod_pos", "n")

    def __init__(self, m1: pd.DataFrame):
        rth = m1[(m1.index.time >= dt.time(9, 30)) & (m1.index.time < dt.time(16, 0))]
        self.ts = rth.index
        self.o = rth["open"].to_numpy(float)
        self.h = rth["high"].to_numpy(float)
        self.l = rth["low"].to_numpy(float)
        self.c = rth["close"].to_numpy(float)
        self.n = len(rth)
        late = np.array([t.time() >= EOD for t in rth.index])
        self.eod_pos = int(np.argmax(late)) if late.any() else self.n - 1


def entry_fill(p: Path, sig: dict, fill_mode: str):
    """-> (entry_px, first_bar_position_for_exit_scan) or None."""
    if p.n == 0:
        return None
    t0 = sig["trig_ts"]
    t1 = t0 + pd.Timedelta(minutes=5)
    if fill_mode == "trigger":
        win = (p.ts >= t0) & (p.ts < t1) & (p.h >= sig["trigger_px"])
        if not win.any():
            return None
        i = int(np.argmax(win))
        entry = max(float(sig["trigger_px"]), float(p.o[i]))
        return entry, i + 1
    nxt = p.ts >= t1                      # repo core/engine.py convention
    if not nxt.any():
        return None
    i = int(np.argmax(nxt))
    return float(p.o[i]), i


def simulate(p: Path, sig: dict, entry: float, start: int, stop_px: float,
             target_mode: str) -> dict | None:
    if start >= p.n or not np.isfinite(stop_px) or stop_px >= entry:
        return None
    R = entry - stop_px
    o, h, l, c = p.o[start:], p.h[start:], p.l[start:], p.c[start:]
    m = len(o)
    if m == 0:
        return None
    eod = max(0, min(m - 1, p.eod_pos - start))

    if target_mode == "trail":
        prev_max = np.maximum.accumulate(np.concatenate(([entry], h[:-1])))
        stop_arr = np.maximum(stop_px, prev_max - R)
    else:
        stop_arr = np.full(m, stop_px)

    gap_stop = o <= stop_arr
    hit_stop = l <= stop_arr
    i_stop = int(np.argmax(hit_stop)) if hit_stop.any() else m

    if target_mode in ("1R", "2R", "3R"):
        tgt = entry + float(target_mode[0]) * R
        hit_t = h >= tgt
        i_tgt = int(np.argmax(hit_t)) if hit_t.any() else m
    else:
        tgt, i_tgt = None, m

    i = min(i_stop, i_tgt, eod)
    if i == i_stop and i_stop <= i_tgt and i_stop <= eod:
        px, why = (o[i], "stop_gap") if gap_stop[i] else (stop_arr[i], "stop")
    elif i == i_tgt and i_tgt <= eod:
        px, why = (o[i], "target_gap") if o[i] >= tgt else (tgt, "target")
    else:
        px, why = c[eod], "eod"
        i = eod
    return _res(entry, float(px), R, why, p.ts[start + i], sig)


def _res(entry, exit_px, R, why, t, sig):
    return {"entry_px": entry, "exit_px": float(exit_px), "R_dollars": R,
            "exit_reason": why, "exit_ts": t,
            "pnl_pct": (exit_px / entry - 1.0) * 100.0,
            "R_mult": (exit_px - entry) / R if R > 0 else np.nan,
            "trig_ts": sig["trig_ts"], "pb_ts": sig["pb_ts"]}
