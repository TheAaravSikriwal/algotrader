"""Market-structure primitives for sources 09, 10 and 11.

Every function here returns, alongside the feature, the bar index at which the
feature was KNOWABLE. That is the whole point: a swing high defined by "k bars
lower on either side" is not knowable until k bars after the pivot, and a model
that marks the pivot and then trades from the pivot bar is reading the future.
Sources 09 and 10 both describe their structure visually, on a finished chart,
which is exactly the setting in which this error is invisible.

Conventions
-----------
* bars are a DataFrame with open/high/low/close/volume, one symbol, ascending.
* "known_at" is an integer positional index into that frame.
* a level/zone becomes tradeable on the bar AFTER known_at, never on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# swings
# ---------------------------------------------------------------------------
@dataclass
class Swing:
    idx: int          # positional index of the pivot bar
    price: float
    kind: str         # "high" | "low"
    known_at: int     # positional index of the bar that confirms it


def swings(df: pd.DataFrame, k: int = 3) -> list[Swing]:
    """Fractal pivots: a high with k strictly lower highs either side.

    Confirmed -- and therefore usable -- only at bar idx + k.
    """
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    n = len(df)
    out: list[Swing] = []
    for i in range(k, n - k):
        wh = highs[i - k:i + k + 1]
        wl = lows[i - k:i + k + 1]
        if highs[i] == wh.max() and (wh[:k] < highs[i]).all() and (wh[k + 1:] < highs[i]).all():
            out.append(Swing(i, float(highs[i]), "high", i + k))
        if lows[i] == wl.min() and (wl[:k] > lows[i]).all() and (wl[k + 1:] > lows[i]).all():
            out.append(Swing(i, float(lows[i]), "low", i + k))
    out.sort(key=lambda s: (s.known_at, s.idx))
    return out


# ---------------------------------------------------------------------------
# break of structure / change of character  (sources 10 and 11)
# ---------------------------------------------------------------------------
@dataclass
class StructureEvent:
    idx: int          # bar whose CLOSE confirmed the break
    kind: str         # "BOS_up" | "BOS_down" | "CHOCH_up" | "CHOCH_down"
    level: float      # the swing level that was broken
    ref_swing: int    # positional index of the broken swing


def structure_events(df: pd.DataFrame, k: int = 3) -> list[StructureEvent]:
    """Close-confirmed breaks of the most recent confirmed swing high/low.

    Source 10 is explicit that the break must CLOSE beyond the level:
    "a candle pushes past that prior swing low and closes beyond it".
    Source 11 gives the same definitions without the close requirement; the
    close version is the stricter and the one implemented.

    BOS   = break in the direction of the prevailing trend.
    CHoCH = the first break against it.
    Trend state is carried forward from the last event; it starts "flat", and
    the first break in either direction is labelled BOS.
    """
    sw = swings(df, k)
    closes = df["close"].to_numpy(float)
    n = len(df)

    # For each bar, the most recent CONFIRMED swing high / low as of that bar.
    # Identity is the swing's own index, not its price: two distinct swings can
    # sit at the same price, and de-duplicating by price would silently drop the
    # second break. That bug suppresses exactly the repeated tests of a level
    # that source 10 calls "contact points".
    last_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)
    last_high_id = np.full(n, -1, dtype=int)
    last_low_id = np.full(n, -1, dtype=int)
    hi = lo = np.nan
    hi_id = lo_id = -1
    ptr = 0
    for i in range(n):
        while ptr < len(sw) and sw[ptr].known_at <= i:
            if sw[ptr].kind == "high":
                hi, hi_id = sw[ptr].price, sw[ptr].idx
            else:
                lo, lo_id = sw[ptr].price, sw[ptr].idx
            ptr += 1
        last_high[i], last_high_id[i] = hi, hi_id
        last_low[i], last_low_id[i] = lo, lo_id

    events: list[StructureEvent] = []
    trend = 0                      # +1 up, -1 down, 0 undecided
    broken_high_id = broken_low_id = -1
    for i in range(n):
        h, l = last_high[i], last_low[i]
        if not np.isnan(h) and closes[i] > h and last_high_id[i] != broken_high_id:
            kind = "BOS_up" if trend >= 0 else "CHOCH_up"
            events.append(StructureEvent(i, kind, float(h), int(last_high_id[i])))
            broken_high_id = int(last_high_id[i])
            trend = 1
        elif not np.isnan(l) and closes[i] < l and last_low_id[i] != broken_low_id:
            kind = "BOS_down" if trend <= 0 else "CHOCH_down"
            events.append(StructureEvent(i, kind, float(l), int(last_low_id[i])))
            broken_low_id = int(last_low_id[i])
            trend = -1
    return events


# ---------------------------------------------------------------------------
# fair value gaps  (sources 10 and 11)
# ---------------------------------------------------------------------------
@dataclass
class FVG:
    idx: int          # positional index of the THIRD candle -- known at its close
    direction: int    # +1 bullish (gap above), -1 bearish (gap below)
    lo: float
    hi: float
    body_atr: float   # size of the middle candle in ATR, a "significance" proxy

    @property
    def mid(self) -> float:
        """Source 10's 'consequent encroachment' -- the 50% level."""
        return (self.lo + self.hi) / 2.0

    @property
    def width(self) -> float:
        return self.hi - self.lo


def fair_value_gaps(df: pd.DataFrame, atr: pd.Series | None = None,
                    min_width_atr: float = 0.0) -> list[FVG]:
    """Three-candle imbalance, exactly as source 10 defines it.

    "A sequence of three candles -- 1, 2, 3 -- where the first candle's high
    wick doesn't overlap with the low of the third candle's wick."

    So bullish: high[i-2] < low[i]; bearish: low[i-2] > high[i]. Knowable at
    the close of candle 3 and not one bar earlier.
    """
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    a = (atr.to_numpy(float) if atr is not None else np.full(len(df), np.nan))
    out: list[FVG] = []
    for i in range(2, len(df)):
        rng = a[i] if not np.isnan(a[i]) and a[i] > 0 else np.nan
        if highs[i - 2] < lows[i]:
            w = lows[i] - highs[i - 2]
            if np.isnan(rng) or w >= min_width_atr * rng:
                out.append(FVG(i, +1, float(highs[i - 2]), float(lows[i]),
                               float(w / rng) if rng == rng else np.nan))
        elif lows[i - 2] > highs[i]:
            w = lows[i - 2] - highs[i]
            if np.isnan(rng) or w >= min_width_atr * rng:
                out.append(FVG(i, -1, float(highs[i]), float(lows[i - 2]),
                               float(w / rng) if rng == rng else np.nan))
    return out


# ---------------------------------------------------------------------------
# supply / demand zones  (source 09, and source 11's "order blocks")
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    idx: int          # the base candle
    direction: int    # +1 demand (expect bounce up), -1 supply (expect drop)
    lo: float
    hi: float
    known_at: int     # end of the impulsive move that defines it
    impulse_atr: float


def zones(df: pd.DataFrame, atr: pd.Series, impulse_bars: int = 3,
          min_impulse_atr: float = 2.0, base_rule: str = "last_opposite",
          lookback: int = 12) -> list[Zone]:
    """Source 09's rule, made mechanical.

    Verbatim: "find a sharp, aggressive move up. The area immediately before
    that move is the demand level... I take the last candle before the
    aggressive move formed and draw a rectangle around it."

    Two knobs, both forced on us by the source and both DECLARED as
    specifications rather than quietly chosen:

    * "sharp, aggressive" is the source's only sizing language. An impulse is a
      move over `impulse_bars` bars of at least `min_impulse_atr` x ATR.
    * "the last candle before the aggressive move" is ambiguous. `base_rule`
      "last" takes it literally -- the immediately preceding candle. The
      conventional reading, "last_opposite", takes the last candle of the
      opposite colour, which is what every chart illustration of the idea
      actually shows. Both are tested.

    If `last_opposite` finds no opposite-colour candle inside `lookback`, the
    zone is SKIPPED rather than silently falling back to the preceding candle.
    The fallback version manufactured zones out of arbitrary bars, which is
    both unfaithful to the source and a free source of extra signals.
    """
    if base_rule not in ("last", "last_opposite"):
        raise ValueError(f"base_rule must be 'last' or 'last_opposite', got {base_rule!r}")
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(df)
    out: list[Zone] = []
    last_start = -1

    for i in range(impulse_bars, n):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        start = i - impulse_bars + 1
        if start <= last_start:
            continue
        move = c[i] - o[start]
        if abs(move) < min_impulse_atr * a[i]:
            continue
        direction = 1 if move > 0 else -1

        if base_rule == "last":
            base = start - 1
        else:
            base = None
            for j in range(start - 1, max(start - 1 - lookback, -1), -1):
                if (direction > 0 and c[j] < o[j]) or (direction < 0 and c[j] > o[j]):
                    base = j
                    break
        if base is None or base < 0:
            continue
        out.append(Zone(base, direction, float(l[base]), float(h[base]), i,
                        float(abs(move) / a[i])))
        last_start = i
    return out


# ---------------------------------------------------------------------------
# stochastic oscillator  (source 09's confirmation, source 11's oscillator)
# ---------------------------------------------------------------------------
def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3):
    """Returns (%K, %D). Source 09 uses "custom settings" shown only on screen
    and never stated, so the parameters must be swept and the sweep declared."""
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    fast = raw.rolling(smooth, min_periods=smooth).mean()
    slow = fast.rolling(d, min_periods=d).mean()
    return fast, slow


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Anchored at each session open, cumulative, causal."""
    day = pd.DatetimeIndex(df.index).normalize()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return (pv / v).rename("vwap")


# ---------------------------------------------------------------------------
# trade simulation: one R-based bracket order, resolved bar by bar
# ---------------------------------------------------------------------------
def simulate_bracket(df: pd.DataFrame, entry_i: int, entry_px: float, stop_px: float,
                     direction: int, r_target: float, max_bars: int = 78,
                     eod_exit: bool = True, breakeven_at_r: float | None = None,
                     resolve_from: int | None = None):
    """Resolve a bracket. Returns a dict or None.

    Pessimistic within a bar, matching core/engine.py: if a bar's range covers
    both the stop and the target, the STOP is assumed to hit first, because one
    bar tells you nothing about the path inside it. A gap through the stop
    fills at the open, not at the stop -- otherwise a backtest reports a capped
    loss on an uncapped move, which flatters exactly the scenarios stops exist
    for.

    `resolve_from` controls the first bar that can close the trade, and the
    default is deliberately NOT always entry_i+1:

      * entry at the OPEN of bar t  -> resolve_from = t. Bar t can stop you out
        and usually should; you held it for the whole bar.
      * entry from a RESTING LIMIT filled intrabar at bar t -> resolve_from = t
        as well. Skipping bar t means a limit that is filled and then blown
        through in the same candle records a fill and no loss. For a setup
        whose stop sits close to its entry -- which is the whole pitch of these
        models -- that single off-by-one manufactures a large fake edge.

    Passing resolve_from = entry_i + 1 is only correct when the entry price is
    the close of bar entry_i and the position is genuinely not held during it.
    """
    risk = (entry_px - stop_px) * direction
    if risk <= 0:
        return None
    target_px = entry_px + direction * r_target * risk

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)
    day = idx[entry_i].normalize()

    cur_stop = stop_px
    moved_be = False
    first = entry_i + 1 if resolve_from is None else max(resolve_from, 0)

    for j in range(first, min(first + max_bars, n)):
        if idx[j].normalize() != day:
            # never carry overnight: both sources are intraday
            px = c[j - 1]
            return _res(entry_i, j - 1, entry_px, px, direction, risk, "eod_rollover", idx)

        if breakeven_at_r is not None and not moved_be:
            reach = (h[j] - entry_px) * direction if direction > 0 else (entry_px - l[j])
            if reach >= breakeven_at_r * risk:
                # only AFTER this bar completes -- applied from the next bar on
                moved_be = True

        hit_stop = (l[j] <= cur_stop) if direction > 0 else (h[j] >= cur_stop)
        hit_tp = (h[j] >= target_px) if direction > 0 else (l[j] <= target_px)

        if hit_stop:
            fill = min(o[j], cur_stop) if direction > 0 else max(o[j], cur_stop)
            return _res(entry_i, j, entry_px, fill, direction, risk,
                        "breakeven" if moved_be and abs(cur_stop - entry_px) < 1e-9 else "stop", idx)
        if hit_tp:
            fill = max(o[j], target_px) if direction > 0 else min(o[j], target_px)
            return _res(entry_i, j, entry_px, fill, direction, risk, "target", idx)

        if moved_be and abs(cur_stop - entry_px) > 1e-9:
            cur_stop = entry_px          # takes effect on the NEXT bar, not this one

        if eod_exit and idx[j].time() >= pd.Timestamp("15:55").time():
            return _res(entry_i, j, entry_px, c[j], direction, risk, "eod", idx)

    j = min(entry_i + max_bars, n - 1)
    return _res(entry_i, j, entry_px, c[j], direction, risk, "timeout", idx)


def _res(ei, xi, epx, xpx, direction, risk, reason, idx):
    pnl = (xpx - epx) * direction
    return {
        "entry_time": idx[ei], "exit_time": idx[xi],
        "entry_i": ei, "exit_i": xi, "bars_held": xi - ei,
        "direction": direction, "entry_px": epx, "exit_px": xpx,
        "risk_px": risk, "R": pnl / risk if risk else np.nan,
        "ret_pct": pnl / epx * 100.0,          # % of NOTIONAL -- bps map directly
        "risk_pct_of_px": risk / epx * 100.0,
        "exit_reason": reason,
    }
