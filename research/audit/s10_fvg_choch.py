"""Source 10: the BOS / CHoCH / fair-value-gap reversal model, backtested.

The source's rules, verbatim, and how each is encoded:

  "we at least need two breaks of structure in a trend in order to confirm"
      -> require >= `min_bos` BOS events in the same direction, no opposing
         CHoCH in between.
  "a CHoCH is a failure to make a new low (in a downtrend) followed by a push
   that closes above the trend's contact level"
      -> structure_events() emits CHOCH_up / CHOCH_down, close-confirmed.
  "a sequence of three candles where the first candle's high wick doesn't
   overlap with the low of the third candle's wick"
      -> fair_value_gaps(), exact.
  "Entry: a limit at the 50% midpoint of the fair value gap"
      -> resting limit at FVG.mid, valid `ttl` bars, filled when the bar's
         range covers mid.
  "Stop-loss: below the prior swing point ... outside critical areas"
      -> the most recent CONFIRMED swing beyond the FVG, else the far edge of
         the FVG itself.
  "Initial take-profit: 4R"
      -> r_target = 4.0 (2R and 3R also swept, since the source calls 4R "a
         starting point" that "moves dynamically" -- that dynamism is not
         codeable and is declared unfalsifiable rather than invented).
  "wait for the first new break of structure in the new direction ... move the
   stop to entry"
      -> breakeven variants.

What is NOT codeable and is therefore not reported as the source's result:
the take-profit refinement (higher-timeframe FVG confluence plus a 61.8%
Fibonacci retracement, chosen by eye, with the source itself saying "there's
not one simple all-purpose solution... you're just reverse engineering your
success"), and the instruction not to size up on high-conviction setups.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402
import structure as S  # noqa: E402
from core.indicators import atr as atr_ind  # noqa: E402


@dataclass
class Spec:
    name: str = "full"
    k: int = 3                     # fractal half-width for swings
    min_bos: int = 2               # "at least two breaks of structure"
    require_choch: bool = True
    require_fvg_after_choch: bool = True
    r_target: float = 4.0
    ttl: int = 12                  # bars a resting limit stays live
    min_fvg_atr: float = 0.0       # "significant" gap filter
    breakeven_at_r: float | None = None
    max_choch_age: int = 20        # bars between CHoCH and FVG
    stop_pad_atr: float = 0.0


def _session_trades(sub: pd.DataFrame, spec: Spec, rng=None,
                    mode: str = "model") -> list[dict]:
    """Walk one session. `mode` selects the model or one of its controls."""
    if len(sub) < 25:
        return []
    a = atr_ind(sub, 14)
    sw = S.swings(sub, spec.k)
    events = S.structure_events(sub, spec.k)
    gaps = S.fair_value_gaps(sub, a, spec.min_fvg_atr)

    ev_by_bar: dict[int, list] = {}
    for e in events:
        ev_by_bar.setdefault(e.idx, []).append(e)
    gap_by_bar: dict[int, list] = {}
    for g in gaps:
        gap_by_bar.setdefault(g.idx, []).append(g)

    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)
    n = len(sub)

    bos_run = 0            # consecutive BOS in the current direction
    bos_dir = 0
    choch_dir = 0
    choch_bar = -10_000
    out: list[dict] = []
    taken = False          # source: "one good trade a day"; one setup per session

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
            if mode in ("model", "skipfill"):
                if spec.require_choch:
                    if choch_dir == 0 or g.direction != choch_dir:
                        continue
                    if i - choch_bar > spec.max_choch_age:
                        continue
                direction = g.direction
            elif mode == "fvg_only":          # control: drop the structure filter
                direction = g.direction
            elif mode == "inverted":          # self-test: trade the wrong way
                if spec.require_choch and (choch_dir == 0 or g.direction != choch_dir):
                    continue
                direction = -g.direction
            elif mode == "coinflip":          # control: same setups, random side
                if spec.require_choch and (choch_dir == 0 or g.direction != choch_dir):
                    continue
                direction = 1 if rng.random() < 0.5 else -1
            else:
                raise ValueError(mode)

            entry_px = g.mid
            # resting limit: filled on the first later bar whose range covers mid
            fill_i = None
            for j in range(i + 1, min(i + 1 + spec.ttl, n)):
                if lows[j] <= entry_px <= highs[j]:
                    fill_i = j
                    break
            if fill_i is None:
                continue

            # stop: most recent confirmed swing beyond the gap, else the gap edge
            pad = spec.stop_pad_atr * (a.iloc[fill_i] if not np.isnan(a.iloc[fill_i]) else 0.0)
            if direction > 0:
                cands = [s.price for s in sw
                         if s.kind == "low" and s.known_at <= fill_i and s.price < entry_px]
                stop_px = min(cands[-1], g.lo) if cands else g.lo
                stop_px -= pad
            else:
                cands = [s.price for s in sw
                         if s.kind == "high" and s.known_at <= fill_i and s.price > entry_px]
                stop_px = max(cands[-1], g.hi) if cands else g.hi
                stop_px += pad
            if (entry_px - stop_px) * direction <= 0:
                continue

            # resolve_from=fill_i: the limit is filled INTRABAR at fill_i, so
            # that same candle must be allowed to stop the trade out. Resolving
            # from fill_i+1 lets a fill-and-blow-through bar book a fill and no
            # loss, which on these tight stops is a large fake edge.
            r = S.simulate_bracket(sub, fill_i, entry_px, stop_px, direction,
                                   spec.r_target, max_bars=n,
                                   breakeven_at_r=spec.breakeven_at_r,
                                   resolve_from=fill_i + 1 if mode == "skipfill" else fill_i)
            if r is None:
                continue
            r.update({"spec": spec.name, "mode": mode, "fvg_bar": i,
                      "choch_dir": choch_dir, "fvg_atr": g.body_atr})
            out.append(r)
            taken = True
            break
    return out


def split_sessions(bars: dict[str, pd.DataFrame]) -> dict:
    """Pre-split into {(symbol, date): frame} once.

    Filtering a 110k-row frame per lookup inside the null loop turns a 200-draw
    test into an overnight job; the null is only useful if it is cheap enough
    to actually run.
    """
    out = {}
    for sym, df in bars.items():
        for day, sub in df.groupby(pd.DatetimeIndex(df.index).normalize()):
            out[(sym, pd.Timestamp(day))] = sub
    return out


def run(bars: dict[str, pd.DataFrame], spec: Spec, mode: str = "model",
        seed: int = 0, sessions: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = sessions if sessions is not None else split_sessions(bars)
    rows = []
    for (sym, day), sub in sessions.items():
        for t in _session_trades(sub, spec, rng, mode):
            t["symbol"] = sym
            t["date"] = day
            rows.append(t)
    return pd.DataFrame(rows)


def random_entry_null(sessions: dict, spec: Spec, real: pd.DataFrame,
                      draws: int = 200, seed: int = 11):
    """THE control that matters: on the SAME sessions the model traded, with the
    SAME direction and the SAME stop distance, enter at a random bar instead.

    This isolates the only thing the source is actually claiming -- that BOS +
    CHoCH + an FVG midpoint picks a better moment than chance. If the model is
    indistinguishable from this, the structure machinery is decoration.
    """
    rng = np.random.default_rng(seed)
    setups = []
    for r in real.itertuples():
        sub = sessions.get((r.symbol, pd.Timestamp(r.date)))
        if sub is not None and len(sub) >= 25:
            setups.append((sub, r.direction, r.risk_px))

    means, wins = [], []
    for _ in range(draws):
        rs = []
        for sub, direction, risk in setups:
            i = int(rng.integers(5, len(sub) - 6))
            entry = float(sub["close"].iloc[i])
            stop = entry - direction * risk
            res = S.simulate_bracket(sub, i, entry, stop, direction,
                                     spec.r_target, max_bars=len(sub))
            if res:
                rs.append(res["R"])
        if rs:
            arr = np.asarray(rs, dtype=float)
            means.append(arr.mean())
            wins.append((arr > 0).mean())
    return np.array(means), np.array(wins)
