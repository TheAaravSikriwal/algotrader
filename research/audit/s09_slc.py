"""Source 09: the SLC system -- Structure, Level, Confirmation -- backtested.

The source's rules, verbatim, and how each is encoded:

  S  "Trade on the 5-minute; read structure on the 4-hour. Uptrend = higher
      highs and higher lows -> longs only. Downtrend = lower highs and lower
      lows -> shorts only. Consolidation -> take no trades."
     -> 4H bars resampled from the full (extended-hours) 5-minute series on a
        midnight-ET grid. Trend from the last two confirmed 4H swing highs and
        lows. A 4H bar is usable only once it has CLOSED.

  L  "Demand level: find a sharp, aggressive move up. The area immediately
      before that move is the demand level... I take the last candle before the
      aggressive move formed and draw a rectangle around it." Only levels
      aligned with the higher-timeframe structure.
     -> structure.zones(). "Sharp, aggressive" is the source's only sizing
        language, so the impulse threshold is swept, not chosen.

  C  "As price taps into the supply level, the blue stochastic line needs to
      break above the upper level and then cross back down."
     -> %K crosses above `upper` and later crosses back below it while price is
        in the zone. THE SETTINGS ARE NEVER STATED -- the source shows them on
        screen and tells the viewer to pause and copy. So the stochastic
        parameters are swept over a declared grid, and the multiple-comparison
        bar is adjusted accordingly. This is the source's own doing, not a
        liberty taken here.

  Trade management: "Entry on confirmation at the level. Stop-loss slightly
  above the supply level. Take-profit fixed at 2R."
     -> entry filled at the OPEN of the bar after the confirming bar (the
        repo's no-lookahead contract), stop = zone edge + pad x ATR, target 2R.

Not codeable, and not invented here: "only mark levels where price has the
highest chance of reacting", and the break-and-retest validity rule "only valid
if a supply level has been broken once -- if it has been chopped through
multiple times it is no longer valid" is implemented as a touch counter, but
"chopped through" is not defined numerically by the source.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import structure as S  # noqa: E402
from core.indicators import atr as atr_ind  # noqa: E402


@dataclass
class Spec:
    name: str = "slc"
    # structure
    htf: str = "4h"
    htf_k: int = 2
    require_structure: bool = True
    # level
    impulse_bars: int = 3
    min_impulse_atr: float = 2.0
    base_rule: str = "last_opposite"
    zone_ttl: int = 78            # bars a zone stays live (one session)
    max_touches: int = 1          # "only valid if broken once"
    # confirmation
    require_confirmation: bool = True
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    stoch_upper: float = 80.0
    stoch_lower: float = 20.0
    confirm_window: int = 6       # bars allowed between the tap and the cross-back
    # management
    r_target: float = 2.0
    stop_pad_atr: float = 0.25


def htf_trend(df5: pd.DataFrame, spec: Spec) -> pd.Series:
    """+1 uptrend, -1 downtrend, 0 consolidation, per 5-minute bar.

    Built from CLOSED higher-timeframe bars only. The value carried on a 5-min
    bar is derived from 4H bars that finished strictly before it, so no part of
    the current 4H candle leaks in -- which is exactly the error a chartist
    makes reading a live 4H candle.
    """
    h = df5.resample(spec.htf, origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(h) < 4 * spec.htf_k + 4:
        return pd.Series(0, index=df5.index)

    sw = S.swings(h, spec.htf_k)
    state = np.zeros(len(h), dtype=int)
    highs: list[float] = []
    lows: list[float] = []
    ptr = 0
    for i in range(len(h)):
        while ptr < len(sw) and sw[ptr].known_at <= i:
            (highs if sw[ptr].kind == "high" else lows).append(sw[ptr].price)
            ptr += 1
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1] > highs[-2]
            hl = lows[-1] > lows[-2]
            lh = highs[-1] < highs[-2]
            ll = lows[-1] < lows[-2]
            state[i] = 1 if (hh and hl) else (-1 if (lh and ll) else 0)

    st = pd.Series(state, index=h.index)
    # a 4H bar stamped T covers [T, T+4h); it is only KNOWN at T+4h.
    st.index = st.index + pd.Timedelta(spec.htf)
    return st.reindex(df5.index.union(st.index)).ffill().reindex(df5.index).fillna(0).astype(int)


def session_trades(sub: pd.DataFrame, trend: pd.Series, spec: Spec,
                   mode: str = "model", rng=None) -> list[dict]:
    if len(sub) < 30:
        return []
    a = atr_ind(sub, 14)
    zs = S.zones(sub, a, spec.impulse_bars, spec.min_impulse_atr, spec.base_rule)
    if not zs:
        return []
    kline, _ = S.stochastic(sub, spec.stoch_k, spec.stoch_d, spec.stoch_smooth)

    o = sub["open"].to_numpy(float)
    hi = sub["high"].to_numpy(float)
    lo = sub["low"].to_numpy(float)
    kv = kline.to_numpy(float)
    n = len(sub)
    tr = trend.reindex(sub.index).fillna(0).to_numpy(int)

    if mode == "random_zone" and rng is not None:
        # keep the count and the widths, move the locations: isolates whether
        # the ZONE LOCATION carries information at all
        px = sub["close"].to_numpy(float)
        shuffled = []
        for z in zs:
            j = int(rng.integers(5, n - 5))
            w = z.hi - z.lo
            c = px[j]
            shuffled.append(S.Zone(j, z.direction, c - w / 2, c + w / 2, j, z.impulse_atr))
        zs = shuffled

    out: list[dict] = []
    touches: dict[int, int] = {}
    broke_above: dict[int, int] = {}

    for zi, z in enumerate(zs):
        direction = z.direction
        if mode == "inverted":
            direction = -direction
        elif mode == "coinflip" and rng is not None:
            direction = 1 if rng.random() < 0.5 else -1

        for i in range(z.known_at + 1, min(z.known_at + 1 + spec.zone_ttl, n - 1)):
            if spec.require_structure and mode != "no_structure" and tr[i] != direction:
                continue
            in_zone = (lo[i] <= z.hi and hi[i] >= z.lo)
            if not in_zone:
                continue
            touches[zi] = touches.get(zi, 0) + 1
            if touches[zi] > spec.max_touches:
                break

            ok = True
            if spec.require_confirmation and mode != "no_confirmation":
                ok = False
                if direction < 0:      # short at supply: %K above upper, then back below
                    for j in range(max(0, i - spec.confirm_window), i + 1):
                        if j > 0 and kv[j - 1] > spec.stoch_upper and kv[j] <= spec.stoch_upper:
                            if (kv[max(0, j - spec.confirm_window):j] > spec.stoch_upper).any():
                                ok = True
                                break
                else:                  # long at demand: mirror image
                    for j in range(max(0, i - spec.confirm_window), i + 1):
                        if j > 0 and kv[j - 1] < spec.stoch_lower and kv[j] >= spec.stoch_lower:
                            if (kv[max(0, j - spec.confirm_window):j] < spec.stoch_lower).any():
                                ok = True
                                break
            if not ok:
                continue

            if i + 1 >= n:
                break
            entry_px = o[i + 1]            # fill at the OPEN of the next bar
            pad = spec.stop_pad_atr * (a.iloc[i] if not np.isnan(a.iloc[i]) else 0.0)
            stop_px = (z.lo - pad) if direction > 0 else (z.hi + pad)
            if (entry_px - stop_px) * direction <= 0:
                break

            r = S.simulate_bracket(sub, i, entry_px, stop_px, direction,
                                   spec.r_target, max_bars=n)
            if r:
                r.update({"spec": spec.name, "mode": mode, "zone_i": zi,
                          "trend": int(tr[i]), "impulse_atr": z.impulse_atr})
                out.append(r)
            break
    return out


def run(bars: dict[str, pd.DataFrame], full: dict[str, pd.DataFrame], spec: Spec,
        mode: str = "model", seed: int = 0, one_per_day: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym, df in bars.items():
        trend = htf_trend(full[sym], spec).reindex(df.index).ffill().fillna(0).astype(int)
        for day, sub in df.groupby(pd.DatetimeIndex(df.index).normalize()):
            ts = session_trades(sub, trend, spec, mode, rng)
            if one_per_day:
                ts = ts[:1]
            for t in ts:
                t["symbol"] = sym
                t["date"] = day
                rows.append(t)
    return pd.DataFrame(rows)
