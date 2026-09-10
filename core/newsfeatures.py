"""Turning scored articles into a feature table aligned to price bars.

Everything here exists to preserve one property: **a feature attached to bar t
was knowable before bar t closed.** News is timestamped continuously while bars
are discrete, and the join between them is where news backtests usually spring
a leak that inflates results and is invisible in the output.

The chain is: article timestamp -> trading session (`core.news.trading_session`,
which rolls post-close stories to the next day) -> bar index. Then the engine
adds its own one-bar execution delay on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sentiment import EVENT_PATTERNS

EVENT_NAMES = list(EVENT_PATTERNS)

BASE_FEATURES = [
    "news_count", "news_sentiment", "news_sentiment_weighted", "news_sentiment_max",
    "news_sentiment_min", "news_dispersion", "news_confidence", "news_intensity",
]


def daily_features(scored: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Collapse per-article rows into one row per trading session."""
    if scored.empty:
        return pd.DataFrame(columns=BASE_FEATURES + [f"event_{e}" for e in EVENT_NAMES])

    df = scored
    if symbol:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return pd.DataFrame(columns=BASE_FEATURES + [f"event_{e}" for e in EVENT_NAMES])

    def collapse(group: pd.DataFrame) -> pd.Series:
        sent = group["sentiment"].astype(float)
        conf = group["confidence"].astype(float)
        weight_sum = conf.sum()

        row = {
            "news_count": float(len(group)),
            "news_sentiment": float(sent.mean()),
            "news_sentiment_weighted": float((sent * conf).sum() / weight_sum)
                                       if weight_sum > 0 else 0.0,
            "news_sentiment_max": float(sent.max()),
            "news_sentiment_min": float(sent.min()),
            "news_dispersion": float(sent.std(ddof=0)) if len(group) > 1 else 0.0,
            "news_confidence": float(conf.mean()),
            # volume x conviction: a burst of confident stories is the real event
            "news_intensity": float(len(group) * conf.mean()),
        }
        tags = [t for tags in group["events"] for t in (tags or [])]
        for event in EVENT_NAMES:
            row[f"event_{event}"] = float(tags.count(event))
        return pd.Series(row)

    out = df.groupby("session", sort=True).apply(collapse, include_groups=False)
    out.index = pd.DatetimeIndex(out.index)
    return out


def add_decay_features(features: pd.DataFrame, halflives=(3, 10)) -> pd.DataFrame:
    """Exponentially weighted history of sentiment.

    Research puts the news edge at a 3-10 day horizon rather than same-day, so
    the useful feature is accumulated narrative rather than today's headline.
    Both windows are causal -- pandas `ewm` only ever looks backwards.
    """
    out = features.copy()
    if out.empty:
        for hl in halflives:
            out[f"news_sentiment_ewm{hl}"] = pd.Series(dtype=float)
            out[f"news_intensity_ewm{hl}"] = pd.Series(dtype=float)
        return out

    for hl in halflives:
        out[f"news_sentiment_ewm{hl}"] = (
            out["news_sentiment_weighted"].ewm(halflife=hl, adjust=False).mean())
        out[f"news_intensity_ewm{hl}"] = (
            out["news_intensity"].ewm(halflife=hl, adjust=False).mean())
    return out


def align_to_bars(features: pd.DataFrame, bar_index: pd.DatetimeIndex,
                  carry_forward: int = 0) -> pd.DataFrame:
    """Map session-dated features onto the bar index.

    A session that is not a trading day (a holiday, or a Sunday that slipped
    through) rolls *forward* to the next bar -- never backwards, which would
    hand a bar information from its own future.
    """
    columns = list(features.columns) if not features.empty else (
        BASE_FEATURES + [f"event_{e}" for e in EVENT_NAMES])
    original = pd.DatetimeIndex(bar_index)
    if features.empty:
        return pd.DataFrame(0.0, index=original, columns=columns)

    normalised = original.normalize()
    if normalised.duplicated().any():
        # Several bars share a calendar day, i.e. this is an intraday index.
        # A session-level aggregate is only complete at that session's close,
        # so attaching it to the 09:30 bar would hand the morning a headline
        # printed at 15:55. Refuse rather than silently leak.
        raise ValueError(
            "daily news features cannot be aligned to intraday bars: a whole "
            "session's aggregate would become readable at the first bar of the "
            "day. Aggregate the news at the same frequency as the bars instead.")

    positions = np.searchsorted(
        normalised.values,
        pd.DatetimeIndex(features.index).normalize().values, side="left")

    # Several sessions can land on one bar -- a market holiday leaves its
    # session without a bar of its own, and it rolls to the next. Summing that
    # is only right for counts. Summing a mean produces values outside its own
    # bounds, and summing an EWM is meaningless.
    buckets: dict[int, list] = {}
    for pos, (_, row) in zip(positions, features.iterrows()):
        if pos >= len(original):
            continue                       # news after the last bar: unusable
        buckets.setdefault(int(pos), []).append(row.reindex(columns).fillna(0.0))

    aligned = pd.DataFrame(0.0, index=original, columns=columns)
    for pos, rows in buckets.items():
        aligned.iloc[pos] = _combine_sessions(rows, columns).to_numpy()

    if carry_forward > 0:
        mask = aligned["news_count"] > 0
        for col in aligned.columns:
            aligned[col] = aligned[col].where(mask).ffill(limit=carry_forward)
        aligned = aligned.fillna(0.0)

    return aligned


# counts genuinely add up across sessions; nothing else does
SUM_COLUMNS = {"news_count", "news_intensity"}
MAX_COLUMNS = {"news_sentiment_max"}
MIN_COLUMNS = {"news_sentiment_min"}


def _combine_sessions(rows: list, columns: list) -> pd.Series:
    """Merge several sessions that land on one bar, per column semantics."""
    block = pd.DataFrame(rows, columns=columns).astype(float)
    if len(block) == 1:
        return block.iloc[0]

    weights = block["news_count"] if "news_count" in block else None
    if weights is None or float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=block.index)

    out = {}
    for col in columns:
        if col in SUM_COLUMNS or col.startswith("event_"):
            out[col] = float(block[col].sum())
        elif col in MAX_COLUMNS:
            out[col] = float(block[col].max())
        elif col in MIN_COLUMNS:
            out[col] = float(block[col].min())
        elif "_ewm" in col:
            # already a time-weighted history; the latest session's value is
            # the current state, and adding two of them means nothing
            out[col] = float(block[col].iloc[-1])
        else:
            out[col] = float((block[col] * weights).sum() / weights.sum())
    return pd.Series(out, index=columns)


def build_news_features(scored: pd.DataFrame, bars: pd.DataFrame,
                        symbol: str | None = None,
                        halflives=(3, 10), carry_forward: int = 0) -> pd.DataFrame:
    """Full pipeline: scored articles -> features indexed exactly like `bars`."""
    daily = daily_features(scored, symbol)
    daily = add_decay_features(daily, halflives)
    return align_to_bars(daily, bars.index, carry_forward)


# ---------------------------------------------------------------------------
# does the signal actually predict anything?
# ---------------------------------------------------------------------------
def forward_returns(bars: pd.DataFrame, horizons=(1, 3, 5, 10, 20),
                    benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Return from this bar's close to the close N bars ahead.

    Pass `benchmark` (a close-price series) to measure returns *relative* to it.
    That matters more than it sounds: over a window where a stock rose 110%,
    every sentiment bucket shows positive forward returns, and the drift swamps
    whatever the news was worth. Subtracting the market answers the question you
    actually care about -- did the news beat simply owning the thing.

    Only ever used for *analysis*. Feeding these into a strategy would be
    lookahead of the purest kind.
    """
    close = bars["close"]
    out = {f"fwd_{h}": close.shift(-h) / close - 1.0 for h in horizons}

    if benchmark is not None and not benchmark.empty:
        bench = benchmark.reindex(bars.index).ffill()
        for h in horizons:
            out[f"fwd_{h}"] = out[f"fwd_{h}"] - (bench.shift(-h) / bench - 1.0)

    return pd.DataFrame(out, index=bars.index)


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t between two samples. Returns 0 when it is undefined."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return 0.0
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else 0.0


def decay_profile(features: pd.DataFrame, bars: pd.DataFrame,
                  feature: str = "news_sentiment_weighted",
                  horizons=(1, 3, 5, 10, 20), quantiles: int = 5,
                  benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Mean forward return by feature quantile, at each horizon.

    This is how you check the 3-10 day claim against your own data instead of
    trusting a paper. A real signal shows a monotonic spread between the top and
    bottom buckets that fades as the horizon grows; noise shows no pattern.

    The final two rows are the ones that matter:

    * ``top-bottom`` -- the spread between the most and least positive buckets.
    * ``t-stat`` -- Welch's t on that spread, **corrected for overlap**. Forward
      windows of h bars share h-1 bars with their neighbours, so the raw t is
      inflated by roughly sqrt(h); it is divided out here. Treat |t| < 2 as
      "indistinguishable from noise" regardless of how good the spread looks.
    """
    fwd = forward_returns(bars, horizons, benchmark)
    joined = features.join(fwd, how="inner").dropna(subset=[feature])
    active = joined[joined["news_count"] > 0] if "news_count" in joined else joined
    if len(active) < quantiles * 4:
        return pd.DataFrame()

    try:
        buckets = pd.qcut(active[feature], quantiles, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows = []
    for bucket in sorted(pd.Series(buckets).dropna().unique()):
        subset = active[buckets == bucket]
        row = {"bucket": int(bucket) + 1, "n": len(subset),
               f"mean_{feature}": float(subset[feature].mean())}
        for h in horizons:
            col = f"fwd_{h}"
            if col in subset:
                row[f"fwd_{h}_%"] = float(subset[col].mean() * 100)
        rows.append(row)

    table = pd.DataFrame(rows)
    if len(table) < 2:
        return table

    bucket_ids = sorted(pd.Series(buckets).dropna().unique())
    bottom = active[buckets == bucket_ids[0]]
    top = active[buckets == bucket_ids[-1]]

    spread = {"bucket": "top-bottom", "n": int(table["n"].sum()),
              f"mean_{feature}": np.nan}
    tstat = {"bucket": "t-stat", "n": np.nan, f"mean_{feature}": np.nan}

    for h in horizons:
        col, out_col = f"fwd_{h}", f"fwd_{h}_%"
        if out_col not in table:
            continue
        spread[out_col] = float(table[out_col].iloc[-1] - table[out_col].iloc[0])
        if col in top and col in bottom:
            # overlapping forward windows share h-1 bars, inflating the raw t
            raw = _welch_t(top[col].to_numpy(), bottom[col].to_numpy())
            tstat[out_col] = round(raw / np.sqrt(h), 2)

    return pd.concat([table, pd.DataFrame([spread, tstat])], ignore_index=True)
