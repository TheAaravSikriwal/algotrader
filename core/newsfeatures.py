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
    empty = pd.DataFrame(0.0, index=bar_index, columns=columns)
    if features.empty:
        return empty

    bar_index = pd.DatetimeIndex(bar_index).normalize()
    positions = np.searchsorted(bar_index.values,
                                pd.DatetimeIndex(features.index).normalize().values,
                                side="left")

    aligned = empty.copy()
    aligned.index = bar_index
    for pos, (_, row) in zip(positions, features.iterrows()):
        if pos >= len(bar_index):
            continue                       # news after the last bar: unusable
        aligned.iloc[pos] += row.reindex(columns).fillna(0.0).to_numpy()

    if carry_forward > 0:
        mask = aligned["news_count"] > 0
        for col in aligned.columns:
            aligned[col] = aligned[col].where(mask).ffill(limit=carry_forward)
        aligned = aligned.fillna(0.0)

    aligned.index = pd.DatetimeIndex(bar_index)
    return aligned


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
def forward_returns(bars: pd.DataFrame, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """Return from this bar's close to the close N bars ahead.

    Only ever used for *analysis*. Feeding these into a strategy would be
    lookahead of the purest kind.
    """
    close = bars["close"]
    return pd.DataFrame(
        {f"fwd_{h}": close.shift(-h) / close - 1.0 for h in horizons},
        index=bars.index)


def decay_profile(features: pd.DataFrame, bars: pd.DataFrame,
                  feature: str = "news_sentiment_weighted",
                  horizons=(1, 3, 5, 10, 20), quantiles: int = 5) -> pd.DataFrame:
    """Mean forward return by feature quantile, at each horizon.

    This is how you check the 3-10 day claim against your own data instead of
    trusting a paper. A real signal shows a monotonic spread between the top and
    bottom buckets that fades as the horizon grows; noise shows no pattern.
    """
    fwd = forward_returns(bars, horizons)
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
    if len(table) >= 2:
        spread = {"bucket": "top-bottom", "n": int(table["n"].sum()),
                  f"mean_{feature}": np.nan}
        for h in horizons:
            col = f"fwd_{h}_%"
            if col in table:
                spread[col] = float(table[col].iloc[-1] - table[col].iloc[0])
        table = pd.concat([table, pd.DataFrame([spread])], ignore_index=True)
    return table
