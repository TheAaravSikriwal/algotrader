"""Event studies: what actually happens after an event, pooled across companies.

This is the tool that separates "SOFI announced a campaign once and rose 8%"
from "across 340 such announcements the five-day drift was +0.4%, t = 1.2,
which is nothing". One company's history of a rare event is an anecdote; the
same event pooled across hundreds of companies over a decade is evidence.

The method is the standard one (Brown and Warner; MacKinlay):

1. Estimate each stock's normal behaviour on a window *before* the event.
2. Predict what it would have done over the event window absent the event.
3. Subtract. The residual is the abnormal return the event is credited with.
4. Average across events and test whether the mean differs from zero.

Step 1 is where these usually go wrong. The estimation window must not touch
the event window -- fit the model on data the event already influenced and you
partly subtract the effect you are trying to measure. `gap` enforces the
separation and a test pins it.

The other trap is calendar clustering. If most events land in the same week,
they share one market move and are not independent observations; the t-test
then reports a sample size it does not have. `clustering` measures this so an
inflated result is visible instead of silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EventWindow:
    """Trading-day offsets relative to the event, which sits at day 0."""
    pre: int = 5                  # bars shown before the event
    post: int = 20                # bars followed after it
    estimation: int = 200         # bars used to learn normal behaviour
    gap: int = 10                 # bars left between estimation and event

    def __post_init__(self):
        if self.pre < 0 or self.post < 1:
            raise ValueError("pre must be >= 0 and post >= 1")
        if self.estimation < 30:
            raise ValueError("estimation window under 30 bars is too short to fit")
        if self.gap < 0:
            raise ValueError("gap must not be negative")

    @property
    def needed_before(self) -> int:
        return self.estimation + self.gap + self.pre


@dataclass
class EventStudyResult:
    events: pd.DataFrame           # one row per usable event
    daily: pd.DataFrame            # mean AR and CAR per relative day, with t
    summary: dict
    skipped: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.events)


def _market_model(stock: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares of stock on market. Returns (alpha, beta)."""
    if len(stock) < 10 or np.allclose(market, market[0]):
        return 0.0, 1.0
    variance = market.var()
    if variance <= 0:
        return 0.0, 1.0
    beta = float(np.cov(stock, market, ddof=1)[0, 1] / (variance * len(market)
                 / max(len(market) - 1, 1)))
    if not np.isfinite(beta):
        beta = 1.0
    beta = float(np.clip(beta, -3.0, 3.0))       # keep a bad fit from exploding
    alpha = float(stock.mean() - beta * market.mean())
    return alpha, beta


def run_event_study(events, bars: dict[str, pd.DataFrame],
                    benchmark: pd.Series | None = None,
                    window: EventWindow | None = None,
                    model: str = "market") -> EventStudyResult:
    """Measure abnormal returns around a set of (symbol, date) events.

    `model` is "market" (fit alpha and beta on the estimation window),
    "market_adjusted" (assume beta 1, alpha 0 -- more robust when events are
    few), or "raw" (no adjustment, which measures drift rather than an effect).
    """
    window = window or EventWindow()
    frame = (events.copy() if isinstance(events, pd.DataFrame)
             else pd.DataFrame(events, columns=["symbol", "date"]))
    if frame.empty:
        raise ValueError("no events supplied")

    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])

    bench_returns = None
    if benchmark is not None and not benchmark.empty:
        bench_returns = benchmark.pct_change()

    offsets = list(range(-window.pre, window.post + 1))
    rows, matrix = [], []
    skipped = {"no_bars": 0, "short_history": 0, "short_future": 0,
               "no_benchmark": 0}

    for record in frame.to_dict("records"):
        symbol, when = record["symbol"], record["date"]
        df = bars.get(symbol)
        if df is None or df.empty:
            skipped["no_bars"] += 1
            continue

        index = df.index
        # first bar on or after the event -- news at 6pm belongs to tomorrow,
        # and a holiday rolls forward, never back
        position = int(index.searchsorted(when, side="left"))
        if position >= len(index):
            skipped["short_future"] += 1
            continue
        if position < window.needed_before:
            skipped["short_history"] += 1
            continue
        if position + window.post >= len(index):
            skipped["short_future"] += 1
            continue

        returns = df["close"].pct_change()
        est_end = position - window.pre - window.gap
        est_start = est_end - window.estimation
        if est_start < 1:
            skipped["short_history"] += 1
            continue

        est_stock = returns.iloc[est_start:est_end].to_numpy(dtype=float)
        event_slice = returns.iloc[position - window.pre:position + window.post + 1]
        actual = event_slice.to_numpy(dtype=float)
        if len(actual) != len(offsets) or np.isnan(actual).any():
            skipped["short_future"] += 1
            continue

        if model == "raw" or bench_returns is None:
            if model != "raw" and bench_returns is None:
                skipped["no_benchmark"] += 1
                continue
            abnormal = actual
            alpha, beta = 0.0, 0.0
        else:
            bench = bench_returns.reindex(index)
            est_market = bench.iloc[est_start:est_end].to_numpy(dtype=float)
            event_market = bench.iloc[position - window.pre:
                                      position + window.post + 1].to_numpy(dtype=float)
            if np.isnan(event_market).any():
                skipped["no_benchmark"] += 1
                continue

            mask = ~(np.isnan(est_stock) | np.isnan(est_market))
            if mask.sum() < 30:
                skipped["short_history"] += 1
                continue

            if model == "market":
                alpha, beta = _market_model(est_stock[mask], est_market[mask])
            else:                                   # market_adjusted
                alpha, beta = 0.0, 1.0
            abnormal = actual - (alpha + beta * event_market)

        matrix.append(abnormal)
        rows.append({
            "symbol": symbol, "event_date": index[position],
            "alpha": alpha, "beta": beta,
            "car": float(np.nansum(abnormal[window.pre:])),
            "ar_0": float(abnormal[window.pre]),
            **{k: v for k, v in record.items() if k not in ("symbol", "date")},
        })

    if not rows:
        raise ValueError(
            "no event had enough clean history. Needs "
            f"{window.needed_before} bars before and {window.post} after; "
            f"skipped {skipped}")

    ar = np.vstack(matrix)
    car = np.cumsum(np.where(np.isnan(ar), 0.0, ar), axis=1)
    n = len(rows)

    daily = pd.DataFrame({
        "day": offsets,
        "mean_ar_%": np.nanmean(ar, axis=0) * 100,
        "mean_car_%": car.mean(axis=0) * 100,
        "positive_share": (ar > 0).mean(axis=0),
    })
    ar_sd = np.nanstd(ar, axis=0, ddof=1)
    car_sd = car.std(axis=0, ddof=1)
    daily["t_ar"] = np.where(ar_sd > 0,
                             np.nanmean(ar, axis=0) / (ar_sd / np.sqrt(n)), 0.0)
    daily["t_car"] = np.where(car_sd > 0, car.mean(axis=0) / (car_sd / np.sqrt(n)), 0.0)

    events_frame = pd.DataFrame(rows)
    dates = events_frame["event_date"].dt.normalize()
    clustering = float(dates.value_counts().max() / n)

    summary = {
        "events": n,
        "symbols": int(events_frame["symbol"].nunique()),
        "model": model,
        "first_event": str(events_frame["event_date"].min().date()),
        "last_event": str(events_frame["event_date"].max().date()),
        "clustering": clustering,
        "skipped": skipped,
    }

    # The event day is reported on its own and kept out of the post-event CAR.
    # Including it is a trap for any condition-selected study: a gap-down
    # screen selects on day 0's move, so a CAR spanning day 0 mostly restates
    # the filter and arrives with a huge t-statistic that means nothing. The
    # tradeable question is what happens from the next bar onward.
    zero = offsets.index(0)
    summary["event_day_%"] = float(ar[:, zero].mean() * 100)
    summary["t_event_day"] = float(daily["t_ar"].iloc[zero])

    for day in (1, 3, 5, 10, 20):
        if day not in offsets:
            continue
        stop = offsets.index(day)
        forward = np.nansum(ar[:, zero + 1:stop + 1], axis=1)
        sd = forward.std(ddof=1)
        summary[f"car_{day}d_%"] = float(forward.mean() * 100)
        summary[f"t_{day}d"] = float(forward.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0

    return EventStudyResult(events=events_frame, daily=daily, summary=summary,
                            skipped=skipped)


# ---------------------------------------------------------------------------
# finding events
# ---------------------------------------------------------------------------
def events_from_news(scored: pd.DataFrame, event_type: str | None = None,
                     keyword: str | None = None,
                     min_sentiment: float | None = None,
                     max_sentiment: float | None = None,
                     min_sources: int = 1,
                     one_per_symbol_per_day: bool = True) -> pd.DataFrame:
    """Turn scored articles into (symbol, date) events.

    `one_per_symbol_per_day` matters: a heavily covered announcement would
    otherwise enter the study several times and count as several independent
    observations, which is the same error deduplication exists to prevent.
    """
    if scored.empty:
        return pd.DataFrame(columns=["symbol", "date"])

    frame = scored.copy()
    if event_type:
        frame = frame[frame["events"].apply(
            lambda tags: event_type in (tags or []))]
    if keyword:
        pattern = keyword.lower()
        frame = frame[frame["text"].astype(str).str.lower().str.contains(
            pattern, regex=False, na=False)]
    if min_sentiment is not None:
        frame = frame[frame["sentiment"] >= min_sentiment]
    if max_sentiment is not None:
        frame = frame[frame["sentiment"] <= max_sentiment]
    if min_sources > 1 and "source_count" in frame:
        frame = frame[frame["source_count"] >= min_sources]

    if frame.empty:
        return pd.DataFrame(columns=["symbol", "date"])

    date_col = "session" if "session" in frame else "timestamp"
    out = pd.DataFrame({
        "symbol": frame["symbol"].astype(str).str.upper(),
        "date": pd.to_datetime(frame[date_col]).dt.normalize(),
        "sentiment": frame.get("sentiment", pd.Series(0.0, index=frame.index)),
    })
    if one_per_symbol_per_day:
        out = out.groupby(["symbol", "date"], as_index=False).agg(
            sentiment=("sentiment", "mean"))
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def events_from_condition(bars: dict[str, pd.DataFrame], condition,
                          min_gap_bars: int = 5) -> pd.DataFrame:
    """Price-based events, e.g. a gap down or a new 52-week high.

    `condition` takes a bar frame and returns a boolean Series. `min_gap_bars`
    stops one long episode registering as a run of near-identical events.
    """
    rows = []
    for symbol, df in bars.items():
        if df is None or df.empty:
            continue
        flags = condition(df).fillna(False).to_numpy()
        last = -10**9
        for i, hit in enumerate(flags):
            if hit and i - last >= min_gap_bars:
                rows.append({"symbol": symbol.upper(), "date": df.index[i]})
                last = i
    if not rows:
        return pd.DataFrame(columns=["symbol", "date"])
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def interpret(result: EventStudyResult, bar: float = 2.0) -> list[str]:
    """Plain-language reading of the result, including the reasons to doubt it."""
    s = result.summary
    notes = []

    significant = [(d, s[f"car_{d}d_%"], s[f"t_{d}d"])
                   for d in (1, 3, 5, 10, 20)
                   if f"t_{d}d" in s and abs(s[f"t_{d}d"]) >= bar]
    if significant:
        for day, value, t in significant:
            notes.append(f"{day}-day CAR {value:+.2f}% (t = {t:+.2f}) clears |t| >= {bar}.")
    else:
        best = max((d for d in (1, 3, 5, 10, 20) if f"t_{d}d" in s),
                   key=lambda d: abs(s[f"t_{d}d"]), default=None)
        if best is not None:
            notes.append(
                f"No horizon reaches |t| >= {bar}. Strongest is {best} days at "
                f"t = {s[f't_{best}d']:+.2f}, which is within what noise produces.")

    if s["events"] < 30:
        notes.append(f"Only {s['events']} events. Under about 30 the test has "
                     "little power, so a null here means 'unknown', not 'nothing'.")
    if s["symbols"] < 10:
        notes.append(f"Only {s['symbols']} distinct symbols. Pooling across more "
                     "companies is what makes this evidence rather than anecdote.")
    if s["clustering"] > 0.2:
        notes.append(
            f"{s['clustering']:.0%} of events share a single calendar date. Those "
            "observations move together, so the effective sample is smaller than "
            f"{s['events']} and the t-statistics are optimistic.")
    return notes
