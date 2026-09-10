"""Event-study checks. Synthetic markets with planted effects, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.eventstudy import (EventWindow, events_from_condition,
                             events_from_news, interpret, run_event_study)


def make_market(symbols, n=900, seed=3, beta=None, event_days=None,
                effect=0.0, effect_span=5):
    """A market factor plus per-symbol noise, with an optional planted bump.

    `effect` is the total abnormal return spread over `effect_span` bars
    starting the day after each event.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    market = rng.normal(0.0003, 0.009, n)
    beta = beta or {s: 1.0 for s in symbols}
    event_days = event_days or {}

    bars = {}
    for s in symbols:
        r = beta[s] * market + rng.normal(0.0, 0.006, n)
        for day in event_days.get(s, []):
            for k in range(1, effect_span + 1):
                if day + k < n:
                    r[day + k] += effect / effect_span
        px = 100 * np.exp(np.cumsum(r))
        bars[s] = pd.DataFrame({"open": px, "high": px * 1.004,
                                "low": px * 0.996, "close": px,
                                "volume": 1e6}, index=idx)

    mkt_px = 100 * np.exp(np.cumsum(market))
    return bars, pd.Series(mkt_px, index=idx, name="SPY"), idx


def events_for(bars, index, spec: dict) -> pd.DataFrame:
    rows = [{"symbol": s, "date": index[d]} for s, days in spec.items() for d in days]
    return pd.DataFrame(rows)


SPEC = {f"S{i}": list(range(300 + i * 7, 800, 60)) for i in range(12)}


def test_detects_a_planted_effect():
    symbols = list(SPEC)
    bars, bench, idx = make_market(symbols, event_days={
        s: days for s, days in SPEC.items()}, effect=0.05)
    result = run_event_study(events_for(bars, idx, SPEC), bars, bench)

    assert result.n > 60, f"too few usable events: {result.n}"
    assert result.summary["car_5d_%"] > 3.0, result.summary
    assert result.summary["t_5d"] > 3.0, "did not detect a 5% planted effect"


def test_finds_nothing_when_there_is_nothing():
    symbols = list(SPEC)
    bars, bench, idx = make_market(symbols, seed=11, effect=0.0)
    result = run_event_study(events_for(bars, idx, SPEC), bars, bench)
    assert abs(result.summary["t_5d"]) < 2.5, (
        f"invented an effect from noise: {result.summary}")


def test_market_model_removes_beta_not_the_event():
    """A high-beta stock in a rising market must not look like an event."""
    symbols = ["HIGHBETA", "LOWBETA"]
    spec = {s: list(range(300, 800, 50)) for s in symbols}
    bars, bench, idx = make_market(symbols, beta={"HIGHBETA": 2.5, "LOWBETA": 0.4},
                                   event_days=spec, effect=0.0)

    adjusted = run_event_study(events_for(bars, idx, spec), bars, bench,
                               model="market")
    raw = run_event_study(events_for(bars, idx, spec), bars, bench, model="raw")

    assert abs(adjusted.summary["car_20d_%"]) < abs(raw.summary["car_20d_%"]) + 1e-9
    betas = adjusted.events.groupby("symbol")["beta"].mean()
    assert betas["HIGHBETA"] > 1.6, betas.to_dict()
    assert betas["LOWBETA"] < 1.0, betas.to_dict()


def test_estimation_window_does_not_touch_the_event_window():
    """Fitting on data the event already moved would subtract the effect."""
    window = EventWindow(pre=5, post=20, estimation=200, gap=10)
    assert window.needed_before == 215
    # the newest estimation bar sits gap+pre bars before the event
    assert window.estimation + window.gap + window.pre == window.needed_before


def test_events_too_close_to_the_start_are_skipped():
    bars, bench, idx = make_market(["AAA"], n=400)
    early = pd.DataFrame([{"symbol": "AAA", "date": idx[20]}])
    try:
        run_event_study(early, bars, bench)
        raise AssertionError("used an event without enough history behind it")
    except ValueError as exc:
        assert "history" in str(exc) or "clean history" in str(exc)


def test_events_too_close_to_the_end_are_skipped():
    bars, bench, idx = make_market(["AAA"], n=400)
    late = pd.DataFrame([{"symbol": "AAA", "date": idx[-3]}])
    try:
        run_event_study(late, bars, bench)
        raise AssertionError("used an event without enough future to measure")
    except ValueError:
        pass


def test_clustering_is_measured():
    """Events sharing one date are not independent observations."""
    bars, bench, idx = make_market([f"S{i}" for i in range(12)])
    same_day = pd.DataFrame([{"symbol": f"S{i}", "date": idx[500]}
                             for i in range(12)])
    result = run_event_study(same_day, bars, bench)
    assert result.summary["clustering"] == 1.0
    assert any("calendar date" in n for n in interpret(result))


def test_day_zero_is_the_event_bar():
    bars, bench, idx = make_market(list(SPEC), event_days=SPEC, effect=0.05)
    result = run_event_study(events_for(bars, idx, SPEC), bars, bench)
    days = result.daily["day"].tolist()
    assert days[0] == -5 and days[-1] == 20
    assert 0 in days
    pre = result.daily[result.daily["day"] < 0]["mean_ar_%"].abs().mean()
    post = result.daily[result.daily["day"] > 0]["mean_ar_%"].mean()
    assert post > pre, "the effect did not appear after the event"


def test_no_drift_before_the_event():
    """The planted effect starts the day after, so pre-event CAR must be flat."""
    bars, bench, idx = make_market(list(SPEC), event_days=SPEC, effect=0.05)
    result = run_event_study(events_for(bars, idx, SPEC), bars, bench)
    pre = result.daily[result.daily["day"] == -1]["mean_car_%"].iloc[0]
    assert abs(pre) < 1.0, f"effect leaked backwards: {pre:.2f}%"


# ---------------------------------------------------------------------------
# event selection
# ---------------------------------------------------------------------------
def scored_news():
    return pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "NVDA", "NVDA"],
        "session": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-04",
                                   "2024-03-05", "2024-04-01"]),
        "sentiment": [0.8, 0.6, -0.5, 0.9, 0.1],
        "events": [["product"], ["product"], ["legal"], ["product"], ["earnings"]],
        "text": ["launches new phone", "unveils phone", "sued over patents",
                 "launches new gpu", "reports results"],
        "source_count": [4, 4, 1, 6, 2],
    })


def test_one_event_per_symbol_per_day():
    """Heavy coverage of one announcement is one observation, not several."""
    events = events_from_news(scored_news(), event_type="product")
    aapl = events[events["symbol"] == "AAPL"]
    assert len(aapl) == 1, "counted the same announcement twice"
    assert abs(aapl["sentiment"].iloc[0] - 0.7) < 1e-9, "did not average duplicates"


def test_filters_by_event_type_and_sentiment():
    news = scored_news()
    assert len(events_from_news(news, event_type="legal")) == 1
    assert len(events_from_news(news, min_sentiment=0.85)) == 1
    assert events_from_news(news, event_type="product",
                            min_sources=5)["symbol"].tolist() == ["NVDA"]


def test_keyword_selection():
    events = events_from_news(scored_news(), keyword="gpu")
    assert events["symbol"].tolist() == ["NVDA"]


def test_condition_events_respect_a_minimum_gap():
    bars, _, idx = make_market(["AAA"], n=300)
    always = lambda df: pd.Series(True, index=df.index)  # noqa: E731
    events = events_from_condition(bars, always, min_gap_bars=10)
    assert len(events) == 30, f"expected 30 spaced events, got {len(events)}"


def test_empty_selection_returns_empty_frame():
    assert events_from_news(pd.DataFrame(), event_type="product").empty
    assert events_from_news(scored_news(), event_type="nonexistent").empty


def test_interpret_flags_small_samples():
    bars, bench, idx = make_market(["AAA", "BBB"], n=900)
    spec = {"AAA": [400, 500], "BBB": [450, 550]}
    result = run_event_study(events_for(bars, idx, spec), bars, bench)
    notes = interpret(result)
    assert any("events" in n and "power" in n for n in notes)
    assert any("symbols" in n for n in notes)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc or 'assertion failed'}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print("\nall green" if not failures else f"\n{failures} failing")
    raise SystemExit(1 if failures else 0)
