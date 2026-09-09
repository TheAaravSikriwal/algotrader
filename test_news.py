"""News pipeline checks. Synthetic data throughout -- no API keys, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.engine import BacktestConfig, run_backtest
from core.news import NewsItem, to_frame, trading_session
from core.newsfeatures import (align_to_bars, build_news_features, daily_features,
                               decay_profile, forward_returns)
from core.sentiment import LexiconScorer, score_frame, tag_events
from core.strategy import get_strategy

import strategies  # noqa: F401


# ---------------------------------------------------------------------------
# session assignment -- the point-in-time boundary
# ---------------------------------------------------------------------------
def test_intraday_news_belongs_to_the_same_session():
    ts = pd.Timestamp("2024-03-12 14:30", tz="America/New_York").tz_convert("UTC")
    assert trading_session(ts) == pd.Timestamp("2024-03-12")


def test_after_close_news_rolls_to_the_next_session():
    """18:40 ET could not have informed that day's close."""
    ts = pd.Timestamp("2024-03-12 18:40", tz="America/New_York").tz_convert("UTC")
    assert trading_session(ts) == pd.Timestamp("2024-03-13")


def test_news_exactly_at_the_close_rolls_forward():
    ts = pd.Timestamp("2024-03-12 16:00", tz="America/New_York").tz_convert("UTC")
    assert trading_session(ts) == pd.Timestamp("2024-03-13")


def test_weekend_news_rolls_to_monday():
    ts = pd.Timestamp("2024-03-16 11:00", tz="America/New_York").tz_convert("UTC")
    assert trading_session(ts) == pd.Timestamp("2024-03-18")  # Saturday -> Monday


def test_friday_evening_news_rolls_to_monday():
    ts = pd.Timestamp("2024-03-15 20:00", tz="America/New_York").tz_convert("UTC")
    assert trading_session(ts) == pd.Timestamp("2024-03-18")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_lexicon_separates_good_from_bad_news():
    s = LexiconScorer()
    good = s.score("Company beats earnings, raises guidance, record profit growth")
    bad = s.score("Company misses earnings, warns of losses, faces lawsuit and probe")
    assert good["sentiment"] > 0.2, good
    assert bad["sentiment"] < -0.2, bad
    assert good["sentiment"] > bad["sentiment"]


def test_lexicon_handles_negation():
    s = LexiconScorer()
    plain = s.score("The company reported strong growth")
    negated = s.score("The company did not report strong growth")
    assert negated["sentiment"] < plain["sentiment"], "negation was ignored"


def test_lexicon_is_neutral_on_empty_or_bland_text():
    s = LexiconScorer()
    assert s.score("")["sentiment"] == 0.0
    assert s.score("The company will hold a meeting on Tuesday")["sentiment"] == 0.0


def test_event_tagging():
    assert "earnings" in tag_events("Apple reports Q3 results, beats estimates")
    assert "analyst" in tag_events("Goldman upgrades Tesla, raises price target")
    assert "mna" in tag_events("Microsoft acquires gaming studio in $2B buyout")
    assert "legal" in tag_events("SEC opens investigation into accounting")
    assert tag_events("A perfectly ordinary sentence about nothing") == []


def test_scoring_is_cached_and_stable():
    frame = pd.DataFrame({"text": ["record profit growth", "heavy losses and fraud"]})
    a = score_frame(frame, LexiconScorer())
    b = score_frame(frame, LexiconScorer())
    assert a["sentiment"].tolist() == b["sentiment"].tolist()
    assert a["sentiment"].iloc[0] > 0 > a["sentiment"].iloc[1]


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def scored_frame(sessions, sentiments, symbol="TEST", events=("earnings",)):
    return pd.DataFrame({
        "session": pd.DatetimeIndex(sessions),
        "symbol": symbol,
        "sentiment": list(sentiments),
        "confidence": 1.0,
        "events": [list(events)] * len(sessions),
        "text": ["synthetic"] * len(sessions),
    })


def test_features_land_on_the_right_bars():
    bars_index = pd.bdate_range("2024-01-01", periods=10)
    scored = scored_frame([bars_index[3]], [0.8])
    aligned = align_to_bars(daily_features(scored, "TEST"), bars_index)
    assert aligned["news_count"].iloc[3] == 1.0
    assert aligned["news_count"].drop(aligned.index[3]).sum() == 0.0


def test_holiday_news_rolls_forward_never_backward():
    """A session with no bar must attach to the NEXT bar. Attaching it to the
    previous one would give that bar tomorrow's information."""
    bars_index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05",
                                   "2024-01-08", "2024-01-09"])
    scored = scored_frame([pd.Timestamp("2024-01-04")], [0.9])   # no bar that day
    aligned = align_to_bars(daily_features(scored, "TEST"), bars_index)
    assert aligned.loc["2024-01-05", "news_count"] == 1.0
    assert aligned.loc["2024-01-03", "news_count"] == 0.0, "leaked backwards"


def test_news_after_the_last_bar_is_dropped():
    bars_index = pd.bdate_range("2024-01-01", periods=5)
    scored = scored_frame([pd.Timestamp("2024-06-01")], [1.0])
    aligned = align_to_bars(daily_features(scored, "TEST"), bars_index)
    assert aligned["news_count"].sum() == 0.0


def test_multiple_articles_aggregate_within_a_session():
    day = pd.Timestamp("2024-01-03")
    scored = scored_frame([day, day, day], [1.0, 0.0, -1.0])
    daily = daily_features(scored, "TEST")
    assert daily["news_count"].iloc[0] == 3.0
    assert abs(daily["news_sentiment"].iloc[0]) < 1e-9
    assert daily["news_dispersion"].iloc[0] > 0
    assert daily["event_earnings"].iloc[0] == 3.0


def test_decay_features_are_causal():
    """An EWM feature at bar t must not move when a LATER bar changes."""
    sessions = pd.bdate_range("2024-01-01", periods=40)
    base = scored_frame(sessions, np.linspace(-1, 1, 40))
    bumped = base.copy()
    bumped.loc[bumped.index[-1], "sentiment"] = 5.0

    a = build_news_features(base, pd.DataFrame(index=sessions), "TEST")
    b = build_news_features(bumped, pd.DataFrame(index=sessions), "TEST")
    assert np.allclose(a["news_sentiment_ewm3"].iloc[:-1],
                       b["news_sentiment_ewm3"].iloc[:-1]), "future leaked backwards"


# ---------------------------------------------------------------------------
# does the analysis actually detect signal, and reject noise?
# ---------------------------------------------------------------------------
def planted_market(n=800, alpha=0.004, seed=7, horizon=5):
    """Prices whose next `horizon` days genuinely respond to today's sentiment."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2021-01-04", periods=n)
    sentiment = rng.uniform(-1, 1, n)

    returns = rng.normal(0, 0.008, n)
    for lag in range(1, horizon + 1):
        returns[lag:] += alpha * sentiment[:-lag] / horizon

    prices = 100 * np.exp(np.cumsum(returns))
    bars = pd.DataFrame({"open": prices, "high": prices * 1.003,
                         "low": prices * 0.997, "close": prices,
                         "volume": np.full(n, 1e6)}, index=sessions)
    return bars, scored_frame(sessions, sentiment)


def test_decay_profile_finds_a_planted_signal():
    bars, scored = planted_market()
    features = build_news_features(scored, bars, "TEST")
    table = decay_profile(features, bars, horizons=(1, 3, 5, 10, 20))
    assert not table.empty

    spread = table[table["bucket"] == "top-bottom"].iloc[0]
    assert spread["fwd_5_%"] > 0.2, f"missed a planted 5-day signal: {spread}"
    assert spread["fwd_5_%"] > spread["fwd_1_%"], "1-day should be weaker than 5-day"


def test_decay_profile_rejects_pure_noise():
    """Sentiment unrelated to returns must not produce a spread."""
    rng = np.random.default_rng(11)
    n = 800
    sessions = pd.bdate_range("2021-01-04", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    bars = pd.DataFrame({"open": prices, "high": prices, "low": prices,
                         "close": prices, "volume": np.full(n, 1e6)}, index=sessions)
    scored = scored_frame(sessions, rng.uniform(-1, 1, n))

    features = build_news_features(scored, bars, "TEST")
    table = decay_profile(features, bars, horizons=(1, 5, 10))
    spread = table[table["bucket"] == "top-bottom"].iloc[0]
    assert abs(spread["fwd_5_%"]) < 0.5, f"invented a signal from noise: {spread}"


def test_forward_returns_are_not_available_to_strategies():
    bars, _ = planted_market(n=50)
    fwd = forward_returns(bars, (1, 5))
    assert fwd["fwd_5"].tail(5).isna().all(), "forward returns leaked into the tail"


# ---------------------------------------------------------------------------
# the strategies
# ---------------------------------------------------------------------------
def test_news_strategy_holds_for_the_configured_window():
    bars, _ = planted_market(n=60)
    features = pd.DataFrame(0.0, index=bars.index,
                            columns=["news_sentiment_weighted", "news_count"])
    features.iloc[10] = [0.9, 3.0]
    enriched = bars.join(features)

    strat = get_strategy("News sentiment")(threshold=0.5, hold_bars=5, min_articles=1)
    signals = strat.generate_signals(enriched)
    assert signals.iloc[9] == 0.0
    assert signals.iloc[10] == 1.0
    assert signals.iloc[14] == 1.0, "exited early"
    assert signals.iloc[15] == 0.0, "held too long"


def test_news_strategy_stays_flat_without_news():
    bars, _ = planted_market(n=60)
    enriched = bars.join(pd.DataFrame(
        0.0, index=bars.index, columns=["news_sentiment_weighted", "news_count"]))
    strat = get_strategy("News sentiment")(threshold=0.3, hold_bars=5)
    assert (strat.generate_signals(enriched) == 0).all()


def test_missing_feature_column_explains_itself():
    bars, _ = planted_market(n=30)
    strat = get_strategy("News sentiment")()
    try:
        strat.generate_signals(bars)
        raise AssertionError("ran without news features")
    except RuntimeError as exc:
        assert "news feature" in str(exc).lower() or "column" in str(exc).lower()


def test_news_strategy_beats_random_on_planted_data():
    """End to end: a planted signal must survive the whole pipeline and the
    engine's execution delay, and beat the same strategy on shuffled news."""
    bars, scored = planted_market(alpha=0.010)
    features = build_news_features(scored, bars, "TEST")
    enriched = bars.join(features)

    shuffled = scored.copy()
    shuffled["sentiment"] = (shuffled["sentiment"]
                             .sample(frac=1.0, random_state=3).to_numpy())
    enriched_shuffled = bars.join(build_news_features(shuffled, bars, "TEST"))

    strat = get_strategy("News sentiment")(threshold=0.3, hold_bars=5, min_articles=1)
    cfg = BacktestConfig(initial_cash=10_000, slippage_bps=1)

    real = run_backtest(enriched, strat.generate_signals(enriched), cfg)
    fake = run_backtest(enriched_shuffled,
                        strat.generate_signals(enriched_shuffled), cfg)
    assert real.equity.iloc[-1] > fake.equity.iloc[-1], (
        f"planted signal did not beat shuffled news: "
        f"{real.equity.iloc[-1]:,.0f} vs {fake.equity.iloc[-1]:,.0f}")


def test_news_item_frame_expands_symbols():
    item = NewsItem(id="1", timestamp=pd.Timestamp("2024-03-12 14:00", tz="UTC"),
                    symbols=["AAPL", "MSFT"], headline="Both firms beat estimates")
    frame = to_frame([item])
    assert set(frame["symbol"]) == {"AAPL", "MSFT"}
    assert (frame["session"] == pd.Timestamp("2024-03-12")).all()


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
