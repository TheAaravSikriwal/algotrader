"""News-driven strategies.

These expect bars enriched with news feature columns -- use
``core.newsfeatures.build_news_features`` and join the result onto your bars,
or let ``news_backtest.py`` do it for you.

The horizon is a parameter on purpose. Research puts the LLM/sentiment news
edge near zero at one day, peaking at 3-10 days and gone by 20, so
``hold_bars`` is the knob that lets the tournament measure that curve on your
own data rather than taking anyone's word for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import Param, Strategy, register

SENTIMENT_FEATURES = [
    "news_sentiment_weighted",
    "news_sentiment",
    "news_sentiment_ewm3",
    "news_sentiment_ewm10",
]


class NewsFeatureMissing(RuntimeError):
    pass


def _require(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise NewsFeatureMissing(
            f"bars carry no '{column}' column. Join news features onto them "
            "first -- see news_backtest.py, or core.newsfeatures.build_news_features."
        )
    return df[column].astype(float).fillna(0.0)


@register
class NewsSentiment(Strategy):
    name = "News sentiment"
    requires_features = ["news_sentiment_weighted", "news_count"]
    description = ("Take a position when news sentiment crosses a threshold and "
                   "hold it for a fixed number of bars. Vary hold_bars to find "
                   "where the signal actually pays.")
    params = [
        Param("threshold", 0.25, "Entry threshold", "float", 0.0, 1.0, 0.05,
              help="Absolute sentiment needed to trigger a position"),
        Param("hold_bars", 5, "Holding period (bars)", "int", 1, 30, 1,
              help="Research suggests the news edge lives around 3-10 days"),
        Param("min_articles", 1, "Minimum articles that day", "int", 1, 20, 1,
              help="Filters out single-story noise"),
        Param("feature_index", 0, "Sentiment feature", "int", 0, 3, 1,
              help="0 confidence-weighted, 1 plain mean, 2 EWM-3, 3 EWM-10"),
        Param("trade_negative", False, "Short on negative news", "bool"),
    ]

    def _feature_name(self) -> str:
        return SENTIMENT_FEATURES[int(self.feature_index) % len(SENTIMENT_FEATURES)]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sentiment = _require(df, self._feature_name()).to_numpy()
        counts = (df["news_count"].astype(float).fillna(0.0).to_numpy()
                  if "news_count" in df.columns else np.ones(len(df)))

        hold = max(int(self.hold_bars), 1)
        threshold = float(self.threshold)
        min_articles = int(self.min_articles)

        signals = np.zeros(len(df))
        remaining, direction = 0, 0.0

        for i in range(len(df)):
            triggered = 0.0
            if counts[i] >= min_articles:
                if sentiment[i] >= threshold:
                    triggered = 1.0
                elif self.trade_negative and sentiment[i] <= -threshold:
                    triggered = -1.0

            if triggered != 0.0:
                # a fresh signal restarts the clock, and can flip the side
                direction, remaining = triggered, hold
            elif remaining > 0:
                remaining -= 1
                if remaining == 0:
                    direction = 0.0

            signals[i] = direction

        return pd.Series(signals, index=df.index)

    def indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {}


@register
class NewsDrift(Strategy):
    name = "News drift"
    requires_features = ["news_sentiment_ewm3", "news_sentiment_ewm10"]
    description = ("Hold while accumulated news sentiment stays positive, exit "
                   "when it fades. Trades the narrative rather than the headline.")
    params = [
        Param("entry", 0.15, "Entry level", "float", 0.0, 1.0, 0.05),
        Param("exit_level", 0.0, "Exit level", "float", -0.5, 0.5, 0.05,
              help="Exits when accumulated sentiment falls back to this"),
        Param("halflife", 0, "Decay window", "int", 0, 1, 1,
              help="0 uses the 3-day EWM feature, 1 uses the 10-day"),
        Param("trade_negative", False, "Short on negative drift", "bool"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        column = "news_sentiment_ewm3" if int(self.halflife) == 0 else "news_sentiment_ewm10"
        drift = _require(df, column).to_numpy()

        entry, exit_level = float(self.entry), float(self.exit_level)
        signals = np.zeros(len(df))
        position = 0.0

        for i in range(len(df)):
            if position > 0 and drift[i] <= exit_level:
                position = 0.0
            elif position < 0 and drift[i] >= -exit_level:
                position = 0.0

            if position == 0.0:
                if drift[i] >= entry:
                    position = 1.0
                elif self.trade_negative and drift[i] <= -entry:
                    position = -1.0

            signals[i] = position

        return pd.Series(signals, index=df.index)


@register
class NewsEventDrift(Strategy):
    name = "News event drift"
    requires_features = ["news_sentiment_weighted", "event_earnings"]
    description = ("Only act on a chosen event category -- earnings, guidance, "
                   "analyst actions and so on -- then hold for a fixed window.")
    params = [
        Param("event_index", 0, "Event category", "int", 0, 9, 1,
              help="0 earnings, 1 guidance, 2 analyst, 3 m&a, 4 legal, "
                   "5 regulatory, 6 management, 7 capital, 8 product, 9 macro"),
        Param("threshold", 0.15, "Sentiment threshold", "float", 0.0, 1.0, 0.05),
        Param("hold_bars", 5, "Holding period (bars)", "int", 1, 30, 1),
        Param("trade_negative", False, "Short on negative events", "bool"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        from core.newsfeatures import EVENT_NAMES

        event = EVENT_NAMES[int(self.event_index) % len(EVENT_NAMES)]
        present = _require(df, f"event_{event}").to_numpy()
        sentiment = _require(df, "news_sentiment_weighted").to_numpy()

        hold = max(int(self.hold_bars), 1)
        threshold = float(self.threshold)
        signals = np.zeros(len(df))
        remaining, direction = 0, 0.0

        for i in range(len(df)):
            triggered = 0.0
            if present[i] > 0:
                if sentiment[i] >= threshold:
                    triggered = 1.0
                elif self.trade_negative and sentiment[i] <= -threshold:
                    triggered = -1.0

            if triggered != 0.0:
                direction, remaining = triggered, hold
            elif remaining > 0:
                remaining -= 1
                if remaining == 0:
                    direction = 0.0

            signals[i] = direction

        return pd.Series(signals, index=df.index)
