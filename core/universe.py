"""Choosing what to trade, by what the strategy actually needs.

Different strategy types are constrained by different things, and picking one
universe for all of them guarantees at least one is mismatched:

  * **short-horizon** rules are constrained by cost. High turnover only
    survives on the tightest spreads -- cluster mean-reversion returned +11%
    frictionless and -6% at 5bps purely on churn.
  * **long-horizon** rules need adequate liquidity and a long clean history,
    and can otherwise ignore microstructure.
  * **reasoning-driven** rules need news coverage. A company nobody writes
    about cannot be traded on what is written about it.

The rule that keeps this honest: **screen on tradability, never on past
returns.** Filtering to names that went up is survivorship bias wearing a
screener's clothes. Liquidity, spread and coverage are constraints on whether a
trade is executable at all -- they say nothing about whether it will profit,
which is exactly why they are safe to select on.

The residual bias this cannot fix: the candidate pool itself. Screening a list
of companies that exist today has already excluded everything that went
bankrupt. Only point-in-time index membership fixes that, and it is not free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A starting pool of liquid US names. Deliberately not a performance ranking --
# it is a breadth-first list across sectors, and `screen` does the selecting.
# It is still survivorship-biased: every name here survived to be typed.
CANDIDATE_POOL = [
    # mega cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD",
    "INTC", "CSCO", "QCOM", "TXN", "ADBE", "CRM", "ORCL", "IBM", "NOW", "MU",
    # financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "SCHW", "AXP", "V", "MA",
    # health
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN", "CVS",
    # consumer
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "LOW", "HD",
    # energy and industrials
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY",
    "CAT", "DE", "BA", "GE", "HON", "UPS", "UNP", "LMT", "RTX", "MMM",
    # communication and media
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # broad and sector ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE",
    "XLI", "XLB", "XLU", "XLRE", "XLC", "TLT", "IEF", "GLD", "SLV", "VNQ",
]

METRICS = ["price", "dollar_volume", "amihud", "range_pct", "ann_vol",
           "history_days", "completeness"]


@dataclass
class Profile:
    """Thresholds a symbol must clear to be worth trading with this style."""
    name: str
    rationale: str
    min_price: float = 5.0
    min_dollar_volume: float = 5e6
    max_amihud: float = np.inf          # lower is more liquid
    max_range_pct: float = np.inf       # intraday range as a spread proxy
    min_history_days: int = 250
    min_completeness: float = 0.95
    min_news_per_day: float = 0.0
    top_n: int = 20
    rank_by: str = "dollar_volume"      # descending
    rank_ascending: bool = False
    asset_types: tuple = ()             # empty means any
    min_bucket_share: float = 0.0       # of coverage in `bucket`
    bucket: str = ""                    # "company" or "macro"


PROFILES = {
    "short_term": Profile(
        name="short_term",
        rationale=("Costs bind. Only the deepest, tightest names survive high "
                   "turnover, so this screens hard on liquidity and range."),
        min_price=10.0, min_dollar_volume=2e8, max_range_pct=0.035,
        min_history_days=500, top_n=15),
    "long_term": Profile(
        name="long_term",
        rationale=("Turnover is low, so microstructure barely matters. What "
                   "matters is a long clean history and enough depth to size."),
        min_price=5.0, min_dollar_volume=2e7, min_history_days=1000,
        min_completeness=0.98, top_n=30),
    "company_reasoning": Profile(
        name="company_reasoning",
        rationale=("Single names with enough corporate coverage to reason "
                   "about. Excludes baskets: news 'about SPY' is market "
                   "commentary, not an event happening to a company."),
        min_price=5.0, min_dollar_volume=5e7, min_news_per_day=0.5,
        min_history_days=500, top_n=20, rank_by="company_news_per_day",
        asset_types=("company",), bucket="company", min_bucket_share=0.25),
    "macro_reasoning": Profile(
        name="macro_reasoning",
        rationale=("Index and sector baskets, traded on macro events. These "
                   "are the instruments a rate decision or a war actually "
                   "moves as a whole, and they pair with the GDELT features."),
        min_price=5.0, min_dollar_volume=1e7, min_news_per_day=0.2,
        min_history_days=500, top_n=15, rank_by="dollar_volume",
        asset_types=("etf",)),
}


def metrics_for(df: pd.DataFrame, window: int = 252) -> dict:
    """Tradability measures for one symbol. No forward-looking quantities."""
    if df is None or len(df) < 30:
        return {}

    recent = df.tail(window)
    close = recent["close"]
    returns = close.pct_change()
    dollar_volume = (close * recent["volume"]).median()

    # Amihud illiquidity: price impact per dollar traded -- how far a trade of
    # a given size moves the market against you. Scaled by 1e12 purely so the
    # numbers are readable; mega caps sit near zero at any smaller scale and
    # the column becomes a row of 0.00. Only the ordering carries meaning.
    daily_dollars = (close * recent["volume"]).replace(0, np.nan)
    amihud = float((returns.abs() / daily_dollars).mean() * 1e12)

    return {
        "price": float(close.iloc[-1]),
        "dollar_volume": float(dollar_volume),
        "amihud": amihud if np.isfinite(amihud) else np.inf,
        "range_pct": float(((recent["high"] - recent["low"]) / close).median()),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(252)),
        "history_days": int(len(df)),
        "completeness": float(close.notna().mean()),
    }


def profile_frame(bars: dict[str, pd.DataFrame],
                  news_counts: dict[str, float] | None = None,
                  window: int = 252,
                  news_profiles: dict[str, dict] | None = None,
                  days: int = 90) -> pd.DataFrame:
    """One row per symbol: tradability, asset type, and its news category mix.

    `news_profiles` carries the per-stock breakdown from
    `core.taxonomy.profile_texts`, so the frame answers not just "how much
    coverage" but "coverage of what".
    """
    from .taxonomy import asset_type

    rows = []
    for symbol, df in bars.items():
        stats = metrics_for(df, window)
        if not stats:
            continue
        symbol = symbol.upper()
        stats["symbol"] = symbol
        stats["asset_type"] = asset_type(symbol)
        stats["news_per_day"] = float((news_counts or {}).get(symbol, 0.0))

        profile = (news_profiles or {}).get(symbol, {})
        stats.update({k: v for k, v in profile.items() if k != "articles"})
        for bucket in ("company", "macro"):
            share = float(profile.get(f"{bucket}_share", 0.0))
            both = float(profile.get("both_share", 0.0))
            # a story tagged both counts toward each bucket's usable coverage
            stats[f"{bucket}_news_per_day"] = stats["news_per_day"] * (share + both)
        rows.append(stats)

    if not rows:
        return pd.DataFrame(columns=["symbol", "asset_type", "news_per_day"] + METRICS)
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def screen(frame: pd.DataFrame, profile: Profile | str) -> pd.DataFrame:
    """Apply a profile's thresholds and rank what survives.

    Returns the passing rows with a `rejected_for` column on the full frame so
    a near-miss is visible rather than silently dropped.
    """
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    if frame.empty:
        return frame

    checks = {
        "price": frame["price"] >= prof.min_price,
        "dollar_volume": frame["dollar_volume"] >= prof.min_dollar_volume,
        "amihud": frame["amihud"] <= prof.max_amihud,
        "range": frame["range_pct"] <= prof.max_range_pct,
        "history": frame["history_days"] >= prof.min_history_days,
        "completeness": frame["completeness"] >= prof.min_completeness,
    }
    if prof.min_news_per_day > 0:
        checks["news"] = frame["news_per_day"] >= prof.min_news_per_day
    if prof.asset_types and "asset_type" in frame:
        checks["asset_type"] = frame["asset_type"].isin(prof.asset_types)
    if prof.bucket and prof.min_bucket_share > 0:
        column = f"{prof.bucket}_share"
        both = frame.get("both_share", pd.Series(0.0, index=frame.index))
        usable = frame.get(column, pd.Series(0.0, index=frame.index)) + both
        checks[f"{prof.bucket}_coverage"] = usable >= prof.min_bucket_share

    passing = pd.Series(True, index=frame.index)
    for mask in checks.values():
        passing &= mask.fillna(False)

    reasons = []
    for symbol in frame.index:
        failed = [name for name, mask in checks.items() if not bool(mask.get(symbol, False))]
        reasons.append(",".join(failed))

    annotated = frame.copy()
    annotated["passes"] = passing
    annotated["rejected_for"] = reasons

    survivors = annotated[passing]
    if survivors.empty:
        return survivors

    key = prof.rank_by if prof.rank_by in survivors.columns else "dollar_volume"
    return survivors.sort_values(key, ascending=prof.rank_ascending).head(prof.top_n)


def describe_profile(profile: Profile | str) -> str:
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    bits = [f"price >= ${prof.min_price:,.0f}",
            f"dollar volume >= ${prof.min_dollar_volume:,.0f}",
            f"history >= {prof.min_history_days} bars"]
    if np.isfinite(prof.max_range_pct):
        bits.append(f"daily range <= {prof.max_range_pct:.1%}")
    if prof.min_news_per_day > 0:
        bits.append(f"news >= {prof.min_news_per_day}/day")
    if prof.asset_types:
        bits.append(f"asset type in {', '.join(prof.asset_types)}")
    if prof.bucket and prof.min_bucket_share > 0:
        bits.append(f"{prof.min_bucket_share:.0%} of coverage {prof.bucket}")
    return f"{prof.rationale}\n  Requires: " + "; ".join(bits)
