"""What a story is about, split into company events and macro events.

Two buckets, because they are different signals that happen to arrive through
the same pipe:

  * **company** news is an event happening *to a firm* -- earnings, a recall, a
    lawsuit. It moves that ticker and its close peers.
  * **macro** news is an event happening *to everyone* -- a rate decision, a
    war, a tariff. It moves the index, and a single stock's exposure to it
    depends on what business it is in.

Mixing them under one "news" label loses the distinction that matters most: a
company-event strategy trading on macro headlines is really just a very
expensive index tracker, and it would look like it was working every time the
market rose.

Classification is regex over the headline, which is crude but has two virtues a
model does not: it is deterministic, so the same article scores the same way in
2020 and 2026, and it cannot be contaminated by knowing what happened next.
"""
from __future__ import annotations

import re

COMPANY_CATEGORIES = {
    "earnings": r"\b(earnings|eps|quarterly results|reports? (?:q[1-4]|results)|beats?|misses?|revenue)\b",
    "guidance": r"\b(guidance|outlook|forecast|raises? (?:its )?(?:full[- ]year|fy)|lowers? (?:its )?(?:full[- ]year|fy)|cuts? (?:its )?outlook)\b",
    "analyst": r"\b(upgrade[sd]?|downgrade[sd]?|price target|initiat(?:es|ed) coverage|reiterat|overweight|underweight|buy rating|sell rating)\b",
    "mna": r"\b(merger|acquisition|acquires?|acquired|takeover|buyout|to buy|stake in|divest|spin[- ]?off|tender offer)\b",
    "legal": r"\b(lawsuit|litigation|settlement|sues?|sued|investigation|probe|subpoena|antitrust|class action)\b",
    "regulatory": r"\b(fda|approval|approved|clearance|regulator|compliance|license|patent|ruling|sec filing)\b",
    "management": r"\b(ceo|cfo|coo|chief executive|chief financial|resign|steps? down|appoints?|names? new|board of directors)\b",
    "capital": r"\b(dividend|buyback|repurchase|offering|secondary|stock split|raises? capital|debt offering|bond sale)\b",
    "product": r"\b(launch(?:es|ed)?|unveil(?:s|ed)?|introduc(?:es|ed)|new product|partnership|contract award|rollout)\b",
    "labor": r"\b(layoff|layoffs|job cuts|strike|union|workforce|hiring freeze|headcount)\b",
    "supply_chain": r"\b(supply chain|shortage|production halt|factory|plant closure|recall|logistics|backlog|inventory)\b",
}

MACRO_CATEGORIES = {
    "monetary": r"\b(federal reserve|fed |fomc|interest rate|rate cut|rate hike|central bank|ecb|bank of japan|basis points|bps)\b",
    "inflation": r"\b(inflation|cpi|ppi|consumer price|producer price|deflation|price pressures)\b",
    "employment": r"\b(jobs report|nonfarm|payrolls|unemployment|jobless claims|labor market|labour market)\b",
    "growth": r"\b(gdp|recession|economic growth|slowdown|contraction|soft landing|expansion)\b",
    "geopolitical": r"\b(war|invasion|missile|airstrike|sanctions|military|ceasefire|nato|conflict|troops|border clash)\b",
    "trade": r"\b(tariff|trade war|export controls|import duties|trade deal|customs|protectionism)\b",
    "energy": r"\b(oil price|crude|opec|natural gas|per barrel|energy prices|refinery|pipeline)\b",
    "fiscal": r"\b(federal budget|deficit|government shutdown|stimulus|debt ceiling|tax bill|spending bill)\b",
    "currency": r"\b(dollar index|greenback|exchange rate|currency market|yuan|yen|euro strengthen)\b",
}

_COMPANY = {k: re.compile(v, re.I) for k, v in COMPANY_CATEGORIES.items()}
_MACRO = {k: re.compile(v, re.I) for k, v in MACRO_CATEGORIES.items()}

# Instruments whose news is market commentary rather than corporate events.
# Reasoning about "SPY news" as though it were a company event is a category
# error -- the underlying is a basket, and the stories are macro.
KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM",
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE",
    "XLC", "TLT", "IEF", "SHY", "AGG", "LQD", "HYG", "GLD", "SLV", "DBC",
    "USO", "UNG", "VNQ", "ARKK", "SMH", "SOXX", "XBI", "IBB", "KRE", "JETS",
}


def asset_type(symbol: str, etfs: set | None = None) -> str:
    """"etf" for baskets, "company" otherwise."""
    return "etf" if symbol.upper() in (etfs or KNOWN_ETFS) else "company"


def classify(text: str) -> dict[str, list[str]]:
    """Every category a story matches, in both buckets.

    A story can be both: "Fed hike hits bank earnings" is monetary *and*
    earnings, and forcing a single label would discard half of that.
    """
    text = text or ""
    return {
        "company": [name for name, rx in _COMPANY.items() if rx.search(text)],
        "macro": [name for name, rx in _MACRO.items() if rx.search(text)],
    }


def bucket_of(text: str) -> str:
    """The dominant bucket: "company", "macro", "both" or "unclassified"."""
    tags = classify(text)
    if tags["company"] and tags["macro"]:
        return "both"
    if tags["company"]:
        return "company"
    if tags["macro"]:
        return "macro"
    return "unclassified"


def profile_texts(texts) -> dict:
    """Per-stock category mix: what this symbol's news is actually about.

    Shares are of total articles, and they do not sum to one -- a story can
    carry several tags, and many carry none.
    """
    texts = [t for t in texts if t]
    total = len(texts)
    profile = {
        "articles": float(total),
        "company_share": 0.0, "macro_share": 0.0,
        "both_share": 0.0, "unclassified_share": 0.0,
    }
    for name in COMPANY_CATEGORIES:
        profile[f"company_{name}"] = 0.0
    for name in MACRO_CATEGORIES:
        profile[f"macro_{name}"] = 0.0
    if not total:
        return profile

    buckets = {"company": 0, "macro": 0, "both": 0, "unclassified": 0}
    counts = {f"company_{k}": 0 for k in COMPANY_CATEGORIES}
    counts.update({f"macro_{k}": 0 for k in MACRO_CATEGORIES})

    for text in texts:
        tags = classify(text)
        for name in tags["company"]:
            counts[f"company_{name}"] += 1
        for name in tags["macro"]:
            counts[f"macro_{name}"] += 1

        if tags["company"] and tags["macro"]:
            buckets["both"] += 1
        elif tags["company"]:
            buckets["company"] += 1
        elif tags["macro"]:
            buckets["macro"] += 1
        else:
            buckets["unclassified"] += 1

    for name, count in buckets.items():
        profile[f"{name}_share"] = count / total
    for name, count in counts.items():
        profile[name] = count / total
    return profile


def top_categories(profile: dict, bucket: str = "company", n: int = 3) -> list[tuple]:
    """The categories this symbol's coverage is actually concentrated in."""
    prefix = f"{bucket}_"
    skip = {f"{bucket}_share"}
    items = [(k[len(prefix):], v) for k, v in profile.items()
             if k.startswith(prefix) and k not in skip and isinstance(v, float)]
    return sorted(items, key=lambda kv: kv[1], reverse=True)[:n]
