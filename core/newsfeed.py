"""Multi-source news ingestion with deduplication.

More sources buys two things, and it is worth being precise about which:

  * **recall** -- events covered by only one outlet stop being invisible, which
    genuinely increases the sample size of an event study;
  * **significance** -- a story carried by forty outlets matters more than one
    carried by two, and coverage breadth is the cleanest measure of that.

It does not buy statistical power directly. Twelve outlets covering the same
announcement is one event reported twelve times, not twelve events.

Which makes `deduplicate` the load-bearing part of this module. Without it a
"news intensity" feature measures how many feeds you subscribed to rather than
how important the news was, and every weight derived from it inherits that
error invisibly. Dedup collapses each cluster to one item and records how many
outlets carried it.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

DEFAULT_TIMEOUT = 25
USER_AGENT = "algotrader-research/0.1"

# Some publishers -- the SEC most strictly -- require a contact address in the
# user agent and will return 403 without one. Read from the environment rather
# than hardcoded: sending an address to a third party is the user's call, not
# something this module should decide on their behalf.
CONTACT = os.getenv("RESEARCH_CONTACT", "").strip()

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "at",
    "by", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "will", "has", "have",
    "after", "amid", "over", "into", "up", "down", "new", "says", "said",
}
TOKEN_RE = re.compile(r"[a-z0-9']+")
# outlet suffixes publishers bolt onto headlines, which defeat exact matching
SUFFIX_RE = re.compile(
    r"\s*[-|–—]\s*(reuters|bloomberg|cnbc|marketwatch|barron'?s|yahoo finance|"
    r"investing\.com|the motley fool|benzinga|seeking alpha|zacks)\s*$", re.I)


class FeedError(RuntimeError):
    pass


@dataclass
class FeedItem:
    title: str
    published: datetime                  # tz-aware UTC
    source: str
    url: str = ""
    summary: str = ""
    symbols: list = field(default_factory=list)
    source_count: int = 1                # outlets carrying this story
    outlets: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()

    def to_row(self) -> dict:
        return {
            "timestamp": pd.Timestamp(self.published),
            "title": self.title, "summary": self.summary, "text": self.text,
            "source": self.source, "url": self.url,
            "symbols": ",".join(self.symbols), "source_count": self.source_count,
        }


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def _get(url: str, contact_required: bool = False, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    agent = USER_AGENT
    if contact_required:
        if not CONTACT:
            raise FeedError(
                "This source requires a contact address in the request header "
                "and refuses requests without one. Set RESEARCH_CONTACT in "
                ".env to an address you are willing to share with it -- it is "
                "sent to that publisher on every request.")
        agent = f"{USER_AGENT} {CONTACT}"

    request = urllib.request.Request(url, headers={"User-Agent": agent})
    try:
        return urllib.request.urlopen(request, timeout=timeout).read()
    except urllib.error.HTTPError as exc:
        raise FeedError(f"HTTP {exc.code} from {url}") from exc
    except Exception as exc:  # noqa: BLE001
        raise FeedError(f"could not reach {url}: {exc}") from exc


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").to_pydatetime()


def _strip_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_feed(blob: bytes, source: str) -> list[FeedItem]:
    """Parse RSS or Atom without a third-party dependency."""
    try:
        root = ET.fromstring(blob)
    except ET.ParseError as exc:
        raise FeedError(f"{source}: malformed feed ({exc})") from exc

    items = []
    for node in root.iter():
        if _strip_tag(node.tag) not in ("item", "entry"):
            continue

        fields: dict[str, str] = {}
        link = ""
        for child in node:
            name = _strip_tag(child.tag)
            if name == "link":
                link = (child.get("href") or child.text or "").strip()
            elif name in ("title", "description", "summary", "content",
                          "pubdate", "published", "updated"):
                fields.setdefault(name, (child.text or "").strip())

        title = fields.get("title", "")
        if not title:
            continue

        published = (_parse_time(fields.get("pubdate", ""))
                     or _parse_time(fields.get("published", ""))
                     or _parse_time(fields.get("updated", "")))
        if published is None:
            continue

        summary = re.sub(r"<[^>]+>", " ",
                         fields.get("description") or fields.get("summary")
                         or fields.get("content") or "")
        items.append(FeedItem(title=title, published=published, source=source,
                              url=link, summary=" ".join(summary.split())[:600]))
    return items


class NewsSource(ABC):
    name: str = "source"
    needs_contact: bool = False

    @abstractmethod
    def fetch(self, symbols: list[str] | None = None) -> list[FeedItem]: ...


class RssSource(NewsSource):
    """A fixed feed, or a per-symbol template containing {symbol}."""

    def __init__(self, name: str, url: str, per_symbol: bool = False,
                 needs_contact: bool = False):
        self.name = name
        self.url = url
        self.per_symbol = per_symbol
        self.needs_contact = needs_contact

    def fetch(self, symbols: list[str] | None = None) -> list[FeedItem]:
        if not self.per_symbol:
            return parse_feed(_get(self.url, self.needs_contact), self.name)

        out = []
        for symbol in (symbols or []):
            try:
                items = parse_feed(
                    _get(self.url.format(symbol=symbol.upper()), self.needs_contact),
                    self.name)
            except FeedError:
                continue
            for item in items:
                item.symbols = [symbol.upper()]
            out.extend(items)
        return out


FEEDS: dict[str, NewsSource] = {
    "yahoo": RssSource(
        "yahoo",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
        "&region=US&lang=en-US", per_symbol=True),
    "cnbc": RssSource(
        "cnbc",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=15839069"),
    "marketwatch": RssSource(
        "marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    "investing": RssSource("investing", "https://www.investing.com/rss/news.rss"),
    "fed": RssSource("fed", "https://www.federalreserve.gov/feeds/press_all.xml"),
}


def available_feeds() -> list[str]:
    return sorted(FEEDS)


def fetch_all(feeds: list[str] | None = None, symbols: list[str] | None = None,
              on_error=None) -> list[FeedItem]:
    """Pull every requested feed, skipping any that fail."""
    chosen = feeds or list(FEEDS)
    collected = []
    for key in chosen:
        source = FEEDS.get(key)
        if source is None:
            continue
        try:
            collected.extend(source.fetch(symbols))
        except FeedError as exc:
            if on_error:
                on_error(key, exc)
    return collected


# ---------------------------------------------------------------------------
# deduplication -- the part that makes multi-source worth having
# ---------------------------------------------------------------------------
def normalise_title(title: str) -> str:
    return SUFFIX_RE.sub("", (title or "").strip()).lower()


def title_tokens(title: str) -> frozenset:
    """Content words from a headline.

    Short tokens are dropped as noise, except digits: "Fed cuts 50bps" and
    "Fed cuts 25bps" are different events, and discarding the number would
    merge them. Two-letter tickers survive for the same reason.
    """
    words = TOKEN_RE.findall(normalise_title(title))
    return frozenset(w for w in words
                     if w not in STOPWORDS and (len(w) > 2 or w.isdigit()))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def same_numbers(a: frozenset, b: frozenset) -> bool:
    """Do two headlines quote the same figures?

    In a financial headline the number usually is the substance. "Fed cuts
    rates 50 bps" and "Fed cuts rates 25 bps" share every other word and score
    0.67 on token overlap -- comfortably inside any threshold loose enough to
    merge genuine rewordings of one story. Text similarity cannot separate
    those two cases; the figures can.
    """
    numbers_a = {t for t in a if t.isdigit()}
    numbers_b = {t for t in b if t.isdigit()}
    if not numbers_a or not numbers_b:
        return True          # nothing to contradict
    return bool(numbers_a & numbers_b)


def deduplicate(items: list[FeedItem], threshold: float = 0.6,
                window_hours: int = 36) -> list[FeedItem]:
    """Collapse the same story from many outlets into one weighted item.

    Compares only within a time window, because unrelated stories months apart
    can share a headline shape ("Apple beats estimates") and merging those
    would erase real events rather than duplicates.
    """
    if not items:
        return []

    ordered = sorted(items, key=lambda i: i.published)
    tokens = [title_tokens(i.title) for i in ordered]
    window = pd.Timedelta(hours=window_hours)

    clusters: list[dict] = []
    for i, item in enumerate(ordered):
        placed = False
        for cluster in reversed(clusters):
            if item.published - cluster["latest"] > window:
                break                      # ordered by time, so no earlier one fits
            if (jaccard(tokens[i], cluster["tokens"]) >= threshold
                    and same_numbers(tokens[i], cluster["tokens"])):
                cluster["members"].append(item)
                cluster["latest"] = max(cluster["latest"], item.published)
                cluster["tokens"] = cluster["tokens"] | tokens[i]
                placed = True
                break
        if not placed:
            clusters.append({"members": [item], "tokens": tokens[i],
                             "latest": item.published})

    merged = []
    for cluster in clusters:
        members = cluster["members"]
        first = min(members, key=lambda m: m.published)   # earliest wins
        outlets = sorted({m.source for m in members})
        symbols = sorted({s for m in members for s in m.symbols})
        merged.append(FeedItem(
            title=first.title, published=first.published, source=first.source,
            url=first.url, summary=first.summary or next(
                (m.summary for m in members if m.summary), ""),
            symbols=symbols, source_count=len(outlets), outlets=outlets))

    return sorted(merged, key=lambda i: (-i.source_count, i.published))


def to_frame(items: list[FeedItem]) -> pd.DataFrame:
    columns = ["timestamp", "title", "summary", "text", "source", "url",
               "symbols", "source_count"]
    if not items:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([i.to_row() for i in items], columns=columns)
    return frame.sort_values("timestamp").reset_index(drop=True)
