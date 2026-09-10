"""Multi-source feed checks, especially deduplication. No network."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.newsfeed import (FeedItem, deduplicate, jaccard, normalise_title,
                           parse_feed, title_tokens, to_frame)

BASE = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)


def item(title, source, minutes=0, symbols=None, summary=""):
    return FeedItem(title=title, published=BASE + timedelta(minutes=minutes),
                    source=source, symbols=symbols or [], summary=summary)


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Apple beats earnings estimates</title>
<link>https://example.com/a</link>
<description>&lt;p&gt;Strong quarter&lt;/p&gt;</description>
<pubDate>Mon, 08 Sep 2026 14:00:00 GMT</pubDate></item>
<item><title>Fed holds rates steady</title><link>https://example.com/b</link>
<pubDate>Mon, 08 Sep 2026 18:30:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Refinery outage reported</title>
<link href="https://example.com/c"/><summary>Details follow</summary>
<published>2026-09-08T15:00:00Z</published></entry></feed>"""


def test_parses_rss():
    items = parse_feed(RSS, "test")
    assert len(items) == 2
    assert items[0].title == "Apple beats earnings estimates"
    assert items[0].published.tzinfo is not None
    assert "Strong quarter" in items[0].summary
    assert "<p>" not in items[0].summary, "html tags survived"


def test_parses_atom():
    items = parse_feed(ATOM, "test")
    assert len(items) == 1
    assert items[0].url == "https://example.com/c"


def test_items_without_a_date_are_dropped():
    blob = b"""<?xml version="1.0"?><rss><channel>
    <item><title>No date here</title></item></channel></rss>"""
    assert parse_feed(blob, "test") == []


def test_outlet_suffixes_are_stripped():
    assert normalise_title("Apple beats estimates - Reuters") == "apple beats estimates"
    assert normalise_title("Oil surges | CNBC") == "oil surges"


def test_stopwords_do_not_drive_similarity():
    """Two headlines sharing only filler words must not look alike."""
    a = title_tokens("The Fed is in the news")
    b = title_tokens("A refinery was on the coast")
    assert jaccard(a, b) == 0.0, "matched on stopwords alone"


def test_numbers_survive_tokenising():
    """Different magnitudes are different events."""
    a = title_tokens("Fed cuts rates 50 bps")
    b = title_tokens("Fed cuts rates 25 bps")
    assert "50" in a and "25" in b
    assert jaccard(a, b) < 1.0, "dropped the number and merged two rate decisions"


# ---------------------------------------------------------------------------
# the property that makes multi-source safe
# ---------------------------------------------------------------------------
def test_same_story_from_many_outlets_becomes_one_item():
    """Twelve outlets on one announcement is one event, not twelve."""
    titles = [
        ("Apple beats earnings estimates, raises guidance", "reuters", 0),
        ("Apple beats earnings estimates and raises guidance", "cnbc", 20),
        ("Apple beats earnings estimates, raises guidance - Bloomberg", "bloomberg", 45),
        ("Apple raises guidance after beating earnings estimates", "yahoo", 90),
    ]
    merged = deduplicate([item(t, s, m) for t, s, m in titles])
    assert len(merged) == 1, f"expected one cluster, got {len(merged)}"
    assert merged[0].source_count == 4
    assert set(merged[0].outlets) == {"reuters", "cnbc", "bloomberg", "yahoo"}


def test_distinct_stories_stay_separate():
    merged = deduplicate([
        item("Apple beats earnings estimates", "reuters", 0),
        item("Boeing halts 737 production after strike", "cnbc", 10),
        item("Fed holds rates steady", "yahoo", 20),
    ])
    assert len(merged) == 3


def test_cluster_keeps_the_earliest_timestamp():
    """The event happened when the first outlet reported it, not the last."""
    merged = deduplicate([
        item("Boeing halts production after strike action", "cnbc", 120),
        item("Boeing halts production after strike action", "reuters", 0),
    ])
    assert len(merged) == 1
    assert merged[0].published == BASE, "used a later report as the event time"


def test_similar_headlines_far_apart_are_not_merged():
    """Quarterly results share a headline shape. Merging across quarters would
    erase real events rather than duplicates."""
    old = item("Apple beats earnings estimates", "reuters", 0)
    new = FeedItem(title="Apple beats earnings estimates", source="reuters",
                   published=BASE + timedelta(days=90))
    assert len(deduplicate([old, new])) == 2


def test_source_count_survives_into_the_frame():
    merged = deduplicate([
        item("Oil spikes on refinery outage", "cnbc", 0),
        item("Oil spikes on refinery outage", "reuters", 15),
        item("Unrelated market wrap", "yahoo", 30),
    ])
    frame = to_frame(merged)
    assert set(frame["source_count"]) == {2, 1}
    assert list(frame.columns)[:4] == ["timestamp", "title", "summary", "text"]


def test_dedup_is_stable_and_loses_nothing():
    items = [item(f"Story number {i}", "cnbc", i * 5) for i in range(20)]
    merged = deduplicate(items)
    assert len(merged) == 20
    assert sum(m.source_count for m in merged) == 20


def test_symbols_are_unioned_across_a_cluster():
    merged = deduplicate([
        item("Chip export curbs widened", "reuters", 0, symbols=["NVDA"]),
        item("Chip export curbs widened", "yahoo", 10, symbols=["AMD"]),
    ])
    assert merged[0].symbols == ["AMD", "NVDA"]


def test_empty_input_is_handled():
    assert deduplicate([]) == []
    assert to_frame([]).empty


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
