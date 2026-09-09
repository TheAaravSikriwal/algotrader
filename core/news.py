"""News ingestion.

Alpaca's news API (Benzinga-sourced) reaches back to 2015 at roughly 130
articles a day, which is what makes a news strategy *backtestable* rather than
an article of faith.

The subtle part here is not fetching -- it is deciding which trading session a
story belongs to. A headline printed at 18:40 ET could not have informed a
decision made at that day's close, so it belongs to the next session. Getting
this wrong invents an edge out of nothing, and it is invisible in the results.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MARKET_TZ = "America/New_York"
SESSION_CUTOFF_HOUR = 16          # 16:00 ET -- the regular-session close

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "news"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class NewsError(RuntimeError):
    pass


@dataclass
class NewsItem:
    id: str
    timestamp: datetime               # always tz-aware UTC
    symbols: list[str]
    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    author: str = ""

    @property
    def text(self) -> str:
        """Headline plus summary -- what the scorer reads."""
        return f"{self.headline}. {self.summary}".strip()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NewsItem":
        ts = pd.Timestamp(d["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        return cls(
            id=str(d.get("id", "")), timestamp=ts.to_pydatetime(),
            symbols=[s.upper() for s in d.get("symbols", [])],
            headline=d.get("headline", ""), summary=d.get("summary", ""),
            source=d.get("source", ""), url=d.get("url", ""),
            author=d.get("author", ""),
        )


def trading_session(ts, cutoff_hour: int = SESSION_CUTOFF_HOUR) -> pd.Timestamp:
    """The session date a story can first act on.

    Anything printed at or after the close belongs to the next calendar day;
    weekend news rolls to Monday. The engine then delays execution by one more
    bar, so a story is never traded on a price that preceded it.
    """
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    local = ts.tz_convert(MARKET_TZ)

    session = local.normalize()
    if local.hour >= cutoff_hour:
        session = session + pd.Timedelta(days=1)
    while session.weekday() >= 5:                  # Sat/Sun -> Monday
        session = session + pd.Timedelta(days=1)
    return session.tz_localize(None)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def _cache_path(symbols: list[str], start: str, end: str) -> Path:
    tag = "-".join(sorted(s.upper() for s in symbols))[:60]
    return CACHE_DIR / f"{tag}_{start}_{end}.jsonl"


def _read_cache(path: Path) -> list[NewsItem]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                items.append(NewsItem.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return items


def _write_cache(path: Path, items: list[NewsItem]):
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), default=str) + "\n")


def fetch_alpaca_news(symbols: list[str], start: str, end: str,
                      limit_per_page: int = 50) -> list[NewsItem]:
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise NewsError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set. News comes "
            "from Alpaca; add your keys to .env (free paper keys work)."
        )
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
    except ImportError as exc:
        raise NewsError("alpaca-py is not installed -- pip install alpaca-py") from exc

    client = NewsClient(key, secret)
    out: list[NewsItem] = []
    page_token = None

    while True:
        request = NewsRequest(
            symbols=",".join(s.upper() for s in symbols),
            start=pd.Timestamp(start).to_pydatetime(),
            end=pd.Timestamp(end).to_pydatetime(),
            limit=limit_per_page,
            include_content=True,
            page_token=page_token,
        )
        response = client.get_news(request)
        raw = getattr(response, "data", response)
        batch = raw.get("news", []) if isinstance(raw, dict) else (raw or [])

        for n in batch:
            out.append(NewsItem(
                id=str(getattr(n, "id", "")),
                timestamp=pd.Timestamp(getattr(n, "created_at", None)).to_pydatetime(),
                symbols=[s.upper() for s in (getattr(n, "symbols", []) or [])],
                headline=getattr(n, "headline", "") or "",
                summary=getattr(n, "summary", "") or "",
                source=getattr(n, "source", "") or "",
                url=getattr(n, "url", "") or "",
                author=getattr(n, "author", "") or "",
            ))

        page_token = getattr(response, "next_page_token", None)
        if not page_token or not batch:
            break

    return out


def load_news(symbols: list[str], start, end, source: str = "alpaca",
              use_cache: bool = True) -> list[NewsItem]:
    """Fetch news, caching to disk as JSONL."""
    symbols = [s.upper() for s in symbols]
    start, end = str(start)[:10], str(end)[:10]
    path = _cache_path(symbols, start, end)

    if use_cache:
        cached = _read_cache(path)
        if cached:
            return cached

    if source != "alpaca":
        raise NewsError(f"unknown news source {source!r}")

    items = fetch_alpaca_news(symbols, start, end)
    items.sort(key=lambda n: n.timestamp)
    if use_cache and items:
        _write_cache(path, items)
    return items


def to_frame(items: list[NewsItem]) -> pd.DataFrame:
    """One row per (article, symbol) pair, with its trading session attached."""
    rows = []
    for n in items:
        session = trading_session(n.timestamp)
        for sym in (n.symbols or [""]):
            rows.append({
                "id": n.id, "timestamp": pd.Timestamp(n.timestamp),
                "session": session, "symbol": sym.upper(),
                "headline": n.headline, "summary": n.summary,
                "text": n.text, "source": n.source,
            })
    if not rows:
        return pd.DataFrame(columns=["id", "timestamp", "session", "symbol",
                                     "headline", "summary", "text", "source"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def clear_news_cache() -> int:
    files = list(CACHE_DIR.glob("*.jsonl"))
    for f in files:
        f.unlink()
    return len(files)
