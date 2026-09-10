"""The research packet: what the app hands you to paste into Claude Code.

The loop this serves is deliberately manual. You are the rate limiter, and that
is a feature -- it is hard to run five hundred variations by hand, and running
five hundred variations is the main way people destroy themselves here.

Two rules the packet enforces, both about keeping the reasoning honest:

**Only recent news.** A model asked what a 2022 invasion did to oil already
knows. Restricting the packet to articles published after the model's training
cutoff means it cannot read the answer off its own memory -- it has to reason.

**Events, not reactions.** "Refinery struck in Ras Tanura" is a forecastable
input. "Energy stocks rallied 4% on the strike" is an answer key wearing a
headline's clothes. The packet filters out the second kind, because pasting one
in turns forecasting into transcription and the timestamp cannot save you.

What the packet cannot enforce is what happens next: the overlay that comes
back is journaled at issue time and scored only on bars that did not exist when
it was written. That is what makes the whole loop testable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .overlay import DEFAULT_EXPIRY_DAYS, MAX_SINGLE_TILT, SCHEMA
from .taxonomy import bucket_of, classify

# Headlines that report the market's response rather than the event. Handing
# one of these to a model and asking what happens next is asking it to read.
REACTION_PATTERNS = re.compile(
    r"\b(stocks?|shares?|futures?|markets?|index|indices|nasdaq|s&p|dow)\b[^.]{0,40}"
    r"\b(rall(?:y|ied)|surge[ds]?|jump(?:ed|s)?|plunge[ds]?|tumble[ds]?|slip(?:ped)?|"
    r"clos(?:ed|ing)|gain(?:ed|s)?|fell|fall(?:s|ing)?|sank|soar(?:ed|s)?|"
    r"drop(?:ped|s)?|slid[e]?|edge[ds]?)\b|"
    r"\b(rose|fell|gained|lost|dropped)\s+\d+(\.\d+)?%",
    re.I)


def is_market_reaction(text: str) -> bool:
    return bool(REACTION_PATTERNS.search(text or ""))


@dataclass
class Briefing:
    generated: str
    bucket: str
    symbols: list
    since: str
    articles: pd.DataFrame
    holdings: pd.Series
    context: dict

    def to_markdown(self, max_articles: int = 40) -> str:
        return render_markdown(self, max_articles)


def normalise_articles(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either news frame schema and collapse per-symbol duplication.

    `core.news.to_frame` emits one row per (article, symbol) with `headline`
    and `symbol`; `core.newsfeed.to_frame` emits one row per story with `title`
    and `symbols`. Feeding the first straight through makes a single article
    mentioning fourteen ETFs look like fourteen separate events, which would
    inflate every count the reasoning layer sees.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    if "title" not in out and "headline" in out:
        out["title"] = out["headline"]
    if "title" not in out:
        return pd.DataFrame()
    out["title"] = out["title"].fillna("").astype(str)
    out = out[out["title"].str.strip() != ""]
    if out.empty:
        return out

    if "symbols" not in out:
        out["symbols"] = out["symbol"].astype(str) if "symbol" in out else ""

    stamp = "timestamp" if "timestamp" in out else "session"
    out[stamp] = pd.to_datetime(out[stamp])

    # one row per story, with the symbols it touched unioned back together
    grouped = []
    for (title, _), rows in out.groupby([out["title"], out[stamp].dt.date],
                                        sort=False):
        symbols = sorted({s.strip().upper()
                          for cell in rows["symbols"].astype(str)
                          for s in cell.split(",") if s.strip()})
        record = rows.iloc[0].to_dict()
        record["title"] = title
        record["symbols"] = ",".join(symbols)
        record["source_count"] = int(rows.get(
            "source_count", pd.Series([1] * len(rows))).max())
        grouped.append(record)

    return pd.DataFrame(grouped)


def build_briefing(articles: pd.DataFrame, holdings: pd.Series,
                   bucket: str = "company", symbols: list | None = None,
                   since_days: int = 7, cutoff: str | None = None,
                   context: dict | None = None,
                   drop_reactions: bool = True) -> Briefing:
    """Assemble a packet from scored news and current positions."""
    frame = normalise_articles(articles)

    if not frame.empty:
        stamp = "timestamp" if "timestamp" in frame else "session"
        frame[stamp] = pd.to_datetime(frame[stamp])
        floor = pd.Timestamp(cutoff) if cutoff else (
            pd.Timestamp.now("UTC").tz_convert(None) - pd.Timedelta(days=since_days))
        if frame[stamp].dt.tz is not None:
            frame[stamp] = frame[stamp].dt.tz_localize(None)
        frame = frame[frame[stamp] >= floor]

        text_col = "text" if "text" in frame else "title"
        frame[text_col] = frame[text_col].fillna("").astype(str)
        frame = frame[frame[text_col].str.strip() != ""]
        if frame.empty:
            return Briefing(
                generated=date.today().isoformat(), bucket=bucket,
                symbols=list(symbols or []),
                since=(date.today() - timedelta(days=since_days)).isoformat(),
                articles=frame, holdings=holdings, context=context or {})
        frame["bucket"] = frame[text_col].astype(str).map(bucket_of)
        frame = frame[frame["bucket"].isin({bucket, "both"})]

        if drop_reactions and not frame.empty:
            frame["is_reaction"] = frame[text_col].astype(str).map(is_market_reaction)
            frame = frame[~frame["is_reaction"]]

        sort_keys = [c for c in ("source_count", stamp) if c in frame]
        if sort_keys:
            frame = frame.sort_values(sort_keys, ascending=False)

    return Briefing(
        generated=date.today().isoformat(), bucket=bucket,
        symbols=list(symbols or []), since=str(
            (date.today() - timedelta(days=since_days)).isoformat()),
        articles=frame, holdings=holdings, context=context or {})


PROMPTS = {
    "company": (
        "You are reasoning about company-specific events for a systematic "
        "trading book. For each event below, decide whether it changes the "
        "expected return of that company's stock over the next few weeks, and "
        "by enough to act on.\n\n"
        "Reason from mechanism, not from memory of what these stocks did. "
        "State the causal chain: what the event changes about the business, "
        "over what horizon, and why the market may not have priced it yet. If "
        "the answer is that it is already priced, say so and issue nothing."),
    "macro": (
        "You are reasoning about macro events for a systematic trading book "
        "holding index and sector ETFs. For each event, decide which "
        "instruments it should move and in which direction.\n\n"
        "Reason from transmission mechanism: a rate decision reaches small caps "
        "through funding costs, an oil shock reaches XLE through revenue and "
        "XLY through consumer spending. Name the channel. Where an event is "
        "already fully reflected in prices, say so and issue nothing."),
}

GUARDRAILS = f"""Rules for your response:

* Return ONE json object and nothing else, matching the schema below.
* Every adjustment needs a `reason` stating the causal mechanism, not a
  restatement of the headline.
* `weight` is an additive tilt in [-{MAX_SINGLE_TILT}, {MAX_SINGLE_TILT}].
  Anything larger is clipped.
* `confidence` in [0, 1] scales the tilt. Use it honestly -- a plausible
  mechanism with weak evidence belongs at 0.3, not 0.9.
* `expires_days` is how long the reasoning should stand. Default
  {DEFAULT_EXPIRY_DAYS}. A view with no expiry is not a view.
* Issuing an empty list is a valid and often correct answer. Most news is
  already priced. You are not being graded on finding something.
* Do not reason from what you remember these tickers doing. Reason from the
  mechanism described in the event.
* Where a "what happened last time" section appears, weigh it by its sample
  size. One precedent is an anecdote and should barely move your confidence;
  a dozen consistent ones is a reason to raise it. Precedents that went against
  the tilt are the most informative rows in the table -- say so if you are
  overriding them."""

SCHEMA_EXAMPLE = json.dumps({
    "schema": SCHEMA,
    "bucket": "macro",
    "rationale": "one paragraph on the overall read",
    "adjustments": [{
        "symbol": "XLE",
        "action": "tilt",
        "weight": 0.08,
        "confidence": 0.5,
        "expires_days": DEFAULT_EXPIRY_DAYS,
        "reason": "supply disruption raises crude, which flows to sector revenue",
        "evidence": "headline the claim rests on",
    }],
}, indent=2)


def render_markdown(briefing: Briefing, max_articles: int = 40) -> str:
    lines = [
        f"# Trading briefing — {briefing.bucket} bucket",
        f"Generated {briefing.generated}. Events since {briefing.since}.",
        "",
        PROMPTS.get(briefing.bucket, PROMPTS["company"]),
        "",
        "## Current book",
    ]

    if briefing.holdings is not None and not briefing.holdings.empty:
        held = briefing.holdings[briefing.holdings.abs() > 1e-6]
        if held.empty:
            lines.append("Flat — no positions held.")
        else:
            lines.append("| symbol | weight |")
            lines.append("|---|---|")
            for symbol, weight in held.sort_values(ascending=False).items():
                lines.append(f"| {symbol} | {weight:.1%} |")
    else:
        lines.append("No position data supplied.")

    if briefing.context:
        lines += ["", "## Market context"]
        for key, value in briefing.context.items():
            shown = f"{value:.3f}" if isinstance(value, float) else value
            lines.append(f"- {key.replace('_', ' ')}: {shown}")

    precedents = briefing.context.pop("_precedents", None) if briefing.context else None
    if precedents is not None and not precedents.empty:
        lines += ["", "## What happened last time", "",
                  "Scored outcomes from earlier reasoning on events of this "
                  "shape. Each was written before its outcome existed, so none "
                  "of it is hindsight. Read the count before the averages.", ""]
        lines.append("| issued | symbol | matched on | tilt | abnormal return | right? |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in precedents.head(12).iterrows():
            lines.append(
                f"| {row['issued']} | {row['symbol']} | {row['matched_on'] or '-'} "
                f"| {row['tilt']:+.3f} | {row['abnormal_return']:+.2%} "
                f"| {'yes' if row['correct'] else 'no'} |")

    lines += ["", f"## Events ({min(len(briefing.articles), max_articles)} shown)"]
    if briefing.articles.empty:
        lines.append("No qualifying events in this window. "
                     "Issuing no adjustments is the correct response.")
    else:
        stamp = "timestamp" if "timestamp" in briefing.articles else "session"
        for _, row in briefing.articles.head(max_articles).iterrows():
            when = pd.Timestamp(row[stamp]).date()
            symbols = row.get("symbols", "") or ""
            outlets = row.get("source_count", 1)
            tags = classify(str(row.get("text", row.get("title", ""))))
            labels = ",".join(tags[briefing.bucket] or tags["company"] or ["-"])
            lines.append(
                f"- **{when}** [{symbols}] ({outlets} outlet"
                f"{'s' if outlets != 1 else ''}, {labels}) "
                f"{str(row.get('title', ''))[:180]}")

    lines += ["", "## Respond with", "", GUARDRAILS, "", "```json",
              SCHEMA_EXAMPLE, "```"]
    return "\n".join(lines)
