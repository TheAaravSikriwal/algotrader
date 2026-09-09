"""Turning headlines into numbers.

Two scorers behind one interface:

* ``LexiconScorer``  -- Loughran-McDonald style financial word lists. Pure
  Python, no dependencies, instant, deterministic. General-purpose sentiment
  tools misread financial language badly ("liability", "aggressive", "cut" read
  neutral or positive in ordinary English and strongly negative in filings),
  which is why finance research leans on domain word lists rather than VADER.
* ``FinBertScorer`` -- a transformer trained on financial text. Better at
  context and negation, needs ``transformers`` + ``torch``, and is several
  orders of magnitude slower.

The lexicon path is the default so the pipeline runs anywhere; FinBERT is an
upgrade you switch on. Scores are cached by content hash either way, because
backtesting ten years of news means scoring each article exactly once.
"""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "scores"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Loughran-McDonald style financial sentiment terms ----------------------
NEGATIVE = {
    "loss", "losses", "losing", "lost", "decline", "declines", "declined",
    "declining", "decrease", "decreased", "decreases", "drop", "drops",
    "dropped", "fall", "falls", "fell", "falling", "weak", "weaker", "weakness",
    "poor", "poorly", "miss", "missed", "misses", "missing", "shortfall",
    "downgrade", "downgraded", "downgrades", "cut", "cuts", "slash", "slashed",
    "plunge", "plunged", "plunges", "slump", "slumped", "tumble", "tumbled",
    "sink", "sank", "sinking", "crash", "crashed", "collapse", "collapsed",
    "bankruptcy", "bankrupt", "insolvency", "default", "defaulted", "delinquent",
    "lawsuit", "lawsuits", "litigation", "sue", "sued", "suing", "settlement",
    "investigation", "investigations", "probe", "probes", "subpoena", "fraud",
    "fraudulent", "misconduct", "violation", "violations", "penalty", "penalties",
    "fine", "fined", "sanction", "sanctions", "recall", "recalls", "recalled",
    "layoff", "layoffs", "restructuring", "impairment", "writedown",
    "deficit", "deficits", "overleveraged", "downturn", "recession",
    "headwind", "headwinds", "pressure", "pressured", "concern", "concerns",
    "risk", "risks", "risky", "uncertain", "uncertainty", "volatile",
    "disappointing", "disappointed", "disappoint", "warn", "warned", "warning",
    "negative", "negatively", "adverse", "unfavorable", "deteriorate",
    "deteriorating", "struggle", "struggling", "struggled", "halt", "halted",
    "suspend", "suspended", "delay", "delayed", "delays", "resign", "resigned",
    "resignation", "departure", "ousted", "terminate", "terminated", "breach",
    "defect", "defective", "failure", "failed", "fails", "failing", "reject",
    "rejected", "denied", "block", "blocked", "underperform", "underperformed",
    "bearish", "selloff", "correction", "bubble",
}

POSITIVE = {
    "gain", "gains", "gained", "gaining", "rise", "rises", "rose", "rising",
    "increase", "increased", "increases", "growth", "grow", "grew", "growing",
    "surge", "surged", "surges", "soar", "soared", "soars", "jump", "jumped",
    "climb", "climbed", "rally", "rallied", "rallies", "advance", "advanced",
    "beat", "beats", "exceeded", "exceed", "exceeds", "outperform",
    "outperformed", "upgrade", "upgraded", "upgrades", "raise", "raised",
    "raises", "boost", "boosted", "boosts", "strong", "stronger", "strength",
    "robust", "solid", "record", "profit", "profits",
    "profitable", "profitability", "margin", "margins",
    "expansion", "expand", "expanded", "expanding", "acquisition", "acquire",
    "acquired", "merger", "partnership", "collaboration", "award",
    "awarded", "win", "wins", "won", "winning", "approval", "approved",
    "approve", "clearance", "cleared", "launch", "launched", "launches",
    "innovation", "breakthrough", "milestone", "success", "successful",
    "successfully", "opportunity", "opportunities", "favorable", "positive",
    "positively", "optimistic", "optimism", "confidence", "confident",
    "improve", "improved", "improving", "improvement", "recovery", "recovered",
    "rebound", "rebounded", "momentum", "bullish", "buyback", "dividend",
    "upside", "attractive", "undervalued", "efficient", "efficiency",
}

NEGATORS = {"not", "no", "never", "without", "fails", "failed", "unable",
            "cannot", "denies", "denied", "avoid", "avoided", "less", "fewer"}

INTENSIFIERS = {"very", "highly", "sharply", "significantly", "substantially",
                "dramatically", "massively", "record", "steeply"}

# --- event taxonomy ---------------------------------------------------------
EVENT_PATTERNS = {
    "earnings": r"\b(earnings|eps|quarterly results|reports? (?:q[1-4]|results)|beats?|misses?)\b",
    "guidance": r"\b(guidance|outlook|forecast|raises? (?:its )?(?:full[- ]year|fy)|lowers? (?:its )?(?:full[- ]year|fy))\b",
    "analyst": r"\b(upgrade[sd]?|downgrade[sd]?|price target|initiat(?:es|ed) coverage|reiterat|overweight|underweight)\b",
    "mna": r"\b(merger|acquisition|acquires?|acquired|takeover|buyout|to buy|stake in|divest|spin[- ]?off)\b",
    "legal": r"\b(lawsuit|litigation|settlement|sues?|sued|investigation|probe|subpoena|antitrust)\b",
    "regulatory": r"\b(fda|approval|approved|clearance|regulator|compliance|license|patent|ruling)\b",
    "management": r"\b(ceo|cfo|chief executive|chief financial|resign|steps? down|appoints?|names? new)\b",
    "capital": r"\b(dividend|buyback|repurchase|offering|secondary|stock split|raises? capital)\b",
    "product": r"\b(launch(?:es|ed)?|unveil(?:s|ed)?|introduc(?:es|ed)|new product|partnership)\b",
    "macro": r"\b(fed|federal reserve|inflation|cpi|interest rate|tariff|jobs report|gdp|recession)\b",
}
COMPILED_EVENTS = {k: re.compile(v, re.I) for k, v in EVENT_PATTERNS.items()}

WORD_RE = re.compile(r"[a-z][a-z\-]+")


def tag_events(text: str) -> list[str]:
    """Which event categories a headline belongs to. Cheap, high precision."""
    return [name for name, rx in COMPILED_EVENTS.items() if rx.search(text or "")]


class Scorer(ABC):
    name = "scorer"

    @abstractmethod
    def score_batch(self, texts: list[str]) -> list[dict]:
        """One dict per text: {'sentiment': -1..1, 'confidence': 0..1}."""

    def score(self, text: str) -> dict:
        return self.score_batch([text])[0]


class LexiconScorer(Scorer):
    """Counts financial-sentiment terms, handling negation and intensifiers."""

    name = "lexicon"

    def __init__(self, window: int = 3):
        self.window = window          # how far back to look for a negator

    def score_batch(self, texts: list[str]) -> list[dict]:
        return [self._score_one(t) for t in texts]

    def _score_one(self, text: str) -> dict:
        words = WORD_RE.findall((text or "").lower())
        if not words:
            return {"sentiment": 0.0, "confidence": 0.0, "hits": 0}

        score, hits = 0.0, 0
        for i, word in enumerate(words):
            polarity = 1.0 if word in POSITIVE else (-1.0 if word in NEGATIVE else 0.0)
            if polarity == 0.0:
                continue
            hits += 1

            context = words[max(0, i - self.window):i]
            if any(w in NEGATORS for w in context):
                polarity = -polarity
            if any(w in INTENSIFIERS for w in context):
                polarity *= 1.5
            score += polarity

        if hits == 0:
            return {"sentiment": 0.0, "confidence": 0.0, "hits": 0}

        sentiment = max(-1.0, min(1.0, score / (hits ** 0.5) / 2.0))
        confidence = min(1.0, hits / 6.0)      # more matched terms -> more signal
        return {"sentiment": round(sentiment, 4),
                "confidence": round(confidence, 4), "hits": hits}


class FinBertScorer(Scorer):
    """ProsusAI/finbert. Better, heavier, optional."""

    name = "finbert"

    def __init__(self, model: str = "ProsusAI/finbert", batch_size: int = 16,
                 device: str = "cpu"):
        try:
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer, pipeline)
        except ImportError as exc:
            raise ImportError(
                "FinBERT needs transformers and torch:\n"
                "  pip install transformers torch --index-url "
                "https://download.pytorch.org/whl/cpu\n"
                "Or use LexiconScorer, which needs nothing."
            ) from exc

        self.batch_size = batch_size
        self._pipe = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            device=-1 if device == "cpu" else 0,
            truncation=True, max_length=512,
        )

    def score_batch(self, texts: list[str]) -> list[dict]:
        clean = [t if (t or "").strip() else "neutral" for t in texts]
        out = []
        for i in range(0, len(clean), self.batch_size):
            for r in self._pipe(clean[i:i + self.batch_size]):
                label = str(r["label"]).lower()
                prob = float(r["score"])
                signed = prob if label.startswith("pos") else (
                    -prob if label.startswith("neg") else 0.0)
                out.append({"sentiment": round(signed, 4),
                            "confidence": round(prob, 4), "hits": 1})
        return out


def get_scorer(name: str = "lexicon", **kwargs) -> Scorer:
    if name == "finbert":
        return FinBertScorer(**kwargs)
    if name == "lexicon":
        return LexiconScorer(**kwargs)
    raise ValueError(f"unknown scorer {name!r} -- use 'lexicon' or 'finbert'")


def _digest(text: str, scorer_name: str) -> str:
    return hashlib.sha1(f"{scorer_name}|{text}".encode("utf-8")).hexdigest()


def score_frame(frame: pd.DataFrame, scorer: Scorer | None = None,
                use_cache: bool = True, text_col: str = "text") -> pd.DataFrame:
    """Attach sentiment, confidence and event tags to a news frame.

    Scores are cached by (scorer, text) hash, so re-running a ten-year backtest
    costs nothing after the first pass -- the difference between an afternoon
    and a week once FinBERT is doing the work.
    """
    scorer = scorer or LexiconScorer()
    out = frame.copy()
    if out.empty:
        out["sentiment"] = pd.Series(dtype="float64")
        out["confidence"] = pd.Series(dtype="float64")
        out["hits"] = pd.Series(dtype="int64")
        out["events"] = pd.Series(dtype="object")
        return out

    cache_file = CACHE_DIR / f"{scorer.name}.json"
    cache = {}
    if use_cache and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    texts = out[text_col].fillna("").astype(str).tolist()
    keys = [_digest(t, scorer.name) for t in texts]

    todo = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in cache]
    if todo:
        fresh = scorer.score_batch([t for _, t in todo])
        for (i, _), result in zip(todo, fresh):
            cache[keys[i]] = result

    results = [cache[k] for k in keys]
    out["sentiment"] = [r["sentiment"] for r in results]
    out["confidence"] = [r["confidence"] for r in results]
    out["hits"] = [r.get("hits", 0) for r in results]
    out["events"] = [tag_events(t) for t in texts]

    if use_cache and todo:
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return out
