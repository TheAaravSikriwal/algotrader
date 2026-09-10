"""GDELT: machine-coded world events as daily macro features.

Alpaca's news is company headlines. This is the other half -- a global event
stream where each record says *who did what to whom*, coded into the CAMEO
taxonomy of 300-odd event types, with a conflict/cooperation classification and
a tone score. It is the closest freely available thing to "a war started, and
here is its measured intensity, by country, by day".

**Point-in-time**: verified, not assumed. Each daily file was checked for
events dated after its own filename and contains none -- a file published on
day D describes day D and earlier, never later. Articles published today about
older events keep their original event date, which is correct: the *knowledge*
arrived today, and `publication_lag` exposes how much of a day's file is
backdated so a strategy can require fresh events only.

Cost warning: each daily file is roughly 7 MB compressed and 120,000 events.
Only the aggregated row survives -- a few hundred bytes per day -- so the
download happens once and every later run reads the cache.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "http://data.gdeltproject.org/events"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "gdelt"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "algotrader-research/0.1"

# column offsets in the GDELT 1.0 export schema (58 columns, tab separated)
COL_DATE = 1
COL_ACTOR1_COUNTRY = 7
COL_ACTOR2_COUNTRY = 17
COL_EVENT_ROOT = 28
COL_QUAD = 29
COL_GOLDSTEIN = 30
COL_MENTIONS = 31
COL_SOURCES = 32
COL_ARTICLES = 33
COL_TONE = 34

QUAD_CLASS = {
    1: "verbal_cooperation",
    2: "material_cooperation",
    3: "verbal_conflict",
    4: "material_conflict",
}

# CAMEO root codes worth tracking on their own. The high numbers are the ones
# that move oil, defence and risk appetite.
ROOT_EVENTS = {
    "10": "demand",
    "13": "threaten",
    "14": "protest",
    "15": "force_posture",
    "16": "reduce_relations",
    "17": "coerce",
    "18": "assault",
    "19": "fight",
    "20": "mass_violence",
}


class GdeltError(RuntimeError):
    pass


def _cache_path(day: date) -> Path:
    return CACHE_DIR / f"{day:%Y%m%d}.json"


def _download(day: date, timeout: int = 180) -> list[list[str]]:
    url = f"{BASE_URL}/{day:%Y%m%d}.export.CSV.zip"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        blob = urllib.request.urlopen(request, timeout=timeout).read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise GdeltError(f"no GDELT file for {day:%Y-%m-%d}") from exc
        raise GdeltError(f"GDELT returned HTTP {exc.code} for {day:%Y-%m-%d}") from exc
    except Exception as exc:  # noqa: BLE001
        raise GdeltError(f"could not fetch GDELT for {day:%Y-%m-%d}: {exc}") from exc

    archive = zipfile.ZipFile(io.BytesIO(blob))
    text = archive.read(archive.namelist()[0]).decode("utf-8", "replace")
    return [line.split("\t") for line in text.splitlines() if line]


def _aggregate(rows: list[list[str]], day: date,
               countries: tuple[str, ...] = ()) -> dict:
    """Collapse ~120,000 events into one row of daily features."""
    total = 0
    tone_sum = weight_sum = goldstein_sum = 0.0
    quad_counts = {v: 0 for v in QUAD_CLASS.values()}
    root_counts = {v: 0 for v in ROOT_EVENTS.values()}
    country_conflict = {c.upper(): 0 for c in countries}
    same_day = 0

    stamp = f"{day:%Y%m%d}"

    for row in rows:
        if len(row) <= COL_TONE:
            continue
        try:
            articles = float(row[COL_ARTICLES] or 0)
            tone = float(row[COL_TONE] or 0)
            goldstein = float(row[COL_GOLDSTEIN] or 0)
            quad = int(row[COL_QUAD] or 0)
        except ValueError:
            continue

        total += 1
        if row[COL_DATE] == stamp:
            same_day += 1

        weight = max(articles, 1.0)          # louder events count for more
        tone_sum += tone * weight
        goldstein_sum += goldstein * weight
        weight_sum += weight

        label = QUAD_CLASS.get(quad)
        if label:
            quad_counts[label] += 1

        root = ROOT_EVENTS.get(row[COL_EVENT_ROOT])
        if root:
            root_counts[root] += 1

        if country_conflict and quad == 4:
            for col in (COL_ACTOR1_COUNTRY, COL_ACTOR2_COUNTRY):
                code = (row[col] or "").upper()
                if code in country_conflict:
                    country_conflict[code] += 1

    if total == 0:
        raise GdeltError(f"GDELT file for {day:%Y-%m-%d} had no usable rows")

    features = {
        "date": f"{day:%Y-%m-%d}",
        "gdelt_events": float(total),
        "gdelt_tone": tone_sum / weight_sum if weight_sum else 0.0,
        "gdelt_goldstein": goldstein_sum / weight_sum if weight_sum else 0.0,
        # how much of today's file is genuinely new, versus backdated coverage
        "gdelt_publication_lag": 1.0 - (same_day / total),
    }
    for label, count in quad_counts.items():
        features[f"gdelt_{label}"] = count / total
    for label, count in root_counts.items():
        features[f"gdelt_{label}"] = count / total
    for code, count in country_conflict.items():
        features[f"gdelt_conflict_{code.lower()}"] = float(count)
    return features


def day_features(day: date, countries: tuple[str, ...] = (),
                 use_cache: bool = True) -> dict:
    """Aggregated features for one day, downloading only when uncached."""
    path = _cache_path(day)
    if use_cache and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            wanted = {f"gdelt_conflict_{c.lower()}" for c in countries}
            if wanted <= set(cached):
                return cached
        except json.JSONDecodeError:
            pass

    features = _aggregate(_download(day), day, countries)
    if use_cache:
        path.write_text(json.dumps(features), encoding="utf-8")
    return features


def load_features(start, end, countries: tuple[str, ...] = (),
                  progress=None, skip_errors: bool = True) -> pd.DataFrame:
    """Daily GDELT features across a date range, indexed by date.

    Weekends are included -- world events do not stop for the market, and the
    alignment step rolls them onto the next session.
    """
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    if start > end:
        raise ValueError("start must not be after end")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    rows, failures = [], 0

    for i, day in enumerate(days):
        try:
            rows.append(day_features(day, countries))
        except GdeltError:
            failures += 1
            if not skip_errors:
                raise
        if progress:
            progress(i + 1, len(days), day)

    if not rows:
        raise GdeltError("no GDELT data could be loaded for that range")

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame.attrs["failed_days"] = failures
    return frame


def add_regime_features(frame: pd.DataFrame, windows=(7, 30)) -> pd.DataFrame:
    """Rolling context: today's conflict against its own recent history.

    The level of global conflict is far less informative than the *change* --
    the world always has wars, so what matters is whether this week is worse
    than the last month. All windows are trailing, so nothing looks forward.
    """
    out = frame.copy()
    if out.empty:
        return out

    for column in ("gdelt_material_conflict", "gdelt_tone", "gdelt_goldstein"):
        if column not in out:
            continue
        for window in windows:
            rolling = out[column].rolling(window, min_periods=max(window // 2, 2))
            out[f"{column}_ma{window}"] = rolling.mean()
            sd = rolling.std(ddof=0)
            out[f"{column}_z{window}"] = (
                (out[column] - rolling.mean()) / sd.replace(0, np.nan)).fillna(0.0)
    return out


def cache_status() -> dict:
    files = sorted(CACHE_DIR.glob("*.json"))
    if not files:
        return {"days_cached": 0, "first": None, "last": None, "bytes": 0}
    return {
        "days_cached": len(files),
        "first": files[0].stem,
        "last": files[-1].stem,
        "bytes": sum(f.stat().st_size for f in files),
    }


def clear_cache() -> int:
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)


# ---------------------------------------------------------------------------
# attaching to price bars
# ---------------------------------------------------------------------------
AVAILABILITY_LAG_DAYS = 1
"""Days between an event date and the file that reports it becoming public.

Measured, not assumed: probing the archive shows today's daily file returns 404
while yesterday's is present, so the file covering day D lands on D+1. Using
day D's features to decide at day D's close would therefore trade on a file
that did not exist yet -- the exact leak this project keeps hunting for. The
engine then adds its own one-bar execution delay on top, so an event on day D
first affects a fill at the open of D+2.
"""


def attach_to_bars(features: pd.DataFrame, bars: pd.DataFrame,
                   availability_lag_days: int = AVAILABILITY_LAG_DAYS,
                   carry_forward: int = 5) -> pd.DataFrame:
    """Map daily GDELT features onto a bar index, honouring publication lag.

    `carry_forward` keeps the last known reading alive across gaps, which is
    right for a slow-moving macro measure: the level of world conflict on a
    Tuesday is still the best estimate on Wednesday if Wednesday is missing.
    """
    from .newsfeatures import align_to_bars

    if features.empty:
        return pd.DataFrame(0.0, index=bars.index, columns=features.columns)

    shifted = features.copy()
    shifted.index = pd.DatetimeIndex(shifted.index) + pd.Timedelta(
        days=int(availability_lag_days))

    aligned = align_to_bars(shifted, bars.index)
    # align_to_bars sums into bars; a macro level should persist, not reset
    live = aligned.replace(0.0, np.nan)
    if carry_forward > 0:
        live = live.ffill(limit=carry_forward)
    return live.fillna(0.0)
