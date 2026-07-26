"""AUTHORITATIVE US macro release timestamps (CPI, NFP), UTC and DST-correct.

Distinct from `macro_calendar_tier1`, which expands approximate recurring rules
for a day-level skip filter and says so. This module carries the real published
schedule fetched from FRED (scripts/data/fetch_us_release_dates.py) and is the
one to use for event-time work, where a date that is off by a day yields a
non-event that still books trades.

Both releases go out at 08:30 America/New_York. That convention is verified
against our own 1-minute FX data rather than assumed: EURUSD |return| peaks at
exactly 13:30 UTC on winter release dates and 12:30 UTC on summer ones, at
25-30x the day's median minute. See tests/data/test_macro_calendar_us.py.
"""
from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

import pandas as pd
import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "macro_calendar"
EVENT_TYPES = ("cpi", "nfp", "fomc")


@functools.lru_cache(maxsize=None)
def _load(event_type: str) -> tuple[tuple[dt.date, ...], str, str]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type!r}. Choices: {EVENT_TYPES}")
    path = _CONFIG_DIR / f"{event_type}_actual.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it with scripts/data/fetch_us_release_dates.py")
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    dates = tuple(dt.date.fromisoformat(d) for d in doc["dates"])
    return dates, doc["release_time_local"], doc["release_timezone"]


def release_dates(event_type: str) -> list[dt.date]:
    return list(_load(event_type)[0])


def release_timestamps(event_type: str, start: dt.date | None = None,
                       end: dt.date | None = None) -> pd.DatetimeIndex:
    """UTC release instants for `event_type`, DST-correct, half-open on [start, end)."""
    dates, local_time, tz = _load(event_type)
    hh, mm = (int(x) for x in local_time.split(":"))
    selected = [d for d in dates
                if (start is None or d >= start) and (end is None or d < end)]
    if not selected:
        return pd.DatetimeIndex([], tz="UTC")
    naive = pd.DatetimeIndex([dt.datetime(d.year, d.month, d.day, hh, mm)
                              for d in selected])
    return naive.tz_localize(tz).tz_convert("UTC")


def all_release_timestamps(start: dt.date | None = None,
                           end: dt.date | None = None) -> pd.DataFrame:
    """Every known US release as (timestamp_utc, event_type), sorted."""
    frames = [pd.DataFrame({"timestamp_utc": release_timestamps(e, start, end),
                            "event_type": e})
              for e in EVENT_TYPES]
    return (pd.concat(frames, ignore_index=True)
              .sort_values("timestamp_utc", ignore_index=True))
