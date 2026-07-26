"""Fetch AUTHORITATIVE US macro release dates from FRED into a committed config.

Why: the existing calendars are rule-generated and unusable for event-time work.
Measured against the real schedule over 2011-2026, `cpi.yaml` ("10th of each
month") is exact in 14% of months with a mean error of 3.9 days, and `nfp.yaml`
("first Friday") is exact in 83% with 24 months off by exactly one week. A +-7d
blackout tolerates that; an entry at T+2 minutes does not, and the failure is
silent -- a wrong date yields a non-event that still books trades.

Source: the FRED releases API, which publishes the actual release calendar.
BLS's own schedule pages return HTTP 403 to automated fetches. FRED is already
this repo's rates source and the key is in .env.

Duplicate months are resolved STRUCTURALLY, never by looking at market outcomes
(that would bias the resolved months toward larger moves, contaminating exactly
the event study this calendar exists to serve):
  - CPI publishes an annual seasonal-adjustment revision in February, 2 days
    before the main release, in all 12 sample years. Take the LATER February date.
  - Employment Situation is the first release of its month; later same-month
    dates are revisions. Take the EARLIEST.

Times are NOT stored. Both releases are 08:30 America/New_York by long-standing
convention, which `src/data/macro_calendar_us.py` applies DST-correctly. That
convention is verified against our own 1-minute FX data, not assumed: see
tests/data/test_macro_calendar_us.py.

Usage: PYTHONPATH=$(pwd) python scripts/data/fetch_us_release_dates.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

from src.utils import logger

_OUT_DIR = Path("config/macro_calendar")
_START, _END = "2011-01-01", "2026-12-31"
RELEASES = {"cpi": (10, "Consumer Price Index"),
            "nfp": (50, "Employment Situation")}


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        m = re.search(r"FRED_API_KEY\s*=\s*(\S+)", env.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    raise RuntimeError("FRED_API_KEY not found in environment or .env")


def fetch_release_dates(release_id: int) -> list[dt.date]:
    url = (f"https://api.stlouisfed.org/fred/release/dates?release_id={release_id}"
           f"&api_key={_api_key()}&file_type=json&realtime_start={_START}"
           f"&realtime_end={_END}&limit=10000")
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    return sorted(dt.date.fromisoformat(d["date"]) for d in payload["release_dates"])


def primary_per_month(dates: list[dt.date], keep: str) -> list[dt.date]:
    """One release per month. `keep` is "first" or "last_in_february"."""
    by_month: dict[tuple[int, int], list[dt.date]] = defaultdict(list)
    for d in dates:
        by_month[(d.year, d.month)].append(d)
    out = []
    for (year, month), group in sorted(by_month.items()):
        group.sort()
        if keep == "last_in_february" and month == 2 and len(group) > 1:
            out.append(group[-1])
        else:
            out.append(group[0])
    return out


def _write_fomc() -> None:
    """FOMC statement instants, from the dates already curated in fomc.yaml.

    Scoped to 2013 onward on purpose. The statement has gone out at 14:00 ET
    since 2013, which our own 1-minute data confirms: EURUSD |return| peaks at
    19:00 UTC in EST months and 18:00 UTC in EDT months, 11-28x the day's median
    minute, across 2013-2024.

    2011-2012 is EXCLUDED. In that era the statement moved to 12:30 ET on
    press-conference meetings and stayed at ~14:15 ET otherwise, and we have no
    per-meeting press-conference flag. The data shows exactly that split (peaks
    at 12:31 ET in EDT months against ~13:20 ET in EST months), so a single
    assumed time for those 13 meetings would be wrong about half the time.
    Rather than encode a silently wrong instant, the era is left out.
    """
    src = _OUT_DIR / "fomc.yaml"
    with open(src, "r", encoding="utf-8") as f:
        dates = [dt.date.fromisoformat(d) for d in yaml.safe_load(f)["dates"]]
    dates = [d for d in dates if d >= dt.date(2013, 1, 1)]
    doc = {
        "event_type": "fomc",
        "source": "federalreserve.gov meeting calendar, via config/macro_calendar/fomc.yaml",
        "fetched": dt.date.today().isoformat(),
        "release_time_local": "14:00",
        "release_timezone": "America/New_York",
        "duplicate_month_rule": "first",
        "description": ("FOMC statement dates from 2013 onward, when the 14:00 ET "
                        "statement time became uniform. 2011-2012 excluded: the "
                        "statement alternated between 12:30 and ~14:15 ET by "
                        "meeting type and no press-conference flag is available."),
        "dates": [d.isoformat() for d in dates],
    }
    path = _OUT_DIR / "fomc_actual.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
    logger.success(f"[us_releases] wrote {path}: {len(dates)} dates (2013+ only)")


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (release_id, description) in RELEASES.items():
        raw = fetch_release_dates(release_id)
        keep = "last_in_february" if name == "cpi" else "first"
        dates = primary_per_month(raw, keep)
        doc = {
            "event_type": name,
            "source": f"FRED releases API, release_id={release_id} ({description})",
            "fetched": dt.date.today().isoformat(),
            "release_time_local": "08:30",
            "release_timezone": "America/New_York",
            "duplicate_month_rule": keep,
            "description": (f"AUTHORITATIVE {name.upper()} release dates {_START[:4]}-{_END[:4]}. "
                            "Real published schedule, not a recurring-rule proxy. "
                            "Times applied by src/data/macro_calendar_us.py."),
            "dates": [d.isoformat() for d in dates],
        }
        path = _OUT_DIR / f"{name}_actual.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
        logger.success(f"[us_releases] wrote {path}: {len(dates)} dates "
                       f"({len(raw) - len(dates)} same-month revisions dropped)")
    _write_fomc()


if __name__ == "__main__":
    main()
