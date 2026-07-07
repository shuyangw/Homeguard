"""FX session / DST clock: exchange-local <-> UTC, session masks, FX day/week,
hour-of-week. Pure functions over stdlib zoneinfo, operating on tz-aware UTC
timestamps. Centralizes DST correctness for all intraday FX work.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

EXCHANGE_TZ: dict[str, ZoneInfo] = {
    "TOKYO": ZoneInfo("Asia/Tokyo"),
    "LONDON": ZoneInfo("Europe/London"),
    "NEW_YORK": ZoneInfo("America/New_York"),
}


@dataclass(frozen=True)
class SessionWindow:
    exchange: str
    start: dt.time
    end: dt.time


SESSION_WINDOWS: dict[str, SessionWindow] = {
    "TOKYO": SessionWindow("TOKYO", dt.time(9, 0), dt.time(15, 0)),
    "ASIAN_RANGE": SessionWindow("LONDON", dt.time(0, 0), dt.time(7, 0)),
    "LONDON": SessionWindow("LONDON", dt.time(8, 0), dt.time(16, 30)),
    "NEW_YORK": SessionWindow("NEW_YORK", dt.time(8, 0), dt.time(17, 0)),
    "WMR_FIX_LONDON": SessionWindow("LONDON", dt.time(15, 58), dt.time(16, 2)),
}

_NONEXISTENT = {"roll_forward": "shift_forward", "roll_backward": "shift_backward"}
_AMBIGUOUS = {"first": True, "second": False}


def _zone_for(exchange: str) -> ZoneInfo:
    tz = EXCHANGE_TZ.get(exchange)
    return tz if tz is not None else ZoneInfo(exchange)


def local_to_utc(exchange: str, local_dt: dt.datetime,
                 nonexistent: str = "roll_forward",
                 ambiguous: str = "first") -> pd.Timestamp:
    ts = pd.Timestamp(local_dt).tz_localize(
        _zone_for(exchange),
        ambiguous=_AMBIGUOUS[ambiguous],
        nonexistent=_NONEXISTENT[nonexistent],
    )
    return ts.tz_convert("UTC")
