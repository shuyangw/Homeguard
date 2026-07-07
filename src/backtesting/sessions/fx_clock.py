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


def _resolve_window(window: "str | SessionWindow") -> SessionWindow:
    if isinstance(window, SessionWindow):
        return window
    return SESSION_WINDOWS[window]


def session_window_utc(window: "str | SessionWindow",
                       day: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    w = _resolve_window(window)
    end_day = day if w.end > w.start else day + dt.timedelta(days=1)
    start = local_to_utc(w.exchange, dt.datetime.combine(day, w.start))
    end = local_to_utc(w.exchange, dt.datetime.combine(end_day, w.end))
    return start, end


def _seconds_of_day(t: dt.time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def in_session_mask(utc_index: pd.DatetimeIndex,
                    window: "str | SessionWindow") -> pd.Series:
    w = _resolve_window(window)
    local = utc_index.tz_convert(_zone_for(w.exchange))
    sod = local.hour * 3600 + local.minute * 60 + local.second
    s, e = _seconds_of_day(w.start), _seconds_of_day(w.end)
    mask = (sod >= s) & (sod < e) if s <= e else (sod >= s) | (sod < e)
    return pd.Series(mask, index=utc_index)


def _fx_day_index(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Shift NY-local time by +7h so the 17:00-ET boundary becomes midnight, then
    # the local calendar date is the FX trading day. DST handled by tz_convert.
    ny = utc_index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return ny


def fx_trading_day(utc_index: pd.DatetimeIndex) -> pd.Series:
    shifted = _fx_day_index(utc_index)
    return pd.Series(shifted.date, index=utc_index)


def is_friday_fx(utc_index: pd.DatetimeIndex) -> pd.Series:
    shifted = _fx_day_index(utc_index)
    return pd.Series(shifted.dayofweek == 4, index=utc_index)


def fx_trading_week_id(utc_index: pd.DatetimeIndex) -> pd.Series:
    shifted = _fx_day_index(utc_index)
    iso = shifted.isocalendar()
    ids = (iso.year.to_numpy() * 100 + iso.week.to_numpy())
    return pd.Series(ids, index=utc_index)


def hour_of_week_utc(utc_index: pd.DatetimeIndex) -> pd.Series:
    how = utc_index.dayofweek * 24 + utc_index.hour
    return pd.Series(how, index=utc_index)


def hour_of_week_anchored(utc_index: pd.DatetimeIndex,
                          anchor: str = "Europe/London") -> pd.Series:
    local = utc_index.tz_convert(_zone_for(anchor))
    how = local.dayofweek * 24 + local.hour
    return pd.Series(how, index=utc_index)
