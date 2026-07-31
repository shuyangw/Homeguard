"""The snapshot-symmetry guard -- ONE clock for marks, signals and hedges.

Registered rule (handoff spec v2, Section 1.3 rule 1):

    The registered snapshot minute is 15:45:00 ET. Same-day option marks come
    from the snapshot record; every signal derived from 1-minute bars uses bars
    through the snapshot minute INCLUSIVE; the daily hedge executes at the
    snapshot minute. Implemented as a SINGLE guarded function in the feature
    layer -- strategies never slice bars themselves.

The snapshot minute itself is owned by `src.data.options.canonical`
(`SNAPSHOT_TIME_ET`) and is re-exported here, never re-declared.

--------------------------------------------------------------------------
The early-close clamp
--------------------------------------------------------------------------
The vendor PADS early-close sessions out to 16:00 with zero volume and stale
quotes. The effective cutoff is therefore

    min(15:45, last real bar of the session)

For bar-START labelled 1-minute data the last real bar of a 13:00 close is the
13:00 - 1min = 12:59 bar, so a padded 13:00 bar can never become the mark.

NOTE (divergence, reported): `canonical._snapshot_cutoff_expr` clamps to the
close TIME itself and filters `ts <= cutoff`, which on an early-close session
admits exactly one padded bar (the bar labelled at the close minute). This
module is strictly tighter. See the Phase-2a doc.

--------------------------------------------------------------------------
Negative control
--------------------------------------------------------------------------
`verify_snapshot_truncation` is the guard's own falsifier: feed it a frame
truncated one minute late and it raises. Every truncation test in this package
runs it, so no truncation test can pass vacuously.
"""
from __future__ import annotations

from datetime import date as _date
from datetime import time as _time
from typing import Dict, Optional

import pandas as pd

from src.data.options.canonical import EASTERN_TZ, SNAPSHOT_TIME_ET
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: First minute of the regular session (bar-START label).
RTH_OPEN_ET: _time = _time(9, 30)

__all__ = [
    "EASTERN_TZ",
    "RTH_OPEN_ET",
    "SNAPSHOT_TIME_ET",
    "SnapshotLeakError",
    "effective_cutoff_et",
    "session_close_map",
    "to_eastern",
    "truncate_to_snapshot",
    "verify_snapshot_truncation",
]


class SnapshotLeakError(RuntimeError):
    """Raised when a frame carries a bar later than its session's cutoff."""


_SCHEDULE_CACHE: Dict[tuple, Dict[_date, pd.Timestamp]] = {}


def session_close_map(start: _date, end: _date) -> Dict[_date, pd.Timestamp]:
    """Real session close (ET) per NYSE trading day in [start, end]."""
    key = (start, end)
    cached = _SCHEDULE_CACHE.get(key)
    if cached is None:
        from src.backtesting.diagnostics.session_bars import session_close_schedule

        cached = session_close_schedule(str(start), str(end))
        _SCHEDULE_CACHE[key] = cached
    return cached


def effective_cutoff_et(
    session: _date, close_schedule: Optional[Dict[_date, pd.Timestamp]] = None
) -> _time:
    """`min(15:45, last real bar of the session)` in ET.

    With no schedule supplied the registered 15:45 is returned unclamped -- the
    caller is then asserting the session is a normal one.
    """
    if close_schedule is None:
        return SNAPSHOT_TIME_ET
    real_close = close_schedule.get(session)
    if real_close is None:
        return SNAPSHOT_TIME_ET
    last_bar = (real_close - pd.Timedelta(minutes=1)).time()
    return min(SNAPSHOT_TIME_ET, last_bar)


def to_eastern(ts: pd.Series) -> pd.Series:
    """Normalize a timestamp column to America/New_York.

    Naive stamps are US/Eastern WALL CLOCK (verified across both 2024 DST
    transitions in `canonical`), so they are localized, never assumed UTC.
    """
    s = pd.to_datetime(ts)
    if s.dt.tz is None:
        return s.dt.tz_localize(EASTERN_TZ)
    return s.dt.tz_convert(EASTERN_TZ)


def truncate_to_snapshot(
    bars: pd.DataFrame,
    ts_col: str = "timestamp",
    close_schedule: Optional[Dict[_date, pd.Timestamp]] = None,
    rth_only: bool = True,
) -> pd.DataFrame:
    """Keep only bars at or before each session's effective snapshot cutoff.

    This is the ONLY sanctioned place 1-minute bars are sliced by time. It adds
    `ts_et` (America/New_York) and `session_date`, and never imputes, pads or
    forward-fills anything.

    Args:
        bars: 1-minute bars with a timestamp column (tz-aware or ET-naive).
        ts_col: name of that column.
        close_schedule: session -> real close (ET). Derived from the project
            market calendar when omitted, so the early-close clamp always binds.
        rth_only: also drop pre-open bars (< 09:30). Extended-hours bars are not
            part of any registered signal.
    """
    if bars.empty:
        out = bars.copy()
        out["ts_et"] = pd.Series(dtype="datetime64[ns, America/New_York]")
        out["session_date"] = pd.Series(dtype="object")
        return out

    out = bars.copy()
    out["ts_et"] = to_eastern(out[ts_col])
    out["session_date"] = out["ts_et"].dt.date

    sessions = out["session_date"]
    if close_schedule is None:
        close_schedule = session_close_map(sessions.min(), sessions.max())

    cutoff = sessions.map(lambda d: effective_cutoff_et(d, close_schedule))
    keep = out["ts_et"].dt.time <= cutoff
    if rth_only:
        keep &= out["ts_et"].dt.time >= RTH_OPEN_ET

    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.debug(f"[+] snapshot truncation dropped {n_dropped:,} of {len(out):,} bars")
    return out[keep].reset_index(drop=True)


def verify_snapshot_truncation(
    df: pd.DataFrame,
    ts_col: str = "ts_et",
    close_schedule: Optional[Dict[_date, pd.Timestamp]] = None,
) -> None:
    """Raise `SnapshotLeakError` if any row is later than its session's cutoff.

    The guard's falsifier. Run it on anything claiming to be snapshot-truncated.
    """
    if df.empty:
        return
    ts = to_eastern(df[ts_col])
    sessions = ts.dt.date
    if close_schedule is None:
        close_schedule = session_close_map(sessions.min(), sessions.max())
    cutoff = sessions.map(lambda d: effective_cutoff_et(d, close_schedule))
    bad = ts.dt.time > cutoff
    if bool(bad.any()):
        worst = ts[bad].max()
        raise SnapshotLeakError(
            f"[-] snapshot leak: {int(bad.sum()):,} bar(s) after the effective "
            f"cutoff; latest = {worst}. The registered snapshot minute is "
            f"{SNAPSHOT_TIME_ET}, clamped to the real session close."
        )
