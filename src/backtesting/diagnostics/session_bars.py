"""Session-level marks extracted from 1-minute underlying equity bars.

Source store: H:/Stock_Data/equities/sip_split/1min/symbol=<SYM>/year=<Y>/month=<M>
(month NOT zero-padded). Split-adjusted (`sip_split`) is used by registration.

Everything here is a measurement of the underlying price series. No positions,
no P&L.

Bar-label convention is INFERRED, not assumed -- see `infer_bar_label_convention`.
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

MINUTE_ROOT = Path("H:/Stock_Data/equities/sip_split/1min")
EASTERN = "America/New_York"

OPEN_MINUTE = "09:30"
OPENING_RANGE_END = "09:59"      # inclusive, bar-start labels -> covers 09:30-10:00
CONFIRM_MINUTE = "10:00"
GAP_ENTRY_MINUTE = "10:01"
FIRST_HOUR_END_MINUTE = "10:30"
FH_ENTRY_MINUTE = "10:31"
SNAPSHOT_MINUTE = "15:45"        # registered snapshot (spec v2 sec 1.3 rule 1)
REGULAR_LAST_MINUTE = "15:59"    # last bar-start of a normal 16:00 session

_REQUIRED_MARKS = ("open_0930", "or_high", "or_low", "c_1000", "c_1001",
                   "c_1030", "c_1031", "c_snap", "c_realclose")


def infer_bar_label_convention(ny_index: pd.Series) -> Tuple[str, str]:
    """Infer whether the timestamp labels the START or the END of the bar.

    Evidence: the extended session runs 04:00-20:00 ET. Bar-START labelling
    yields a first bar at 04:00 and a last at 19:59; bar-END labelling yields
    04:01 .. 20:00.
    """
    times = pd.Series(pd.to_datetime(ny_index).dt.strftime("%H:%M"))
    first, last = times.min(), times.max()
    if first == "04:00" and last == "19:59":
        conv = "bar_start"
    elif first == "04:01" and last == "20:00":
        conv = "bar_end"
    else:
        conv = "unknown"
    return conv, f"first_bar={first} last_bar={last}"


def session_close_schedule(start: str, end: str) -> Dict[_date, pd.Timestamp]:
    """Real session close (ET) per trading day, from the project market calendar."""
    from src.backtesting.utils.market_calendar import MarketCalendar

    sched = MarketCalendar().calendar.schedule(start_date=start, end_date=end)
    closes = sched["market_close"].dt.tz_convert(EASTERN)
    return {ts.date(): close for ts, close in zip(sched.index, closes)}


def load_minute_bars(symbol: str, years: Iterable[int]) -> pd.DataFrame:
    """Load 1m bars for `symbol` over `years`, tz-converted to America/New_York."""
    frames = []
    for year in years:
        ydir = MINUTE_ROOT / f"symbol={symbol}" / f"year={year}"
        if not ydir.exists():
            logger.warning(f"[!] missing year dir {ydir}")
            continue
        for month in range(1, 13):
            fp = ydir / f"month={month}" / "data.parquet"
            if not fp.exists():
                continue
            frames.append(pd.read_parquet(
                fp, columns=["timestamp", "open", "high", "low", "close", "volume"]))
    if not frames:
        raise FileNotFoundError(f"no 1m bars found for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    df["ny"] = df["timestamp"].dt.tz_convert(EASTERN)
    df = df.sort_values("ny").drop_duplicates(subset="ny").reset_index(drop=True)
    return df


def build_session_marks(bars: pd.DataFrame,
                        close_schedule: Dict[_date, pd.Timestamp]) -> pd.DataFrame:
    """Reduce 1m bars to one row per regular-hours session.

    Columns:
      open_0930   open of the 09:30 bar (official session open)
      or_high/or_low  high/low over bars 09:30..09:59 (the first 30 minutes)
      c_1000      close of the 10:00 bar (gap confirmation mark)
      c_1001      close of the 10:01 bar (gap-arm entry mark, t+1 convention)
      c_1030      close of the 10:30 bar (first-hour end mark)
      c_1031      close of the 10:31 bar (first-hour-arm entry mark)
      c_snap      close of the registered 15:45 bar; on early closes the last
                  bar at or before the real close
      c_realclose close of the last regular bar (15:59, or early-close equivalent)
      intraday_rv sum of squared 1m log CLOSE-to-CLOSE returns inside the regular
                  session. Close-to-close within the session structurally excludes
                  the overnight gap (the "drop the first bar" requirement).
      quarantined True when any required mark is missing -> row is excluded from
                  measurement but COUNTED and reported.

    No imputation, no forward-fill, no smoothing anywhere.
    """
    df = bars.copy()
    df["session_date"] = df["ny"].dt.date
    df["hhmm"] = df["ny"].dt.strftime("%H:%M")
    reg = df[(df["hhmm"] >= OPEN_MINUTE) & (df["hhmm"] <= REGULAR_LAST_MINUTE)]

    rows = []
    for sess, g in reg.groupby("session_date", sort=True):
        real_close = close_schedule.get(sess)
        if real_close is None:
            continue
        close_hhmm = real_close.strftime("%H:%M")
        is_early = close_hhmm < "16:00"
        last_bar_label = (real_close - pd.Timedelta(minutes=1)).strftime("%H:%M")
        snap_label = last_bar_label if is_early else SNAPSHOT_MINUTE

        g = g[g["hhmm"] <= last_bar_label]
        by = g.set_index("hhmm")
        px = by["close"]

        def at(label, col="close"):
            if label in by.index:
                return float(by.loc[label, col])
            return np.nan

        oR = g[(g["hhmm"] >= OPEN_MINUTE) & (g["hhmm"] <= OPENING_RANGE_END)]
        r = np.diff(np.log(px.to_numpy())) if len(px) > 1 else np.array([])
        row = {
            "session_date": sess,
            "open_0930": at(OPEN_MINUTE, "open"),
            "or_high": float(oR["high"].max()) if len(oR) else np.nan,
            "or_low": float(oR["low"].min()) if len(oR) else np.nan,
            "c_1000": at(CONFIRM_MINUTE),
            "c_1001": at(GAP_ENTRY_MINUTE),
            "c_1030": at(FIRST_HOUR_END_MINUTE),
            "c_1031": at(FH_ENTRY_MINUTE),
            "c_snap": at(snap_label),
            "c_realclose": at(last_bar_label),
            "snapshot_minute": snap_label,
            "is_early_close": bool(is_early),
            "n_regular_bars": int(len(g)),
            "intraday_rv": float((r ** 2).sum()) if len(r) else np.nan,
            "n_rv_returns": int(len(r)),
        }
        row["quarantined"] = bool(
            any(not np.isfinite(row[c]) for c in _REQUIRED_MARKS))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("session_date").reset_index(drop=True)
