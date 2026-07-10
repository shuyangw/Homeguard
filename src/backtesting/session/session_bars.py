"""Extract + cache the ratio-adjusted 1-min closes at the session-boundary ET
times, per date, per root.

The intraday session engine only needs the close at a handful of ET times each
day (RTH open/close, FOMC time, hour-slice bounds). Extracting them ONCE per root
into a small parquet avoids re-loading/re-adjusting the multi-million-row 1-min
series on every walk-forward run (the A1 OOM lesson). Ratio-adjusted closes give
roll-clean overnight returns."""
from __future__ import annotations

import os
from datetime import date, time
from pathlib import Path

import pandas as pd

from src.backtesting.sessions.equity_index_clock import (
    et_to_utc, SLICE_START, SLICE_END, RTH_OPEN, RTH_CLOSE, FOMC_TIME)
from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.data.futures.paths import session_bars_dir
from src.utils import logger

SESSION_TIMES: dict[str, time] = {
    "et_0200": SLICE_START, "et_0500": SLICE_END,
    "et_0930": RTH_OPEN, "et_1400": FOMC_TIME, "et_1600": RTH_CLOSE,
}
_TOL = pd.Timedelta(minutes=15)  # accept the first bar within 15 min at/after the target


def extract_from_minute_frame(mf: pd.DataFrame, times_et: dict[str, time]) -> pd.DataFrame:
    """mf: DatetimeIndex (UTC) with a 'close' column. Returns date-indexed closes
    at/after each ET time (first bar within _TOL), NaN if none."""
    et_dates = mf.index.tz_convert("America/New_York").date
    dates = sorted(set(et_dates))
    cols: dict[str, pd.Series] = {}
    for name, t in times_et.items():
        vals: dict[date, float] = {}
        for d in dates:
            target = et_to_utc(d, t)
            window = mf.loc[target: target + _TOL]
            vals[d] = float(window["close"].iloc[0]) if len(window) else float("nan")
        cols[name] = pd.Series(vals)
    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out


def drop_all_nan_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop dates where EVERY session-time close is NaN.

    ES/NQ Globex opens Sunday 18:00 ET, so a Sunday ET date carries bars only
    after 18:00 -- all five SESSION_TIMES (02:00-16:00 ET) precede the open and
    the whole row is NaN. Keeping these all-NaN rows would inject phantom
    "trading days" into the date index, so a strategy's next-index lookup maps
    Friday -> Sunday (not Monday) and skips the Friday-close -> Monday-open
    weekend overnight. Rows with at least one non-NaN close are kept."""
    return df.dropna(how="all")


def extract_session_bars(root: str, start=None, end=None) -> pd.DataFrame:
    df = ContinuousContractDataLoader().load(root, method="ratio_adjusted", start=start, end=end)
    if df.is_empty():
        return pd.DataFrame(columns=list(SESSION_TIMES))
    mf = df.select(["timestamp", "close"]).to_pandas()
    mf["timestamp"] = pd.to_datetime(mf["timestamp"], utc=True)
    mf = mf.set_index("timestamp").sort_index()
    out = extract_from_minute_frame(mf, SESSION_TIMES)
    out = drop_all_nan_dates(out)
    del mf, df  # free the large minute frame
    return out


def build_session_bars_cache(root: str) -> Path:
    out = extract_session_bars(root)
    d = session_bars_dir()
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{root}.parquet"
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    out.reset_index().to_parquet(tmp)
    os.replace(tmp, fp)
    logger.info(f"[session_bars] wrote {len(out)} dates for {root} -> {fp}")
    return fp


def load_session_bars(root: str) -> pd.DataFrame:
    fp = session_bars_dir() / f"{root}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"session-bars cache missing for {root}: {fp} "
                                 "(run build_session_bars_cache)")
    df = pd.read_parquet(fp)
    return df.set_index("date")
