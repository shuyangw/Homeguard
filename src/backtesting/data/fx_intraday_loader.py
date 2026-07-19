"""1-minute spot-FX bar loader.

Reads the canonical 8-column 1m parquet cache (tz-aware UTC) for a pair over a
date range. Pure reads; no cleaning beyond sort + dedupe (the 1m cache is
already spike-cleaned upstream).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.utils import logger

_COLS = ["open", "high", "low", "close", "volume"]


def load_fx_1min(pair: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "fx" / "massive" / "1min" / f"symbol={pair}"
    if not base.exists() or not any(base.glob("**/*.parquet")):
        logger.warning(f"[load_fx_1min] no 1m data for {pair}")
        return pd.DataFrame(columns=_COLS)
    df = pl.scan_parquet(base / "**/*.parquet").collect().to_pandas()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    out = pd.DataFrame({c: df[c].astype(float) for c in _COLS})
    out.index = ts
    out.index.name = "timestamp"
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    out = out[(out.index >= lo) & (out.index < hi)]
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out
