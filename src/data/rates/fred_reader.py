"""Point-in-time reader for downloaded FRED series (alt_data/fred/{id}/daily.parquet)."""
from __future__ import annotations

from datetime import date

import polars as pl

from src.settings import get_local_storage_dir


def get_fred_series(series_id: str, d: date) -> float:
    """Value of `series_id` as of the latest date <= d (causal forward-fill).

    Raises FileNotFoundError if the series is not downloaded, ValueError if
    `d` precedes the series' first observation.
    """
    fp = get_local_storage_dir() / "alt_data" / "fred" / series_id / "daily.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"FRED series not downloaded: {fp}")
    df = pl.read_parquet(fp, columns=["date", "value"]).filter(pl.col("date") <= d)
    if df.height == 0:
        raise ValueError(f"no {series_id} observation on or before {d}")
    return float(df.sort("date")["value"][-1])
