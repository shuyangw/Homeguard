"""FRED rates downloader via pandas-datareader.

Series go to alt_data/fred/{series_id}/daily.parquet with schema:
  date (pl.Date), value (pl.Float64)

Reuses pandas_datareader.data.DataReader (already in requirements.txt).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
from pandas_datareader.data import DataReader

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fred_to_parquet(series: pd.Series, series_id: str, root: Path) -> Path:
    """Write a single FRED series to alt_data/fred/{id}/daily.parquet."""
    out_dir = root / "fred" / series_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "daily.parquet"

    df = pl.DataFrame({
        "date": [d.date() for d in series.index],
        "value": series.to_list(),
    })
    df = df.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
    )
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, out)
    return out


class FREDRatesPlugin:
    """Pull FRED economic data series."""

    def __init__(self, storage_root: Path | None = None) -> None:
        self._root = storage_root if storage_root is not None else (get_local_storage_dir() / "alt_data")

    def fetch_series(
        self,
        series_id: str,
        start: date,
        end: date,
        *,
        skip_existing: bool = True,
    ) -> dict:
        out = self._root / "fred" / series_id / "daily.parquet"
        if skip_existing and out.exists():
            logger.info(f"[skip-existing] {series_id}")
            return {"series_id": series_id, "skipped": True, "rows": 0,
                    "out_path": str(out)}
        try:
            df = DataReader(series_id, "fred", start, end)
        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
            return {"series_id": series_id, "skipped": False, "rows": 0,
                    "error": str(e), "out_path": None}

        series = df[series_id] if series_id in df.columns else df.iloc[:, 0]
        series = series.dropna()
        path = fred_to_parquet(series, series_id, self._root)
        logger.info(f"[wrote] {series_id}: {len(series)} rows -> {path}")
        return {"series_id": series_id, "skipped": False, "rows": len(series),
                "out_path": str(path)}
