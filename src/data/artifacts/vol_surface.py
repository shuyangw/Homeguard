from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from src.utils import logger


def hour_of_week(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    return ts.dayofweek * 24 + ts.hour


def build_surface(minute_df: pd.DataFrame) -> pd.DataFrame:
    df = minute_df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["how"] = ts.dt.dayofweek * 24 + ts.dt.hour
    df["abs_ret"] = df["close"].pct_change(fill_method=None).abs()
    g = df.dropna(subset=["abs_ret"]).groupby("how")["abs_ret"]
    med = g.median()
    mad = g.apply(lambda s: float(np.median(np.abs(s - np.median(s)))))
    surf = pd.DataFrame({"hour_of_week": range(168)})
    surf["median_abs_ret"] = surf["hour_of_week"].map(med).fillna(0.0)
    surf["mad"] = surf["hour_of_week"].map(mad).fillna(0.0)
    return surf


class VolSurface(ArtifactBuilder):
    name = "vol_surface"
    output_subdir = "vol_surface"

    def inputs(self) -> list[str]:
        return ["minute"]

    def build(self, start: date, end: date) -> Path:
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        src_root = get_local_storage_dir() / "fx_1min"
        for pair in DailyOhlcCache().target_pairs():
            sym = src_root / f"symbol={pair}"
            if not sym.exists():
                continue
            mdf = pl.scan_parquet(sym / "**/*.parquet").collect().to_pandas()
            surf = build_surface(mdf)
            tmp = out_dir / f"{pair}.parquet.tmp"
            pl.from_pandas(surf).write_parquet(tmp)
            os.replace(tmp, out_dir / f"{pair}.parquet")
        logger.info("[vol_surface] built surfaces")
        return out_dir
