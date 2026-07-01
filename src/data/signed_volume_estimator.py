"""Adaptation A imbalance signal: tick-rule signed volume from 1-minute bars.

Replaces the deferred-permanently trades-schema-based MBP-1 signal.
Captures roughly 70-80% of the fidelity at zero PAYG cost.

Tick rule: for each bar, sign = sign(close - prior_close). First bar of
the day has no prior so signed_volume = 0. Multiply volume * sign to get
the signed (buyer- vs seller-initiated) volume estimate.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir
from src.data.futures.paths import continuous_1min_dir


def _storage_root() -> Path:
    return get_local_storage_dir()


def estimate_signed_volume_from_bars(symbol: str, d: date) -> pl.DataFrame:
    """Estimate signed volume per minute using the tick rule on bar closes.

    Returns the original DataFrame with added columns:
        tick_sign  -- {-1, 0, +1}
        signed_volume -- volume * tick_sign (cast to Int64 to allow negatives)

    Returns an empty DataFrame if no data file exists for the requested
    (symbol, date).
    """
    sym_dir = continuous_1min_dir() / f"symbol={symbol}"
    f = sym_dir / f"year={d.year}" / f"month={d.month}" / "data.parquet"
    if not f.exists():
        return pl.DataFrame()
    df = pl.read_parquet(f)
    df = df.filter(pl.col("timestamp").dt.date() == d).sort("timestamp")
    if df.is_empty():
        return df
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1))
        .sign()
        .fill_null(0)
        .cast(pl.Int8)
        .alias("tick_sign")
    )
    df = df.with_columns(
        (pl.col("volume").cast(pl.Int64) * pl.col("tick_sign"))
        .alias("signed_volume")
    )
    return df
