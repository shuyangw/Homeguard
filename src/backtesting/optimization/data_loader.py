"""
Data loading utilities for optimization runs.

Provides efficient data loading using Polars for large datasets (minute data)
and pandas-compatible outputs for existing backtest functions.
"""

import os
from typing import Dict, Optional, Any
import numpy as np

from src.utils import logger


def get_default_workers() -> int:
    """Return 80% of available CPU threads.

    Returns:
        Number of worker threads to use (minimum 1).
    """
    cpu_count = os.cpu_count() or 1
    return max(1, int(cpu_count * 0.8))


def load_minute_cache_polars(
    cache_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load minute cache using Polars (10-20x faster than pandas).

    Args:
        cache_path: Path to parquet cache file.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).

    Returns:
        Nested dict: {symbol: {date_str: {'high': array, 'low': array, 'close': array}}}
        Compatible with existing backtest functions like ramp_minute_parallel.py.

    Note:
        Uses Polars partition_by which returns tuple keys like ('AAPL',).
        This function handles the tuple extraction automatically.
    """
    import polars as pl

    logger.info(f"Loading minute cache from {cache_path}...")
    df = pl.read_parquet(cache_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Add date string column for grouping and filtering
    df = df.with_columns([
        pl.col('timestamp').dt.strftime('%Y-%m-%d').alias('date_str')
    ])

    # Filter by date range using date strings (avoids timezone issues)
    if start_date:
        df = df.filter(pl.col('date_str') >= start_date)
    if end_date:
        df = df.filter(pl.col('date_str') <= end_date)

    df = df.sort(['symbol', 'date_str', 'timestamp'])

    # Partition by symbol - CRITICAL: keys are tuples like ('AAPL',)!
    symbol_dfs = df.partition_by('symbol', as_dict=True, maintain_order=True)

    minute_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for i, (symbol_key, symbol_df) in enumerate(symbol_dfs.items()):
        # Extract string from tuple key
        symbol = symbol_key[0] if isinstance(symbol_key, tuple) else symbol_key

        minute_cache[symbol] = {}

        # Partition by date within symbol
        date_groups = symbol_df.partition_by('date_str', as_dict=True, maintain_order=True)

        for date_key, date_df in date_groups.items():
            date_str = date_key[0] if isinstance(date_key, tuple) else date_key
            minute_cache[symbol][date_str] = {
                'high': date_df['high'].to_numpy(),
                'low': date_df['low'].to_numpy(),
                'close': date_df['close'].to_numpy(),
            }

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(symbol_dfs)} symbols")

    logger.info(f"Loaded {len(minute_cache)} symbols into minute cache")
    return minute_cache


def load_daily_data_polars(path: str) -> Any:
    """Load daily data using Polars.

    Args:
        path: Path to parquet file.

    Returns:
        Polars DataFrame. Can be converted to pandas with .to_pandas() if needed.

    Note:
        Daily data (375K rows for 3 years x 500 symbols) is small enough
        that both Polars and pandas load in <1 second.
    """
    import polars as pl
    return pl.read_parquet(path)


def load_daily_data_pandas(path: str) -> Any:
    """Load daily data using pandas.

    Args:
        path: Path to parquet file.

    Returns:
        Pandas DataFrame.
    """
    import pandas as pd
    return pd.read_parquet(path)
