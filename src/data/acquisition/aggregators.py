"""Aggregation utilities for reconstructing OHLCV bars from raw trade data."""

import numpy as np
import pandas as pd

from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.utils.logger import get_logger

logger = get_logger(__name__)


def trades_to_ohlcv_1m(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Resample raw trades into 1-minute OHLCV bars.

    Args:
        trades_df: DataFrame with columns [timestamp, price, size].
                   Timestamps should be timezone-aware (UTC).

    Returns:
        DataFrame with canonical 8-column OHLCV schema.
        Minutes with no trades are excluded (no forward-fill).
    """
    if trades_df.empty:
        return pd.DataFrame(columns=CANONICAL_OHLCV_SCHEMA)

    df = trades_df.copy()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by timestamp for correct open/close
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Deduplicate exact duplicate rows (same timestamp, price, size)
    df = df.drop_duplicates(subset=["timestamp", "price", "size"])

    # Floor timestamps to 1-minute boundaries
    df["minute"] = df["timestamp"].dt.floor("min")

    # Pre-compute price * size for VWAP
    df["_pv"] = df["price"] * df["size"]

    # Group by minute
    grouped = df.groupby("minute")

    bars = pd.DataFrame({
        "timestamp": grouped["minute"].first(),
        "open": grouped["price"].first(),
        "high": grouped["price"].max(),
        "low": grouped["price"].min(),
        "close": grouped["price"].last(),
        "volume": grouped["size"].sum(),
        "trade_count": grouped["price"].count(),
        "_pv_sum": grouped["_pv"].sum(),
    })

    # VWAP: sum(price*size) / sum(size), NaN if volume is 0
    bars["vwap"] = np.where(
        bars["volume"] > 0,
        bars["_pv_sum"] / bars["volume"],
        np.nan,
    )
    bars = bars.drop(columns=["_pv_sum"])

    # Enforce float64 for all numeric columns
    for col in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        bars[col] = bars[col].astype(np.float64)

    # Sort by timestamp and reset index
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    # Enforce canonical column order
    bars = bars[CANONICAL_OHLCV_SCHEMA]

    return bars
