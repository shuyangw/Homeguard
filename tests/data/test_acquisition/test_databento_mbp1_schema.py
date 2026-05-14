"""Tests for MBP-1 schema support in DatabentoFuturesPlugin."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from src.data.acquisition.plugins.databento_futures import (
    DatabentoFuturesPlugin,
    MBP1_CANONICAL_COLUMNS,
)


def test_mbp1_canonical_columns():
    """MBP-1 partition must have exactly these columns."""
    assert MBP1_CANONICAL_COLUMNS == [
        "ts_event", "bid_px", "ask_px", "bid_sz", "ask_sz",
    ]


def test_mbp1_schema_accepted_by_plugin(tmp_path: Path):
    """schema='mbp-1' must not raise; plugin routes to MBP-1 writer."""
    plugin = DatabentoFuturesPlugin.__new__(DatabentoFuturesPlugin)
    assert plugin._is_supported_schema("mbp-1") is True
    assert plugin._is_supported_schema("ohlcv-1m") is True
    assert plugin._is_supported_schema("trades") is True
    assert plugin._is_supported_schema("mbp-10") is False


@patch("src.data.acquisition.plugins.databento_futures.db")
@patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
def test_mbp1_writes_to_futures_mbp1_partition(mock_db, tmp_path: Path):
    """MBP-1 data lands at futures_mbp1/symbol={SYM}/year={Y}/month={M}/data.parquet."""
    df = pd.DataFrame({
        "ts_event": pd.to_datetime([datetime(2024, 6, 3, 14, 0)]),
        "bid_px": [1.085],
        "ask_px": [1.0851],
        "bid_sz": [10],
        "ask_sz": [12],
    })
    plugin = DatabentoFuturesPlugin(output_dir=tmp_path, schema="mbp-1")
    plugin._write_mbp1_partition(df, "6E", 2024, 6, tmp_path)
    out = tmp_path / "futures_mbp1" / "symbol=6E" / "year=2024" / "month=6" / "data.parquet"
    assert out.exists()
    read = pl.read_parquet(out)
    assert read.columns == MBP1_CANONICAL_COLUMNS
