"""Tests for FRED rates plugin."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from src.data.acquisition.plugins.fred_rates import (
    FREDRatesPlugin,
    fred_to_parquet,
)


def test_fred_to_parquet_writes_canonical_layout(tmp_path: Path):
    """One series fetched from FRED should land at alt_data/fred/{id}/daily.parquet."""
    series = pd.Series(
        [5.25, 5.30, 5.28], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    series.name = "SOFR"

    out = fred_to_parquet(series, "SOFR", tmp_path)
    assert out.exists()
    assert out == tmp_path / "fred" / "SOFR" / "daily.parquet"
    df = pl.read_parquet(out)
    assert df.columns == ["date", "value"]
    assert df.height == 3
    assert df.dtypes[0] == pl.Date
    assert df.dtypes[1] == pl.Float64


def test_plugin_skips_existing_when_skip_existing_true(tmp_path: Path):
    """If a daily.parquet already exists, skip refetching unless --no-skip-existing."""
    (tmp_path / "fred" / "SOFR").mkdir(parents=True)
    (tmp_path / "fred" / "SOFR" / "daily.parquet").touch()

    plugin = FREDRatesPlugin(storage_root=tmp_path)
    with patch("src.data.acquisition.plugins.fred_rates.DataReader") as mock_dr:
        result = plugin.fetch_series("SOFR", date(2020, 1, 1), date(2026, 1, 1),
                                     skip_existing=True)
    assert result["skipped"] is True
    mock_dr.assert_not_called()


def test_plugin_writes_canonical_when_no_existing(tmp_path: Path):
    """No prior file -> fetch + write."""
    plugin = FREDRatesPlugin(storage_root=tmp_path)
    series = pd.Series(
        [5.25], index=pd.to_datetime(["2024-01-02"]),
    )
    series.name = "SOFR"
    with patch("src.data.acquisition.plugins.fred_rates.DataReader",
               return_value=series.to_frame()):
        result = plugin.fetch_series("SOFR", date(2020, 1, 1), date(2026, 1, 1),
                                     skip_existing=True)
    assert result["skipped"] is False
    assert result["rows"] == 1
    assert (tmp_path / "fred" / "SOFR" / "daily.parquet").exists()
