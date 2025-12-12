"""Tests for data_converter module."""

from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.adapters.nautilus.config import DataConversionConfig
from src.backtesting.adapters.nautilus.data_converter import (
    load_homeguard_bars,
    NAUTILUS_AVAILABLE,
)


class TestLoadHomeguardBars:
    """Tests for loading Homeguard parquet data."""

    def test_load_existing_data(self, sample_homeguard_parquet):
        """Test loading data from Hive-partitioned structure."""
        df = load_homeguard_bars(
            source_dir=sample_homeguard_parquet,
            symbol="TEST",
            timeframe="1min",
        )

        assert not df.empty
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "timestamp" in df.columns

    def test_load_nonexistent_symbol_raises(self, sample_homeguard_parquet):
        """Test loading non-existent symbol raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_homeguard_bars(
                source_dir=sample_homeguard_parquet,
                symbol="NONEXISTENT",
                timeframe="1min",
            )

    def test_load_with_date_filter(self, sample_homeguard_parquet, sample_dataframe):
        """Test loading with date filters."""
        # The sample data is for 2024-01-02
        df = load_homeguard_bars(
            source_dir=sample_homeguard_parquet,
            symbol="TEST",
            timeframe="1min",
            start_date="2024-01-02",
            end_date="2024-01-03",
        )

        assert not df.empty

    def test_load_with_restrictive_filter_empty(self, sample_homeguard_parquet):
        """Test loading with restrictive filter returns empty or filtered data."""
        # Filter for a date that doesn't exist in sample data
        df = load_homeguard_bars(
            source_dir=sample_homeguard_parquet,
            symbol="TEST",
            timeframe="1min",
            start_date="2025-01-01",  # Future date
            end_date="2025-12-31",
        )

        # Should be empty or very small
        assert len(df) < 100  # Original sample has 100 rows


class TestDataConversionConfig:
    """Tests for DataConversionConfig."""

    def test_config_defaults(self, tmp_path):
        """Test config with default values."""
        config = DataConversionConfig(
            source_dir=tmp_path,
            symbols=["AAPL", "MSFT"],
        )

        assert config.catalog_path == tmp_path / "nautilus_catalog"
        assert config.timeframes == ["1min", "1day"]
        assert config.venue == "XNAS"

    def test_config_custom_catalog_path(self, tmp_path):
        """Test config with custom catalog path."""
        custom_path = tmp_path / "custom_catalog"
        config = DataConversionConfig(
            source_dir=tmp_path,
            symbols=["AAPL"],
            catalog_path=custom_path,
        )

        assert config.catalog_path == custom_path

    def test_config_validates_empty_symbols(self, tmp_path):
        """Test config raises on empty symbols."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DataConversionConfig(
                source_dir=tmp_path,
                symbols=[],
            )

    def test_config_validates_source_dir(self, tmp_path):
        """Test config raises on non-existent source_dir."""
        with pytest.raises(ValueError, match="does not exist"):
            DataConversionConfig(
                source_dir=tmp_path / "nonexistent",
                symbols=["AAPL"],
            )

    def test_config_validates_timeframes(self, tmp_path):
        """Test config raises on invalid timeframes."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            DataConversionConfig(
                source_dir=tmp_path,
                symbols=["AAPL"],
                timeframes=["invalid"],
            )

    def test_config_validates_date_format(self, tmp_path):
        """Test config raises on invalid date format."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            DataConversionConfig(
                source_dir=tmp_path,
                symbols=["AAPL"],
                start_date="01-02-2024",  # Wrong format
            )

    def test_get_source_pattern(self, tmp_path):
        """Test source pattern generation."""
        config = DataConversionConfig(
            source_dir=tmp_path,
            symbols=["AAPL"],
        )

        pattern = config.get_source_pattern("AAPL", "1min")
        assert "equities_1min" in pattern
        assert "symbol=AAPL" in pattern
        assert "data.parquet" in pattern


class TestTimestampConversion:
    """Tests for timestamp handling."""

    def test_timestamps_are_utc(self, sample_homeguard_parquet):
        """Test loaded timestamps are UTC."""
        df = load_homeguard_bars(
            source_dir=sample_homeguard_parquet,
            symbol="TEST",
            timeframe="1min",
        )

        # Check timestamp column is UTC
        if df["timestamp"].dt.tz is not None:
            assert str(df["timestamp"].dt.tz) == "UTC"


@pytest.mark.skipif(not NAUTILUS_AVAILABLE, reason="NautilusTrader not installed")
class TestNautilusConversion:
    """Tests requiring NautilusTrader to be installed."""

    def test_create_equity_instrument(self):
        """Test creating NautilusTrader Equity instrument."""
        from src.backtesting.adapters.nautilus.data_converter import create_equity_instrument

        instrument = create_equity_instrument("AAPL", venue="XNAS")

        assert instrument is not None
        assert "AAPL" in str(instrument.id)

    def test_create_bar_type(self):
        """Test creating NautilusTrader BarType."""
        from src.backtesting.adapters.nautilus.data_converter import create_bar_type

        bar_type = create_bar_type("AAPL", "1min", "XNAS")

        assert bar_type is not None

    def test_end_to_end_conversion(self, sample_homeguard_parquet, tmp_path):
        """Test full conversion pipeline."""
        from src.backtesting.adapters.nautilus.data_converter import homeguard_to_nautilus_catalog

        config = DataConversionConfig(
            source_dir=sample_homeguard_parquet,
            symbols=["TEST"],
            catalog_path=tmp_path / "test_catalog",
            timeframes=["1min"],
        )

        catalog = homeguard_to_nautilus_catalog(config)

        assert catalog is not None
        assert (tmp_path / "test_catalog").exists()
