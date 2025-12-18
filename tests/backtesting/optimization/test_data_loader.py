"""
Unit tests for optimization data_loader module.
"""

import os
import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.backtesting.optimization.data_loader import (
    get_default_workers,
    load_minute_cache_polars,
    load_daily_data_polars,
    load_daily_data_pandas
)


class TestGetDefaultWorkers:
    """Tests for get_default_workers function."""

    def test_returns_positive_integer(self):
        """Should return a positive integer."""
        workers = get_default_workers()
        assert isinstance(workers, int)
        assert workers >= 1

    def test_returns_80_percent_of_cpu(self):
        """Should return approximately 80% of CPU count."""
        cpu_count = os.cpu_count() or 1
        expected = max(1, int(cpu_count * 0.8))
        workers = get_default_workers()
        assert workers == expected

    def test_minimum_is_one(self):
        """Should never return less than 1."""
        workers = get_default_workers()
        assert workers >= 1


class TestLoadMinuteCachePolars:
    """Tests for load_minute_cache_polars function."""

    @pytest.fixture
    def sample_minute_parquet(self, tmp_path):
        """Create a sample minute data parquet file."""
        import polars as pl
        from datetime import datetime, timedelta

        # Create sample data for 2 symbols, 2 days, 3 minutes each
        rows = []
        base_time = datetime(2024, 1, 2, 9, 30)

        for symbol in ['AAPL', 'MSFT']:
            for day_offset in range(2):
                day_time = base_time + timedelta(days=day_offset)
                for minute in range(3):
                    ts = day_time + timedelta(minutes=minute)
                    rows.append({
                        'timestamp': ts,
                        'symbol': symbol,
                        'open': 100.0 + minute,
                        'high': 101.0 + minute,
                        'low': 99.0 + minute,
                        'close': 100.5 + minute,
                        'volume': 1000.0
                    })

        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col('timestamp').cast(pl.Datetime('us', 'UTC')))

        parquet_path = tmp_path / 'minute_cache.parquet'
        df.write_parquet(parquet_path)
        return parquet_path

    def test_loads_all_symbols(self, sample_minute_parquet):
        """Should load all symbols from parquet."""
        cache = load_minute_cache_polars(str(sample_minute_parquet))
        assert 'AAPL' in cache
        assert 'MSFT' in cache
        assert len(cache) == 2

    def test_loads_all_dates(self, sample_minute_parquet):
        """Should load all dates for each symbol."""
        cache = load_minute_cache_polars(str(sample_minute_parquet))
        assert len(cache['AAPL']) == 2  # 2 days
        assert len(cache['MSFT']) == 2

    def test_returns_numpy_arrays(self, sample_minute_parquet):
        """Should return numpy arrays for price data."""
        cache = load_minute_cache_polars(str(sample_minute_parquet))

        # Get first symbol's first date's data
        first_symbol = list(cache.keys())[0]
        first_date = list(cache[first_symbol].keys())[0]
        day_data = cache[first_symbol][first_date]

        assert 'high' in day_data
        assert 'low' in day_data
        assert 'close' in day_data
        assert isinstance(day_data['high'], np.ndarray)
        assert isinstance(day_data['low'], np.ndarray)
        assert isinstance(day_data['close'], np.ndarray)

    def test_correct_data_values(self, sample_minute_parquet):
        """Should have correct data values."""
        cache = load_minute_cache_polars(str(sample_minute_parquet))

        first_symbol = list(cache.keys())[0]
        first_date = list(cache[first_symbol].keys())[0]
        day_data = cache[first_symbol][first_date]

        # Should have 3 minutes of data per day
        assert len(day_data['high']) == 3
        assert len(day_data['low']) == 3
        assert len(day_data['close']) == 3

    def test_date_filter_start(self, sample_minute_parquet):
        """Should filter by start date."""
        cache = load_minute_cache_polars(
            str(sample_minute_parquet),
            start_date='2024-01-03'
        )
        # Only second day should be included
        first_symbol = list(cache.keys())[0]
        assert len(cache[first_symbol]) == 1

    def test_date_filter_end(self, sample_minute_parquet):
        """Should filter by end date."""
        cache = load_minute_cache_polars(
            str(sample_minute_parquet),
            end_date='2024-01-02'
        )
        # Only first day should be included
        first_symbol = list(cache.keys())[0]
        assert len(cache[first_symbol]) == 1


class TestLoadDailyData:
    """Tests for daily data loading functions."""

    @pytest.fixture
    def sample_daily_parquet(self, tmp_path):
        """Create a sample daily data parquet file."""
        import polars as pl
        from datetime import date

        rows = []
        for symbol in ['SPY', 'QQQ']:
            for day_offset in range(5):
                rows.append({
                    'date': date(2024, 1, 2 + day_offset),
                    'symbol': symbol,
                    'close': 400.0 + day_offset,
                    'signal': 1 if day_offset % 2 == 0 else -1
                })

        df = pl.DataFrame(rows)
        parquet_path = tmp_path / 'daily_data.parquet'
        df.write_parquet(parquet_path)
        return parquet_path

    def test_load_daily_polars_returns_dataframe(self, sample_daily_parquet):
        """Should return a Polars DataFrame."""
        import polars as pl
        df = load_daily_data_polars(str(sample_daily_parquet))
        assert isinstance(df, pl.DataFrame)

    def test_load_daily_polars_correct_shape(self, sample_daily_parquet):
        """Should have correct number of rows."""
        df = load_daily_data_polars(str(sample_daily_parquet))
        assert len(df) == 10  # 2 symbols x 5 days

    def test_load_daily_pandas_returns_dataframe(self, sample_daily_parquet):
        """Should return a Pandas DataFrame."""
        import pandas as pd
        df = load_daily_data_pandas(str(sample_daily_parquet))
        assert isinstance(df, pd.DataFrame)

    def test_load_daily_pandas_correct_shape(self, sample_daily_parquet):
        """Should have correct number of rows."""
        df = load_daily_data_pandas(str(sample_daily_parquet))
        assert len(df) == 10  # 2 symbols x 5 days
