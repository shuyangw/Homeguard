"""
Unit tests for StreamingDataLoader module.

Tests cover:
- SymbolCache LRU cache behavior
- Single symbol loading
- Symbol iteration (sequential and parallel)
- Multi-symbol synchronized loading
- Optimization cache format compatibility
- Memory estimation and utilities
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.backtesting.engine.streaming_data_loader import (
    StreamingDataLoader,
    SymbolCache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_hive_data(tmp_path):
    """
    Create sample hive-partitioned data for testing.

    Structure: equities_1min/symbol={SYM}/year={YYYY}/month={MM}/data.parquet
    """
    data_dir = tmp_path / "equities_1min"

    symbols = ["AAPL", "MSFT", "GOOGL"]
    base_time = datetime(2024, 1, 2, 9, 30)

    for symbol in symbols:
        for month in [1, 2]:
            # Create partition directory
            partition_dir = data_dir / f"symbol={symbol}" / "year=2024" / f"month={month}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            # Create sample data for this partition
            rows = []
            month_start = datetime(2024, month, 1, 9, 30)

            for day in range(1, 4):  # 3 days
                for minute in range(5):  # 5 minutes per day
                    ts = month_start.replace(day=day) + timedelta(minutes=minute)
                    rows.append({
                        "timestamp": ts,
                        "open": 100.0 + minute + day * 0.1,
                        "high": 101.0 + minute + day * 0.1,
                        "low": 99.0 + minute + day * 0.1,
                        "close": 100.5 + minute + day * 0.1,
                        "volume": 1000.0 * (minute + 1),
                        "trade_count": 100.0 * (minute + 1),
                        "vwap": 100.25 + minute,
                    })

            df = pl.DataFrame(rows)
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            df.write_parquet(partition_dir / "data.parquet")

    return tmp_path


@pytest.fixture
def loader(sample_hive_data):
    """Create StreamingDataLoader with test data."""
    return StreamingDataLoader(
        base_path=sample_hive_data,
        cache_size_mb=100,
        filter_market_days=False,  # Disable for testing
        timeframe="1min",
    )


# =============================================================================
# SymbolCache Tests
# =============================================================================


class TestSymbolCache:
    """Tests for SymbolCache LRU implementation."""

    def test_cache_stores_and_retrieves(self):
        """Should store and retrieve DataFrames."""
        cache = SymbolCache(max_size_mb=100)
        df = pl.DataFrame({"a": [1, 2, 3]})

        cache.put("key1", df)
        result = cache.get("key1")

        assert result is not None
        assert result.equals(df)

    def test_cache_miss_returns_none(self):
        """Should return None for missing keys."""
        cache = SymbolCache(max_size_mb=100)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_evicts_lru(self):
        """Should evict least recently used entries when full."""
        # Create very small cache (1KB)
        cache = SymbolCache(max_size_mb=0.001)

        # Add entries that exceed cache size
        for i in range(10):
            df = pl.DataFrame({"data": list(range(100))})
            cache.put(f"key{i}", df)

        # Cache should have evicted older entries
        stats = cache.stats()
        assert stats["entries"] < 10

    def test_cache_moves_to_end_on_access(self):
        """Should move accessed items to end (most recently used)."""
        cache = SymbolCache(max_size_mb=100)

        df1 = pl.DataFrame({"a": [1]})
        df2 = pl.DataFrame({"b": [2]})

        cache.put("key1", df1)
        cache.put("key2", df2)

        # Access key1 to move it to end
        cache.get("key1")

        # Verify key1 is now more recent than key2
        keys = list(cache._cache.keys())
        assert keys[-1] == "key1"

    def test_cache_stats(self):
        """Should track hit/miss statistics."""
        cache = SymbolCache(max_size_mb=100)
        df = pl.DataFrame({"a": [1, 2, 3]})

        cache.put("key1", df)
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_cache_clear(self):
        """Should clear all entries."""
        cache = SymbolCache(max_size_mb=100)
        cache.put("key1", pl.DataFrame({"a": [1]}))
        cache.put("key2", pl.DataFrame({"b": [2]}))

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.stats()["entries"] == 0


# =============================================================================
# Single Symbol Loading Tests
# =============================================================================


class TestLoadSymbol:
    """Tests for single symbol loading."""

    def test_load_single_symbol(self, loader):
        """Should load data for a single symbol."""
        df = loader.load_symbol("AAPL", "2024-01-01", "2024-01-31")

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()
        assert "timestamp" in df.columns
        assert "close" in df.columns

    def test_load_symbol_date_filtering(self, loader):
        """Should filter data by date range."""
        df = loader.load_symbol("AAPL", "2024-01-01", "2024-01-15")

        timestamps = df["timestamp"].to_list()
        for ts in timestamps:
            assert ts.month == 1

    def test_load_symbol_missing_raises(self, loader):
        """Should raise FileNotFoundError for missing symbol."""
        with pytest.raises(FileNotFoundError):
            loader.load_symbol("INVALID", "2024-01-01", "2024-01-31")

    def test_load_symbol_uses_cache(self, loader):
        """Should use cache on second load."""
        # First load
        df1 = loader.load_symbol("AAPL", "2024-01-01", "2024-01-31")

        # Check cache stats
        stats = loader.get_cache_stats()
        assert stats["misses"] == 1

        # Second load (should hit cache)
        df2 = loader.load_symbol("AAPL", "2024-01-01", "2024-01-31")

        stats = loader.get_cache_stats()
        assert stats["hits"] == 1

    def test_load_symbol_column_selection(self, loader):
        """Should select only specified columns."""
        df = loader.load_symbol(
            "AAPL", "2024-01-01", "2024-01-31", columns=["close", "volume"]
        )

        assert "timestamp" in df.columns  # Always included
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "open" not in df.columns

    def test_load_symbol_lazy(self, loader):
        """Should return LazyFrame for lazy loading."""
        lf = loader.load_symbol_lazy("AAPL", "2024-01-01", "2024-01-31")

        assert isinstance(lf, pl.LazyFrame)

        # Collect to verify
        df = lf.collect()
        assert not df.is_empty()


# =============================================================================
# Symbol Iterator Tests
# =============================================================================


class TestIterSymbols:
    """Tests for symbol iteration."""

    def test_iter_symbols_yields_all(self, loader):
        """Should yield data for all symbols."""
        symbols = ["AAPL", "MSFT"]
        results = list(loader.iter_symbols(symbols, "2024-01-01", "2024-01-31"))

        assert len(results) == 2
        symbols_loaded = [sym for sym, _ in results]
        assert "AAPL" in symbols_loaded
        assert "MSFT" in symbols_loaded

    def test_iter_symbols_returns_tuples(self, loader):
        """Should yield (symbol, dataframe) tuples."""
        for symbol, df in loader.iter_symbols(["AAPL"], "2024-01-01", "2024-01-31"):
            assert isinstance(symbol, str)
            assert isinstance(df, pl.DataFrame)

    def test_iter_symbols_skips_missing(self, loader):
        """Should skip missing symbols with warning."""
        symbols = ["AAPL", "INVALID", "MSFT"]
        results = list(loader.iter_symbols(symbols, "2024-01-01", "2024-01-31"))

        # Should only get 2 results (AAPL and MSFT)
        assert len(results) == 2

    def test_iter_symbols_parallel(self, loader):
        """Should load symbols in parallel."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = list(
            loader.iter_symbols_parallel(
                symbols, "2024-01-01", "2024-01-31", n_workers=2
            )
        )

        assert len(results) == 3

    def test_iter_symbols_parallel_preserves_order(self, loader):
        """Should return results in original symbol order."""
        symbols = ["GOOGL", "AAPL", "MSFT"]
        results = list(
            loader.iter_symbols_parallel(
                symbols, "2024-01-01", "2024-01-31", n_workers=2
            )
        )

        result_symbols = [sym for sym, _ in results]
        assert result_symbols == symbols


# =============================================================================
# Multi-Symbol Synchronized Loading Tests
# =============================================================================


class TestLoadSymbolsSynchronized:
    """Tests for synchronized multi-symbol loading."""

    def test_synchronized_returns_dict(self, loader):
        """Should return dict of DataFrames."""
        result = loader.load_symbols_synchronized(
            ["AAPL", "MSFT"], "2024-01-01", "2024-01-31"
        )

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_synchronized_common_timestamps(self, loader):
        """Should align all symbols to common timestamps."""
        result = loader.load_symbols_synchronized(
            ["AAPL", "MSFT"], "2024-01-01", "2024-01-31"
        )

        aapl_ts = set(result["AAPL"]["timestamp"].to_list())
        msft_ts = set(result["MSFT"]["timestamp"].to_list())

        assert aapl_ts == msft_ts

    def test_panel_format(self, loader):
        """Should create wide-format panel."""
        panel = loader.load_symbols_panel(
            ["AAPL", "MSFT"], "2024-01-01", "2024-01-31", price_column="close"
        )

        assert isinstance(panel, pl.DataFrame)
        assert "timestamp" in panel.columns
        assert "AAPL" in panel.columns
        assert "MSFT" in panel.columns


# =============================================================================
# Optimization Cache Interface Tests
# =============================================================================


class TestLoadMinuteCache:
    """Tests for optimization cache format compatibility."""

    def test_minute_cache_format(self, loader):
        """Should return nested dict with numpy arrays."""
        cache = loader.load_minute_cache(["AAPL"], "2024-01-01", "2024-01-31")

        assert "AAPL" in cache
        assert isinstance(cache["AAPL"], dict)

        # Check first date
        first_date = list(cache["AAPL"].keys())[0]
        day_data = cache["AAPL"][first_date]

        assert "high" in day_data
        assert "low" in day_data
        assert "close" in day_data
        assert isinstance(day_data["high"], np.ndarray)

    def test_minute_cache_all_symbols(self, loader):
        """Should load all requested symbols."""
        symbols = ["AAPL", "MSFT"]
        cache = loader.load_minute_cache(symbols, "2024-01-01", "2024-01-31")

        assert len(cache) == 2
        assert "AAPL" in cache
        assert "MSFT" in cache


# =============================================================================
# Utility Methods Tests
# =============================================================================


class TestUtilities:
    """Tests for utility methods."""

    def test_get_available_symbols(self, loader):
        """Should list all available symbols."""
        symbols = loader.get_available_symbols()

        assert len(symbols) == 3
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

    def test_check_symbols_availability(self, loader):
        """Should identify available and missing symbols."""
        result = loader.check_symbols_availability(["AAPL", "INVALID", "MSFT"])

        assert "AAPL" in result["available"]
        assert "MSFT" in result["available"]
        assert "INVALID" in result["missing"]

    def test_get_symbol_date_range(self, loader):
        """Should return date range for symbol."""
        start, end = loader.get_symbol_date_range("AAPL")

        assert "2024-01" in start
        assert "2024-02" in end

    def test_estimate_memory_mb(self, loader):
        """Should estimate memory usage."""
        estimate = loader.estimate_memory_mb(["AAPL"], "2024-01-01", "2024-01-31")

        assert isinstance(estimate, float)
        assert estimate > 0

    def test_clear_cache(self, loader):
        """Should clear all cached data."""
        # Load to populate cache
        loader.load_symbol("AAPL", "2024-01-01", "2024-01-31")
        assert loader.get_cache_stats()["entries"] > 0

        # Clear
        loader.clear_cache()
        assert loader.get_cache_stats()["entries"] == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_symbol_list(self, loader):
        """Should handle empty symbol list."""
        results = list(loader.iter_symbols([], "2024-01-01", "2024-01-31"))
        assert len(results) == 0

    def test_invalid_date_format(self, loader):
        """Should handle invalid date formats gracefully."""
        with pytest.raises(ValueError):
            loader.load_symbol("AAPL", "invalid-date", "2024-01-31")

    def test_end_before_start(self, loader):
        """Should handle end date before start date."""
        # When end < start, no partitions match, so FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            loader.load_symbol("AAPL", "2024-02-01", "2024-01-01")

    def test_timeframe_validation(self, sample_hive_data):
        """Should validate timeframe parameter."""
        with pytest.raises(ValueError):
            StreamingDataLoader(
                base_path=sample_hive_data,
                timeframe="invalid",
            )


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_cache_access(self, loader):
        """Should handle concurrent cache access."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def load_symbol(symbol):
            return loader.load_symbol(symbol, "2024-01-01", "2024-01-31")

        symbols = ["AAPL", "MSFT", "GOOGL"] * 3  # Load each 3 times

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_symbol, s) for s in symbols]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed
        assert len(results) == 9
        for df in results:
            assert not df.is_empty()
