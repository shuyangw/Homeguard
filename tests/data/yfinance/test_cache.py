"""
Tests for FundamentalsCache persistent cache.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from src.data.yfinance.cache import FundamentalsCache, CacheEntry
from src.data.yfinance.fundamentals import FundamentalData


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_is_expired_false(self):
        """Test entry is not expired when within TTL."""
        data = FundamentalData(symbol="AAPL", market_cap=3000.0)
        entry = CacheEntry(data=data, timestamp=time.time(), ttl_hours=24)
        assert entry.is_expired is False

    def test_is_expired_true(self):
        """Test entry is expired when past TTL."""
        data = FundamentalData(symbol="AAPL", market_cap=3000.0)
        # Timestamp from 25 hours ago
        entry = CacheEntry(
            data=data,
            timestamp=time.time() - (25 * 3600),
            ttl_hours=24,
        )
        assert entry.is_expired is True

    def test_age_hours(self):
        """Test age calculation in hours."""
        data = FundamentalData(symbol="AAPL")
        # 2 hours ago
        entry = CacheEntry(
            data=data,
            timestamp=time.time() - (2 * 3600),
            ttl_hours=24,
        )
        assert 1.9 < entry.age_hours < 2.1


class TestFundamentalsCache:
    """Tests for FundamentalsCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache with temporary directory."""
        return FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=24)

    @pytest.fixture
    def sample_data(self):
        """Sample FundamentalData for testing."""
        return FundamentalData(
            symbol="AAPL",
            market_cap=3000.0,
            pe_ratio=28.5,
            sector="Technology",
        )

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that init creates cache directory."""
        subdir = os.path.join(temp_cache_dir, "subdir")
        cache = FundamentalsCache(cache_dir=subdir)
        assert os.path.exists(subdir)

    def test_set_and_get(self, cache, sample_data):
        """Test basic set and get."""
        cache.set("AAPL", sample_data)
        result = cache.get("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.market_cap == 3000.0
        assert result.sector == "Technology"

    def test_get_nonexistent(self, cache):
        """Test get returns None for nonexistent symbol."""
        result = cache.get("NONEXISTENT")
        assert result is None

    def test_get_expired(self, temp_cache_dir):
        """Test that expired entries return None."""
        # Create cache with very short TTL
        cache = FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=0.0001)
        data = FundamentalData(symbol="AAPL", market_cap=3000.0)
        cache.set("AAPL", data)

        # Wait for expiration
        time.sleep(0.5)

        result = cache.get("AAPL")
        assert result is None

    def test_set_batch(self, cache):
        """Test batch set operation."""
        data = {
            "AAPL": FundamentalData(symbol="AAPL", market_cap=3000.0),
            "MSFT": FundamentalData(symbol="MSFT", market_cap=2800.0),
            "GOOGL": FundamentalData(symbol="GOOGL", market_cap=1900.0),
        }
        cache.set_batch(data)

        assert cache.get("AAPL").market_cap == 3000.0
        assert cache.get("MSFT").market_cap == 2800.0
        assert cache.get("GOOGL").market_cap == 1900.0

    def test_get_batch(self, cache):
        """Test batch get operation."""
        data = {
            "AAPL": FundamentalData(symbol="AAPL", market_cap=3000.0),
            "MSFT": FundamentalData(symbol="MSFT", market_cap=2800.0),
        }
        cache.set_batch(data)

        result = cache.get_batch(["AAPL", "MSFT", "NONEXISTENT"])

        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        assert "NONEXISTENT" not in result

    def test_clear(self, cache, sample_data):
        """Test clearing cache."""
        cache.set("AAPL", sample_data)
        cache.clear()

        assert cache.get("AAPL") is None
        assert len(cache) == 0

    def test_evict_expired(self, temp_cache_dir):
        """Test evicting expired entries."""
        cache = FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=0.0001)

        data = {
            "AAPL": FundamentalData(symbol="AAPL", market_cap=3000.0),
            "MSFT": FundamentalData(symbol="MSFT", market_cap=2800.0),
        }
        cache.set_batch(data)

        # Wait for expiration
        time.sleep(0.5)

        evicted = cache.evict_expired()
        assert evicted == 2
        assert len(cache) == 0

    def test_contains(self, cache, sample_data):
        """Test __contains__ method."""
        cache.set("AAPL", sample_data)

        assert "AAPL" in cache
        assert "MSFT" not in cache

    def test_len(self, cache):
        """Test __len__ method."""
        assert len(cache) == 0

        cache.set("AAPL", FundamentalData(symbol="AAPL"))
        assert len(cache) == 1

        cache.set("MSFT", FundamentalData(symbol="MSFT"))
        assert len(cache) == 2

    def test_get_cached_symbols(self, cache):
        """Test getting list of cached symbols."""
        cache.set("AAPL", FundamentalData(symbol="AAPL"))
        cache.set("MSFT", FundamentalData(symbol="MSFT"))

        symbols = cache.get_cached_symbols()
        assert set(symbols) == {"AAPL", "MSFT"}

    def test_stats_tracking(self, cache, sample_data):
        """Test hit/miss statistics."""
        # Miss
        cache.get("NONEXISTENT")
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Set and hit
        cache.set("AAPL", sample_data)
        cache.get("AAPL")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_hit_rate(self, cache, sample_data):
        """Test hit rate calculation."""
        cache.set("AAPL", sample_data)

        # 1 hit, 1 miss
        cache.get("AAPL")  # Hit
        cache.get("MSFT")  # Miss

        stats = cache.get_stats()
        assert stats["hit_rate"] == 50.0


class TestCachePersistence:
    """Tests for cache persistence to disk."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_and_load(self, temp_cache_dir):
        """Test that cache persists across instances."""
        # Create cache and add data
        cache1 = FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=24)
        cache1.set("AAPL", FundamentalData(symbol="AAPL", market_cap=3000.0, sector="Technology"))
        cache1.set("MSFT", FundamentalData(symbol="MSFT", market_cap=2800.0))

        # Create new cache instance (should load from disk)
        cache2 = FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=24)

        assert len(cache2) == 2
        assert cache2.get("AAPL").market_cap == 3000.0
        assert cache2.get("AAPL").sector == "Technology"
        assert cache2.get("MSFT").market_cap == 2800.0

    def test_load_skips_expired(self, temp_cache_dir):
        """Test that loading from disk skips expired entries."""
        # Create cache with data
        cache1 = FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=0.0001)
        cache1.set("AAPL", FundamentalData(symbol="AAPL", market_cap=3000.0))

        # Wait for expiration
        time.sleep(0.5)

        # Load should skip expired
        cache2 = FundamentalsCache(cache_dir=temp_cache_dir, ttl_hours=0.0001)
        assert len(cache2) == 0

    def test_clear_removes_file(self, temp_cache_dir):
        """Test that clear removes the cache file."""
        cache = FundamentalsCache(cache_dir=temp_cache_dir)
        cache.set("AAPL", FundamentalData(symbol="AAPL"))

        cache_file = Path(temp_cache_dir) / "fundamentals_cache.parquet"
        assert cache_file.exists()

        cache.clear()
        assert not cache_file.exists()
