"""
Tests for screening cache module.
"""

import time
from threading import Thread
from typing import List

import pytest

from src.screening.cache import (
    CacheEntry,
    CacheStats,
    ScreenerCache,
    hash_config,
)


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_is_expired_false(self):
        """Test entry is not expired when within TTL."""
        entry = CacheEntry(data="test", timestamp=time.time(), ttl_seconds=60)
        assert entry.is_expired is False

    def test_is_expired_true(self):
        """Test entry is expired when past TTL."""
        entry = CacheEntry(data="test", timestamp=time.time() - 100, ttl_seconds=60)
        assert entry.is_expired is True

    def test_age_seconds(self):
        """Test age calculation."""
        entry = CacheEntry(data="test", timestamp=time.time() - 30, ttl_seconds=60)
        assert 29 < entry.age_seconds < 31

    def test_remaining_seconds(self):
        """Test remaining TTL calculation."""
        entry = CacheEntry(data="test", timestamp=time.time(), ttl_seconds=60)
        assert 59 < entry.remaining_seconds <= 60

    def test_remaining_seconds_expired(self):
        """Test remaining returns 0 when expired."""
        entry = CacheEntry(data="test", timestamp=time.time() - 100, ttl_seconds=60)
        assert entry.remaining_seconds == 0


class TestScreenerCache:
    """Tests for ScreenerCache class."""

    def test_init_default_ttls(self):
        """Test default TTL values."""
        cache = ScreenerCache()
        stats = cache.get_detailed_stats()
        assert stats["snapshots"]["ttl_seconds"] == 60
        assert stats["historical"]["ttl_seconds"] == 3600
        assert stats["results"]["ttl_seconds"] == 30
        assert stats["assets"]["ttl_seconds"] == 3600

    def test_init_custom_ttls(self):
        """Test custom TTL values."""
        cache = ScreenerCache(
            snapshot_ttl=120,
            historical_ttl=7200,
            result_ttl=60,
            assets_ttl=1800,
        )
        stats = cache.get_detailed_stats()
        assert stats["snapshots"]["ttl_seconds"] == 120
        assert stats["historical"]["ttl_seconds"] == 7200


class TestSnapshotCache:
    """Tests for snapshot caching."""

    def test_set_and_get_snapshot(self):
        """Test storing and retrieving snapshot."""
        cache = ScreenerCache(snapshot_ttl=60)
        cache.set_snapshot("AAPL", {"price": 175.50})

        result = cache.get_snapshot("AAPL")
        assert result == {"price": 175.50}

    def test_get_nonexistent_snapshot(self):
        """Test getting non-existent snapshot returns None."""
        cache = ScreenerCache()
        result = cache.get_snapshot("NONEXISTENT")
        assert result is None

    def test_snapshot_expiration(self):
        """Test snapshot expires after TTL."""
        cache = ScreenerCache(snapshot_ttl=1)  # 1 second TTL
        cache.set_snapshot("AAPL", {"price": 175.50})

        # Should exist immediately
        assert cache.get_snapshot("AAPL") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get_snapshot("AAPL") is None

    def test_batch_operations(self):
        """Test batch set and get operations."""
        cache = ScreenerCache()
        snapshots = {
            "AAPL": {"price": 175.50},
            "MSFT": {"price": 380.25},
            "GOOGL": {"price": 140.00},
        }
        cache.set_snapshots_batch(snapshots)

        result = cache.get_snapshots_batch(["AAPL", "MSFT", "GOOGL", "NOTFOUND"])
        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result
        assert "NOTFOUND" not in result


class TestHistoricalCache:
    """Tests for historical bars caching."""

    def test_set_and_get_historical(self):
        """Test storing and retrieving historical data."""
        cache = ScreenerCache(historical_ttl=3600)
        cache.set_historical("AAPL", {"close": [100, 101, 102]})

        result = cache.get_historical("AAPL")
        assert result == {"close": [100, 101, 102]}

    def test_batch_historical(self):
        """Test batch historical operations."""
        cache = ScreenerCache()
        bars = {
            "AAPL": {"close": [100, 101]},
            "MSFT": {"close": [200, 201]},
        }
        cache.set_historical_batch(bars)

        result = cache.get_historical_batch(["AAPL", "MSFT"])
        assert len(result) == 2


class TestResultCache:
    """Tests for screening result caching."""

    def test_set_and_get_result(self):
        """Test storing and retrieving results."""
        cache = ScreenerCache(result_ttl=30)
        cache.set_result("config_hash_123", ["AAPL", "MSFT"])

        result = cache.get_result("config_hash_123")
        assert result == ["AAPL", "MSFT"]

    def test_result_expiration(self):
        """Test result expires after TTL."""
        cache = ScreenerCache(result_ttl=1)
        cache.set_result("hash", ["AAPL"])

        assert cache.get_result("hash") is not None
        time.sleep(1.1)
        assert cache.get_result("hash") is None


class TestAssetsCache:
    """Tests for tradable assets caching."""

    def test_set_and_get_assets(self):
        """Test storing and retrieving assets list."""
        cache = ScreenerCache(assets_ttl=3600)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        cache.set_assets(symbols)

        result = cache.get_assets()
        assert result == symbols

    def test_assets_not_cached(self):
        """Test assets returns None when not cached."""
        cache = ScreenerCache()
        assert cache.get_assets() is None


class TestCacheManagement:
    """Tests for cache management operations."""

    def test_clear_all(self):
        """Test clearing all caches."""
        cache = ScreenerCache()
        cache.set_snapshot("AAPL", {"price": 100})
        cache.set_historical("AAPL", {"close": [100]})
        cache.set_result("hash", ["AAPL"])
        cache.set_assets(["AAPL"])

        cache.clear()

        assert cache.get_snapshot("AAPL") is None
        assert cache.get_historical("AAPL") is None
        assert cache.get_result("hash") is None
        assert cache.get_assets() is None

    def test_clear_snapshots_only(self):
        """Test clearing only snapshot cache."""
        cache = ScreenerCache()
        cache.set_snapshot("AAPL", {"price": 100})
        cache.set_historical("AAPL", {"close": [100]})

        cache.clear_snapshots()

        assert cache.get_snapshot("AAPL") is None
        assert cache.get_historical("AAPL") is not None

    def test_evict_expired(self):
        """Test evicting expired entries."""
        cache = ScreenerCache(snapshot_ttl=1)
        cache.set_snapshot("AAPL", {"price": 100})

        time.sleep(1.1)

        evicted = cache.evict_expired()
        assert evicted >= 1


class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_tracking(self):
        """Test hit/miss tracking."""
        cache = ScreenerCache()

        # Miss
        cache.get_snapshot("NOTFOUND")
        stats = cache.get_stats()
        assert stats.misses == 1

        # Set and hit
        cache.set_snapshot("AAPL", {"price": 100})
        cache.get_snapshot("AAPL")
        stats = cache.get_stats()
        assert stats.hits == 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == 80.0

    def test_hit_rate_zero_total(self):
        """Test hit rate with zero requests."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self):
        """Test concurrent cache access."""
        cache = ScreenerCache()
        errors: List[Exception] = []

        def writer():
            try:
                for i in range(100):
                    cache.set_snapshot(f"SYM{i}", {"price": i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get_snapshot(f"SYM{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            Thread(target=writer),
            Thread(target=reader),
            Thread(target=writer),
            Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestHashConfig:
    """Tests for config hashing."""

    def test_hash_produces_string(self):
        """Test hash produces string output."""
        result = hash_config('{"universe": ["AAPL"]}')
        assert isinstance(result, str)
        assert len(result) == 16  # SHA256 truncated to 16 chars

    def test_same_config_same_hash(self):
        """Test same config produces same hash."""
        config = '{"universe": ["AAPL"], "max_results": 50}'
        hash1 = hash_config(config)
        hash2 = hash_config(config)
        assert hash1 == hash2

    def test_different_config_different_hash(self):
        """Test different configs produce different hashes."""
        hash1 = hash_config('{"universe": ["AAPL"]}')
        hash2 = hash_config('{"universe": ["MSFT"]}')
        assert hash1 != hash2
