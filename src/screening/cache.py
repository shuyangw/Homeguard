"""
In-Memory TTL Cache for Stock Screener.

Provides thread-safe caching with TTL expiration for:
- Snapshots (60s default)
- Historical bars (1 hour default)
- Screening results (30s default)
- Tradable assets list (1 hour default)
"""

import hashlib
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with TTL tracking."""

    data: T
    timestamp: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get the age of this entry in seconds."""
        return time.time() - self.timestamp

    @property
    def remaining_seconds(self) -> float:
        """Get remaining TTL in seconds (0 if expired)."""
        remaining = self.ttl_seconds - self.age_seconds
        return max(0, remaining)


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class ScreenerCache:
    """
    Thread-safe in-memory cache for screener data.

    Provides separate caches for different data types with configurable TTLs.

    Usage:
        cache = ScreenerCache()

        # Store and retrieve snapshots
        cache.set_snapshot('AAPL', snapshot_data)
        snapshot = cache.get_snapshot('AAPL')  # Returns None if expired

        # Store and retrieve historical bars
        cache.set_historical('AAPL', bars_dataframe)
        bars = cache.get_historical('AAPL')

        # Cache screening results by config hash
        cache.set_result(config_hash, ['AAPL', 'MSFT'])
        result = cache.get_result(config_hash)
    """

    def __init__(
        self,
        snapshot_ttl: int = 60,
        historical_ttl: int = 3600,
        result_ttl: int = 30,
        assets_ttl: int = 3600,
    ):
        """
        Initialize the cache with TTL settings.

        Args:
            snapshot_ttl: TTL for snapshot data in seconds (default: 60)
            historical_ttl: TTL for historical bars in seconds (default: 3600)
            result_ttl: TTL for screening results in seconds (default: 30)
            assets_ttl: TTL for tradable assets list in seconds (default: 3600)
        """
        self._snapshot_ttl = snapshot_ttl
        self._historical_ttl = historical_ttl
        self._result_ttl = result_ttl
        self._assets_ttl = assets_ttl

        # Separate caches for different data types
        self._snapshots: Dict[str, CacheEntry] = {}
        self._historical: Dict[str, CacheEntry] = {}
        self._results: Dict[str, CacheEntry] = {}
        self._assets: Optional[CacheEntry[List[str]]] = None

        # Thread safety
        self._lock = Lock()

        # Statistics
        self._stats = CacheStats()

    # -------------------------------------------------------------------------
    # Snapshot Cache (individual symbol snapshots)
    # -------------------------------------------------------------------------

    def get_snapshot(self, symbol: str) -> Optional[Any]:
        """
        Get cached snapshot for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Snapshot data if cached and not expired, None otherwise
        """
        with self._lock:
            entry = self._snapshots.get(symbol)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.is_expired:
                del self._snapshots[symbol]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            self._stats.hits += 1
            return entry.data

    def set_snapshot(self, symbol: str, data: Any) -> None:
        """
        Cache snapshot data for a symbol.

        Args:
            symbol: Stock symbol
            data: Snapshot data to cache
        """
        with self._lock:
            self._snapshots[symbol] = CacheEntry(
                data=data, timestamp=time.time(), ttl_seconds=self._snapshot_ttl
            )

    def get_snapshots_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get cached snapshots for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict of symbol -> snapshot for cached entries
        """
        result = {}
        with self._lock:
            for symbol in symbols:
                entry = self._snapshots.get(symbol)
                if entry is not None and not entry.is_expired:
                    result[symbol] = entry.data
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1
                    if entry is not None:
                        del self._snapshots[symbol]
                        self._stats.evictions += 1
        return result

    def set_snapshots_batch(self, snapshots: Dict[str, Any]) -> None:
        """
        Cache multiple snapshots at once.

        Args:
            snapshots: Dict of symbol -> snapshot data
        """
        timestamp = time.time()
        with self._lock:
            for symbol, data in snapshots.items():
                self._snapshots[symbol] = CacheEntry(
                    data=data, timestamp=timestamp, ttl_seconds=self._snapshot_ttl
                )

    # -------------------------------------------------------------------------
    # Historical Bars Cache
    # -------------------------------------------------------------------------

    def get_historical(self, symbol: str) -> Optional[Any]:
        """
        Get cached historical bars for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Historical bars DataFrame if cached and not expired, None otherwise
        """
        with self._lock:
            entry = self._historical.get(symbol)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.is_expired:
                del self._historical[symbol]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            self._stats.hits += 1
            return entry.data

    def set_historical(self, symbol: str, data: Any) -> None:
        """
        Cache historical bars for a symbol.

        Args:
            symbol: Stock symbol
            data: Historical bars DataFrame
        """
        with self._lock:
            self._historical[symbol] = CacheEntry(
                data=data, timestamp=time.time(), ttl_seconds=self._historical_ttl
            )

    def get_historical_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get cached historical bars for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict of symbol -> bars DataFrame for cached entries
        """
        result = {}
        with self._lock:
            for symbol in symbols:
                entry = self._historical.get(symbol)
                if entry is not None and not entry.is_expired:
                    result[symbol] = entry.data
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1
                    if entry is not None:
                        del self._historical[symbol]
                        self._stats.evictions += 1
        return result

    def set_historical_batch(self, bars: Dict[str, Any]) -> None:
        """
        Cache multiple historical bars at once.

        Args:
            bars: Dict of symbol -> bars DataFrame
        """
        timestamp = time.time()
        with self._lock:
            for symbol, data in bars.items():
                self._historical[symbol] = CacheEntry(
                    data=data, timestamp=timestamp, ttl_seconds=self._historical_ttl
                )

    # -------------------------------------------------------------------------
    # Screening Results Cache
    # -------------------------------------------------------------------------

    def get_result(self, config_hash: str) -> Optional[List[str]]:
        """
        Get cached screening result.

        Args:
            config_hash: Hash of the ScreenerConfig

        Returns:
            List of matching symbols if cached and not expired, None otherwise
        """
        with self._lock:
            entry = self._results.get(config_hash)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.is_expired:
                del self._results[config_hash]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            self._stats.hits += 1
            return entry.data

    def set_result(self, config_hash: str, symbols: List[str]) -> None:
        """
        Cache screening result.

        Args:
            config_hash: Hash of the ScreenerConfig
            symbols: List of matching symbols
        """
        with self._lock:
            self._results[config_hash] = CacheEntry(
                data=symbols, timestamp=time.time(), ttl_seconds=self._result_ttl
            )

    # -------------------------------------------------------------------------
    # Tradable Assets Cache
    # -------------------------------------------------------------------------

    def get_assets(self) -> Optional[List[str]]:
        """
        Get cached list of tradable assets.

        Returns:
            List of tradable symbols if cached and not expired, None otherwise
        """
        with self._lock:
            if self._assets is None:
                self._stats.misses += 1
                return None
            if self._assets.is_expired:
                self._assets = None
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            self._stats.hits += 1
            return self._assets.data

    def set_assets(self, symbols: List[str]) -> None:
        """
        Cache list of tradable assets.

        Args:
            symbols: List of tradable symbols
        """
        with self._lock:
            self._assets = CacheEntry(
                data=symbols, timestamp=time.time(), ttl_seconds=self._assets_ttl
            )

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._snapshots.clear()
            self._historical.clear()
            self._results.clear()
            self._assets = None
            self._stats = CacheStats()

    def clear_snapshots(self) -> None:
        """Clear only snapshot cache."""
        with self._lock:
            self._snapshots.clear()

    def clear_historical(self) -> None:
        """Clear only historical bars cache."""
        with self._lock:
            self._historical.clear()

    def clear_results(self) -> None:
        """Clear only results cache."""
        with self._lock:
            self._results.clear()

    def evict_expired(self) -> int:
        """
        Evict all expired entries from caches.

        Returns:
            Number of entries evicted
        """
        evicted = 0
        with self._lock:
            # Snapshots
            expired_symbols = [
                s for s, e in self._snapshots.items() if e.is_expired
            ]
            for symbol in expired_symbols:
                del self._snapshots[symbol]
                evicted += 1

            # Historical
            expired_historical = [
                s for s, e in self._historical.items() if e.is_expired
            ]
            for symbol in expired_historical:
                del self._historical[symbol]
                evicted += 1

            # Results
            expired_results = [
                h for h, e in self._results.items() if e.is_expired
            ]
            for config_hash in expired_results:
                del self._results[config_hash]
                evicted += 1

            # Assets
            if self._assets is not None and self._assets.is_expired:
                self._assets = None
                evicted += 1

            self._stats.evictions += evicted
        return evicted

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats with hit/miss/eviction counts
        """
        with self._lock:
            self._stats.entries = (
                len(self._snapshots)
                + len(self._historical)
                + len(self._results)
                + (1 if self._assets is not None else 0)
            )
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                entries=self._stats.entries,
            )

    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed cache statistics by type.

        Returns:
            Dict with stats for each cache type
        """
        with self._lock:
            return {
                "snapshots": {
                    "count": len(self._snapshots),
                    "ttl_seconds": self._snapshot_ttl,
                },
                "historical": {
                    "count": len(self._historical),
                    "ttl_seconds": self._historical_ttl,
                },
                "results": {
                    "count": len(self._results),
                    "ttl_seconds": self._result_ttl,
                },
                "assets": {
                    "cached": self._assets is not None,
                    "ttl_seconds": self._assets_ttl,
                    "remaining_seconds": (
                        self._assets.remaining_seconds if self._assets else 0
                    ),
                },
                "totals": {
                    "hits": self._stats.hits,
                    "misses": self._stats.misses,
                    "evictions": self._stats.evictions,
                    "hit_rate_pct": self._stats.hit_rate,
                },
            }


def hash_config(config_json: str) -> str:
    """
    Generate a hash for a ScreenerConfig JSON string.

    Args:
        config_json: JSON representation of ScreenerConfig

    Returns:
        SHA256 hash of the config
    """
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]
