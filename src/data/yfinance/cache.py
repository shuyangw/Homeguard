"""
Persistent cache for YFinance fundamental data.

Uses parquet files for storage since fundamentals don't change frequently.
Cache is stored in the project data directory.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.yfinance.fundamentals import FundamentalData
from src.utils.logger import logger


@dataclass
class CacheEntry:
    """Cache entry with timestamp for TTL checking."""

    data: FundamentalData
    timestamp: float
    ttl_hours: float

    @property
    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL."""
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > self.ttl_hours

    @property
    def age_hours(self) -> float:
        """Get age of entry in hours."""
        return (time.time() - self.timestamp) / 3600


class FundamentalsCache:
    """
    Persistent cache for fundamental data using parquet storage.

    Features:
    - 24-hour default TTL (fundamentals change slowly)
    - Parquet storage for durability across sessions
    - Thread-safe read/write operations
    - Automatic cleanup of expired entries
    """

    DEFAULT_TTL_HOURS = 24
    CACHE_FILENAME = "fundamentals_cache.parquet"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_hours: float = DEFAULT_TTL_HOURS,
    ):
        """
        Initialize fundamentals cache.

        Args:
            cache_dir: Directory for cache file (default: .cache in project root)
            ttl_hours: Hours before entries expire (default: 24)
        """
        self._ttl_hours = ttl_hours

        # Set cache directory
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            # Default to .cache in project root
            self._cache_dir = Path(__file__).parent.parent.parent.parent / ".cache"

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / self.CACHE_FILENAME

        # In-memory cache (loaded from disk)
        self._cache: Dict[str, CacheEntry] = {}
        self._load_from_disk()

        # Stats
        self._hits = 0
        self._misses = 0

    def _load_from_disk(self) -> None:
        """Load cache from parquet file."""
        if not self._cache_path.exists():
            return

        try:
            df = pd.read_parquet(self._cache_path)

            for _, row in df.iterrows():
                symbol = row["symbol"]
                timestamp = row["timestamp"]

                # Skip expired entries
                age_hours = (time.time() - timestamp) / 3600
                if age_hours > self._ttl_hours:
                    continue

                # Reconstruct FundamentalData from row
                data_dict = row.to_dict()
                del data_dict["timestamp"]

                # Filter out NaN values
                data_dict = {k: v for k, v in data_dict.items() if pd.notna(v)}

                data = FundamentalData(**data_dict)
                self._cache[symbol] = CacheEntry(
                    data=data,
                    timestamp=timestamp,
                    ttl_hours=self._ttl_hours,
                )

            logger.debug(f"[FundamentalsCache] Loaded {len(self._cache)} entries from disk")

        except Exception as e:
            logger.warning(f"[FundamentalsCache] Failed to load cache: {e}")
            self._cache = {}

    def _save_to_disk(self) -> None:
        """Save cache to parquet file."""
        if not self._cache:
            # Remove file if cache is empty
            if self._cache_path.exists():
                self._cache_path.unlink()
            return

        try:
            rows = []
            for symbol, entry in self._cache.items():
                if not entry.is_expired:
                    row = entry.data.to_dict()
                    row["timestamp"] = entry.timestamp
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                df.to_parquet(self._cache_path, index=False)
                logger.debug(f"[FundamentalsCache] Saved {len(rows)} entries to disk")

        except Exception as e:
            logger.error(f"[FundamentalsCache] Failed to save cache: {e}")

    def get(self, symbol: str) -> Optional[FundamentalData]:
        """
        Get cached fundamentals for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            FundamentalData if cached and not expired, else None
        """
        entry = self._cache.get(symbol)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            del self._cache[symbol]
            self._misses += 1
            return None

        self._hits += 1
        return entry.data

    def get_batch(self, symbols: list) -> Dict[str, FundamentalData]:
        """
        Get cached fundamentals for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to FundamentalData (only cached symbols)
        """
        result = {}
        for symbol in symbols:
            data = self.get(symbol)
            if data is not None:
                result[symbol] = data
        return result

    def set(self, symbol: str, data: FundamentalData) -> None:
        """
        Cache fundamentals for a symbol.

        Args:
            symbol: Stock symbol
            data: FundamentalData to cache
        """
        self._cache[symbol] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl_hours=self._ttl_hours,
        )
        self._save_to_disk()

    def set_batch(self, data: Dict[str, FundamentalData]) -> None:
        """
        Cache fundamentals for multiple symbols.

        Args:
            data: Dict mapping symbol to FundamentalData
        """
        now = time.time()
        for symbol, fundamental in data.items():
            self._cache[symbol] = CacheEntry(
                data=fundamental,
                timestamp=now,
                ttl_hours=self._ttl_hours,
            )
        self._save_to_disk()

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        if self._cache_path.exists():
            self._cache_path.unlink()
        logger.debug("[FundamentalsCache] Cache cleared")

    def evict_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries evicted
        """
        expired = [symbol for symbol, entry in self._cache.items() if entry.is_expired]
        for symbol in expired:
            del self._cache[symbol]

        if expired:
            self._save_to_disk()

        return len(expired)

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with hit/miss stats
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0

        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_hours": self._ttl_hours,
        }

    def get_cached_symbols(self) -> list:
        """Get list of all cached symbols (including expired)."""
        return list(self._cache.keys())

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in cache (and not expired)."""
        entry = self._cache.get(symbol)
        return entry is not None and not entry.is_expired
