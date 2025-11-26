"""
Smart caching system for grid search optimization results.

Provides two-tier caching:
1. In-memory FIFO cache for fast access within session
2. SQLite disk cache for persistence across sessions

Cache keys are generated from:
- Strategy class and parameters
- Data context (symbols, dates, price type)
- Engine configuration (fees, risk settings, etc.)
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.settings import get_log_output_dir
from src.utils import logger


def _safe_json_dumps(obj: Any) -> str:
    """
    Safely serialize objects to JSON, handling NumPy types.

    Args:
        obj: Object to serialize

    Returns:
        JSON string
    """
    def convert(o):
        if isinstance(o, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64, np.float32, np.float16)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=convert, sort_keys=True)


@dataclass
class CacheConfig:
    """Configuration for result cache."""
    enabled: bool = True
    memory_cache_size: int = 1000
    disk_cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    ttl_days: int = 30  # Cache entries older than this are invalid


class ResultCache:
    """
    Two-tier caching system for optimization results.

    Features:
    - In-memory FIFO cache for fast access within session
    - SQLite disk cache for persistence across sessions
    - Automatic cache invalidation based on TTL
    - Cache statistics tracking

    Usage:
        cache = ResultCache()

        # Check cache before running test
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        # Run test...
        result = run_test(params)

        # Store in cache
        cache.put(cache_key, result)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize result cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'invalidations': 0
        }

        if self.config.disk_cache_enabled:
            self._init_disk_cache()

    def _init_disk_cache(self) -> None:
        """Initialize SQLite disk cache."""
        try:
            # Determine cache directory
            if self.config.cache_dir is None:
                base_dir = get_log_output_dir()
                self.cache_dir = base_dir / '.cache'
            else:
                self.cache_dir = self.config.cache_dir

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.cache_dir / 'optimization_cache.db'

            # Create database schema with timeout
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_cache (
                    cache_key TEXT PRIMARY KEY,
                    params_json TEXT NOT NULL,
                    metric_value REAL,
                    stats_json TEXT,
                    error TEXT,
                    created_timestamp REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)

            # Create index on timestamp for TTL cleanup
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_timestamp
                ON optimization_cache(created_timestamp)
            """)

            conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to initialize disk cache: {e}")
            logger.warning("Disk caching will be disabled")
            self.config.disk_cache_enabled = False
        finally:
            if 'conn' in locals():
                conn.close()

    @staticmethod
    def generate_cache_key(
        strategy_class: type,
        params: Dict[str, Any],
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_type: str,
        engine_config: 'Any',  # _EngineConfig
        metric: str
    ) -> str:
        """
        Generate unique cache key for a parameter test.

        Args:
            strategy_class: Strategy class
            params: Parameter dictionary
            symbols: List of symbols
            start_date: Start date string
            end_date: End date string
            price_type: Price column name
            engine_config: Engine configuration
            metric: Optimization metric

        Returns:
            SHA256 hash string (cache key)
        """
        # Build cache context dictionary
        cache_context = {
            'strategy': strategy_class.__name__,
            'params': sorted(params.items()),  # Sort for consistency
            'symbols': sorted(symbols),
            'start_date': start_date,
            'end_date': end_date,
            'price_type': price_type,
            'metric': metric,
            'engine': {
                'initial_capital': engine_config.initial_capital,
                'fees': engine_config.fees,
                'slippage': engine_config.slippage,
                'freq': engine_config.freq,
                'market_hours_only': engine_config.market_hours_only,
                'enable_regime_analysis': engine_config.enable_regime_analysis,
                # Include relevant risk config fields
                'risk_config': engine_config.risk_config_dict
            }
        }

        # Convert to JSON string (sorted keys for consistency, handle NumPy types)
        json_str = _safe_json_dumps(cache_context)

        # Generate SHA256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve result from cache.

        Args:
            cache_key: Cache key generated by generate_cache_key()

        Returns:
            Cached result dictionary or None if not found/expired
        """
        if not self.config.enabled:
            return None

        # Check memory cache first
        if cache_key in self._memory_cache:
            self._cache_stats['hits'] += 1
            return self._memory_cache[cache_key]

        # Check disk cache
        if self.config.disk_cache_enabled:
            result = self._get_from_disk(cache_key)
            if result:
                # Populate memory cache
                self._memory_cache[cache_key] = result
                self._cache_stats['hits'] += 1

                # Enforce memory cache size limit (simple FIFO)
                if len(self._memory_cache) >= self.config.memory_cache_size:
                    # Remove oldest entry (first in dict)
                    oldest_key = next(iter(self._memory_cache))
                    del self._memory_cache[oldest_key]

                return result

        self._cache_stats['misses'] += 1
        return None

    def _get_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve result from SQLite disk cache."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            # Check TTL
            cutoff_timestamp = time.time() - (self.config.ttl_days * 24 * 3600)

            cursor.execute("""
                SELECT params_json, metric_value, stats_json, error, created_timestamp
                FROM optimization_cache
                WHERE cache_key = ? AND created_timestamp > ?
            """, (cache_key, cutoff_timestamp))

            row = cursor.fetchone()

            if row:
                # Update access statistics
                cursor.execute("""
                    UPDATE optimization_cache
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                """, (time.time(), cache_key))
                conn.commit()

                # Parse result
                params_json, metric_value, stats_json, error, created_timestamp = row

                result = {
                    'params': json.loads(params_json),
                    'value': metric_value,
                    'stats': json.loads(stats_json) if stats_json else None,
                    'error': error
                }

                return result

            return None
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.warning(f"Error retrieving from disk cache: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()

    def put(
        self,
        cache_key: str,
        params: Dict[str, Any],
        metric_value: float,
        stats: Optional[Dict[str, Any]],
        error: Optional[str]
    ) -> None:
        """
        Store result in cache.

        Args:
            cache_key: Cache key generated by generate_cache_key()
            params: Parameter dictionary
            metric_value: Metric value
            stats: Portfolio stats dictionary (or None if error)
            error: Error message (or None if success)
        """
        if not self.config.enabled:
            return

        result = {
            'params': params,
            'value': metric_value,
            'stats': stats,
            'error': error
        }

        # Store in memory cache
        self._memory_cache[cache_key] = result

        # Enforce memory cache size limit
        if len(self._memory_cache) >= self.config.memory_cache_size:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        # Store in disk cache
        if self.config.disk_cache_enabled:
            self._put_to_disk(cache_key, params, metric_value, stats, error)

        self._cache_stats['puts'] += 1

    def _put_to_disk(
        self,
        cache_key: str,
        params: Dict[str, Any],
        metric_value: float,
        stats: Optional[Dict[str, Any]],
        error: Optional[str]
    ) -> None:
        """Store result in SQLite disk cache."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            now = time.time()

            # Use INSERT OR REPLACE to handle duplicates
            cursor.execute("""
                INSERT OR REPLACE INTO optimization_cache
                (cache_key, params_json, metric_value, stats_json, error,
                 created_timestamp, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                cache_key,
                _safe_json_dumps(params),
                metric_value,
                _safe_json_dumps(stats) if stats else None,
                error,
                now,
                now
            ))

            conn.commit()
        except (sqlite3.Error, TypeError) as e:
            logger.warning(f"Error writing to disk cache: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def clear(self) -> None:
        """Clear all cached results (both memory and disk)."""
        self._memory_cache.clear()

        if self.config.disk_cache_enabled:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM optimization_cache")
                conn.commit()
            except sqlite3.Error as e:
                logger.warning(f"Error clearing disk cache: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()

        self._cache_stats['invalidations'] += 1
        logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries based on TTL.

        Returns:
            Number of entries removed
        """
        if not self.config.disk_cache_enabled:
            return 0

        try:
            cutoff_timestamp = time.time() - (self.config.ttl_days * 24 * 3600)

            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM optimization_cache
                WHERE created_timestamp < ?
            """, (cutoff_timestamp,))

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")

            return deleted_count
        except sqlite3.Error as e:
            logger.warning(f"Error cleaning up expired cache entries: {e}")
            return 0
        finally:
            if 'conn' in locals():
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit rate, etc.)
        """
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        stats = {
            **self._cache_stats,
            'total_requests': total_requests,
            'hit_rate_pct': hit_rate,
            'memory_cache_size': len(self._memory_cache)
        }

        # Add disk cache stats if enabled
        if self.config.disk_cache_enabled:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM optimization_cache")
                disk_count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT SUM(access_count) FROM optimization_cache
                """)
                total_accesses = cursor.fetchone()[0] or 0

                stats['disk_cache_size'] = disk_count
                stats['total_disk_accesses'] = total_accesses
            except sqlite3.Error as e:
                logger.warning(f"Error retrieving disk cache stats: {e}")
                stats['disk_cache_size'] = 0
                stats['total_disk_accesses'] = 0
            finally:
                if 'conn' in locals():
                    conn.close()

        return stats

    def print_stats(self) -> None:
        """Print cache statistics to logger."""
        stats = self.get_stats()

        logger.blank()
        logger.separator()
        logger.header("CACHE STATISTICS")
        logger.separator()
        logger.metric(f"Cache hits: {stats['hits']}")
        logger.metric(f"Cache misses: {stats['misses']}")
        logger.metric(f"Hit rate: {stats['hit_rate_pct']:.1f}%")
        logger.metric(f"Memory cache size: {stats['memory_cache_size']} entries")

        if self.config.disk_cache_enabled:
            logger.metric(f"Disk cache size: {stats['disk_cache_size']} entries")
            logger.metric(f"Total disk accesses: {stats['total_disk_accesses']}")

        logger.separator()
        logger.blank()
