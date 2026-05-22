"""
Streaming data loader for backtesting using Polars LazyFrames.

This module REPLACES the old DuckDB-based data_loader.py with a memory-efficient
streaming approach that:
- Loads data symbol-by-symbol to avoid loading entire universe into RAM
- Uses Polars LazyFrames with streaming collection
- Provides LRU cache for optimization loops
- Supports both single-symbol and multi-symbol strategies
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from src.backtesting.utils.market_calendar import MarketCalendar
from src.settings import OS_ENVIRONMENT, settings
from src.utils import logger


class SymbolCache:
    """
    LRU cache for symbol data with configurable memory limit.

    Thread-safe cache that automatically evicts least-recently-used entries
    when memory limit is exceeded.
    """

    def __init__(self, max_size_mb: int = 4096):
        """
        Initialize the LRU cache.

        Args:
            max_size_mb: Maximum cache size in megabytes (default 4GB)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cache: OrderedDict[str, pl.DataFrame] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._current_size = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _estimate_size(self, df: pl.DataFrame) -> int:
        """Estimate DataFrame memory usage in bytes."""
        return df.estimated_size()

    def get(self, key: str) -> Optional[pl.DataFrame]:
        """
        Get item from cache, moving it to end (most recently used).

        Args:
            key: Cache key (typically symbol:start:end)

        Returns:
            DataFrame if found, None otherwise
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, df: pl.DataFrame) -> None:
        """
        Add item to cache, evicting LRU entries if needed.

        Args:
            key: Cache key
            df: DataFrame to cache
        """
        size = self._estimate_size(df)

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._current_size -= self._sizes[key]
                del self._cache[key]
                del self._sizes[key]

            # Evict LRU entries until we have space
            while self._current_size + size > self.max_size_bytes and self._cache:
                oldest_key, _ = self._cache.popitem(last=False)
                self._current_size -= self._sizes.pop(oldest_key)

            # Add new entry
            self._cache[key] = df
            self._sizes[key] = size
            self._current_size += size

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._current_size = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            return {
                "entries": len(self._cache),
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class StreamingDataLoader:
    """
    Polars-based streaming data loader for memory-efficient backtesting.

    Features:
    - Symbol-by-symbol iterator for large-universe sweeps
    - Synchronized multi-symbol loading for portfolio strategies
    - LRU cache for optimization loops
    - Lazy evaluation with streaming collection
    - Support for both symbol-first and time-first partitioning
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        cache_size_mb: int = 4096,
        filter_market_days: bool = True,
        timeframe: str = "1min",
        prefer_time_partitioned: bool = True,
    ):
        """
        Initialize the streaming data loader.

        Args:
            base_path: Base directory for data storage (defaults to OS-specific setting)
            cache_size_mb: Maximum cache size in MB (default 4GB)
            filter_market_days: Filter weekends/holidays (default True)
            timeframe: Data timeframe - '1min', '1hour', or '1day' (default '1min')
            prefer_time_partitioned: If True, use time-first partitioning when available (default True)
        """
        if base_path is None:
            base_path = settings[OS_ENVIRONMENT]["local_storage_dir"]
        self._base_path = Path(base_path)

        # Map timeframe to directory name
        timeframe_dirs = {
            "1min": "equities_1min",
            "1hour": "equities_1hour",
            "1day": "equities_1day",
            # Crypto data directories
            "crypto_1min": "crypto_1min",
            "crypto_1hour": "crypto_1hour",
            "crypto_1day": "crypto_1day",
        }
        if timeframe not in timeframe_dirs:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(timeframe_dirs.keys())}")

        self.timeframe = timeframe
        self._timeframe_dir = timeframe_dirs[timeframe]
        self.data_path = self._base_path / self._timeframe_dir
        self.data_path_by_date = self._base_path / f"{self._timeframe_dir}_by_date"
        self.prefer_time_partitioned = prefer_time_partitioned
        self.filter_market_days = filter_market_days
        self.market_calendar = MarketCalendar("NYSE") if filter_market_days else None
        self._cache = SymbolCache(max_size_mb=cache_size_mb)
        self._date_cache = SymbolCache(max_size_mb=cache_size_mb)  # Separate cache for date-partitioned data
        self._lock = threading.RLock()

    @property
    def base_path(self) -> Path:
        """Base path for data storage."""
        return self._base_path

    @base_path.setter
    def base_path(self, value: Path) -> None:
        """Set base path (for testing)."""
        self._base_path = Path(value)
        self.data_path = self._base_path / self._timeframe_dir
        self.data_path_by_date = self._base_path / f"{self._timeframe_dir}_by_date"

    def has_time_partitioned_data(self) -> bool:
        """Check if time-partitioned data is available."""
        return self.data_path_by_date.exists() and any(self.data_path_by_date.glob("*.parquet"))

    def use_time_partitioned(self) -> bool:
        """Determine whether to use time-partitioned data."""
        return self.prefer_time_partitioned and self.has_time_partitioned_data()

    def _get_partition_paths(self, symbol: str, start_date: str, end_date: str) -> List[Path]:
        """
        Get list of partition paths for a symbol within date range.

        Uses hive partitioning structure: symbol={SYM}/year={YYYY}/month={MM}/data.parquet
        """
        symbol_path = self.data_path / f"symbol={symbol}"
        if not symbol_path.exists():
            return []

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        paths = []
        current_year = start_dt.year
        current_month = start_dt.month

        while (current_year, current_month) <= (end_dt.year, end_dt.month):
            month_dir = symbol_path / f"year={current_year}" / f"month={current_month}"
            if month_dir.exists():
                # Check for data.parquet first, then data_*.parquet files
                partition_path = month_dir / "data.parquet"
                if partition_path.exists():
                    paths.append(partition_path)
                else:
                    # Look for data_0.parquet, data_1.parquet, etc.
                    data_files = sorted(month_dir.glob("data_*.parquet"))
                    paths.extend(data_files)

            # Move to next month
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1

        return paths

    def _scan_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
    ) -> pl.LazyFrame:
        """
        Create a LazyFrame for a single symbol with date filtering.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Optional list of columns to select

        Returns:
            LazyFrame with filtered data
        """
        paths = self._get_partition_paths(symbol, start_date, end_date)
        if not paths:
            raise FileNotFoundError(f"No data found for symbol {symbol} between {start_date} and {end_date}")

        # Scan parquet files - handle potential schema mismatches between files
        # by loading each file separately and concatenating if needed
        if len(paths) == 1:
            lf = pl.scan_parquet(paths[0])
        else:
            # Load files individually to handle schema mismatches
            dfs = []
            for p in paths:
                try:
                    df = pl.read_parquet(p)
                    # Normalize timestamp to microseconds if needed
                    if "ns" in str(df["timestamp"].dtype):
                        df = df.with_columns(
                            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                        )
                    dfs.append(df)
                except (pl.exceptions.ComputeError, OSError, ValueError) as e:
                    logger.debug(f"Skipping file {p}: {e}")
                    continue
            if not dfs:
                raise FileNotFoundError(f"No data found for symbol {symbol} between {start_date} and {end_date}")
            combined = pl.concat(dfs)
            lf = combined.lazy()

        # Filter by date range using date strings to avoid timezone issues
        # Convert timestamp to date string for comparison
        lf = lf.filter(
            (pl.col("timestamp").dt.strftime("%Y-%m-%d") >= start_date)
            & (pl.col("timestamp").dt.strftime("%Y-%m-%d") <= end_date)
        )

        # Select columns if specified
        if columns:
            # Always include timestamp
            cols = ["timestamp"] + [c for c in columns if c != "timestamp"]
            lf = lf.select(cols)

        # Sort by timestamp
        lf = lf.sort("timestamp")

        return lf

    def _filter_trading_days(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter DataFrame to only include trading days."""
        if not self.filter_market_days or self.market_calendar is None:
            return df

        if df.is_empty():
            return df

        # Get date range from data
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        # Get all trading days in range at once (fast batch operation)
        trading_days_index = self.market_calendar.get_trading_days(
            min_date.strftime("%Y-%m-%d"),
            max_date.strftime("%Y-%m-%d")
        )

        # Convert to set of date objects for fast lookup
        trading_dates = set(d.date() for d in trading_days_index)

        if not trading_dates:
            return df.clear()

        return df.filter(pl.col("timestamp").dt.date().is_in(list(trading_dates)))

    # =========================================================================
    # Time-Partitioned Loading (date-first structure)
    # =========================================================================

    def _get_date_partition_paths(self, start_date: str, end_date: str) -> List[Path]:
        """
        Get list of date partition files within date range.

        Time-partitioned structure: equities_1min_by_date/2024-01-02.parquet
        """
        if not self.data_path_by_date.exists():
            return []

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        paths = []
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")
            partition_path = self.data_path_by_date / f"{date_str}.parquet"
            if partition_path.exists():
                paths.append(partition_path)
            current += pd.Timedelta(days=1)

        return paths

    def load_date(
        self,
        date: str,
        symbols: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Load all symbols for a single date from time-partitioned data.

        Args:
            date: Date string (YYYY-MM-DD)
            symbols: Optional list of symbols to filter (None = all)
            columns: Optional list of columns to select
            use_cache: Whether to use cache

        Returns:
            DataFrame with all symbols for the date
        """
        cache_key = f"date:{date}:{symbols}:{columns}"

        if use_cache:
            cached = self._date_cache.get(cache_key)
            if cached is not None:
                return cached

        partition_path = self.data_path_by_date / f"{date}.parquet"
        if not partition_path.exists():
            return pl.DataFrame()

        df = pl.read_parquet(partition_path)

        # Filter symbols if specified
        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))

        # Select columns if specified
        if columns:
            cols = ["timestamp", "symbol"] + [c for c in columns if c not in ("timestamp", "symbol")]
            available = set(df.columns)
            cols = [c for c in cols if c in available]
            df = df.select(cols)

        if use_cache and not df.is_empty():
            self._date_cache.put(cache_key, df)

        return df

    def load_date_range(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load all symbols for a date range from time-partitioned data.

        Much more efficient than symbol-first loading for walk-forward backtests.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: Optional list of symbols to filter
            columns: Optional list of columns to select

        Returns:
            DataFrame with all symbols for the date range
        """
        paths = self._get_date_partition_paths(start_date, end_date)

        if not paths:
            return pl.DataFrame()

        # Scan all date files
        lf = pl.scan_parquet(paths)

        # Filter symbols if specified
        if symbols:
            lf = lf.filter(pl.col("symbol").is_in(symbols))

        # Select columns if specified
        if columns:
            cols = ["timestamp", "symbol"] + [c for c in columns if c not in ("timestamp", "symbol")]
            lf = lf.select(cols)

        # Sort by timestamp and symbol
        lf = lf.sort(["timestamp", "symbol"])

        return lf.collect(engine="streaming")

    def load_date_range_by_symbol(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, pl.DataFrame]:
        """
        Load date range and return as dict keyed by symbol.

        Efficient for walk-forward backtests with time-partitioned data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: Optional list of symbols to filter
            columns: Optional list of columns to select

        Returns:
            Dict mapping symbol -> DataFrame
        """
        df = self.load_date_range(start_date, end_date, symbols, columns)

        if df.is_empty():
            return {}

        result = {}
        for symbol in df["symbol"].unique().to_list():
            symbol_df = df.filter(pl.col("symbol") == symbol).drop("symbol")
            result[symbol] = symbol_df

        return result

    def iter_dates(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, pl.DataFrame]]:
        """
        Iterate over dates one at a time.

        Memory efficient for walk-forward processing.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: Optional list of symbols to filter

        Yields:
            (date_string, dataframe) tuples
        """
        paths = self._get_date_partition_paths(start_date, end_date)

        for path in paths:
            date_str = path.stem  # e.g., "2024-01-02"
            try:
                df = self.load_date(date_str, symbols=symbols)
                if not df.is_empty():
                    yield (date_str, df)
            except Exception as e:
                logger.warning(f"Error loading date {date_str}: {e}")

    # =========================================================================
    # Single Symbol Loading
    # =========================================================================

    def load_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        use_cache: bool = True,
        streaming: bool = True,
    ) -> pl.DataFrame:
        """
        Load data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Optional list of columns to select
            use_cache: Whether to use LRU cache (default True)
            streaming: Use streaming collection for lower memory (default True)

        Returns:
            Polars DataFrame with timestamp index and OHLCV columns
        """
        cache_key = f"{symbol}:{start_date}:{end_date}:{columns}"

        # Check cache
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Load data
        lf = self._scan_symbol(symbol, start_date, end_date, columns)

        if streaming:
            # Use new_streaming engine for memory-efficient collection
            df = lf.collect(engine="streaming")
        else:
            df = lf.collect()

        # Filter trading days
        df = self._filter_trading_days(df)

        # Cache result
        if use_cache:
            self._cache.put(cache_key, df)

        return df

    def load_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1min",
    ) -> pd.DataFrame:
        """
        Load data for a single symbol and return as pandas DataFrame.

        This method provides backward compatibility with the old DataLoader API.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe (ignored - uses loader's timeframe)

        Returns:
            pandas DataFrame with timestamp index and OHLCV columns
        """
        pl_df = self.load_symbol(symbol, start_date, end_date)
        pdf = pl_df.to_pandas()
        pdf = pdf.set_index("timestamp")
        return pdf

    def load_symbol_lazy(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
    ) -> pl.LazyFrame:
        """
        Return lazy reference for deferred execution.

        Useful for chaining operations before collection.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Optional columns to select

        Returns:
            Polars LazyFrame
        """
        return self._scan_symbol(symbol, start_date, end_date, columns)

    # =========================================================================
    # Symbol Iterator (Memory-Efficient Sweeps)
    # =========================================================================

    def iter_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Iterator[Tuple[str, pl.DataFrame]]:
        """
        Iterate over symbols one at a time.

        Memory efficient - only one symbol's data loaded at a time.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Optional columns to select
            use_cache: Whether to cache loaded data

        Yields:
            (symbol, dataframe) tuples
        """
        for symbol in symbols:
            try:
                df = self.load_symbol(symbol, start_date, end_date, columns, use_cache=use_cache)
                yield (symbol, df)
            except FileNotFoundError:
                logger.warning(f"No data found for {symbol}, skipping")
                continue
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue

    def iter_symbols_parallel(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        n_workers: int = 4,
        prefetch: int = 2,
    ) -> Iterator[Tuple[str, pl.DataFrame]]:
        """
        Parallel iterator with prefetching.

        Background threads load next symbols while current is being processed.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Optional columns to select
            n_workers: Number of parallel workers
            prefetch: Number of symbols to prefetch

        Yields:
            (symbol, dataframe) tuples in order
        """
        # Use a queue-based approach with ThreadPoolExecutor
        results: Dict[str, pl.DataFrame] = {}
        errors: Dict[str, Exception] = {}

        def load_one(sym: str) -> Tuple[str, Optional[pl.DataFrame], Optional[Exception]]:
            try:
                df = self.load_symbol(sym, start_date, end_date, columns, use_cache=True)
                return (sym, df, None)
            except Exception as e:
                return (sym, None, e)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit initial batch
            futures = {}
            submitted_idx = 0
            batch_size = min(prefetch * n_workers, len(symbols))

            for i in range(batch_size):
                if submitted_idx < len(symbols):
                    future = executor.submit(load_one, symbols[submitted_idx])
                    futures[future] = symbols[submitted_idx]
                    submitted_idx += 1

            # Yield results in order
            for symbol in symbols:
                # Wait for this symbol's result
                while symbol not in results and symbol not in errors:
                    # Get next completed future
                    done = next(as_completed(futures))
                    sym, df, err = done.result()
                    del futures[done]

                    if err:
                        errors[sym] = err
                    else:
                        results[sym] = df

                    # Submit next symbol if available
                    if submitted_idx < len(symbols):
                        future = executor.submit(load_one, symbols[submitted_idx])
                        futures[future] = symbols[submitted_idx]
                        submitted_idx += 1

                # Yield result for this symbol
                if symbol in errors:
                    logger.error(f"Error loading {symbol}: {errors[symbol]}")
                    continue
                elif symbol in results:
                    yield (symbol, results.pop(symbol))

    # =========================================================================
    # Multi-Symbol Synchronized Loading
    # =========================================================================

    def load_symbols_synchronized(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, pl.DataFrame]:
        """
        Load multiple symbols with aligned timestamps.

        All symbols share the same timestamp index (intersection).

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Optional columns to select

        Returns:
            Dict mapping symbol to DataFrame with synchronized timestamps
        """
        # Load all symbols
        dfs = {}
        for symbol in symbols:
            try:
                df = self.load_symbol(symbol, start_date, end_date, columns, use_cache=True)
                if not df.is_empty():
                    dfs[symbol] = df.sort("timestamp")
            except Exception as e:
                logger.warning(f"Could not load {symbol}: {e}")

        if not dfs:
            raise ValueError(f"No data loaded for any symbols: {symbols}")

        # Find common timestamps using Polars join operations to avoid type conversion issues
        # (Python datetime objects from to_list() have different timezone representation)
        symbols_list = list(dfs.keys())
        first_symbol = symbols_list[0]

        # Normalize all timestamp columns to microsecond precision to avoid join errors
        # Different data sources may have ns vs us precision
        for symbol in dfs:
            ts_dtype = dfs[symbol]["timestamp"].dtype
            if "ns" in str(ts_dtype):
                # Cast nanosecond timestamps to microsecond
                dfs[symbol] = dfs[symbol].with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )

        # Start with first symbol's timestamps as a Series
        common_ts_df = dfs[first_symbol].select("timestamp").unique()

        # Inner join with each other symbol to get intersection
        for symbol in symbols_list[1:]:
            other_ts_df = dfs[symbol].select("timestamp").unique()
            common_ts_df = common_ts_df.join(other_ts_df, on="timestamp", how="inner")

        if common_ts_df.is_empty():
            logger.warning("No common timestamps found across symbols")
            return dfs

        # Get the common timestamps as a Polars Series
        common_ts_series = common_ts_df["timestamp"]

        # Filter all dataframes to common timestamps using semi join
        for symbol in dfs:
            dfs[symbol] = dfs[symbol].join(
                common_ts_df, on="timestamp", how="semi"
            ).sort("timestamp")

        logger.info(f"Synchronized {len(dfs)} symbols to {len(common_ts_series)} common timestamps")
        return dfs

    def load_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ):
        """
        Load multiple symbols as a single MultiIndex pandas DataFrame
        with levels (symbol, timestamp). Provided for back-compat with
        the optimization callers (grid_search, random_search) that pre-date
        the polars migration; new code should prefer
        load_symbols_synchronized() (per-symbol dict) or load_symbols_panel().
        """
        import pandas as pd

        dfs = self.load_symbols_synchronized(symbols, start_date, end_date)
        if not dfs:
            return pd.DataFrame()
        frames = []
        for symbol, pl_df in dfs.items():
            pdf = pl_df.to_pandas()
            pdf["symbol"] = symbol
            frames.append(pdf)
        combined = pd.concat(frames, ignore_index=True)
        return combined.set_index(["symbol", "timestamp"]).sort_index()

    def load_symbols_panel(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_column: str = "close",
    ) -> pl.DataFrame:
        """
        Load as wide-format panel (timestamps x symbols).

        Efficient for portfolio strategies that need all prices at each timestamp.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            price_column: Which price column to use (default 'close')

        Returns:
            DataFrame with timestamp index and symbol columns
        """
        dfs = self.load_symbols_synchronized(symbols, start_date, end_date, columns=[price_column])

        # Build panel using concat + pivot (more efficient than sequential joins)
        all_dfs = []
        for symbol, df in dfs.items():
            symbol_df = df.select(["timestamp", price_column]).with_columns(
                pl.lit(symbol).alias("symbol")
            )
            all_dfs.append(symbol_df)

        if not all_dfs:
            raise ValueError("No data loaded")

        # Concat all and pivot
        combined = pl.concat(all_dfs)
        result = combined.pivot(
            on="symbol",
            index="timestamp",
            values=price_column,
        )

        return result.sort("timestamp")

    # =========================================================================
    # Optimization Cache Interface
    # =========================================================================

    def load_minute_cache(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Load into nested dict format for existing backtest functions.

        Compatible with load_minute_cache_polars() output format:
        {symbol: {date_str: {'high': array, 'low': array, 'close': array}}}

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Nested dict with numpy arrays for each symbol/date
        """
        minute_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        for i, (symbol, df) in enumerate(self.iter_symbols(symbols, start_date, end_date)):
            if df.is_empty():
                continue

            # Add date string column
            df = df.with_columns([pl.col("timestamp").dt.strftime("%Y-%m-%d").alias("date_str")])

            minute_cache[symbol] = {}

            # Group by date
            for date_str in df["date_str"].unique().to_list():
                date_df = df.filter(pl.col("date_str") == date_str)
                minute_cache[symbol][date_str] = {
                    "high": date_df["high"].to_numpy(),
                    "low": date_df["low"].to_numpy(),
                    "close": date_df["close"].to_numpy(),
                }

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} symbols into minute cache")

        logger.info(f"Loaded {len(minute_cache)} symbols into minute cache")
        return minute_cache

    def preload_to_cache(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Preload symbols into LRU cache for optimization runs.

        Call before parameter sweep to avoid repeated I/O.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info(f"Preloading {len(symbols)} symbols to cache...")
        loaded = 0
        for symbol, _ in self.iter_symbols(symbols, start_date, end_date, use_cache=True):
            loaded += 1
            if loaded % 100 == 0:
                logger.info(f"Preloaded {loaded}/{len(symbols)} symbols")
        logger.info(f"Preload complete: {loaded} symbols cached")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def estimate_memory_mb(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> float:
        """
        Estimate memory required without loading data.

        Based on partition file sizes and typical compression ratio.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Estimated memory in MB
        """
        total_size = 0
        compression_ratio = 3.0  # Parquet typically compresses ~3x

        for symbol in symbols:
            paths = self._get_partition_paths(symbol, start_date, end_date)
            for path in paths:
                if path.exists():
                    total_size += path.stat().st_size

        return (total_size * compression_ratio) / (1024 * 1024)

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols available in storage.

        Returns:
            Sorted list of symbol strings
        """
        if not self.data_path.exists():
            return []

        symbol_dirs = [d.name.replace("symbol=", "") for d in self.data_path.iterdir() if d.is_dir()]
        return sorted(symbol_dirs)

    def get_symbol_date_range(self, symbol: str) -> Tuple[str, str]:
        """
        Get available date range for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (start_date, end_date) as strings (YYYY-MM-DD)
        """
        symbol_path = self.data_path / f"symbol={symbol}"
        if not symbol_path.exists():
            raise FileNotFoundError(f"No data found for symbol {symbol}")

        # Find min/max year-month partitions
        min_year, min_month = 9999, 13
        max_year, max_month = 0, 0

        for year_dir in symbol_path.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.replace("year=", ""))

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                month = int(month_dir.name.replace("month=", ""))

                if (year, month) < (min_year, min_month):
                    min_year, min_month = year, month
                if (year, month) > (max_year, max_month):
                    max_year, max_month = year, month

        if min_year == 9999:
            raise FileNotFoundError(f"No data partitions found for symbol {symbol}")

        start_date = f"{min_year:04d}-{min_month:02d}-01"
        # Use last day of max month
        if max_month == 12:
            end_date = f"{max_year:04d}-12-31"
        else:
            from calendar import monthrange

            last_day = monthrange(max_year, max_month)[1]
            end_date = f"{max_year:04d}-{max_month:02d}-{last_day:02d}"

        return start_date, end_date

    def check_symbols_availability(self, symbols: Union[str, List[str]]) -> Dict[str, List[str]]:
        """
        Check which symbols are available in storage.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Dict with 'available' and 'missing' lists
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        available_symbols = set(self.get_available_symbols())
        requested_symbols = set(symbols)

        return {
            "available": sorted(list(requested_symbols & available_symbols)),
            "missing": sorted(list(requested_symbols - available_symbols)),
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._date_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics for both caches."""
        symbol_stats = self._cache.stats()
        date_stats = self._date_cache.stats()
        return {
            "symbol_cache": symbol_stats,
            "date_cache": date_stats,
            # Combined stats for backward compatibility
            "entries": symbol_stats["entries"] + date_stats["entries"],
            "size_mb": symbol_stats["size_mb"] + date_stats["size_mb"],
            "hits": symbol_stats["hits"] + date_stats["hits"],
            "misses": symbol_stats["misses"] + date_stats["misses"],
            "hit_rate": (
                (symbol_stats["hits"] + date_stats["hits"]) /
                max(symbol_stats["hits"] + symbol_stats["misses"] + date_stats["hits"] + date_stats["misses"], 1)
            ),
        }
