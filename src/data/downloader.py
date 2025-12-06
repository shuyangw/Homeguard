"""
Alpaca Data Downloader Module.

Provides a reusable AlpacaDownloader class for downloading historical OHLCV data
from Alpaca API with robust error handling, multithreading, and schema compliance.

Usage:
    from src.data.downloader import AlpacaDownloader, Timeframe

    downloader = AlpacaDownloader(start_date='2017-01-01')
    result = downloader.download_symbols(['AAPL', 'MSFT'], timeframe=Timeframe.MINUTE)
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest

from src.api_key import API_KEY, API_SECRET
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thread-local storage for Alpaca clients
_thread_local = threading.local()


class Timeframe(Enum):
    """Supported timeframes for data download."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


# Map our Timeframe enum to Alpaca's TimeFrame
_ALPACA_TIMEFRAME_MAP = {
    Timeframe.MINUTE: TimeFrame.Minute,
    Timeframe.HOUR: TimeFrame.Hour,
    Timeframe.DAY: TimeFrame.Day,
}

# Output directory suffixes by timeframe
_OUTPUT_DIR_SUFFIX = {
    Timeframe.MINUTE: "equities_1min",
    Timeframe.HOUR: "equities_1hour",
    Timeframe.DAY: "equities_1day",
}


@dataclass
class DownloadResult:
    """Result of a batch download operation."""
    total_symbols: int
    succeeded: int
    failed: int
    total_bars: int
    elapsed_seconds: float
    failed_symbols: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.succeeded / self.total_symbols) * 100


def _get_client() -> StockHistoricalDataClient:
    """Get thread-local Alpaca client."""
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = StockHistoricalDataClient(API_KEY, API_SECRET)
    return _thread_local.client


def _get_thread_id() -> str:
    """Get short thread ID for logging."""
    name = threading.current_thread().name
    if '_' in name:
        return f"T{name.split('_')[-1]}"
    return name


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable time string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class AlpacaDownloader:
    """
    Download historical OHLCV data from Alpaca API.

    Features:
    - Multithreaded downloads (default 6 threads)
    - Retry logic with exponential backoff
    - End-of-run retry rounds for failed symbols
    - Skip-existing support
    - Canonical 8-column schema enforcement
    - Hive partitioned output format

    Example:
        downloader = AlpacaDownloader(start_date='2020-01-01')
        result = downloader.download_symbols(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            timeframe=Timeframe.MINUTE,
            skip_existing=True
        )
        print(f"Downloaded {result.total_bars} bars")
    """

    # Default configuration
    DEFAULT_START_DATE = '2017-01-01'
    DEFAULT_NUM_THREADS = 6
    MAX_RETRIES_PER_SYMBOL = 3
    END_RETRY_ROUNDS = 3
    RETRY_DELAY = 5  # seconds

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        num_threads: int = DEFAULT_NUM_THREADS,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the downloader.

        Args:
            start_date: Start date in YYYY-MM-DD format (default: 2017-01-01)
            end_date: End date in YYYY-MM-DD format (default: today)
            num_threads: Number of download threads (default: 6)
            output_dir: Base output directory (default: from settings.ini)
        """
        self.start_date = start_date or self.DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.num_threads = num_threads
        self.base_output_dir = output_dir or get_local_storage_dir()

    def _get_output_dir(self, timeframe: Timeframe) -> Path:
        """Get output directory for a specific timeframe."""
        return self.base_output_dir / _OUTPUT_DIR_SUFFIX[timeframe]

    def get_existing_symbols(self, timeframe: Timeframe) -> Set[str]:
        """
        Get symbols that already have data downloaded.

        Args:
            timeframe: The timeframe to check

        Returns:
            Set of symbol names with existing data
        """
        output_dir = self._get_output_dir(timeframe)
        existing = set()
        if output_dir.exists():
            for symbol_dir in output_dir.glob('symbol=*'):
                symbol = symbol_dir.name.replace('symbol=', '')
                # Check if there's at least one parquet file
                if list(symbol_dir.glob('**/data*.parquet')):
                    existing.add(symbol)
        return existing

    def _download_single_symbol(
        self,
        symbol: str,
        timeframe: Timeframe,
        output_dir: Path,
    ) -> Tuple[str, bool, int, Optional[str]]:
        """
        Download data for a single symbol with retry logic.

        Args:
            symbol: Stock symbol to download
            timeframe: Timeframe for the data
            output_dir: Directory to save data

        Returns:
            Tuple of (symbol, success, bar_count, error_message)
        """
        tid = _get_thread_id()
        alpaca_timeframe = _ALPACA_TIMEFRAME_MAP[timeframe]

        for attempt in range(1, self.MAX_RETRIES_PER_SYMBOL + 1):
            try:
                if attempt == 1:
                    logger.info(f"[{tid}] {symbol}: Starting download...")
                else:
                    logger.info(f"[{tid}] {symbol}: Retry {attempt}/{self.MAX_RETRIES_PER_SYMBOL}...")

                client = _get_client()

                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=alpaca_timeframe,
                    start=datetime.strptime(self.start_date, '%Y-%m-%d'),
                    end=datetime.strptime(self.end_date, '%Y-%m-%d'),
                )

                bars = client.get_stock_bars(request)
                df = bars.df

                if df.empty:
                    logger.warning(f"[{tid}] {symbol}: No data returned")
                    return (symbol, False, 0, "No data")

                logger.info(f"[{tid}] {symbol}: Received {len(df):,} bars, processing...")

                # Reset index to get symbol out of MultiIndex
                df = df.reset_index()

                # Enforce canonical schema: 8 columns, lowercase names
                # Columns: timestamp, open, high, low, close, volume, trade_count, vwap
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']]

                # Save to Hive partitioned format: symbol=XXX/year=YYYY/month=MM/data.parquet
                df['year'] = pd.to_datetime(df['timestamp']).dt.year
                df['month'] = pd.to_datetime(df['timestamp']).dt.month

                partitions_created = 0
                for (year, month), group in df.groupby(['year', 'month']):
                    partition_dir = output_dir / f'symbol={symbol}' / f'year={year}' / f'month={month}'
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    partition_file = partition_dir / 'data.parquet'
                    data_to_save = group.drop(columns=['year', 'month'])
                    data_to_save.to_parquet(partition_file, index=False)
                    partitions_created += 1

                logger.info(f"[{tid}] {symbol}: Saved {len(df):,} bars to {partitions_created} partitions")
                return (symbol, True, len(df), None)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{tid}] {symbol}: Attempt {attempt} FAILED - {error_msg}")

                if attempt < self.MAX_RETRIES_PER_SYMBOL:
                    time.sleep(self.RETRY_DELAY * attempt)  # Exponential backoff
                else:
                    return (symbol, False, 0, error_msg)

        return (symbol, False, 0, "Max retries exceeded")

    def _download_batch(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        phase_name: str,
    ) -> Tuple[int, List[Tuple[str, str]], int]:
        """
        Download a batch of symbols with progress tracking.

        Args:
            symbols: List of symbols to download
            timeframe: Timeframe for the data
            phase_name: Name for logging (e.g., "INITIAL", "RETRY-1")

        Returns:
            Tuple of (success_count, failed_list, total_bars)
        """
        output_dir = self._get_output_dir(timeframe)
        output_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = []
        total_bars = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._download_single_symbol, sym, timeframe, output_dir): sym
                for sym in symbols
            }

            for i, future in enumerate(as_completed(futures)):
                symbol, ok, bars, error = future.result()

                if ok:
                    success += 1
                    total_bars += bars
                else:
                    failed.append((symbol, error))

                completed = i + 1
                elapsed = time.time() - start_time
                avg_per_symbol = elapsed / completed if completed > 0 else 0
                remaining = len(symbols) - completed
                eta = avg_per_symbol * remaining
                pct = completed / len(symbols) * 100

                progress_line = (
                    f"[{phase_name}] {completed}/{len(symbols)} ({pct:.1f}%) | "
                    f"OK: {success} | FAIL: {len(failed)} | "
                    f"Elapsed: {_format_time(elapsed)} | ETA: {_format_time(eta)}"
                )
                print(progress_line, flush=True)

        return success, failed, total_bars

    def download_symbols(
        self,
        symbols: List[str],
        timeframe: Timeframe = Timeframe.MINUTE,
        skip_existing: bool = False,
    ) -> DownloadResult:
        """
        Download historical data for a list of symbols.

        Args:
            symbols: List of stock symbols to download
            timeframe: Data timeframe (MINUTE, HOUR, or DAY)
            skip_existing: If True, skip symbols that already have data

        Returns:
            DownloadResult with statistics and failed symbols
        """
        output_dir = self._get_output_dir(timeframe)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"ALPACA DATA DOWNLOADER - {timeframe.value.upper()} DATA")
        logger.info("=" * 60)
        logger.info(f"Total symbols: {len(symbols)}")

        # Filter existing symbols if requested
        if skip_existing:
            existing = self.get_existing_symbols(timeframe)
            symbols_to_download = [s for s in symbols if s not in existing]
            logger.info(f"Already downloaded: {len(existing)}")
            logger.info(f"To download: {len(symbols_to_download)}")
        else:
            symbols_to_download = symbols
            logger.info("Mode: Download all (overwrite existing)")

        if not symbols_to_download:
            logger.info("No symbols to download!")
            # All symbols were skipped (already exist), so 0 downloaded
            return DownloadResult(
                total_symbols=0,
                succeeded=0,
                failed=0,
                total_bars=0,
                elapsed_seconds=0.0,
            )

        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Threads: {self.num_threads}")
        logger.info(f"Retries per symbol: {self.MAX_RETRIES_PER_SYMBOL}")
        logger.info(f"End-of-run retry rounds: {self.END_RETRY_ROUNDS}")
        logger.info("")

        # Initial download
        logger.info("=" * 60)
        logger.info("PHASE: INITIAL DOWNLOAD")
        logger.info("=" * 60)

        total_success = 0
        total_bars = 0
        overall_start = time.time()

        success, failed, bars = self._download_batch(symbols_to_download, timeframe, "INITIAL")
        total_success += success
        total_bars += bars

        logger.info("")
        logger.info(f"Initial download: {success} succeeded, {len(failed)} failed")

        # Retry rounds at the end
        for retry_round in range(1, self.END_RETRY_ROUNDS + 1):
            if not failed:
                break

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"PHASE: RETRY ROUND {retry_round}/{self.END_RETRY_ROUNDS} ({len(failed)} symbols)")
            logger.info("=" * 60)

            # Wait before retry round to allow rate limits to reset
            wait_time = 10 * retry_round
            logger.info(f"Waiting {wait_time}s before retry round...")
            time.sleep(wait_time)

            retry_symbols = [sym for sym, _ in failed]
            success, failed, bars = self._download_batch(retry_symbols, timeframe, f"RETRY-{retry_round}")
            total_success += success
            total_bars += bars

            logger.info("")
            logger.info(f"Retry round {retry_round}: {success} succeeded, {len(failed)} failed")

        # Final summary
        elapsed_total = time.time() - overall_start
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total succeeded: {total_success}/{len(symbols_to_download)}")
        logger.info(f"Total failed: {len(failed)}")
        logger.info(f"Total bars downloaded: {total_bars:,}")
        logger.info(f"Total time: {_format_time(elapsed_total)}")

        if failed:
            logger.info("")
            logger.info("PERMANENTLY FAILED SYMBOLS:")
            for sym, err in failed:
                logger.info(f"  {sym}: {err}")

            # Save failed symbols to file
            failed_file = output_dir / f'failed_symbols_{timeframe.value}.txt'
            with open(failed_file, 'w') as f:
                for sym, err in failed:
                    f.write(f"{sym},{err}\n")
            logger.info(f"Failed symbols saved to {failed_file}")
        else:
            logger.info("")
            logger.info("All symbols downloaded successfully!")

        return DownloadResult(
            total_symbols=len(symbols_to_download),
            succeeded=total_success,
            failed=len(failed),
            total_bars=total_bars,
            elapsed_seconds=elapsed_total,
            failed_symbols=failed,
        )
