"""
ThetaData Options Downloader.

Downloads historical options data from ThetaData REST API following the "Liquid 10TB" plan.
Merges OHLC, Quote, and Greeks data into a unified schema with ZSTD compression.

Features:
    - 16-thread parallel downloads
    - Robust resume capability with download manifest
    - Atomic file writes (temp file + rename)
    - Tracks complete/incomplete/failed downloads
    - Reconciles state on startup and before each download

Prerequisites:
    1. ThetaData Standard subscription ($80/month)
    2. Java 21+ installed
    3. Theta Terminal running on localhost:25503

Usage:
    # Download all symbols (full 2017-present)
    python scripts/download_options.py

    # Download specific symbols
    python scripts/download_options.py --symbols SPY,QQQ,AAPL

    # Specific date range
    python scripts/download_options.py --start 2020-01-01 --end 2024-12-31

    # Resume interrupted download (default behavior)
    python scripts/download_options.py --resume

    # Force re-download of failed entries
    python scripts/download_options.py --retry-failed

    # Adjust thread count
    python scripts/download_options.py --threads 8
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import queue
import signal
import time
import threading
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any, Set
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict

import httpx
import pandas as pd
import polars as pl

from src.settings import get_options_data_dir

# Use standard Python logging (not project's custom logger)
# This allows proper handler configuration with timestamps
logger = logging.getLogger("download_options")
logger.setLevel(logging.INFO)


# =============================================================================
# Configuration
# =============================================================================

# Default universe of liquid options
LIQUID_UNIVERSE = [
    # US Equity Indices
    "SPY", "QQQ", "IWM",
    # Technology/AI
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN",
    # Interest Rates
    "TLT",
    # Commodities
    "GLD", "SLV", "XLE",
    # Emerging Markets
    "EEM", "FXI",
    # Volatility
    "VIX",
    # Financials
    "XLF",
]

# ThetaData REST API base URLs
THETA_BASE_URL = "http://localhost:25503/v3"
THETA_BULK_URL = "http://localhost:25510/v2"  # PRO tier bulk endpoints with exp=0 support

# Default date range
DEFAULT_START_DATE = "2017-01-01"

# Moneyness filter threshold (30% OTM max)
MAX_MONEYNESS = 0.30

# Threading
DEFAULT_NUM_THREADS = 16

# Request settings
DEFAULT_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]  # seconds for each retry
RATE_LIMIT_WAIT = 60  # seconds to wait on 429

# Thread-local storage for HTTP clients
_thread_local = threading.local()


# =============================================================================
# Enums and Data Classes
# =============================================================================

class DownloadState(str, Enum):
    """State of a download entry."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class DownloadEntry:
    """Represents a single download unit (symbol + month)."""
    symbol: str
    year: int
    month: int
    state: DownloadState = DownloadState.PENDING
    bars_count: int = 0
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    file_hash: Optional[str] = None  # For integrity verification

    @property
    def key(self) -> str:
        """Unique key for this entry."""
        return f"{self.symbol}_{self.year}_{self.month:02d}"

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "year": self.year,
            "month": self.month,
            "state": self.state.value,
            "bars_count": self.bars_count,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "file_hash": self.file_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DownloadEntry":
        return cls(
            symbol=data["symbol"],
            year=data["year"],
            month=data["month"],
            state=DownloadState(data.get("state", "pending")),
            bars_count=data.get("bars_count", 0),
            error_message=data.get("error_message"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            file_hash=data.get("file_hash"),
        )


@dataclass
class DownloadManifest:
    """
    Tracks all download entries and their states.

    Persisted to disk for resume capability.
    """
    entries: Dict[str, DownloadEntry] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DownloadManifest":
        manifest = cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
        for key, entry_data in data.get("entries", {}).items():
            manifest.entries[key] = DownloadEntry.from_dict(entry_data)
        return manifest

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about download states."""
        stats = defaultdict(int)
        for entry in self.entries.values():
            stats[entry.state.value] += 1
        return dict(stats)


@dataclass
class DownloadResult:
    """Result of a download operation."""
    total_entries: int
    completed: int
    failed: int
    skipped: int
    total_bars: int
    elapsed_seconds: float


# =============================================================================
# Progress Tracker
# =============================================================================

class ProgressTracker:
    """
    Tracks and reports download progress with detailed metrics.

    Thread-safe progress tracking with periodic disk writes.
    """

    def __init__(self, log_dir: Path, total_entries: int):
        self.log_dir = log_dir
        self.total_entries = total_entries
        self.lock = threading.RLock()  # Reentrant lock to avoid deadlock

        # Counters
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.total_bars = 0
        self.total_bytes = 0

        # Timing
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.report_interval = 30  # seconds between detailed reports

        # Per-symbol tracking
        self.symbol_progress: Dict[str, Dict] = {}

        # Progress file
        self.progress_file = log_dir / "progress.txt"

        # Write initial status
        self._write_progress_file()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes as human-readable string."""
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f} KB"
        elif bytes_val < 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"

    def _calculate_eta(self) -> str:
        """Calculate estimated time remaining."""
        with self.lock:
            processed = self.completed + self.failed + self.skipped
            if processed == 0:
                return "calculating..."

            elapsed = time.time() - self.start_time
            rate = processed / elapsed  # entries per second
            remaining = self.total_entries - processed

            if rate > 0:
                eta_seconds = remaining / rate
                return self._format_time(eta_seconds)
            return "unknown"

    def _write_progress_file(self) -> None:
        """Write current progress to disk."""
        with self.lock:
            elapsed = time.time() - self.start_time
            processed = self.completed + self.failed + self.skipped
            pct = (processed / self.total_entries * 100) if self.total_entries > 0 else 0

            lines = [
                f"=" * 70,
                f"OPTIONS DOWNLOAD PROGRESS",
                f"=" * 70,
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Elapsed: {self._format_time(elapsed)}",
                f"ETA: {self._calculate_eta()}",
                f"",
                f"OVERALL PROGRESS: {processed}/{self.total_entries} ({pct:.1f}%)",
                f"  Completed: {self.completed}",
                f"  Failed: {self.failed}",
                f"  Skipped: {self.skipped}",
                f"  Remaining: {self.total_entries - processed}",
                f"",
                f"DATA:",
                f"  Total bars: {self.total_bars:,}",
                f"  Total size: {self._format_bytes(self.total_bytes)}",
                f"",
                f"THROUGHPUT:",
            ]

            if elapsed > 0:
                bars_per_sec = self.total_bars / elapsed
                lines.append(f"  Bars/sec: {bars_per_sec:,.0f}")
                entries_per_min = (processed / elapsed) * 60
                lines.append(f"  Entries/min: {entries_per_min:.1f}")

            lines.extend([
                f"",
                f"PER-SYMBOL STATUS:",
            ])

            for symbol, data in sorted(self.symbol_progress.items()):
                status = data.get("status", "unknown")
                sym_completed = data.get("completed", 0)
                sym_total = data.get("total", 0)
                sym_bars = data.get("bars", 0)
                sym_pct = (sym_completed / sym_total * 100) if sym_total > 0 else 0
                lines.append(
                    f"  {symbol:6s}: {status:12s} | "
                    f"{sym_completed}/{sym_total} ({sym_pct:5.1f}%) | "
                    f"{sym_bars:,} bars"
                )

            lines.append(f"=" * 70)

        # Write to file
        try:
            with open(self.progress_file, "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.error(f"Failed to write progress file: {e}")

    def start_symbol(self, symbol: str, total_months: int) -> None:
        """Mark a symbol as started."""
        with self.lock:
            self.symbol_progress[symbol] = {
                "status": "downloading",
                "total": total_months,
                "completed": 0,
                "failed": 0,
                "bars": 0,
                "started_at": time.time(),
            }

    def complete_entry(
        self,
        symbol: str,
        entry_key: str,
        bars: int,
        bytes_written: int = 0,
    ) -> None:
        """Mark an entry as completed."""
        with self.lock:
            self.completed += 1
            self.total_bars += bars
            self.total_bytes += bytes_written

            if symbol in self.symbol_progress:
                self.symbol_progress[symbol]["completed"] += 1
                self.symbol_progress[symbol]["bars"] += bars

        self._maybe_report()

    def fail_entry(self, symbol: str, entry_key: str, error: str) -> None:
        """Mark an entry as failed."""
        with self.lock:
            self.failed += 1

            if symbol in self.symbol_progress:
                self.symbol_progress[symbol]["failed"] += 1

        logger.error(f"[FAILED] {entry_key}: {error}")
        self._maybe_report()

    def skip_entry(self, symbol: str, entry_key: str) -> None:
        """Mark an entry as skipped."""
        with self.lock:
            self.skipped += 1

            if symbol in self.symbol_progress:
                self.symbol_progress[symbol]["completed"] += 1

        self._maybe_report()

    def complete_symbol(self, symbol: str) -> None:
        """Mark a symbol as completed."""
        with self.lock:
            if symbol in self.symbol_progress:
                data = self.symbol_progress[symbol]
                elapsed = time.time() - data.get("started_at", time.time())
                data["status"] = "complete"
                data["elapsed"] = elapsed

                logger.info(
                    f"[{symbol}] COMPLETE: "
                    f"{data['completed']}/{data['total']} entries, "
                    f"{data['bars']:,} bars in {self._format_time(elapsed)}"
                )

        self._write_progress_file()

    def _maybe_report(self) -> None:
        """Write progress report if enough time has passed."""
        now = time.time()
        with self.lock:
            if now - self.last_report_time >= self.report_interval:
                self.last_report_time = now
                self._write_progress_file()
                self._log_summary()

    def _log_summary(self) -> None:
        """Log a summary to the console and log file."""
        with self.lock:
            elapsed = time.time() - self.start_time
            processed = self.completed + self.failed + self.skipped
            pct = (processed / self.total_entries * 100) if self.total_entries > 0 else 0

            logger.info(
                f"[PROGRESS] {processed}/{self.total_entries} ({pct:.1f}%) | "
                f"OK: {self.completed} | FAIL: {self.failed} | SKIP: {self.skipped} | "
                f"Bars: {self.total_bars:,} | "
                f"Elapsed: {self._format_time(elapsed)} | "
                f"ETA: {self._calculate_eta()}"
            )

    def final_report(self) -> None:
        """Write final progress report."""
        self._write_progress_file()

        with self.lock:
            elapsed = time.time() - self.start_time

            logger.info("")
            logger.info("=" * 70)
            logger.info("FINAL DOWNLOAD REPORT")
            logger.info("=" * 70)
            logger.info(f"Total time: {self._format_time(elapsed)}")
            logger.info(f"Entries processed: {self.completed + self.failed + self.skipped}")
            logger.info(f"  Completed: {self.completed}")
            logger.info(f"  Failed: {self.failed}")
            logger.info(f"  Skipped: {self.skipped}")
            logger.info(f"Total bars: {self.total_bars:,}")
            logger.info(f"Total size: {self._format_bytes(self.total_bytes)}")

            if elapsed > 0:
                logger.info(f"Throughput: {self.total_bars / elapsed:,.0f} bars/sec")

            logger.info("")
            logger.info("Per-symbol summary:")
            for symbol, data in sorted(self.symbol_progress.items()):
                sym_elapsed = data.get("elapsed", 0)
                logger.info(
                    f"  {symbol:6s}: {data['completed']}/{data['total']} entries, "
                    f"{data['bars']:,} bars, {self._format_time(sym_elapsed)}"
                )

            logger.info("=" * 70)
            logger.info(f"Progress file: {self.progress_file}")
            logger.info("=" * 70)


# =============================================================================
# File Logging Setup
# =============================================================================

class FlushingFileHandler(RotatingFileHandler):
    """File handler that flushes after every write for real-time logging."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class TimestampFormatter(logging.Formatter):
    """Formatter that adds HH:MM:SS timestamp to all messages."""

    def format(self, record):
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Get the original message
        original = super().format(record)
        # Prepend timestamp
        return f"{timestamp} {original}"


def setup_file_logging(output_dir: Path) -> Path:
    """
    Set up file logging to the options data directory.

    Returns the log directory path.
    """
    global logger

    log_dir = output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")

    # Main log file (flushes immediately)
    main_log = log_dir / f"download_{today}.log"
    main_handler = FlushingFileHandler(
        main_log,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Error log file (flushes immediately)
    error_log = log_dir / f"errors_{today}.log"
    error_handler = FlushingFileHandler(
        error_log,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Reconfigure the module logger with timestamp formatter
    # Clear existing handlers and add new ones with timestamps
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Console handler with timestamps
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(TimestampFormatter("%(message)s"))
    logger.addHandler(console_handler)

    # Also add file handlers to our logger
    logger.addHandler(main_handler)
    logger.addHandler(error_handler)

    logger.info(f"File logging enabled: {log_dir}")
    logger.info(f"  Main log: {main_log}")
    logger.info(f"  Error log: {error_log}")

    return log_dir


# =============================================================================
# ThetaData Client (Thread-Safe)
# =============================================================================

def _get_thread_client(timeout: int = DEFAULT_TIMEOUT) -> httpx.Client:
    """Get thread-local HTTP client."""
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = httpx.Client(timeout=timeout)
    return _thread_local.client


def _get_thread_id() -> str:
    """Get short thread ID for logging."""
    name = threading.current_thread().name
    if '_' in name:
        return f"T{name.split('_')[-1]}"
    return name[:8]


class ThetaDataClient:
    """
    Thread-safe REST API client for ThetaData with retry logic.

    Each thread gets its own HTTP client via thread-local storage.
    """

    def __init__(
        self,
        base_url: str = THETA_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _request_with_retry(
        self,
        endpoint: str,
        params: Dict[str, Any],
        symbol: str = "",
        date_str: str = "",
        use_bulk_url: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Make a request with retry logic (thread-safe).

        Args:
            endpoint: API endpoint path
            params: Query parameters
            symbol: Symbol for logging
            date_str: Date for logging
            use_bulk_url: If True, use THETA_BULK_URL (port 25510) instead of base_url

        Returns:
            Tuple of (DataFrame or None, error_message or None)
        """
        client = _get_thread_client(self.timeout)
        base = THETA_BULK_URL if use_bulk_url else self.base_url
        url = f"{base}{endpoint}"
        tid = _get_thread_id()

        for attempt in range(self.max_retries):
            try:
                response = client.get(url, params=params)

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"[{tid}] Rate limited on {endpoint}, waiting {RATE_LIMIT_WAIT}s..."
                        )
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    return None, "Rate limit exceeded after retries"

                # Handle server errors
                if response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        wait_time = RETRY_BACKOFF[attempt]
                        logger.warning(
                            f"[{tid}] Server error {response.status_code}, "
                            f"retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    return None, f"Server error {response.status_code}"

                # Handle client errors
                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}"

                # Parse response
                try:
                    if params.get("format", "csv") == "csv":
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text))
                    else:
                        data = response.json()
                        df = pd.DataFrame(data)

                    if df.empty:
                        return None, None  # Empty but not an error

                    return df, None

                except Exception as e:
                    return None, f"Parse error: {str(e)}"

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    wait_time = RETRY_BACKOFF[attempt]
                    logger.warning(f"[{tid}] Timeout, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None, "Request timed out"

            except httpx.ConnectError as e:
                return None, f"Connection failed: Is Theta Terminal running?"

            except Exception as e:
                return None, f"Unexpected error: {str(e)}"

        return None, "Max retries exceeded"

    def verify_connection(self) -> bool:
        """Verify that Theta Terminal is running and accessible."""
        try:
            df, error = self._request_with_retry(
                "/option/list/expirations",
                {"symbol": "SPY", "format": "csv"},
                symbol="SPY",
            )
            if error and "Connection" in error:
                return False
            return True
        except Exception:
            return False

    def get_expirations(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        """Get all available expiration dates for a symbol."""
        df, error = self._request_with_retry(
            "/option/list/expirations",
            {"symbol": symbol, "format": "csv"},
            symbol=symbol,
        )
        if error:
            return [], error
        if df is None or df.empty:
            return [], None
        return df["expiration"].tolist(), None

    def get_ohlc(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
        interval: str = "1m",
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get OHLC data for all strikes on a specific date."""
        return self._request_with_retry(
            "/option/history/ohlc",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "interval": interval,
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
            symbol=symbol,
            date_str=date_str,
        )

    def get_quotes(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
        interval: str = "1m",
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get quote (NBBO) data for all strikes on a specific date."""
        return self._request_with_retry(
            "/option/history/quote",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "interval": interval,
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
            symbol=symbol,
            date_str=date_str,
        )

    def get_greeks(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
        interval: str = "1m",
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get first-order Greeks (delta, theta, vega) for all strikes on a specific date."""
        return self._request_with_retry(
            "/option/history/greeks/first_order",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "interval": interval,
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
            symbol=symbol,
            date_str=date_str,
        )

    def get_greeks_second_order(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
        interval: str = "1m",
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get second-order Greeks (gamma, vanna, charm, vomma, veta) for all strikes."""
        return self._request_with_retry(
            "/option/history/greeks/second_order",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "interval": interval,
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
            symbol=symbol,
            date_str=date_str,
        )

    def get_open_interest(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Get EOD open interest for all strikes on a specific date.

        Note: OI is reported once per day (~06:30 ET) by OPRA.
        """
        return self._request_with_retry(
            "/option/history/open_interest",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
            symbol=symbol,
            date_str=date_str,
        )

    # =========================================================================
    # BULK ENDPOINTS (PRO tier - port 25510, v2 API with exp=0)
    # These fetch ALL expirations in a single call - 250x faster!
    # =========================================================================
    
    def get_bulk_ohlc(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Get OHLC data for ALL expirations on a specific date (PRO tier).
        Uses exp=0 on v2 bulk endpoint - single call instead of 250+.
        """
        # Format date as YYYYMMDD integer
        date_int = date_str.replace("-", "")
        return self._request_with_retry(
            "/bulk_hist/option/ohlc",
            {
                "root": symbol,
                "exp": 0,  # Magic: 0 = ALL expirations
                "start_date": date_int,
                "end_date": date_int,
                "ivl": 60000,  # 1 minute in milliseconds
                "use_csv": "true",
            },
            symbol=symbol,
            date_str=date_str,
            use_bulk_url=True,
        )

    def get_bulk_quotes(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Get quote data for ALL expirations on a specific date (PRO tier).
        """
        date_int = date_str.replace("-", "")
        return self._request_with_retry(
            "/bulk_hist/option/quote",
            {
                "root": symbol,
                "exp": 0,
                "start_date": date_int,
                "end_date": date_int,
                "ivl": 60000,
                "use_csv": "true",
            },
            symbol=symbol,
            date_str=date_str,
            use_bulk_url=True,
        )

    def get_bulk_greeks(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Get Greeks data for ALL expirations on a specific date (PRO tier).
        """
        date_int = date_str.replace("-", "")
        return self._request_with_retry(
            "/bulk_hist/option/greeks",
            {
                "root": symbol,
                "exp": 0,
                "start_date": date_int,
                "end_date": date_int,
                "ivl": 60000,
                "use_csv": "true",
            },
            symbol=symbol,
            date_str=date_str,
            use_bulk_url=True,
        )

    def verify_bulk_connection(self) -> bool:
        """
        Verify that bulk endpoints are available (PRO tier required).

        Returns True if bulk endpoints respond, False otherwise.
        """
        try:
            client = _get_thread_client(self.timeout)
            # Try a simple bulk request for SPY on a recent date
            url = f"{THETA_BULK_URL}/bulk_hist/option/ohlc"
            params = {
                "root": "SPY",
                "exp": 0,
                "start_date": "20241201",
                "end_date": "20241201",
                "ivl": 60000,
                "use_csv": "true",
            }
            response = client.get(url, params=params, timeout=30)
            if response.status_code == 200:
                logger.info("[+] Bulk endpoints available (PRO tier confirmed)")
                return True
            else:
                logger.warning(f"Bulk endpoint returned {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Bulk endpoints not available: {e}")
            return False


# =============================================================================
# Manifest Manager
# =============================================================================

class ManifestManager:
    """
    Manages the download manifest with thread-safe operations.

    The manifest tracks all download entries and their states,
    enabling robust resume capability.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.lock = threading.Lock()
        self.manifest = self._load_or_create()

    def _load_or_create(self) -> DownloadManifest:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                manifest = DownloadManifest.from_dict(data)
                logger.info(f"Loaded manifest with {len(manifest.entries)} entries")
                return manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}, creating new one")

        manifest = DownloadManifest(created_at=datetime.now().isoformat())
        return manifest

    def save(self) -> None:
        """Save manifest to disk atomically."""
        with self.lock:
            self.manifest.updated_at = datetime.now().isoformat()

            # Write to temp file first, then rename (atomic on most filesystems)
            temp_path = self.manifest_path.with_suffix('.tmp')
            try:
                with open(temp_path, "w") as f:
                    json.dump(self.manifest.to_dict(), f, indent=2)
                shutil.move(str(temp_path), str(self.manifest_path))
            except Exception as e:
                logger.error(f"Failed to save manifest: {e}")
                if temp_path.exists():
                    temp_path.unlink()

    def get_entry(self, key: str) -> Optional[DownloadEntry]:
        """Get an entry by key (thread-safe)."""
        with self.lock:
            return self.manifest.entries.get(key)

    def set_entry(self, entry: DownloadEntry) -> None:
        """Set an entry (thread-safe)."""
        with self.lock:
            self.manifest.entries[entry.key] = entry

    def update_state(
        self,
        key: str,
        state: DownloadState,
        bars_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Update entry state (thread-safe)."""
        with self.lock:
            if key in self.manifest.entries:
                entry = self.manifest.entries[key]
                entry.state = state
                entry.bars_count = bars_count
                entry.error_message = error_message
                if state == DownloadState.COMPLETE:
                    entry.completed_at = datetime.now().isoformat()

    def get_pending_entries(
        self,
        include_failed: bool = False,
        include_incomplete: bool = True,
    ) -> List[DownloadEntry]:
        """Get entries that need to be downloaded."""
        with self.lock:
            result = []
            for entry in self.manifest.entries.values():
                if entry.state == DownloadState.PENDING:
                    result.append(entry)
                elif entry.state == DownloadState.IN_PROGRESS and include_incomplete:
                    result.append(entry)
                elif entry.state == DownloadState.FAILED and include_failed:
                    result.append(entry)
            return result

    def get_stats(self) -> Dict[str, int]:
        """Get download statistics."""
        with self.lock:
            return self.manifest.get_stats()

    def initialize_entries(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Initialize entries for all symbol/month combinations.

        Returns count of new entries added.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        new_count = 0
        with self.lock:
            for symbol in symbols:
                current = start.replace(day=1)
                while current <= end:
                    entry = DownloadEntry(
                        symbol=symbol,
                        year=current.year,
                        month=current.month,
                    )
                    if entry.key not in self.manifest.entries:
                        self.manifest.entries[entry.key] = entry
                        new_count += 1

                    # Move to next month
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)

        return new_count


# =============================================================================
# ThetaData Downloader
# =============================================================================

class ThetaDataDownloader:
    """
    Downloads and processes options data from ThetaData.

    Features:
    - 16-thread parallel downloads
    - Robust resume capability with manifest tracking
    - Atomic file writes
    - Detailed progress tracking with disk logging
    - Merges OHLC, Quote, and Greeks data
    - Applies moneyness filter
    - Saves to Hive-partitioned parquet with ZSTD compression
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_moneyness: float = MAX_MONEYNESS,
        num_threads: int = DEFAULT_NUM_THREADS,
        retry_failed: bool = False,
        log_dir: Optional[Path] = None,
        write_daily: bool = False,
        use_bulk: bool = True,  # Use PRO bulk endpoints (exp=0) - 250x faster!
    ):
        self.output_dir = output_dir or get_options_data_dir()
        self.data_dir = self.output_dir / "options_1min"
        self.log_dir = log_dir or (self.output_dir / "_logs")
        self.symbols = symbols or LIQUID_UNIVERSE
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.max_moneyness = max_moneyness
        self.num_threads = num_threads
        self.retry_failed = retry_failed
        self.write_daily = write_daily
        self.use_bulk = use_bulk  # Will be verified at runtime

        self.client = ThetaDataClient()
        self._shutdown_requested = False
        self._active_futures: Set = set()
        self.progress: Optional[ProgressTracker] = None
        self._bulk_available: Optional[bool] = None  # Cached bulk availability

        # Thread-safe expiration cache for work queue parallelization
        self._expiration_cache: Dict[str, List[str]] = {}
        self._expiration_cache_lock = threading.Lock()

        # Work queue coordination
        self._work_queue: Optional[queue.Queue] = None
        self._rate_limit_event = threading.Event()  # Set when NOT rate limited
        self._rate_limit_event.set()  # Start in non-limited state
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_until = 0.0  # Timestamp when rate limit expires
        self._worker_stats: Dict[str, Dict] = {}  # Per-thread statistics
        self._stats_lock = threading.Lock()
        self._in_flight: Set[str] = set()  # Entry keys currently being processed
        self._in_flight_lock = threading.Lock()  # Protects _in_flight set

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize manifest manager
        manifest_path = self.output_dir / "_manifest.json"
        self.manifest = ManifestManager(manifest_path)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C."""
        if self._shutdown_requested:
            logger.warning("Force shutdown requested")
            sys.exit(1)

        logger.warning("Shutdown requested. Completing current downloads...")
        self._shutdown_requested = True

    def _get_expirations_cached(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        """Get expirations for a symbol with thread-safe caching."""
        with self._expiration_cache_lock:
            if symbol in self._expiration_cache:
                return self._expiration_cache[symbol], None

        # Fetch outside lock to allow parallel fetches for different symbols
        expirations, error = self.client.get_expirations(symbol)

        if not error and expirations:
            with self._expiration_cache_lock:
                self._expiration_cache[symbol] = expirations

        return expirations or [], error

    def _get_partition_path(self, symbol: str, year: int, month: int, day: Optional[int] = None) -> Path:
        """Get the path to a partition file."""
        base = (
            self.data_dir
            / f"root={symbol}"
            / f"year={year}"
            / f"month={month:02d}"
        )
        if day is not None:
            return base / f"day_{day:02d}.parquet"
        return base / "data.parquet"

    def _partition_exists_and_valid(self, symbol: str, year: int, month: int) -> bool:
        """Check if a partition exists and is valid (non-empty)."""
        path = self._get_partition_path(symbol, year, month)
        if not path.exists():
            return False

        try:
            # Check file size > 0 and can be read
            if path.stat().st_size == 0:
                return False

            # Try to read metadata
            df = pl.scan_parquet(path)
            schema = df.collect_schema()
            return len(schema) > 0
        except Exception:
            return False

    def _reconcile_entry(self, entry: DownloadEntry) -> DownloadEntry:
        """
        Reconcile entry state with actual files on disk.

        Called before processing each entry to ensure consistency.
        """
        # Check if file exists and is valid
        file_exists = self._partition_exists_and_valid(
            entry.symbol, entry.year, entry.month
        )

        if file_exists:
            # File exists - mark as complete if not already
            if entry.state != DownloadState.COMPLETE:
                logger.info(
                    f"Reconciled {entry.key}: file exists, marking complete"
                )
                entry.state = DownloadState.COMPLETE
                entry.completed_at = datetime.now().isoformat()
        else:
            # File doesn't exist
            if entry.state == DownloadState.COMPLETE:
                # Was marked complete but file is missing - needs re-download
                logger.warning(
                    f"Reconciled {entry.key}: marked complete but file missing, "
                    f"marking as pending"
                )
                entry.state = DownloadState.PENDING
                entry.completed_at = None
            elif entry.state == DownloadState.IN_PROGRESS:
                # Was in progress - incomplete, needs retry
                logger.info(
                    f"Reconciled {entry.key}: was in_progress, marking pending"
                )
                entry.state = DownloadState.PENDING

        return entry

    def _get_dates_for_month(self, year: int, month: int) -> List[str]:
        """Get all trading days in a month."""
        from calendar import monthrange

        _, last_day = monthrange(year, month)
        start = datetime(year, month, 1)
        end = datetime(year, month, last_day)

        # Clamp to our date range
        range_start = datetime.strptime(self.start_date, "%Y-%m-%d")
        range_end = datetime.strptime(self.end_date, "%Y-%m-%d")

        start = max(start, range_start)
        end = min(end, range_end)

        dates = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return dates

    # Expected columns in final schema
    EXPECTED_COLUMNS = [
        "timestamp", "expiration", "strike", "right",
        "open", "high", "low", "close", "volume", "trade_count", "vwap",
        "bid_close", "ask_close",
        "implied_vol", "delta", "gamma", "theta", "vega",
        "underlying_px", "open_interest"
    ]

    def _ensure_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure DataFrame has all expected columns (add nulls for missing)."""
        for col in self.EXPECTED_COLUMNS:
            if col not in df.columns:
                # Add missing column with nulls
                if col in ["volume", "trade_count", "open_interest"]:
                    df = df.with_columns(pl.lit(None).cast(pl.Int32).alias(col))
                elif col == "right":
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
                elif col == "expiration":
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
                elif col == "timestamp":
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
                else:
                    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # Reorder to match expected schema
        available = [c for c in self.EXPECTED_COLUMNS if c in df.columns]
        return df.select(available)

    def _merge_data(
        self,
        ohlc_df: Optional[pd.DataFrame],
        quote_df: Optional[pd.DataFrame],
        greeks_df: Optional[pd.DataFrame],
    ) -> Optional[pl.DataFrame]:
        """Merge OHLC, Quote, and Greeks data into unified schema."""
        if ohlc_df is None or ohlc_df.empty:
            return None

        try:
            ohlc = pl.from_pandas(ohlc_df)

            # Rename columns to match our schema
            if "count" in ohlc.columns:
                ohlc = ohlc.rename({"count": "trade_count"})

            # Select OHLC columns
            ohlc_cols = [
                "timestamp", "expiration", "strike", "right",
                "open", "high", "low", "close", "volume", "trade_count", "vwap"
            ]
            ohlc = ohlc.select([c for c in ohlc_cols if c in ohlc.columns])

            result = ohlc

            # Merge quotes if available
            if quote_df is not None and not quote_df.empty:
                quotes = pl.from_pandas(quote_df)
                if "bid" in quotes.columns:
                    quotes = quotes.rename({"bid": "bid_close", "ask": "ask_close"})
                quote_cols = ["timestamp", "expiration", "strike", "right", "bid_close", "ask_close"]
                quotes = quotes.select([c for c in quote_cols if c in quotes.columns])

                result = result.join(
                    quotes,
                    on=["timestamp", "expiration", "strike", "right"],
                    how="left",
                )

            # Merge Greeks if available
            if greeks_df is not None and not greeks_df.empty:
                greeks = pl.from_pandas(greeks_df)
                rename_map = {}
                if "implied_volatility" in greeks.columns:
                    rename_map["implied_volatility"] = "implied_vol"
                if "underlying_price" in greeks.columns:
                    rename_map["underlying_price"] = "underlying_px"
                if rename_map:
                    greeks = greeks.rename(rename_map)

                greeks_cols = [
                    "timestamp", "expiration", "strike", "right",
                    "implied_vol", "delta", "theta", "vega",
                    "underlying_px"
                ]
                greeks = greeks.select([c for c in greeks_cols if c in greeks.columns])

                result = result.join(
                    greeks,
                    on=["timestamp", "expiration", "strike", "right"],
                    how="left",
                )

            # Ensure consistent schema
            result = self._ensure_schema(result)
            return result

        except Exception as e:
            logger.error(f"Failed to merge data: {e}")
            return None

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply moneyness filter and optimize types."""
        if df is None or df.is_empty():
            return df

        # Apply moneyness filter if we have underlying price
        if "underlying_px" in df.columns:
            df = df.filter(
                (pl.col("strike") - pl.col("underlying_px")).abs()
                / pl.col("underlying_px")
                <= self.max_moneyness
            )

        # Optimize types for storage
        if "right" in df.columns:
            df = df.with_columns(pl.col("right").cast(pl.Categorical))

        if "expiration" in df.columns:
            try:
                df = df.with_columns(pl.col("expiration").str.to_date("%Y-%m-%d"))
            except Exception:
                pass  # Already date type

        # Cast floats to float64
        float_cols = [
            "open", "high", "low", "close", "vwap", "strike",
            "bid_close", "ask_close", "underlying_px",
            "implied_vol", "delta", "gamma", "theta", "vega"
        ]
        for col in float_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        # Cast ints to int32
        int_cols = ["volume", "trade_count", "open_interest"]
        for col in int_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Int32))

        return df

    def _save_partition_atomic(
        self,
        df: pl.DataFrame,
        symbol: str,
        year: int,
        month: int,
        day: Optional[int] = None,
    ) -> bool:
        """
        Save data to partition atomically.

        Writes to temp file first, then renames to final location.
        Returns True if successful.
        """
        partition_dir = (
            self.data_dir
            / f"root={symbol}"
            / f"year={year}"
            / f"month={month:02d}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        if day is not None:
            filename = f"day_{day:02d}.parquet"
        else:
            filename = "data.parquet"

        final_path = partition_dir / filename
        temp_path = partition_dir / f"{filename}.{uuid.uuid4().hex[:8]}.tmp"

        try:
            # Write to temp file
            df.write_parquet(
                temp_path,
                compression="zstd",
                use_pyarrow=True,
            )

            # Atomic rename
            shutil.move(str(temp_path), str(final_path))
            return True

        except Exception as e:
            logger.error(f"Failed to save partition {symbol}/{year}/{month}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def _download_month(
        self,
        entry: DownloadEntry,
        expirations: List[str],
    ) -> Tuple[int, Optional[str]]:
        """
        Download all data for a single month.

        Returns:
            Tuple of (bars_count, error_message or None)
        """
        tid = _get_thread_id()
        symbol = entry.symbol
        year = entry.year
        month = entry.month
        month_start = time.time()

        # Reconcile entry state with disk
        entry = self._reconcile_entry(entry)
        self.manifest.set_entry(entry)

        # Skip if already complete
        if entry.state == DownloadState.COMPLETE:
            logger.info(f"[{tid}] {entry.key}: Already complete, skipping")
            if self.progress:
                self.progress.skip_entry(symbol, entry.key)
            return entry.bars_count, None

        # Mark as in progress
        entry.state = DownloadState.IN_PROGRESS
        entry.started_at = datetime.now().isoformat()
        self.manifest.set_entry(entry)
        self.manifest.save()

        # Get dates for this month
        dates = self._get_dates_for_month(year, month)
        if not dates:
            logger.warning(f"[{tid}] {entry.key}: No trading days in range")
            entry.state = DownloadState.COMPLETE
            entry.bars_count = 0
            self.manifest.set_entry(entry)
            if self.progress:
                self.progress.complete_entry(symbol, entry.key, 0)
            return 0, None

        # Filter expirations to only those that could exist during this month
        # Options typically list 6-12 months ahead, so filter to expirations
        # within a reasonable window (1 year) from this month
        month_start_str = f"{year}-{month:02d}-01"
        month_end = datetime(year, month, 1) + timedelta(days=365)  # 1 year ahead
        month_end_str = month_end.strftime("%Y-%m-%d")

        relevant_expirations = [
            exp for exp in expirations
            if exp >= month_start_str and exp <= month_end_str
        ]

        logger.info(
            f"[{tid}] {entry.key}: {len(dates)} days, "
            f"{len(relevant_expirations)}/{len(expirations)} relevant expirations"
        )

        all_data = []  # Only used when not write_daily
        total_bars = 0
        total_bytes = 0
        dates_processed = 0
        expirations_processed = 0
        api_calls = 0

        for date_str in dates:
            if self._shutdown_requested:
                return total_bars, "Shutdown requested"

            # In daily mode, skip days that already have files (resume support)
            if self.write_daily:
                day_num = int(date_str.split("-")[2])
                existing_path = self._get_partition_path(symbol, year, month, day=day_num)
                if existing_path.exists() and existing_path.stat().st_size > 0:
                    logger.info(f"[{tid}] {entry.key} {date_str}: already exists, skipping")
                    dates_processed += 1
                    continue

            # Filter to expirations >= current date (can't trade expired options)
            valid_expirations = [exp for exp in relevant_expirations if exp >= date_str]
            day_data = []
            date_bars = 0
            date_start = time.time()

            for expiration in valid_expirations:
                if self._shutdown_requested:
                    break

                # Fetch OHLC
                ohlc_df, ohlc_error = self.client.get_ohlc(
                    symbol, expiration, date_str
                )
                api_calls += 1
                if ohlc_error:
                    continue  # Skip this expiration

                if ohlc_df is None or ohlc_df.empty:
                    continue

                # Fetch Quotes
                quote_df, _ = self.client.get_quotes(symbol, expiration, date_str)
                api_calls += 1

                # Fetch first-order Greeks (delta, theta, vega)
                greeks_df, _ = self.client.get_greeks(symbol, expiration, date_str)
                api_calls += 1

                # Merge data
                merged = self._merge_data(ohlc_df, quote_df, greeks_df)
                if merged is not None and not merged.is_empty():
                    day_data.append(merged)
                    date_bars += len(merged)
                    expirations_processed += 1

            # Write daily if enabled, otherwise accumulate
            if self.write_daily and day_data:
                combined = pl.concat(day_data)
                filtered = self._apply_filters(combined)
                if filtered is not None and not filtered.is_empty():
                    # Parse day from date_str
                    day_num = int(date_str.split("-")[2])
                    if self._save_partition_atomic(filtered, symbol, year, month, day=day_num):
                        partition_path = self._get_partition_path(symbol, year, month, day=day_num)
                        if partition_path.exists():
                            total_bytes += partition_path.stat().st_size
                    date_bars = len(filtered)
                # Clear memory immediately
                day_data.clear()
                del combined, filtered
            elif day_data:
                # Accumulate for monthly write
                all_data.extend(day_data)

            total_bars += date_bars
            dates_processed += 1
            date_elapsed = time.time() - date_start

            # Log per-date progress (every date for visibility)
            logger.info(
                f"[{tid}] {entry.key} {date_str}: "
                f"{len(valid_expirations)} exp, {date_bars:,} bars, "
                f"{date_elapsed:.1f}s ({dates_processed}/{len(dates)} days)"
            )
            
            # Trigger progress update periodically (even mid-entry)
            if self.progress:
                self.progress._maybe_report()

        if self._shutdown_requested:
            # Don't save incomplete data
            return total_bars, "Shutdown requested"

        # For monthly mode, combine and save all data at once
        bytes_written = total_bytes
        if not self.write_daily and all_data:
            combined = pl.concat(all_data)
            filtered = self._apply_filters(combined)

            if filtered is not None and not filtered.is_empty():
                # Save atomically
                if self._save_partition_atomic(filtered, symbol, year, month):
                    # Get file size for progress tracking
                    partition_path = self._get_partition_path(symbol, year, month)
                    if partition_path.exists():
                        bytes_written = partition_path.stat().st_size
                    total_bars = len(filtered)
                else:
                    entry.state = DownloadState.FAILED
                    entry.error_message = "Failed to save partition"
                    self.manifest.set_entry(entry)
                    if self.progress:
                        self.progress.fail_entry(symbol, entry.key, "Failed to save")
                    return 0, "Failed to save partition"

        # Mark entry complete
        entry.state = DownloadState.COMPLETE
        entry.bars_count = total_bars
        entry.completed_at = datetime.now().isoformat()
        self.manifest.set_entry(entry)
        self.manifest.save()

        elapsed = time.time() - month_start
        if total_bars > 0:
            logger.info(
                f"[+] [{tid}] {entry.key}: Saved {total_bars:,} bars "
                f"({bytes_written / 1024 / 1024:.1f} MB) in {elapsed:.1f}s "
                f"({api_calls} API calls)"
            )
        else:
            logger.info(f"[{tid}] {entry.key}: No data (empty month) in {elapsed:.1f}s")

        if self.progress:
            self.progress.complete_entry(symbol, entry.key, total_bars, bytes_written)
        return total_bars, None

    def _download_month_bulk(
        self,
        entry: DownloadEntry,
    ) -> Tuple[int, Optional[str]]:
        """
        Download all data for a single month using BULK endpoints (PRO tier).

        This uses exp=0 on port 25510 to get ALL expirations in one API call,
        reducing API calls from ~750/day to just 3/day (250x speedup!).

        Returns:
            Tuple of (bars_count, error_message or None)
        """
        tid = _get_thread_id()
        symbol = entry.symbol
        year = entry.year
        month = entry.month
        month_start = time.time()

        # Reconcile entry state with disk
        entry = self._reconcile_entry(entry)
        self.manifest.set_entry(entry)

        # Skip if already complete
        if entry.state == DownloadState.COMPLETE:
            logger.info(f"[{tid}] {entry.key}: Already complete, skipping")
            if self.progress:
                self.progress.skip_entry(symbol, entry.key)
            return entry.bars_count, None

        # Mark as in progress
        entry.state = DownloadState.IN_PROGRESS
        entry.started_at = datetime.now().isoformat()
        self.manifest.set_entry(entry)
        self.manifest.save()

        # Get dates for this month
        dates = self._get_dates_for_month(year, month)
        if not dates:
            logger.warning(f"[{tid}] {entry.key}: No trading days in range")
            entry.state = DownloadState.COMPLETE
            entry.bars_count = 0
            self.manifest.set_entry(entry)
            if self.progress:
                self.progress.complete_entry(symbol, entry.key, 0)
            return 0, None

        logger.info(f"[{tid}] {entry.key}: {len(dates)} days (BULK mode - 3 API calls/day)")

        all_data = []
        total_bars = 0
        total_bytes = 0
        dates_processed = 0
        api_calls = 0

        for date_str in dates:
            if self._shutdown_requested:
                return total_bars, "Shutdown requested"

            # In daily mode, skip days that already have files
            if self.write_daily:
                day_num = int(date_str.split("-")[2])
                existing_path = self._get_partition_path(symbol, year, month, day=day_num)
                if existing_path.exists() and existing_path.stat().st_size > 0:
                    logger.info(f"[{tid}] {entry.key} {date_str}: already exists, skipping")
                    dates_processed += 1
                    continue

            date_start = time.time()
            date_bars = 0

            # BULK: Fetch ALL expirations in ONE call each!
            ohlc_df, ohlc_error = self.client.get_bulk_ohlc(symbol, date_str)
            api_calls += 1

            if ohlc_error:
                logger.warning(f"[{tid}] {entry.key} {date_str}: OHLC error: {ohlc_error}")
                dates_processed += 1
                continue

            if ohlc_df is None or ohlc_df.empty:
                logger.info(f"[{tid}] {entry.key} {date_str}: No OHLC data")
                dates_processed += 1
                continue

            # Fetch quotes (bulk)
            quote_df, _ = self.client.get_bulk_quotes(symbol, date_str)
            api_calls += 1

            # Fetch Greeks (bulk)
            greeks_df, _ = self.client.get_bulk_greeks(symbol, date_str)
            api_calls += 1

            # Merge all data
            merged = self._merge_data(ohlc_df, quote_df, greeks_df)
            if merged is not None and not merged.is_empty():
                date_bars = len(merged)

                if self.write_daily:
                    # Apply filters and write immediately
                    filtered = self._apply_filters(merged)
                    if filtered is not None and not filtered.is_empty():
                        day_num = int(date_str.split("-")[2])
                        if self._save_partition_atomic(filtered, symbol, year, month, day=day_num):
                            partition_path = self._get_partition_path(symbol, year, month, day=day_num)
                            if partition_path.exists():
                                total_bytes += partition_path.stat().st_size
                        date_bars = len(filtered)
                    del merged, filtered
                else:
                    # Accumulate for monthly write
                    all_data.append(merged)

            total_bars += date_bars
            dates_processed += 1
            date_elapsed = time.time() - date_start

            logger.info(
                f"[{tid}] {entry.key} {date_str}: "
                f"{date_bars:,} bars, {date_elapsed:.1f}s, "
                f"3 API calls ({dates_processed}/{len(dates)} days)"
            )

            # Trigger progress update periodically
            if self.progress:
                self.progress._maybe_report()

        if self._shutdown_requested:
            return total_bars, "Shutdown requested"

        # For monthly mode, combine and save all data at once
        bytes_written = total_bytes
        if not self.write_daily and all_data:
            combined = pl.concat(all_data)
            filtered = self._apply_filters(combined)

            if filtered is not None and not filtered.is_empty():
                if self._save_partition_atomic(filtered, symbol, year, month):
                    partition_path = self._get_partition_path(symbol, year, month)
                    if partition_path.exists():
                        bytes_written = partition_path.stat().st_size
                    total_bars = len(filtered)
                else:
                    entry.state = DownloadState.FAILED
                    entry.error_message = "Failed to save partition"
                    self.manifest.set_entry(entry)
                    if self.progress:
                        self.progress.fail_entry(symbol, entry.key, "Failed to save")
                    return 0, "Failed to save partition"

        # Mark entry complete
        entry.state = DownloadState.COMPLETE
        entry.bars_count = total_bars
        entry.completed_at = datetime.now().isoformat()
        self.manifest.set_entry(entry)
        self.manifest.save()

        elapsed = time.time() - month_start
        if total_bars > 0:
            logger.info(
                f"[+] [{tid}] {entry.key}: Saved {total_bars:,} bars "
                f"({bytes_written / 1024 / 1024:.1f} MB) in {elapsed:.1f}s "
                f"({api_calls} API calls - BULK MODE)"
            )
        else:
            logger.info(f"[{tid}] {entry.key}: No data (empty month) in {elapsed:.1f}s")

        if self.progress:
            self.progress.complete_entry(symbol, entry.key, total_bars, bytes_written)
        return total_bars, None

    def _download_single_entry(
        self,
        entry: DownloadEntry,
    ) -> Tuple[int, Optional[str]]:
        """
        Download a single (symbol, month) entry using cached expirations.
        
        This method is designed for work-queue parallelization where any
        thread can pick up any entry.
        
        Returns: (bars_count, error_message)
        """
        tid = _get_thread_id()
        symbol = entry.symbol
        
        # CHECK DISK FIRST before making any API calls
        # This prevents unnecessary expiration fetches for already-downloaded data
        entry = self._reconcile_entry(entry)
        if entry.state == DownloadState.COMPLETE:
            logger.info(f"[{tid}] {entry.key}: File exists on disk, skipping")
            if self.progress:
                self.progress.skip_entry(symbol, entry.key)
            return entry.bars_count, None
        
        # Get expirations from cache (or fetch and cache)
        expirations, exp_error = self._get_expirations_cached(symbol)
        if exp_error:
            logger.error(f"[{tid}] Failed to get expirations for {symbol}: {exp_error}")
            entry.state = DownloadState.FAILED
            entry.error_message = exp_error
            self.manifest.set_entry(entry)
            if self.progress:
                self.progress.fail_entry(symbol, entry.key, exp_error)
            return 0, exp_error
            
        if not expirations:
            logger.warning(f"[{tid}] No expirations found for {symbol}")
            entry.state = DownloadState.COMPLETE
            entry.bars_count = 0
            self.manifest.set_entry(entry)
            if self.progress:
                self.progress.complete_entry(symbol, entry.key, 0, 0)
            return 0, None
        
        # Download this month
        bars, error = self._download_month(entry, expirations)
        return bars, error

    def _download_symbol_months(
        self,
        symbol: str,
        entries: List[DownloadEntry],
    ) -> Tuple[int, int, int]:
        """
        Download all months for a symbol.

        Uses bulk endpoints (exp=0) when available, falling back to standard
        per-expiration download if bulk is not available.

        Returns: (completed, failed, total_bars)
        """
        tid = _get_thread_id()

        # Register symbol with progress tracker
        if self.progress:
            self.progress.start_symbol(symbol, len(entries))

        mode = "BULK" if self._bulk_available else "STANDARD"
        logger.info(f"[{tid}] Starting {symbol}: {len(entries)} months ({mode} mode)")

        # For BULK mode, we don't need expirations - the API returns all!
        expirations = []
        if not self._bulk_available:
            # Get expirations for this symbol (only needed for standard mode)
            expirations, exp_error = self.client.get_expirations(symbol)
            if exp_error:
                logger.error(f"[{tid}] Failed to get expirations for {symbol}: {exp_error}")
                for entry in entries:
                    entry.state = DownloadState.FAILED
                    entry.error_message = exp_error
                    self.manifest.set_entry(entry)
                    if self.progress:
                        self.progress.fail_entry(symbol, entry.key, exp_error)
                if self.progress:
                    self.progress.complete_symbol(symbol)
                return 0, len(entries), 0

            if not expirations:
                logger.warning(f"[{tid}] No expirations found for {symbol}")
                for entry in entries:
                    entry.state = DownloadState.COMPLETE
                    entry.bars_count = 0
                    self.manifest.set_entry(entry)
                    if self.progress:
                        self.progress.complete_entry(symbol, entry.key, 0)
                if self.progress:
                    self.progress.complete_symbol(symbol)
                return len(entries), 0, 0

            logger.info(f"[{tid}] {symbol}: Found {len(expirations)} expirations")

        completed = 0
        failed = 0
        total_bars = 0

        for i, entry in enumerate(entries):
            if self._shutdown_requested:
                break

            logger.info(
                f"[{tid}] {symbol}: Processing month {i + 1}/{len(entries)} "
                f"({entry.year}-{entry.month:02d}) - {mode} mode"
            )

            # Use bulk or standard download based on availability
            if self._bulk_available:
                bars, error = self._download_month_bulk(entry)
            else:
                bars, error = self._download_month(entry, expirations)

            total_bars += bars

            if error:
                failed += 1
            else:
                completed += 1

        # Mark symbol as complete
        if self.progress:
            self.progress.complete_symbol(symbol)

        return completed, failed, total_bars

    # =========================================================================
    # Work Queue Implementation
    # =========================================================================

    def _signal_rate_limit(self, wait_seconds: float = 60.0) -> None:
        """
        Signal that we hit a rate limit. All threads should back off.

        Thread-safe: multiple threads can call this concurrently.
        """
        with self._rate_limit_lock:
            new_until = time.time() + wait_seconds
            if new_until > self._rate_limit_until:
                self._rate_limit_until = new_until
                self._rate_limit_event.clear()  # Block all threads
                logger.warning(f"[RATE LIMIT] All threads backing off for {wait_seconds:.0f}s")

    def _wait_for_rate_limit(self, timeout: float = 120.0) -> bool:
        """
        Wait if we're in a rate-limited state. Returns True if OK to proceed.

        Uses Event.wait() which is efficient (no busy-waiting).
        """
        # Quick check without blocking
        if self._rate_limit_event.is_set():
            return True

        tid = _get_thread_id()

        # We're rate limited - wait for it to clear
        while not self._shutdown_requested:
            with self._rate_limit_lock:
                now = time.time()
                if now >= self._rate_limit_until:
                    # Rate limit expired, signal all threads
                    self._rate_limit_event.set()
                    return True
                wait_remaining = self._rate_limit_until - now

            logger.info(f"[{tid}] Waiting {wait_remaining:.0f}s for rate limit...")

            # Wait for event or timeout (whichever comes first)
            if self._rate_limit_event.wait(timeout=min(wait_remaining + 1, timeout)):
                return True

        return False  # Shutdown requested

    def _worker(self, worker_id: int, startup_delay: float = 0.0) -> Dict[str, int]:
        """
        Worker thread that pulls entries from the work queue.

        Args:
            worker_id: Unique worker identifier
            startup_delay: Seconds to wait before starting (stagger)

        Returns:
            Dict with statistics (completed, failed, bars)
        """
        tid = f"W{worker_id}"
        threading.current_thread().name = f"Worker_{worker_id}"

        # Stagger startup to avoid thundering herd
        if startup_delay > 0:
            time.sleep(startup_delay)

        stats = {"completed": 0, "failed": 0, "bars": 0}

        logger.info(f"[{tid}] Worker started")

        while not self._shutdown_requested:
            # Wait for rate limit to clear
            if not self._wait_for_rate_limit():
                break

            # Get next entry from queue (with timeout to check shutdown)
            try:
                entry = self._work_queue.get(timeout=2.0)
            except queue.Empty:
                # Check if we're done
                if self._work_queue.empty():
                    break
                continue

            if entry is None:  # Poison pill
                self._work_queue.task_done()
                break

            symbol = entry.symbol
            entry_key = entry.key

            # Check if this entry is already being processed (duplicate prevention)
            with self._in_flight_lock:
                if entry_key in self._in_flight:
                    logger.warning(f"[{tid}] {entry_key}: Already in-flight, skipping")
                    self._work_queue.task_done()
                    continue
                self._in_flight.add(entry_key)

            try:
                # Get expirations (cached)
                expirations, exp_error = self._get_expirations_cached(symbol)
                if exp_error:
                    logger.error(f"[{tid}] {entry_key}: Expiration fetch failed: {exp_error}")
                    entry.state = DownloadState.FAILED
                    entry.error_message = exp_error
                    self.manifest.set_entry(entry)
                    if self.progress:
                        self.progress.fail_entry(symbol, entry_key, exp_error)
                    stats["failed"] += 1
                    self._work_queue.task_done()
                    continue

                # Download this entry
                if self._bulk_available:
                    bars, error = self._download_month_bulk(entry)
                else:
                    bars, error = self._download_month(entry, expirations)

                if error:
                    if "rate" in error.lower() or "429" in error:
                        # Rate limit - signal global backoff and requeue
                        self._signal_rate_limit(RATE_LIMIT_WAIT)
                        # Remove from in-flight before requeue (so it can be picked up again)
                        with self._in_flight_lock:
                            self._in_flight.discard(entry_key)
                        # Put back in queue for retry
                        self._work_queue.put(entry)
                    else:
                        stats["failed"] += 1
                else:
                    stats["completed"] += 1
                    stats["bars"] += bars

            except Exception as e:
                logger.error(f"[{tid}] {entry_key}: Exception: {e}")
                entry.state = DownloadState.FAILED
                entry.error_message = str(e)
                self.manifest.set_entry(entry)
                stats["failed"] += 1

            finally:
                # Remove from in-flight when done (unless requeued - handled above)
                with self._in_flight_lock:
                    self._in_flight.discard(entry_key)

            self._work_queue.task_done()

        logger.info(
            f"[{tid}] Worker finished: {stats['completed']} OK, "
            f"{stats['failed']} failed, {stats['bars']:,} bars"
        )

        # Store stats
        with self._stats_lock:
            self._worker_stats[tid] = stats

        return stats

    def _download_with_work_queue(self, pending: List[DownloadEntry]) -> Tuple[int, int, int]:
        """
        Download entries using a shared work queue.

        All entries go into one queue, and worker threads pull from it.
        This ensures all threads stay busy until work is done.

        Returns:
            Tuple of (completed, failed, total_bars)
        """
        # Create and populate work queue
        self._work_queue = queue.Queue()

        # Sort entries for optimal ordering:
        # - Recent data first (more likely to be useful)
        # - Spread symbols across the queue (avoid all SPY entries together)
        entries_by_symbol = defaultdict(list)
        for entry in pending:
            entries_by_symbol[entry.symbol].append(entry)

        # Sort within each symbol (newest first)
        for symbol in entries_by_symbol:
            entries_by_symbol[symbol].sort(
                key=lambda e: (e.year, e.month),
                reverse=True,
            )

        # Interleave symbols in queue (round-robin)
        symbol_iterators = {sym: iter(entries) for sym, entries in entries_by_symbol.items()}
        symbols = list(symbol_iterators.keys())
        added = 0

        while symbol_iterators:
            for sym in list(symbols):
                try:
                    entry = next(symbol_iterators[sym])
                    self._work_queue.put(entry)
                    added += 1
                except StopIteration:
                    del symbol_iterators[sym]
                    symbols.remove(sym)

        logger.info(f"Work queue populated: {added} entries")

        # Start worker threads with staggered startup
        threads = []
        stagger_delay = 0.5  # 500ms between thread starts

        for i in range(self.num_threads):
            t = threading.Thread(
                target=self._worker,
                args=(i, i * stagger_delay),
                name=f"Worker_{i}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        logger.info(f"Started {len(threads)} workers (staggered by {stagger_delay}s each)")

        # Wait for all work to complete (with periodic status updates)
        while not self._work_queue.empty() and not self._shutdown_requested:
            time.sleep(5.0)
            remaining = self._work_queue.qsize()
            if remaining > 0:
                logger.info(f"[QUEUE] {remaining} entries remaining...")

        # Wait for threads to finish
        for t in threads:
            t.join(timeout=30.0)

        # Aggregate stats
        completed = 0
        failed = 0
        total_bars = 0

        with self._stats_lock:
            for stats in self._worker_stats.values():
                completed += stats.get("completed", 0)
                failed += stats.get("failed", 0)
                total_bars += stats.get("bars", 0)

        return completed, failed, total_bars

    def download_all(self) -> DownloadResult:
        """Download all symbols with parallel execution."""
        logger.info("=" * 70)
        logger.info("THETADATA OPTIONS DOWNLOADER")
        logger.info("=" * 70)
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info(f"Symbols list: {', '.join(self.symbols)}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Log dir: {self.log_dir}")
        logger.info(f"Moneyness filter: {self.max_moneyness * 100:.0f}%")
        logger.info(f"Threads: {self.num_threads}")
        logger.info(f"Retry failed: {self.retry_failed}")
        logger.info("=" * 70)

        # Verify connection
        if not self.client.verify_connection():
            logger.error(
                "Cannot connect to Theta Terminal. "
                "Please ensure it is running on localhost:25503"
            )
            return DownloadResult(
                total_entries=0,
                completed=0,
                failed=0,
                skipped=0,
                total_bars=0,
                elapsed_seconds=0,
            )

        logger.info("[+] Connected to Theta Terminal")

        # Check for bulk endpoint availability (PRO tier)
        if self.use_bulk:
            self._bulk_available = self.client.verify_bulk_connection()
            if self._bulk_available:
                logger.info("[+] BULK MODE ENABLED - using exp=0 endpoints (250x faster!)")
            else:
                logger.warning(
                    "[!] Bulk endpoints not available - using standard mode. "
                    "PRO tier required for bulk endpoints."
                )
                self._bulk_available = False
        else:
            self._bulk_available = False
            logger.info("[!] Bulk mode disabled - using standard per-expiration download")

        # Initialize manifest entries
        new_entries = self.manifest.initialize_entries(
            self.symbols, self.start_date, self.end_date
        )
        logger.info(f"Initialized {new_entries} new entries in manifest")

        # Get pending entries
        pending = self.manifest.get_pending_entries(
            include_failed=self.retry_failed,
            include_incomplete=True,
        )

        # Show current stats
        stats = self.manifest.get_stats()
        logger.info(f"Manifest stats: {stats}")
        logger.info(f"Entries to process: {len(pending)}")

        if not pending:
            logger.info("No pending entries - all downloads complete!")
            return DownloadResult(
                total_entries=len(self.manifest.manifest.entries),
                completed=stats.get("complete", 0),
                failed=stats.get("failed", 0),
                skipped=stats.get("complete", 0),
                total_bars=sum(e.bars_count for e in self.manifest.manifest.entries.values()),
                elapsed_seconds=0,
            )

        # Initialize progress tracker
        logger.info("Initializing progress tracker...")
        self.progress = ProgressTracker(self.log_dir, len(pending))
        logger.info(f"Progress tracking enabled: {self.progress.progress_file}")

        # Show entries per symbol
        entries_by_symbol: Dict[str, List[DownloadEntry]] = defaultdict(list)
        for entry in pending:
            entries_by_symbol[entry.symbol].append(entry)
        logger.info(f"Entries across {len(entries_by_symbol)} symbols:")
        for sym, entries in sorted(entries_by_symbol.items()):
            logger.info(f"  {sym}: {len(entries)} months")

        start_time = time.time()
        logger.info(f"Starting work-queue download with {self.num_threads} threads...")
        logger.info("  - Threads pull from shared queue (no idle threads)")
        logger.info("  - Staggered startup (0.5s between threads)")
        logger.info("  - Coordinated rate-limit backoff")

        # Use work queue for better load balancing
        completed, failed, total_bars = self._download_with_work_queue(pending)
        skipped = 0

        # Save manifest
        self.manifest.save()

        elapsed = time.time() - start_time

        # Final stats
        final_stats = self.manifest.get_stats()

        # Write final progress report
        if self.progress:
            self.progress.final_report()

        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total bars: {total_bars:,}")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info(f"Final manifest stats: {final_stats}")
        logger.info("=" * 70)

        if self._shutdown_requested:
            logger.warning("Download was interrupted - run again to resume")

        return DownloadResult(
            total_entries=len(pending),
            completed=completed,
            failed=failed,
            skipped=skipped,
            total_bars=total_bars,
            elapsed_seconds=elapsed,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download historical options data from ThetaData"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: LIQUID_UNIVERSE)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from manifest (default behavior)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed entries",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f"Number of download threads (default: {DEFAULT_NUM_THREADS})",
    )
    parser.add_argument(
        "--moneyness",
        type=float,
        default=MAX_MONEYNESS,
        help=f"Max OTM percentage (default: {MAX_MONEYNESS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: from settings.ini)",
    )
    parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="Show manifest stats and exit",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset manifest (clear all progress) and start fresh",
    )
    parser.add_argument(
        "--write-monthly",
        action="store_true",
        help="Write data per-month instead of per-day (higher memory, fewer files)",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolidate daily parquet files into monthly (run after download)",
    )
    parser.add_argument(
        "--no-bulk",
        action="store_true",
        help="Disable bulk endpoints (use standard per-expiration download)",
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Parse output directory
    output_dir = Path(args.output) if args.output else get_options_data_dir()

    # Set up file logging
    log_dir = setup_file_logging(output_dir)

    # Handle reset
    if args.reset:
        manifest_path = output_dir / "_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
            logger.info(f"Manifest reset: {manifest_path}")
        else:
            logger.info("No manifest to reset")

    # Show manifest and exit if requested
    if args.show_manifest:
        manifest_path = output_dir / "_manifest.json"
        if manifest_path.exists():
            manager = ManifestManager(manifest_path)
            stats = manager.get_stats()
            print(f"\nManifest: {manifest_path}")
            print(f"Stats: {stats}")
            print(f"Total entries: {len(manager.manifest.entries)}")

            # Show sample of each state
            for state in DownloadState:
                entries = [
                    e for e in manager.manifest.entries.values()
                    if e.state == state
                ]
                if entries:
                    print(f"\n{state.value.upper()} ({len(entries)}):")
                    for e in entries[:5]:
                        print(f"  {e.key}: {e.bars_count:,} bars")
                    if len(entries) > 5:
                        print(f"  ... and {len(entries) - 5} more")
        else:
            print(f"No manifest found at {manifest_path}")
        return 0

    # Consolidate daily files into monthly
    if args.consolidate:
        data_dir = output_dir / "options_1min"
        if not data_dir.exists():
            logger.error(f"No data directory found: {data_dir}")
            return 1

        # Load manifest to check which months are complete
        manifest_path = output_dir / "_manifest.json"
        complete_entries = set()
        if manifest_path.exists():
            manager = ManifestManager(manifest_path)
            for key, entry in manager.manifest.entries.items():
                if entry.state == DownloadState.COMPLETE:
                    complete_entries.add(key)
            logger.info(f"Loaded manifest: {len(complete_entries)} complete entries")
        else:
            logger.warning("No manifest found - will consolidate all months with daily files")

        logger.info("Consolidating daily parquet files into monthly...")
        consolidated = 0
        skipped_incomplete = 0
        errors = 0

        # Find all month directories
        for symbol_dir in sorted(data_dir.glob("root=*")):
            symbol = symbol_dir.name.replace("root=", "")
            for year_dir in sorted(symbol_dir.glob("year=*")):
                year_str = year_dir.name.replace("year=", "")
                for month_dir in sorted(year_dir.glob("month=*")):
                    month_str = month_dir.name.replace("month=", "")

                    # Find daily files
                    daily_files = sorted(month_dir.glob("day_*.parquet"))
                    if not daily_files:
                        continue

                    monthly_file = month_dir / "data.parquet"
                    if monthly_file.exists():
                        logger.info(f"  {symbol}/{year_str}/{month_str}: already consolidated, skipping")
                        continue

                    # Check if this month is complete in manifest
                    entry_key = f"{symbol}_{year_str}_{month_str}"
                    if complete_entries and entry_key not in complete_entries:
                        logger.warning(
                            f"  {symbol}/{year_str}/{month_str}: not complete in manifest, skipping "
                            f"({len(daily_files)} daily files)"
                        )
                        skipped_incomplete += 1
                        continue

                    try:
                        # Read and combine all daily files
                        logger.info(f"  {symbol}/{year_str}/{month_str}: consolidating {len(daily_files)} daily files...")
                        dfs = [pl.read_parquet(f) for f in daily_files]
                        combined = pl.concat(dfs)

                        # Write monthly file
                        combined.write_parquet(
                            monthly_file,
                            compression="zstd",
                            use_pyarrow=True,
                        )

                        # Delete daily files
                        for f in daily_files:
                            f.unlink()

                        logger.info(
                            f"  [+] {symbol}/{year_str}/{month_str}: {len(combined):,} bars "
                            f"({monthly_file.stat().st_size / 1024 / 1024:.1f} MB)"
                        )
                        consolidated += 1

                    except Exception as e:
                        logger.error(f"  [-] {symbol}/{year_str}/{month_str}: {e}")
                        errors += 1

        logger.info(
            f"Consolidation complete: {consolidated} months consolidated, "
            f"{skipped_incomplete} skipped (incomplete), {errors} errors"
        )
        return 0 if errors == 0 else 1

    # Create and run downloader
    downloader = ThetaDataDownloader(
        output_dir=output_dir,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        max_moneyness=args.moneyness,
        num_threads=args.threads,
        retry_failed=args.retry_failed,
        write_daily=not args.write_monthly,  # Daily is default (lower memory)
        use_bulk=not args.no_bulk,  # Bulk is default (PRO tier - 250x faster!)
    )

    result = downloader.download_all()

    # Exit with appropriate code
    if result.failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
