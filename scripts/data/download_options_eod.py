"""
ThetaData Options EOD Downloader (Gamma + Open Interest).

Downloads EOD gamma and open interest data from ThetaData REST API.
Stores separately from 1-min data for easy backfilling.

Storage: options_eod/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet

Schema (6 columns):
    - date: Date of the EOD snapshot
    - expiration: Option expiration date
    - strike: Strike price
    - right: C or P
    - gamma: Second-order Greek
    - open_interest: EOD open interest

Prerequisites:
    1. ThetaData Standard subscription
    2. Java 21+ installed
    3. Theta Terminal running on localhost:25503

Usage:
    # Download all symbols (full 2017-present)
    python scripts/download_options_eod.py

    # Download specific symbols
    python scripts/download_options_eod.py --symbols SPY,QQQ,AAPL

    # Specific date range
    python scripts/download_options_eod.py --start 2020-01-01 --end 2024-12-31

    # Resume interrupted download
    python scripts/download_options_eod.py --resume
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

# Use standard Python logging
logger = logging.getLogger("download_options_eod")
logger.setLevel(logging.INFO)


# =============================================================================
# Configuration
# =============================================================================

LIQUID_UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN",
    "TLT",
    "GLD", "SLV", "XLE",
    "EEM", "FXI",
    "VIX",
    "XLF",
]

THETA_BASE_URL = "http://localhost:25503/v3"
DEFAULT_START_DATE = "2017-01-01"
DEFAULT_NUM_THREADS = 16
DEFAULT_TIMEOUT = 30  # seconds per request (reduced from 120)
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]
RATE_LIMIT_WAIT = 60

_thread_local = threading.local()


def _get_thread_id() -> str:
    """Get a short thread identifier for logging."""
    name = threading.current_thread().name
    if name.startswith("Worker_"):
        return f"W{name.split('_')[-1]}"
    if '_' in name:
        return f"T{name.split('_')[-1]}"
    return name[:8]


# =============================================================================
# Data Classes
# =============================================================================

class DownloadState(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class DownloadEntry:
    symbol: str
    year: int
    month: int
    state: DownloadState = DownloadState.PENDING
    rows_count: int = 0
    error_message: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.symbol}_{self.year}_{self.month:02d}"

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "year": self.year,
            "month": self.month,
            "state": self.state.value,
            "rows_count": self.rows_count,
            "error_message": self.error_message,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DownloadEntry":
        return cls(
            symbol=data["symbol"],
            year=data["year"],
            month=data["month"],
            state=DownloadState(data.get("state", "pending")),
            rows_count=data.get("rows_count", 0),
            error_message=data.get("error_message"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class DownloadManifest:
    entries: Dict[str, DownloadEntry] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DownloadManifest":
        manifest = cls(
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
        for key, entry_data in data.get("entries", {}).items():
            manifest.entries[key] = DownloadEntry.from_dict(entry_data)
        return manifest


# =============================================================================
# Logging Setup
# =============================================================================

class TimestampFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now().strftime("%H:%M:%S")
        original = super().format(record)
        return f"{timestamp} {original}"


def setup_logging(output_dir: Path) -> Path:
    global logger

    log_dir = output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")

    # File handler
    file_handler = RotatingFileHandler(
        log_dir / f"download_eod_{today}.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Console handler
    logger.handlers.clear()
    logger.propagate = False
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(TimestampFormatter("%(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return log_dir


# =============================================================================
# ThetaData Client
# =============================================================================

def _get_thread_client(timeout: int = DEFAULT_TIMEOUT) -> httpx.Client:
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = httpx.Client(timeout=timeout)
    return _thread_local.client


def _get_thread_id() -> str:
    name = threading.current_thread().name
    if '_' in name:
        return f"T{name.split('_')[-1]}"
    return name[:8]


class ThetaDataClient:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def _request(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        client = _get_thread_client(self.timeout)
        url = f"{THETA_BASE_URL}{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                response = client.get(url, params=params, timeout=self.timeout)

                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limited, waiting {RATE_LIMIT_WAIT}s...")
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    return None, "Rate limit exceeded"

                if response.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None, f"Server error {response.status_code}"

                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}"

                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                if df.empty:
                    return None, None
                return df, None

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None, "Timeout"
            except httpx.ConnectError:
                return None, "Connection failed - is Theta Terminal running?"
            except Exception as e:
                return None, str(e)

        return None, "Max retries exceeded"

    def verify_connection(self) -> bool:
        df, error = self._request(
            "/option/list/expirations",
            {"symbol": "SPY", "format": "csv"},
        )
        return error is None or "Connection" not in str(error)

    def get_expirations(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        df, error = self._request(
            "/option/list/expirations",
            {"symbol": symbol, "format": "csv"},
        )
        if error:
            return [], error
        if df is None or df.empty:
            return [], None
        return df["expiration"].tolist(), None

    def get_greeks_second_order(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get second-order Greeks (gamma) for all strikes at EOD."""
        return self._request(
            "/option/history/greeks/second_order",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "interval": "1h",  # Largest available interval
                "start_time": "15:30:00",  # Get last hour of trading
                "end_time": "16:00:00",
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
        )

    def get_open_interest(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get EOD open interest for all strikes."""
        return self._request(
            "/option/history/open_interest",
            {
                "symbol": symbol,
                "expiration": expiration,
                "date": date_str,
                "strike": "*",
                "right": "both",
                "format": "csv",
            },
        )

    def get_bulk_eod_greeks(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Get bulk EOD Greeks for ALL expirations on a single date.
        Uses v3 API with expiration=* to get all expirations in one call.

        Returns gamma + all Greeks for all contracts on that date.
        """
        client = _get_thread_client(self.timeout)
        # v3 API endpoint - greeks/eod has gamma and supports expiration=*
        url = f"{THETA_BASE_URL}/option/history/greeks/eod"

        date_fmt = date_str.replace("-", "")

        params = {
            "symbol": symbol,  # v3 uses 'symbol' not 'root'
            "expiration": "*",  # v3 syntax: * = all expirations
            "start_date": date_fmt,  # greeks/eod uses start_date/end_date
            "end_date": date_fmt,
            "strike": "*",
            "right": "both",
            "format": "csv",
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = client.get(url, params=params, timeout=self.timeout)

                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Bulk EOD Greeks rate limited, waiting {RATE_LIMIT_WAIT}s...")
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    return None, "Rate limit exceeded"

                if response.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None, f"Server error {response.status_code}"

                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}: {response.text[:200]}"

                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                if df.empty:
                    return None, None
                return df, None

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None, "Timeout"
            except httpx.ConnectError:
                return None, "Connection failed - is Theta Terminal running?"
            except Exception as e:
                return None, str(e)

        return None, "Max retries exceeded"

    def get_bulk_open_interest(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Get bulk open interest for ALL expirations on a single date.
        Uses v3 API with expiration=* to get all expirations in one call.
        """
        client = _get_thread_client(self.timeout)
        # v3 API endpoint
        url = f"{THETA_BASE_URL}/option/history/open_interest"

        date_fmt = date_str.replace("-", "")

        params = {
            "symbol": symbol,  # v3 uses 'symbol' not 'root'
            "expiration": "*",  # v3 syntax: * = all expirations
            "date": date_fmt,  # v3 uses 'date' not 'start_date/end_date'
            "strike": "*",
            "right": "both",
            "format": "csv",
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = client.get(url, params=params, timeout=self.timeout)

                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Bulk OI rate limited, waiting {RATE_LIMIT_WAIT}s...")
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    return None, "Rate limit exceeded"

                if response.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None, f"Server error {response.status_code}"

                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}: {response.text[:200]}"

                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                if df.empty:
                    return None, None
                return df, None

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None, "Timeout"
            except httpx.ConnectError:
                return None, "Connection failed - is Theta Terminal running?"
            except Exception as e:
                return None, str(e)

        return None, "Max retries exceeded"


# =============================================================================
# Manifest Manager
# =============================================================================

class ManifestManager:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.lock = threading.Lock()
        self.manifest = self._load_or_create()

    def _load_or_create(self) -> DownloadManifest:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return DownloadManifest.from_dict(json.load(f))
            except Exception:
                pass
        return DownloadManifest(created_at=datetime.now().isoformat())

    def save(self) -> None:
        with self.lock:
            self.manifest.updated_at = datetime.now().isoformat()
            temp_path = self.manifest_path.with_suffix('.tmp')
            with open(temp_path, "w") as f:
                json.dump(self.manifest.to_dict(), f, indent=2)
            shutil.move(str(temp_path), str(self.manifest_path))

    def initialize_entries(self, symbols: List[str], start_date: str, end_date: str) -> int:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        new_count = 0

        with self.lock:
            for symbol in symbols:
                current = start.replace(day=1)
                while current <= end:
                    entry = DownloadEntry(symbol=symbol, year=current.year, month=current.month)
                    if entry.key not in self.manifest.entries:
                        self.manifest.entries[entry.key] = entry
                        new_count += 1
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)
        return new_count

    def get_pending_entries(self, include_failed: bool = False) -> List[DownloadEntry]:
        with self.lock:
            result = []
            for entry in self.manifest.entries.values():
                if entry.state == DownloadState.PENDING:
                    result.append(entry)
                elif entry.state == DownloadState.IN_PROGRESS:
                    result.append(entry)
                elif entry.state == DownloadState.FAILED and include_failed:
                    result.append(entry)
            return result

    def get_stats(self) -> Dict[str, int]:
        with self.lock:
            stats = defaultdict(int)
            for entry in self.manifest.entries.values():
                stats[entry.state.value] += 1
            return dict(stats)


# =============================================================================
# EOD Downloader
# =============================================================================

class EODDownloader:
    """Downloads EOD gamma and open interest data."""

    SCHEMA = ["date", "expiration", "strike", "right", "gamma", "open_interest"]

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        num_threads: int = DEFAULT_NUM_THREADS,
        retry_failed: bool = False,
    ):
        self.output_dir = output_dir or get_options_data_dir()
        self.data_dir = self.output_dir / "options_eod"
        self.symbols = symbols or LIQUID_UNIVERSE
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.num_threads = num_threads
        self.retry_failed = retry_failed

        self.client = ThetaDataClient()
        self._shutdown_requested = False
        self._bulk_available: Optional[bool] = None  # Checked at runtime

        # Work queue coordination
        self._work_queue: Optional[queue.Queue] = None
        self._rate_limit_event = threading.Event()
        self._rate_limit_event.set()  # Start in non-limited state
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_until = 0.0
        self._worker_stats: Dict[str, Dict] = {}
        self._stats_lock = threading.Lock()
        self._in_flight: Set[str] = set()
        self._in_flight_lock = threading.Lock()

        # Expiration cache
        self._expiration_cache: Dict[str, List[str]] = {}
        self._expiration_cache_lock = threading.Lock()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = ManifestManager(self.output_dir / "_manifest_eod.json")

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _handle_shutdown(self, signum, frame):
        if self._shutdown_requested:
            sys.exit(1)
        logger.warning("Shutdown requested...")
        self._shutdown_requested = True

    def _check_bulk_availability(self) -> bool:
        """Check if wildcard expiration (PRO tier) is available on v3 API."""
        try:
            client = _get_thread_client(10)  # Short timeout for test
            # v3 API: use expiration=* to get all expirations
            # Test with greeks/eod which is what we use for gamma
            url = f"{THETA_BASE_URL}/option/history/greeks/eod"
            params = {
                "symbol": "SPY",  # v3 uses 'symbol' not 'root'
                "expiration": "*",  # v3 syntax for all expirations
                "start_date": "20241231",  # greeks/eod uses start_date/end_date
                "end_date": "20241231",
                "format": "csv",
            }
            response = client.get(url, params=params, timeout=10)
            # 200 = success, 403 = no access (not PRO)
            if response.status_code == 200:
                logger.info("[+] Wildcard expiration available (PRO tier detected)")
                return True
            elif response.status_code == 403:
                logger.info("[-] Wildcard expiration not available (Standard tier)")
                return False
            else:
                logger.info(f"[-] Bulk check returned {response.status_code}: {response.text[:100]}")
                return False
        except Exception as e:
            logger.info(f"[-] Bulk check failed: {e}")
            return False

    def _get_partition_path(self, symbol: str, year: int, month: int) -> Path:
        return (
            self.data_dir
            / f"root={symbol}"
            / f"year={year}"
            / f"month={month:02d}"
            / "data.parquet"
        )

    def _partition_exists(self, symbol: str, year: int, month: int) -> bool:
        path = self._get_partition_path(symbol, year, month)
        return path.exists() and path.stat().st_size > 0

    def _get_dates_for_month(self, year: int, month: int) -> List[str]:
        from calendar import monthrange
        _, last_day = monthrange(year, month)
        start = datetime(year, month, 1)
        end = datetime(year, month, last_day)

        range_start = datetime.strptime(self.start_date, "%Y-%m-%d")
        range_end = datetime.strptime(self.end_date, "%Y-%m-%d")
        start = max(start, range_start)
        end = min(end, range_end)

        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    def _merge_gamma_oi(
        self,
        date_str: str,
        gamma_df: Optional[pd.DataFrame],
        oi_df: Optional[pd.DataFrame],
    ) -> Optional[pl.DataFrame]:
        """Merge gamma and open interest data for a single date.

        Handles both standard API format (expiration column) and
        bulk API format (exp column).
        """

        def get_exp(row):
            """Get expiration from either 'expiration' or 'exp' column."""
            if "expiration" in row.index:
                return row["expiration"]
            elif "exp" in row.index:
                return row["exp"]
            return None

        records = []

        # Process gamma data
        gamma_dict = {}
        if gamma_df is not None and not gamma_df.empty:
            for _, row in gamma_df.iterrows():
                exp = get_exp(row)
                key = (exp, row.get("strike"), row.get("right"))
                gamma_dict[key] = row.get("gamma")

        # Process OI data and merge
        if oi_df is not None and not oi_df.empty:
            for _, row in oi_df.iterrows():
                exp = get_exp(row)
                strike = row.get("strike")
                right = row.get("right")
                oi = row.get("open_interest")

                key = (exp, strike, right)
                gamma = gamma_dict.get(key)

                records.append({
                    "date": date_str,
                    "expiration": exp,
                    "strike": strike,
                    "right": right,
                    "gamma": gamma,
                    "open_interest": oi,
                })

        # Also include gamma-only records (contracts with gamma but no OI)
        if gamma_df is not None and not gamma_df.empty:
            oi_keys = set()
            if oi_df is not None and not oi_df.empty:
                oi_keys = {(get_exp(r), r.get("strike"), r.get("right"))
                          for _, r in oi_df.iterrows()}

            for _, row in gamma_df.iterrows():
                exp = get_exp(row)
                key = (exp, row.get("strike"), row.get("right"))
                if key not in oi_keys:
                    records.append({
                        "date": date_str,
                        "expiration": exp,
                        "strike": key[1],
                        "right": key[2],
                        "gamma": row.get("gamma"),
                        "open_interest": None,
                    })

        if not records:
            return None

        df = pl.DataFrame(records)

        # Cast types
        df = df.with_columns([
            pl.col("strike").cast(pl.Float64),
            pl.col("gamma").cast(pl.Float64),
            pl.col("open_interest").cast(pl.Int64),
        ])

        return df

    def _download_month(
        self,
        entry: DownloadEntry,
        expirations: List[str],
    ) -> Tuple[int, Optional[str]]:
        """Download EOD gamma + OI for a single month."""
        tid = _get_thread_id()
        symbol = entry.symbol
        year = entry.year
        month = entry.month

        # Check if already exists
        if self._partition_exists(symbol, year, month):
            logger.info(f"[{tid}] {entry.key}: Already exists, skipping")
            entry.state = DownloadState.COMPLETE
            return 0, None

        entry.state = DownloadState.IN_PROGRESS
        self.manifest.save()

        dates = self._get_dates_for_month(year, month)
        if not dates:
            entry.state = DownloadState.COMPLETE
            entry.rows_count = 0
            return 0, None

        # Filter relevant expirations
        month_start = f"{year}-{month:02d}-01"
        month_end = (datetime(year, month, 1) + timedelta(days=365)).strftime("%Y-%m-%d")
        relevant_exps = [e for e in expirations if month_start <= e <= month_end]

        logger.info(f"[{tid}] {entry.key}: {len(dates)} days, {len(relevant_exps)} expirations")

        all_data = []
        api_calls = 0

        # Choose between bulk (PRO) and standard (per-day) mode
        if self._bulk_available:
            # BULK MODE: Day-by-day with exp=0 to get ALL expirations per call
            # ~44 API calls per month (22 days x 2 endpoints) vs 4400+ without bulk
            logger.info(f"[{tid}] {entry.key}: Using BULK API (exp=0, ~{len(dates) * 2} calls)")

            for date_idx, date_str in enumerate(dates):
                if self._shutdown_requested:
                    return 0, "Shutdown"

                if date_idx % 5 == 0:
                    logger.info(f"[{tid}] {entry.key}: Day {date_idx + 1}/{len(dates)}, {api_calls} API calls")

                # Bulk EOD Greeks for ALL expirations on this date
                gamma_df, gamma_err = self.client.get_bulk_eod_greeks(symbol, date_str)
                api_calls += 1

                if gamma_err:
                    if "rate" in gamma_err.lower():
                        return 0, gamma_err
                    # Log other errors but continue
                    logger.debug(f"[{tid}] {entry.key} {date_str}: Gamma error: {gamma_err}")

                # Bulk OI for ALL expirations on this date
                oi_df, oi_err = self.client.get_bulk_open_interest(symbol, date_str)
                api_calls += 1

                if oi_err:
                    if "rate" in oi_err.lower():
                        return 0, oi_err
                    # Log other errors but continue
                    logger.debug(f"[{tid}] {entry.key} {date_str}: OI error: {oi_err}")

                # Merge gamma and OI for this date
                merged = self._merge_gamma_oi(date_str, gamma_df, oi_df)
                if merged is not None:
                    all_data.append(merged)
        else:
            # STANDARD MODE: Per-day API calls (slower but works without PRO)
            total_calls = sum(1 for d in dates for e in relevant_exps if e >= d) * 2
            logger.info(f"[{tid}] {entry.key}: Using STANDARD API (~{total_calls} calls)")

            for date_idx, date_str in enumerate(dates):
                if self._shutdown_requested:
                    return 0, "Shutdown"

                valid_exps = [e for e in relevant_exps if e >= date_str]

                if date_idx % 5 == 0:
                    logger.info(f"[{tid}] {entry.key}: Day {date_idx + 1}/{len(dates)}, {len(valid_exps)} exps, {api_calls} calls")

                for expiration in valid_exps:
                    if self._shutdown_requested:
                        break

                    gamma_df, gamma_err = self.client.get_greeks_second_order(symbol, expiration, date_str)
                    api_calls += 1

                    if gamma_err and "rate" in gamma_err.lower():
                        return 0, gamma_err

                    oi_df, oi_err = self.client.get_open_interest(symbol, expiration, date_str)
                    api_calls += 1

                    if oi_err and "rate" in oi_err.lower():
                        return 0, oi_err

                    merged = self._merge_gamma_oi(date_str, gamma_df, oi_df)
                    if merged is not None:
                        all_data.append(merged)

        if self._shutdown_requested:
            return 0, "Shutdown"

        # Save
        total_rows = 0
        if all_data:
            combined = pl.concat(all_data)
            total_rows = len(combined)

            partition_dir = self._get_partition_path(symbol, year, month).parent
            partition_dir.mkdir(parents=True, exist_ok=True)

            combined.write_parquet(
                self._get_partition_path(symbol, year, month),
                compression="zstd",
            )

        entry.state = DownloadState.COMPLETE
        entry.rows_count = total_rows
        entry.completed_at = datetime.now().isoformat()
        self.manifest.save()

        logger.info(f"[+] [{tid}] {entry.key}: {total_rows:,} rows, {api_calls} API calls")
        return total_rows, None

    def _download_symbol(self, symbol: str, entries: List[DownloadEntry]) -> Tuple[int, int, int]:
        """Download all months for a symbol."""
        tid = _get_thread_id()
        logger.info(f"[{tid}] Starting {symbol}: {len(entries)} months")

        expirations, error = self.client.get_expirations(symbol)
        if error:
            logger.error(f"[{tid}] Failed to get expirations for {symbol}: {error}")
            for entry in entries:
                entry.state = DownloadState.FAILED
                entry.error_message = error
            return 0, len(entries), 0

        completed = 0
        failed = 0
        total_rows = 0

        for entry in entries:
            if self._shutdown_requested:
                break

            rows, error = self._download_month(entry, expirations)
            total_rows += rows
            if error:
                failed += 1
            else:
                completed += 1

        return completed, failed, total_rows

    # =========================================================================
    # Work Queue Implementation
    # =========================================================================

    def _get_expirations_cached(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        """Get expirations with thread-safe caching."""
        with self._expiration_cache_lock:
            if symbol in self._expiration_cache:
                return self._expiration_cache[symbol], None

        expirations, error = self.client.get_expirations(symbol)
        if not error and expirations:
            with self._expiration_cache_lock:
                self._expiration_cache[symbol] = expirations

        return expirations, error

    def _signal_rate_limit(self, wait_seconds: float = 60.0) -> None:
        """Signal global rate limit - all threads back off."""
        with self._rate_limit_lock:
            new_until = time.time() + wait_seconds
            if new_until > self._rate_limit_until:
                self._rate_limit_until = new_until
                self._rate_limit_event.clear()
                logger.warning(f"[RATE LIMIT] All threads backing off for {wait_seconds:.0f}s")

    def _wait_for_rate_limit(self, timeout: float = 120.0) -> bool:
        """Wait if rate-limited. Returns True if OK to proceed."""
        if self._rate_limit_event.is_set():
            return True

        tid = _get_thread_id()

        while not self._shutdown_requested:
            with self._rate_limit_lock:
                now = time.time()
                if now >= self._rate_limit_until:
                    self._rate_limit_event.set()
                    return True
                wait_remaining = self._rate_limit_until - now

            logger.info(f"[{tid}] Waiting {wait_remaining:.0f}s for rate limit...")
            if self._rate_limit_event.wait(timeout=min(wait_remaining + 1, timeout)):
                return True

        return False

    def _worker(self, worker_id: int, startup_delay: float = 0.0) -> Dict[str, int]:
        """Worker thread that pulls entries from the work queue."""
        tid = f"W{worker_id}"
        threading.current_thread().name = f"Worker_{worker_id}"

        if startup_delay > 0:
            time.sleep(startup_delay)

        stats = {"completed": 0, "failed": 0, "rows": 0}
        logger.info(f"[{tid}] Worker started")

        while not self._shutdown_requested:
            # Check rate limit
            logger.debug(f"[{tid}] Checking rate limit...")
            if not self._wait_for_rate_limit():
                logger.info(f"[{tid}] Exiting due to shutdown during rate limit wait")
                break

            try:
                logger.debug(f"[{tid}] Waiting for queue entry...")
                entry = self._work_queue.get(timeout=2.0)
            except queue.Empty:
                if self._work_queue.empty():
                    logger.info(f"[{tid}] Queue empty, exiting")
                    break
                continue

            if entry is None:
                self._work_queue.task_done()
                break

            entry_key = entry.key
            symbol = entry.symbol
            logger.info(f"[{tid}] Got entry: {entry_key}")

            # Duplicate prevention
            with self._in_flight_lock:
                if entry_key in self._in_flight:
                    logger.warning(f"[{tid}] {entry_key}: Already in-flight, skipping")
                    self._work_queue.task_done()
                    continue
                self._in_flight.add(entry_key)

            try:
                expirations, exp_error = self._get_expirations_cached(symbol)
                if exp_error:
                    logger.error(f"[{tid}] {entry_key}: Expiration fetch failed: {exp_error}")
                    entry.state = DownloadState.FAILED
                    entry.error_message = exp_error
                    stats["failed"] += 1
                    self._work_queue.task_done()
                    continue

                rows, error = self._download_month(entry, expirations)

                if error:
                    if "rate" in error.lower() or "429" in error:
                        self._signal_rate_limit(RATE_LIMIT_WAIT)
                        with self._in_flight_lock:
                            self._in_flight.discard(entry_key)
                        self._work_queue.put(entry)
                    else:
                        stats["failed"] += 1
                else:
                    stats["completed"] += 1
                    stats["rows"] += rows

            except Exception as e:
                logger.error(f"[{tid}] {entry_key}: Exception: {e}")
                entry.state = DownloadState.FAILED
                entry.error_message = str(e)
                stats["failed"] += 1

            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(entry_key)

            self._work_queue.task_done()

        logger.info(f"[{tid}] Worker finished: {stats['completed']} OK, {stats['failed']} failed")

        with self._stats_lock:
            self._worker_stats[tid] = stats

        return stats

    def _download_with_work_queue(self, pending: List[DownloadEntry]) -> Tuple[int, int, int]:
        """Download entries using a shared work queue."""
        self._work_queue = queue.Queue()

        # Interleave symbols in queue (round-robin)
        entries_by_symbol = defaultdict(list)
        for entry in pending:
            entries_by_symbol[entry.symbol].append(entry)

        for symbol in entries_by_symbol:
            entries_by_symbol[symbol].sort(key=lambda e: (e.year, e.month), reverse=True)

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

        # Start workers with staggered startup
        threads = []
        stagger_delay = 0.5

        for i in range(self.num_threads):
            t = threading.Thread(
                target=self._worker,
                args=(i, i * stagger_delay),
                daemon=True,
            )
            threads.append(t)
            t.start()

        logger.info(f"Started {len(threads)} workers (staggered by {stagger_delay}s)")

        # Track progress for ETA
        total_entries = added
        start_time = time.time()

        # Wait for completion
        while not self._work_queue.empty() and not self._shutdown_requested:
            time.sleep(5.0)
            remaining = self._work_queue.qsize()
            if remaining > 0:
                # Calculate ETA
                processed = total_entries - remaining
                elapsed = time.time() - start_time
                if processed > 0:
                    rate = processed / elapsed
                    eta_seconds = remaining / rate
                    eta_str = self._format_time(eta_seconds)
                    elapsed_str = self._format_time(elapsed)
                    pct = (processed / total_entries) * 100
                    logger.info(
                        f"[PROGRESS] {processed}/{total_entries} ({pct:.1f}%) | "
                        f"Remaining: {remaining} | Elapsed: {elapsed_str} | ETA: {eta_str}"
                    )
                else:
                    logger.info(f"[QUEUE] {remaining} entries remaining...")

        for t in threads:
            t.join(timeout=30.0)

        # Aggregate stats
        completed = 0
        failed = 0
        total_rows = 0

        with self._stats_lock:
            for stats in self._worker_stats.values():
                completed += stats.get("completed", 0)
                failed += stats.get("failed", 0)
                total_rows += stats.get("rows", 0)

        return completed, failed, total_rows

    def download_all(self):
        """Download all symbols."""
        logger.info("=" * 60)
        logger.info("THETADATA EOD DOWNLOADER (Gamma + Open Interest)")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Output: {self.data_dir}")
        logger.info(f"Threads: {self.num_threads}")
        logger.info("=" * 60)

        if not self.client.verify_connection():
            logger.error("Cannot connect to Theta Terminal")
            return

        logger.info("[+] Connected to Theta Terminal")

        # Check for PRO tier (bulk endpoints)
        self._bulk_available = self._check_bulk_availability()
        if self._bulk_available:
            logger.info("[+] PRO tier detected - using fast bulk endpoints")
        else:
            logger.info("[!] Standard tier - using per-day endpoints (slower)")
            logger.info("    Tip: Upgrade to PRO for ~100x faster downloads")

        new_entries = self.manifest.initialize_entries(
            self.symbols, self.start_date, self.end_date
        )
        logger.info(f"Initialized {new_entries} new entries")

        pending = self.manifest.get_pending_entries(include_failed=self.retry_failed)
        stats = self.manifest.get_stats()
        logger.info(f"Stats: {stats}")
        logger.info(f"To process: {len(pending)}")

        if not pending:
            logger.info("Nothing to download!")
            return

        # Show entries per symbol
        by_symbol = defaultdict(list)
        for entry in pending:
            by_symbol[entry.symbol].append(entry)
        logger.info(f"Entries across {len(by_symbol)} symbols:")
        for sym, entries in sorted(by_symbol.items()):
            logger.info(f"  {sym}: {len(entries)} months")

        start_time = time.time()
        logger.info(f"Starting work-queue download with {self.num_threads} threads...")
        logger.info("  - Threads pull from shared queue (no idle threads)")
        logger.info("  - Staggered startup (0.5s between threads)")
        logger.info("  - Coordinated rate-limit backoff")

        # Use work queue for better load balancing
        completed, failed, total_rows = self._download_with_work_queue(pending)

        self.manifest.save()
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"Completed: {completed}, Failed: {failed}")
        logger.info(f"Total rows: {total_rows:,}")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download EOD gamma + open interest from ThetaData"
    )
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: LIQUID_UNIVERSE)")
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE,
                        help=f"Start date (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (default: today)")
    parser.add_argument("--threads", type=int, default=DEFAULT_NUM_THREADS,
                        help=f"Thread count (default: {DEFAULT_NUM_THREADS})")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry failed entries")
    parser.add_argument("--show-manifest", action="store_true",
                        help="Show manifest stats and exit")
    parser.add_argument("--reset", action="store_true",
                        help="Reset manifest and start fresh")

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    output_dir = get_options_data_dir()
    setup_logging(output_dir)

    if args.reset:
        manifest_path = output_dir / "_manifest_eod.json"
        if manifest_path.exists():
            manifest_path.unlink()
            logger.info("Manifest reset")

    if args.show_manifest:
        manifest_path = output_dir / "_manifest_eod.json"
        if manifest_path.exists():
            manager = ManifestManager(manifest_path)
            stats = manager.get_stats()
            print(f"Manifest: {manifest_path}")
            print(f"Stats: {stats}")
        else:
            print("No manifest found")
        return

    downloader = EODDownloader(
        output_dir=output_dir,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        num_threads=args.threads,
        retry_failed=args.retry_failed,
    )
    downloader.download_all()


if __name__ == "__main__":
    main()
