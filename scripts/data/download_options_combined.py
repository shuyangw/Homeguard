"""
ThetaData Combined Options Downloader.

Downloads 1-min OHLCV + EOD gamma/open_interest in one pass,
saving directly to the combined schema.

Output: options_combined/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet

Schema (20 columns):
    - timestamp, expiration, strike, right
    - open, high, low, close, volume, trade_count, vwap
    - bid_close, ask_close, implied_vol, delta, theta, vega, underlying_px
    - gamma_eod, open_interest_eod

Usage:
    # Download all symbols
    python scripts/download_options_combined.py

    # Download specific symbols
    python scripts/download_options_combined.py --symbols SPY,QQQ

    # Specific date range
    python scripts/download_options_combined.py --start 2024-01-01 --end 2024-12-31

    # Resume interrupted download
    python scripts/download_options_combined.py --resume
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import queue
import signal
import time
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict

import httpx
import pandas as pd
import polars as pl

from src.settings import get_options_data_dir


logger = logging.getLogger("download_options_combined")
logger.setLevel(logging.INFO)


# =============================================================================
# Configuration
# =============================================================================

LIQUID_UNIVERSE = [
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Index Options
    "SPX",
    # Tech Mega-caps
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", "GOOGL", "AVGO", "PLTR",
    # Sector ETFs
    "XLF", "XLK", "XLV", "XLI", "XLE", "SMH",
    # Bonds
    "TLT",
    # Commodities
    "GLD", "SLV",
    # International
    "EEM", "FXI",
    # Crypto-related
    "IBIT", "MSTR", "COIN",
    # Volatility
    "VIX",
]

THETA_BASE_URL = "http://localhost:25503/v3"
DEFAULT_START_DATE = "2012-06-01"
DEFAULT_NUM_THREADS = 8
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]
RATE_LIMIT_WAIT = 60

_thread_local = threading.local()


def _get_thread_client(timeout: int = DEFAULT_TIMEOUT) -> httpx.Client:
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = httpx.Client(timeout=timeout)
    return _thread_local.client


def _get_thread_id() -> str:
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
    NO_DATA = "no_data"  # Data doesn't exist (pre-IPO, ticker change, etc.)


# IPO/listing dates for symbols - don't request data before these dates
# Format: "SYMBOL": "YYYY-MM-DD" (first month with options data available)
SYMBOL_DATA_START_DATES = {
    "IBIT": "2024-01-01",   # iShares Bitcoin Trust ETF launched Jan 2024
    "COIN": "2021-04-01",   # Coinbase IPO'd April 14, 2021
    "PLTR": "2020-10-01",   # Palantir IPO'd Sept 30, 2020 (options Oct)
    "META": "2021-11-01",   # Ticker changed from FB to META Oct 28, 2021
    "FB": "2012-06-01",     # Facebook IPO'd May 18, 2012 (options June)
    "MSTR": "2000-01-01",   # MicroStrategy - long history, no restriction
    "AVGO": "2012-06-01",   # Broadcom (as AVGO) - options available from 2009
    "SMH": "2012-06-01",    # VanEck Semiconductor ETF - long history
}

# Symbols that had ticker changes - map old ticker to new ticker and change date
TICKER_CHANGES = {
    # Old ticker -> (new ticker, change date, note)
    "FB": ("META", "2021-10-28", "Facebook renamed to Meta"),
}


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

    file_handler = RotatingFileHandler(
        log_dir / f"download_combined_{today}.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

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

class ThetaDataClient:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def _request(
        self,
        endpoint: str,
        params: Dict,
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

                if response.status_code == 472:
                    # No data found - not an error
                    return None, None

                if response.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None, f"Server error {response.status_code}"

                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}: {response.text[:100]}"

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

    def get_1min_ohlcv(
        self,
        symbol: str,
        expiration: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get 1-minute OHLCV + Quote + Greeks for all strikes (3 API calls merged)."""
        date_fmt = date_str.replace("-", "")
        base_params = {
            "symbol": symbol,
            "expiration": expiration,
            "date": date_fmt,
            "interval": "1m",
            "strike": "*",
            "right": "both",
            "format": "csv",
        }

        # 1. Get OHLC data
        ohlc_df, ohlc_err = self._request("/option/history/ohlc", base_params)
        if ohlc_err or ohlc_df is None or ohlc_df.empty:
            return None, ohlc_err

        # Rename 'count' to 'trade_count'
        if "count" in ohlc_df.columns:
            ohlc_df = ohlc_df.rename(columns={"count": "trade_count"})

        # 2. Get Quote data (bid/ask)
        quote_df, _ = self._request("/option/history/quote", base_params)
        if quote_df is not None and not quote_df.empty:
            # Rename bid/ask to bid_close/ask_close
            quote_df = quote_df.rename(columns={"bid": "bid_close", "ask": "ask_close"})
            # Select only needed columns
            quote_cols = ["timestamp", "expiration", "strike", "right", "bid_close", "ask_close"]
            quote_df = quote_df[[c for c in quote_cols if c in quote_df.columns]]
            # Merge
            ohlc_df = ohlc_df.merge(
                quote_df,
                on=["timestamp", "expiration", "strike", "right"],
                how="left"
            )

        # 3. Get First-order Greeks (delta, theta, vega, implied_vol)
        greeks_df, _ = self._request("/option/history/greeks/first_order", base_params)
        if greeks_df is not None and not greeks_df.empty:
            # Rename columns
            rename_map = {
                "implied_vol": "implied_vol",
                "underlying_price": "underlying_px",
            }
            greeks_df = greeks_df.rename(columns={k: v for k, v in rename_map.items() if k in greeks_df.columns})
            # Select only needed columns
            greeks_cols = ["timestamp", "expiration", "strike", "right", "implied_vol", "delta", "theta", "vega", "underlying_px"]
            greeks_df = greeks_df[[c for c in greeks_cols if c in greeks_df.columns]]
            # Merge
            ohlc_df = ohlc_df.merge(
                greeks_df,
                on=["timestamp", "expiration", "strike", "right"],
                how="left"
            )

        return ohlc_df, None

    def get_eod_greeks(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get EOD Greeks for ALL expirations on a single date (PRO tier with expiration=*)."""
        client = _get_thread_client(self.timeout)
        url = f"{THETA_BASE_URL}/option/history/greeks/eod"

        date_fmt = date_str.replace("-", "")

        params = {
            "symbol": symbol,
            "expiration": "*",
            "start_date": date_fmt,
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
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    return None, "Rate limit exceeded"

                if response.status_code == 472:
                    return None, None

                if response.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None, f"Server error {response.status_code}"

                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}: {response.text[:100]}"

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
                return None, "Connection failed"
            except Exception as e:
                return None, str(e)

        return None, "Max retries exceeded"

    def get_eod_open_interest(
        self,
        symbol: str,
        date_str: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get EOD open interest for ALL expirations on a single date."""
        client = _get_thread_client(self.timeout)
        url = f"{THETA_BASE_URL}/option/history/open_interest"

        date_fmt = date_str.replace("-", "")

        params = {
            "symbol": symbol,
            "expiration": "*",
            "date": date_fmt,
            "strike": "*",
            "right": "both",
            "format": "csv",
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = client.get(url, params=params, timeout=self.timeout)

                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    return None, "Rate limit exceeded"

                if response.status_code == 472:
                    return None, None

                if response.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None, f"Server error {response.status_code}"

                if response.status_code >= 400:
                    return None, f"Client error {response.status_code}: {response.text[:100]}"

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
                return None, "Connection failed"
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
        skipped_pre_ipo = 0

        with self.lock:
            for symbol in symbols:
                # Check if symbol has a known data start date (IPO, ticker change, etc.)
                symbol_start = start
                if symbol in SYMBOL_DATA_START_DATES:
                    ipo_date = datetime.strptime(SYMBOL_DATA_START_DATES[symbol], "%Y-%m-%d")
                    if ipo_date > start:
                        symbol_start = ipo_date
                        logger.debug(
                            f"{symbol}: Skipping dates before {SYMBOL_DATA_START_DATES[symbol]} (data start)"
                        )

                current = symbol_start.replace(day=1)
                while current <= end:
                    entry = DownloadEntry(symbol=symbol, year=current.year, month=current.month)
                    if entry.key not in self.manifest.entries:
                        self.manifest.entries[entry.key] = entry
                        new_count += 1
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)

                # Count skipped months for this symbol
                if symbol_start > start:
                    months_skipped = (symbol_start.year - start.year) * 12 + (symbol_start.month - start.month)
                    skipped_pre_ipo += months_skipped

        if skipped_pre_ipo > 0:
            logger.info(f"Skipped {skipped_pre_ipo} pre-IPO/pre-data entries based on SYMBOL_DATA_START_DATES")

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
# Combined Downloader
# =============================================================================

class CombinedDownloader:
    """Downloads 1-min + EOD data and saves in combined schema."""

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
        self.data_dir = self.output_dir / "options_combined"
        self.symbols = symbols or LIQUID_UNIVERSE
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.num_threads = num_threads
        self.retry_failed = retry_failed

        self.client = ThetaDataClient()
        self._shutdown_requested = False

        # Work queue coordination
        self._work_queue: Optional[queue.Queue] = None
        self._rate_limit_event = threading.Event()
        self._rate_limit_event.set()
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
        self.manifest = ManifestManager(self.output_dir / "_manifest_combined.json")

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def _handle_shutdown(self, signum, frame):
        if self._shutdown_requested:
            sys.exit(1)
        logger.warning("Shutdown requested...")
        self._shutdown_requested = True

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

    def _get_expirations_cached(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        with self._expiration_cache_lock:
            if symbol in self._expiration_cache:
                return self._expiration_cache[symbol], None

        expirations, error = self.client.get_expirations(symbol)
        if not error and expirations:
            with self._expiration_cache_lock:
                self._expiration_cache[symbol] = expirations

        return expirations, error

    def _merge_1min_with_eod(
        self,
        intraday_df: pl.DataFrame,
        eod_greeks_df: Optional[pd.DataFrame],
        eod_oi_df: Optional[pd.DataFrame],
        date_str: str,
    ) -> pl.DataFrame:
        """Merge 1-min data with EOD gamma and open interest."""

        # Drop intraday gamma/OI if present
        cols_to_drop = [c for c in ["gamma", "open_interest"] if c in intraday_df.columns]
        if cols_to_drop:
            intraday_df = intraday_df.drop(cols_to_drop)

        # Add date column for joining
        intraday_df = intraday_df.with_columns(
            pl.lit(date_str).alias("_date")
        )

        # Normalize join columns
        if "expiration" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("expiration").cast(pl.Utf8).alias("expiration")
            )
        if "right" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("right").cast(pl.Utf8).alias("right")
            )
        if "strike" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("strike").cast(pl.Float64).alias("strike")
            )

        # Prepare EOD data
        eod_data = {}

        # Process EOD Greeks (for gamma)
        if eod_greeks_df is not None and not eod_greeks_df.empty:
            for _, row in eod_greeks_df.iterrows():
                exp = str(row.get("expiration", ""))
                strike = float(row.get("strike", 0))
                right = str(row.get("right", ""))
                gamma = row.get("gamma")
                key = (exp, strike, right)
                if key not in eod_data:
                    eod_data[key] = {"gamma_eod": None, "open_interest_eod": None}
                eod_data[key]["gamma_eod"] = gamma

        # Process EOD Open Interest
        if eod_oi_df is not None and not eod_oi_df.empty:
            for _, row in eod_oi_df.iterrows():
                exp = str(row.get("expiration", ""))
                strike = float(row.get("strike", 0))
                right = str(row.get("right", ""))
                oi = row.get("open_interest")
                key = (exp, strike, right)
                if key not in eod_data:
                    eod_data[key] = {"gamma_eod": None, "open_interest_eod": None}
                eod_data[key]["open_interest_eod"] = oi

        # Create EOD dataframe for joining
        if eod_data:
            eod_records = []
            for (exp, strike, right), values in eod_data.items():
                eod_records.append({
                    "expiration": exp,
                    "strike": strike,
                    "right": right,
                    "gamma_eod": values["gamma_eod"],
                    "open_interest_eod": values["open_interest_eod"],
                })
            eod_df = pl.DataFrame(eod_records)

            # Join
            combined = intraday_df.join(
                eod_df,
                on=["expiration", "strike", "right"],
                how="left"
            )
        else:
            combined = intraday_df.with_columns([
                pl.lit(None).cast(pl.Float64).alias("gamma_eod"),
                pl.lit(None).cast(pl.Int64).alias("open_interest_eod"),
            ])

        # Drop temporary date column
        if "_date" in combined.columns:
            combined = combined.drop("_date")

        # Ensure columns exist
        if "gamma_eod" not in combined.columns:
            combined = combined.with_columns(pl.lit(None).cast(pl.Float64).alias("gamma_eod"))
        if "open_interest_eod" not in combined.columns:
            combined = combined.with_columns(pl.lit(None).cast(pl.Int64).alias("open_interest_eod"))

        return combined

    def _download_day(
        self,
        symbol: str,
        date_str: str,
        expirations: List[str],
    ) -> Tuple[Optional[pl.DataFrame], int, Optional[str]]:
        """Download and merge 1-min + EOD data for a single day."""
        tid = _get_thread_id()
        api_calls = 0

        # Filter relevant expirations (not expired)
        valid_exps = [e for e in expirations if e >= date_str]

        # Download 1-min data for each expiration
        all_1min = []
        for expiration in valid_exps:
            if self._shutdown_requested:
                return None, api_calls, "Shutdown"

            df, err = self.client.get_1min_ohlcv(symbol, expiration, date_str)
            api_calls += 1

            if err:
                if "rate" in err.lower():
                    return None, api_calls, err
                # Skip this expiration
                continue

            if df is not None and not df.empty:
                all_1min.append(pl.from_pandas(df))

        if not all_1min:
            return None, api_calls, None

        intraday_df = pl.concat(all_1min)

        # Download EOD Greeks (with expiration=*)
        eod_greeks_df, greeks_err = self.client.get_eod_greeks(symbol, date_str)
        api_calls += 1
        if greeks_err and "rate" in greeks_err.lower():
            return None, api_calls, greeks_err

        # Download EOD Open Interest (with expiration=*)
        eod_oi_df, oi_err = self.client.get_eod_open_interest(symbol, date_str)
        api_calls += 1
        if oi_err and "rate" in oi_err.lower():
            return None, api_calls, oi_err

        # Merge
        combined = self._merge_1min_with_eod(intraday_df, eod_greeks_df, eod_oi_df, date_str)

        return combined, api_calls, None

    def _download_month(
        self,
        entry: DownloadEntry,
        expirations: List[str],
    ) -> Tuple[int, Optional[str]]:
        """Download combined data for a single month."""
        tid = _get_thread_id()
        symbol = entry.symbol
        year = entry.year
        month = entry.month

        if self._partition_exists(symbol, year, month):
            entry.state = DownloadState.COMPLETE
            return 0, None  # Already pre-filtered, shouldn't happen often

        entry.state = DownloadState.IN_PROGRESS
        self.manifest.save()

        dates = self._get_dates_for_month(year, month)
        if not dates:
            entry.state = DownloadState.COMPLETE
            entry.rows_count = 0
            return 0, None

        logger.info(f"[{tid}] {entry.key}: {len(dates)} trading days")

        all_data = []
        total_api_calls = 0

        for date_idx, date_str in enumerate(dates):
            if self._shutdown_requested:
                return 0, "Shutdown"

            if date_idx % 5 == 0:
                logger.info(f"[{tid}] {entry.key}: Day {date_idx + 1}/{len(dates)}")

            combined, api_calls, error = self._download_day(symbol, date_str, expirations)
            total_api_calls += api_calls

            if error:
                if "rate" in error.lower():
                    return 0, error
                logger.debug(f"[{tid}] {entry.key} {date_str}: {error}")
                continue

            if combined is not None:
                all_data.append(combined)

        if self._shutdown_requested:
            return 0, "Shutdown"

        # Save
        total_rows = 0
        if all_data:
            final_df = pl.concat(all_data, how="diagonal_relaxed")
            total_rows = len(final_df)

            partition_dir = self._get_partition_path(symbol, year, month).parent
            partition_dir.mkdir(parents=True, exist_ok=True)

            final_df.write_parquet(
                self._get_partition_path(symbol, year, month),
                compression="zstd",
            )

            entry.state = DownloadState.COMPLETE
            entry.rows_count = total_rows
            entry.completed_at = datetime.now().isoformat()
            self.manifest.save()

            logger.info(f"[+] [{tid}] {entry.key}: {total_rows:,} rows, {total_api_calls} API calls")
        else:
            # No data returned - mark as failed so we can retry
            entry.state = DownloadState.FAILED
            entry.rows_count = 0
            entry.error_message = f"No data returned ({total_api_calls} API calls)"
            self.manifest.save()

            logger.warning(f"[-] [{tid}] {entry.key}: No data returned ({total_api_calls} API calls)")

        return total_rows, None

    def _worker(self, worker_id: int, startup_delay: float = 0.0) -> Dict[str, int]:
        """Worker thread."""
        tid = f"W{worker_id}"
        threading.current_thread().name = f"Worker_{worker_id}"

        if startup_delay > 0:
            time.sleep(startup_delay)

        stats = {"completed": 0, "failed": 0, "rows": 0}
        logger.info(f"[{tid}] Worker started")

        while not self._shutdown_requested:
            try:
                entry = self._work_queue.get(timeout=2.0)
            except queue.Empty:
                if self._work_queue.empty():
                    break
                continue

            if entry is None:
                self._work_queue.task_done()
                break

            entry_key = entry.key
            symbol = entry.symbol

            with self._in_flight_lock:
                if entry_key in self._in_flight:
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
                    if "rate" in error.lower():
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
        """Download entries using work queue."""
        self._work_queue = queue.Queue()

        # Interleave symbols
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

        # Start workers
        threads = []
        stagger_delay = 0.3

        for i in range(self.num_threads):
            t = threading.Thread(
                target=self._worker,
                args=(i, i * stagger_delay),
                daemon=True,
            )
            threads.append(t)
            t.start()

        logger.info(f"Started {len(threads)} workers")

        # Monitor progress
        total_entries = added
        start_time = time.time()

        while not self._work_queue.empty() and not self._shutdown_requested:
            time.sleep(5.0)
            remaining = self._work_queue.qsize()
            if remaining > 0:
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
        logger.info("THETADATA COMBINED DOWNLOADER (1-min + EOD)")
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

        new_entries = self.manifest.initialize_entries(
            self.symbols, self.start_date, self.end_date
        )
        logger.info(f"Initialized {new_entries} new entries")

        pending = self.manifest.get_pending_entries(include_failed=self.retry_failed)
        stats = self.manifest.get_stats()
        logger.info(f"Stats: {stats}")
        logger.info(f"Pending in manifest: {len(pending)}")

        # Fast pre-filter: skip entries where file already exists
        logger.info("Checking for existing files...")
        filtered = []
        skipped = 0
        for entry in pending:
            if self._partition_exists(entry.symbol, entry.year, entry.month):
                entry.state = DownloadState.COMPLETE
                skipped += 1
            else:
                filtered.append(entry)

        if skipped > 0:
            self.manifest.save()
            logger.info(f"Skipped {skipped} already-downloaded partitions")

        pending = filtered
        logger.info(f"To process: {len(pending)}")

        if not pending:
            logger.info("Nothing to download!")
            return

        start_time = time.time()
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
        description="Download combined 1-min + EOD options data from ThetaData"
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
        manifest_path = output_dir / "_manifest_combined.json"
        if manifest_path.exists():
            manifest_path.unlink()
            logger.info("Manifest reset")

    if args.show_manifest:
        manifest_path = output_dir / "_manifest_combined.json"
        if manifest_path.exists():
            manager = ManifestManager(manifest_path)
            stats = manager.get_stats()
            print(f"Manifest: {manifest_path}")
            print(f"Stats: {stats}")
        else:
            print("No manifest found")
        return

    downloader = CombinedDownloader(
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
