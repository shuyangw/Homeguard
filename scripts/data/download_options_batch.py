"""
ThetaData Async Batch Options Downloader.

Downloads 1-min OHLCV + EOD gamma/open_interest using:
  - Monthly batch API calls (start_date/end_date) -> ~20x fewer round trips
  - Async httpx with streaming responses -> eliminates timeout/abort errors
  - asyncio.Semaphore concurrency -> efficient I/O without thread overhead

Output: options_combined/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet

Schema (20 columns):
    - timestamp, expiration, strike, right
    - open, high, low, close, volume, trade_count, vwap
    - bid_close, ask_close, implied_vol, delta, theta, vega, underlying_px
    - gamma_eod, open_interest_eod

Usage:
    # Download all symbols
    python scripts/download_options_batch.py

    # Download specific symbols
    python scripts/download_options_batch.py --symbols SPY,QQQ

    # Specific date range
    python scripts/download_options_batch.py --start 2012-06-01 --end 2016-12-31

    # Retry previously failed entries
    python scripts/download_options_batch.py --retry-failed

    # Increase concurrency (match HTTP_CONCURRENCY in terminal config)
    python scripts/download_options_batch.py --concurrency 12
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import asyncio
import concurrent.futures
import json
import signal
import time
import shutil
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import List, Optional, Dict, Tuple
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict

import httpx
import pandas as pd
import polars as pl

from src.settings import get_options_data_dir


logger = logging.getLogger("download_options_batch")
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
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT = 300  # 5 min for large batch streaming responses
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]
RATE_LIMIT_WAIT = 60

# IPO/listing dates - don't request data before these dates
SYMBOL_DATA_START_DATES = {
    "IBIT": "2024-01-01",
    "COIN": "2021-04-01",
    "PLTR": "2020-10-01",
    "META": "2021-11-01",
    "FB": "2012-06-01",
    "MSTR": "2000-01-01",
    "AVGO": "2012-06-01",
    "SMH": "2012-06-01",
}

TICKER_CHANGES = {
    "FB": ("META", "2021-10-28", "Facebook renamed to Meta"),
}


# =============================================================================
# Data Classes
# =============================================================================

class DownloadState(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    NO_DATA = "no_data"


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
        log_dir / f"download_batch_{today}.log",
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
# Manifest Manager (sync - called from async context via locks)
# =============================================================================

class ManifestManager:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.lock = asyncio.Lock()
        self.manifest = self._load_or_create()

    def _load_or_create(self) -> DownloadManifest:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return DownloadManifest.from_dict(json.load(f))
            except Exception:
                pass
        return DownloadManifest(created_at=datetime.now().isoformat())

    async def save(self) -> None:
        async with self.lock:
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

        for symbol in symbols:
            symbol_start = start
            if symbol in SYMBOL_DATA_START_DATES:
                ipo_date = datetime.strptime(SYMBOL_DATA_START_DATES[symbol], "%Y-%m-%d")
                if ipo_date > start:
                    symbol_start = ipo_date

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

            if symbol_start > start:
                months_skipped = (symbol_start.year - start.year) * 12 + (symbol_start.month - start.month)
                skipped_pre_ipo += months_skipped

        if skipped_pre_ipo > 0:
            logger.info(f"Skipped {skipped_pre_ipo} pre-IPO/pre-data entries based on SYMBOL_DATA_START_DATES")

        return new_count

    def get_pending_entries(self, include_failed: bool = False) -> List[DownloadEntry]:
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
        stats = defaultdict(int)
        for entry in self.manifest.entries.values():
            stats[entry.state.value] += 1
        return dict(stats)


# =============================================================================
# Async ThetaData Client
# =============================================================================

class AsyncThetaClient:
    """Async ThetaData API client with streaming response support."""

    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)
        self._client: Optional[httpx.AsyncClient] = None
        # ThreadPool (not ProcessPool) because Polars releases the GIL,
        # giving true parallelism without expensive IPC serialization.
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=30.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.thread_pool.shutdown(wait=False)

    def _parse_bytes_in_thread(self, content: bytes) -> Optional[pd.DataFrame]:
        """Parse CSV bytes using Polars (GIL-free) and convert to Pandas.

        Running in a thread does NOT block the asyncio loop because
        Polars releases the GIL during the heavy parsing phase.
        """
        if not content:
            return None
        try:
            df = pl.read_csv(content, ignore_errors=True)
            if df.is_empty():
                return None
            return df.to_pandas()
        except Exception:
            return None

    async def _stream_request(
        self,
        endpoint: str,
        params: Dict,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Make a streaming request with pipelined download and parse phases.

        The semaphore only wraps the network I/O phase so that parsing
        (offloaded to a GIL-free Polars thread) does not block new downloads.
        """
        content = b""
        error: Optional[str] = None

        # --- DOWNLOAD PHASE (semaphore held) ---
        async with self.semaphore:
            client = await self._get_client()
            url = f"{THETA_BASE_URL}{endpoint}"

            for attempt in range(MAX_RETRIES):
                try:
                    async with client.stream("GET", url, params=params, timeout=self.timeout) as response:
                        if response.status_code == 429:
                            if attempt < MAX_RETRIES - 1:
                                logger.warning(f"Rate limited, waiting {RATE_LIMIT_WAIT}s...")
                                await asyncio.sleep(RATE_LIMIT_WAIT)
                                continue
                            return None, "Rate limit exceeded"

                        if response.status_code == 472:
                            return None, None

                        if response.status_code >= 500:
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(RETRY_BACKOFF[attempt])
                                continue
                            return None, f"Server error {response.status_code}"

                        if response.status_code >= 400:
                            body = b""
                            async for chunk in response.aiter_bytes():
                                body += chunk
                                if len(body) > 200:
                                    break
                            return None, f"Client error {response.status_code}: {body.decode('utf-8', errors='replace')[:200]}"

                        # Stream response body into buffer
                        buffer = BytesIO()
                        async for chunk in response.aiter_bytes():
                            buffer.write(chunk)

                    # Download complete - grab raw bytes
                    buffer.seek(0)
                    content = buffer.getvalue()
                    break  # Exit retry loop on success

                except httpx.TimeoutException:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BACKOFF[attempt])
                        continue
                    error = "Timeout"
                except httpx.ConnectError:
                    error = "Connection failed - is Theta Terminal running?"
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BACKOFF[attempt])
                        continue
                    error = str(e)
            else:
                if not content and not error:
                    error = "Max retries exceeded"

        # --- Semaphore released - network slot is free for next download ---

        if error:
            return None, error
        if not content:
            return None, None

        # --- PARSING PHASE (no semaphore, GIL-free Polars in thread pool) ---
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                self.thread_pool, self._parse_bytes_in_thread, content
            )
            return df, None
        except Exception as e:
            return None, f"Parsing error: {e}"

    async def verify_connection(self) -> bool:
        df, error = await self._stream_request(
            "/option/list/expirations",
            {"symbol": "SPY", "format": "csv"},
        )
        return error is None or "Connection" not in str(error)

    async def get_expirations(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        df, error = await self._stream_request(
            "/option/list/expirations",
            {"symbol": symbol, "format": "csv"},
        )
        if error:
            return [], error
        if df is None or df.empty:
            return [], None
        return df["expiration"].tolist(), None

    async def get_1min_batch(
        self,
        symbol: str,
        expiration: str,
        year: int,
        month: int,
    ) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
        """Get 1-min OHLCV + Quote + Greeks for a full month (3 async streaming calls)."""
        _, last_day = monthrange(year, month)
        start_date = f"{year}{month:02d}01"
        end_date = f"{year}{month:02d}{last_day:02d}"

        base_params = {
            "symbol": symbol,
            "expiration": expiration,
            "start_date": start_date,
            "end_date": end_date,
            "interval": "1m",
            "strike": "*",
            "right": "both",
            "format": "csv",
        }

        # Fire all 3 requests concurrently
        ohlc_task = self._stream_request("/option/history/ohlc", base_params)
        quote_task = self._stream_request("/option/history/quote", base_params)
        greeks_task = self._stream_request("/option/history/greeks/first_order", base_params)

        results = await asyncio.gather(ohlc_task, quote_task, greeks_task)
        api_calls = 3

        ohlc_df, ohlc_err = results[0]
        quote_df, _ = results[1]
        greeks_df, _ = results[2]

        # OHLC is required
        if ohlc_err:
            return None, api_calls, ohlc_err
        if ohlc_df is None or ohlc_df.empty:
            return None, api_calls, None

        if "count" in ohlc_df.columns:
            ohlc_df = ohlc_df.rename(columns={"count": "trade_count"})

        # Merge quote data
        if quote_df is not None and not quote_df.empty:
            quote_df = quote_df.rename(columns={"bid": "bid_close", "ask": "ask_close"})
            quote_cols = ["timestamp", "expiration", "strike", "right", "bid_close", "ask_close"]
            quote_df = quote_df[[c for c in quote_cols if c in quote_df.columns]]
            ohlc_df = ohlc_df.merge(
                quote_df,
                on=["timestamp", "expiration", "strike", "right"],
                how="left",
            )

        # Merge greeks data
        if greeks_df is not None and not greeks_df.empty:
            rename_map = {"underlying_price": "underlying_px"}
            greeks_df = greeks_df.rename(
                columns={k: v for k, v in rename_map.items() if k in greeks_df.columns}
            )
            greeks_cols = [
                "timestamp", "expiration", "strike", "right",
                "implied_vol", "delta", "theta", "vega", "underlying_px",
            ]
            greeks_df = greeks_df[[c for c in greeks_cols if c in greeks_df.columns]]
            ohlc_df = ohlc_df.merge(
                greeks_df,
                on=["timestamp", "expiration", "strike", "right"],
                how="left",
            )

        return ohlc_df, api_calls, None

    async def get_eod_batch(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], int, Optional[str]]:
        """Get EOD Greeks + OI for a full month (2 async streaming calls)."""
        _, last_day = monthrange(year, month)
        start_date = f"{year}{month:02d}01"
        end_date = f"{year}{month:02d}{last_day:02d}"

        # Fire both EOD requests concurrently
        greeks_task = self._stream_request(
            "/option/history/greeks/eod",
            {
                "symbol": symbol, "expiration": "*",
                "start_date": start_date, "end_date": end_date,
                "strike": "*", "right": "both", "format": "csv",
            },
        )
        oi_task = self._stream_request(
            "/option/history/open_interest",
            {
                "symbol": symbol, "expiration": "*",
                "start_date": start_date, "end_date": end_date,
                "strike": "*", "right": "both", "format": "csv",
            },
        )

        results = await asyncio.gather(greeks_task, oi_task)
        api_calls = 2

        greeks_df, greeks_err = results[0]
        oi_df, oi_err = results[1]

        error = None
        if greeks_err and "rate" in greeks_err.lower():
            error = greeks_err
        elif oi_err and "rate" in oi_err.lower():
            error = oi_err

        return greeks_df, oi_df, api_calls, error


# =============================================================================
# Async Batch Downloader
# =============================================================================

class AsyncBatchDownloader:
    """Downloads 1-min + EOD data using async streaming batch API calls."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout: int = DEFAULT_TIMEOUT,
        retry_failed: bool = False,
    ):
        self.output_dir = output_dir or get_options_data_dir()
        self.data_dir = self.output_dir / "options_combined"
        self.symbols = symbols or LIQUID_UNIVERSE
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.concurrency = concurrency
        self.retry_failed = retry_failed

        self.client = AsyncThetaClient(concurrency=concurrency, timeout=timeout)
        self._shutdown_requested = False

        # Expiration cache
        self._expiration_cache: Dict[str, List[str]] = {}

        # Progress tracking
        self._completed = 0
        self._failed = 0
        self._total_rows = 0
        self._total_entries = 0
        self._start_time = 0.0

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = ManifestManager(self.output_dir / "_manifest_combined.json")

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

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

    async def _get_expirations_cached(self, symbol: str) -> Tuple[List[str], Optional[str]]:
        if symbol in self._expiration_cache:
            return self._expiration_cache[symbol], None

        expirations, error = await self.client.get_expirations(symbol)
        if not error and expirations:
            self._expiration_cache[symbol] = expirations

        return expirations, error

    def _filter_expirations_for_month(
        self,
        expirations: List[str],
        year: int,
        month: int,
    ) -> List[str]:
        month_start = f"{year}{month:02d}01"
        return [e for e in expirations if str(e) >= month_start]

    def _merge_with_eod(
        self,
        intraday_df: pl.DataFrame,
        eod_greeks_df: Optional[pd.DataFrame],
        eod_oi_df: Optional[pd.DataFrame],
    ) -> pl.DataFrame:
        """Merge monthly 1-min data with monthly EOD gamma/OI."""
        cols_to_drop = [c for c in ["gamma", "open_interest"] if c in intraday_df.columns]
        if cols_to_drop:
            intraday_df = intraday_df.drop(cols_to_drop)

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

        # Extract date from timestamp for EOD join
        if "timestamp" in intraday_df.columns:
            ts_col = intraday_df["timestamp"]
            if ts_col.dtype in (pl.Int64, pl.UInt64, pl.Float64):
                intraday_df = intraday_df.with_columns(
                    (pl.col("timestamp") // 86400000 * 86400000)
                    .cast(pl.Datetime("ms"))
                    .dt.strftime("%Y%m%d")
                    .alias("_eod_date")
                )
            else:
                intraday_df = intraday_df.with_columns(
                    pl.col("timestamp").cast(pl.Utf8).str.slice(0, 10)
                    .str.replace_all("-", "").alias("_eod_date")
                )

        # Build EOD lookup
        eod_records = {}

        if eod_greeks_df is not None and not eod_greeks_df.empty:
            date_col = None
            for col_name in ["date", "timestamp", "trade_date"]:
                if col_name in eod_greeks_df.columns:
                    date_col = col_name
                    break

            for _, row in eod_greeks_df.iterrows():
                exp = str(row.get("expiration", ""))
                strike = float(row.get("strike", 0))
                right = str(row.get("right", ""))
                gamma = row.get("gamma")
                date_val = str(row.get(date_col, "")) if date_col else ""
                date_key = date_val.replace("-", "")[:8]
                key = (date_key, exp, strike, right)
                if key not in eod_records:
                    eod_records[key] = {"gamma_eod": None, "open_interest_eod": None}
                eod_records[key]["gamma_eod"] = gamma

        if eod_oi_df is not None and not eod_oi_df.empty:
            date_col = None
            for col_name in ["date", "timestamp", "trade_date"]:
                if col_name in eod_oi_df.columns:
                    date_col = col_name
                    break

            for _, row in eod_oi_df.iterrows():
                exp = str(row.get("expiration", ""))
                strike = float(row.get("strike", 0))
                right = str(row.get("right", ""))
                oi = row.get("open_interest")
                date_val = str(row.get(date_col, "")) if date_col else ""
                date_key = date_val.replace("-", "")[:8]
                key = (date_key, exp, strike, right)
                if key not in eod_records:
                    eod_records[key] = {"gamma_eod": None, "open_interest_eod": None}
                eod_records[key]["open_interest_eod"] = oi

        if eod_records:
            eod_list = []
            for (date_key, exp, strike, right), values in eod_records.items():
                eod_list.append({
                    "_eod_date": date_key,
                    "expiration": exp,
                    "strike": strike,
                    "right": right,
                    "gamma_eod": values["gamma_eod"],
                    "open_interest_eod": values["open_interest_eod"],
                })
            eod_df = pl.DataFrame(eod_list)

            if "strike" in eod_df.columns:
                eod_df = eod_df.with_columns(pl.col("strike").cast(pl.Float64))

            combined = intraday_df.join(
                eod_df,
                on=["_eod_date", "expiration", "strike", "right"],
                how="left",
            )
        else:
            combined = intraday_df.with_columns([
                pl.lit(None).cast(pl.Float64).alias("gamma_eod"),
                pl.lit(None).cast(pl.Int64).alias("open_interest_eod"),
            ])

        if "_eod_date" in combined.columns:
            combined = combined.drop("_eod_date")

        if "gamma_eod" not in combined.columns:
            combined = combined.with_columns(pl.lit(None).cast(pl.Float64).alias("gamma_eod"))
        if "open_interest_eod" not in combined.columns:
            combined = combined.with_columns(pl.lit(None).cast(pl.Int64).alias("open_interest_eod"))

        return combined

    async def _download_month(self, entry: DownloadEntry) -> Tuple[int, Optional[str]]:
        """Download combined data for a single month using async batch calls."""
        symbol = entry.symbol
        year = entry.year
        month = entry.month

        if self._partition_exists(symbol, year, month):
            entry.state = DownloadState.COMPLETE
            return 0, None

        entry.state = DownloadState.IN_PROGRESS
        await self.manifest.save()

        # Get expirations
        expirations, exp_error = await self._get_expirations_cached(symbol)
        if exp_error:
            entry.state = DownloadState.FAILED
            entry.error_message = exp_error
            await self.manifest.save()
            return 0, exp_error

        valid_exps = self._filter_expirations_for_month(expirations, year, month)
        if not valid_exps:
            entry.state = DownloadState.COMPLETE
            entry.rows_count = 0
            entry.completed_at = datetime.now().isoformat()
            await self.manifest.save()
            logger.info(f"{entry.key}: No active expirations")
            return 0, None

        logger.info(f"{entry.key}: {len(valid_exps)} expirations (async batch)")

        # Download 1-min data: fire all expirations concurrently
        # (semaphore in client controls actual concurrency)
        tasks = [
            self.client.get_1min_batch(symbol, exp, year, month)
            for exp in valid_exps
        ]
        results = await asyncio.gather(*tasks)

        all_1min = []
        total_api_calls = 0
        rate_limited = False

        for df, api_calls, error in results:
            total_api_calls += api_calls
            if error and "rate" in error.lower():
                rate_limited = True
            if df is not None and not df.empty:
                all_1min.append(pl.from_pandas(df))

        if rate_limited:
            return 0, "Rate limit exceeded"

        if self._shutdown_requested:
            return 0, "Shutdown"

        if not all_1min:
            entry.state = DownloadState.FAILED
            entry.rows_count = 0
            entry.error_message = f"No data returned ({total_api_calls} API calls)"
            await self.manifest.save()
            logger.warning(f"[-] {entry.key}: No data returned ({total_api_calls} API calls)")
            return 0, None

        intraday_df = pl.concat(all_1min, how="diagonal_relaxed")

        # Download EOD data (both endpoints concurrently)
        eod_greeks_df, eod_oi_df, eod_api_calls, eod_error = await self.client.get_eod_batch(
            symbol, year, month
        )
        total_api_calls += eod_api_calls
        if eod_error and "rate" in eod_error.lower():
            return 0, eod_error

        # Merge
        combined = self._merge_with_eod(intraday_df, eod_greeks_df, eod_oi_df)
        total_rows = len(combined)

        # Atomic write
        partition_path = self._get_partition_path(symbol, year, month)
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = partition_path.with_suffix(".tmp")
        combined.write_parquet(tmp_path, compression="zstd")
        shutil.move(str(tmp_path), str(partition_path))

        entry.state = DownloadState.COMPLETE
        entry.rows_count = total_rows
        entry.completed_at = datetime.now().isoformat()
        await self.manifest.save()

        logger.info(
            f"[+] {entry.key}: {total_rows:,} rows, "
            f"{len(valid_exps)} exps, {total_api_calls} API calls"
        )

        return total_rows, None

    async def _process_entry(self, entry: DownloadEntry) -> None:
        """Process a single download entry with error handling."""
        try:
            rows, error = await self._download_month(entry)
            if error:
                if "rate" in error.lower():
                    logger.warning(f"Rate limited on {entry.key}, waiting {RATE_LIMIT_WAIT}s...")
                    await asyncio.sleep(RATE_LIMIT_WAIT)
                    # Retry once
                    rows, error = await self._download_month(entry)
                if error:
                    self._failed += 1
                    return

            self._completed += 1
            self._total_rows += rows

        except Exception as e:
            logger.error(f"{entry.key}: Exception: {e}")
            entry.state = DownloadState.FAILED
            entry.error_message = str(e)
            await self.manifest.save()
            self._failed += 1

    async def _progress_monitor(self) -> None:
        """Periodically log progress."""
        while not self._shutdown_requested:
            await asyncio.sleep(15.0)
            processed = self._completed + self._failed
            if processed > 0 and processed < self._total_entries:
                elapsed = time.time() - self._start_time
                rate = processed / elapsed
                remaining = self._total_entries - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = self._format_time(eta_seconds)
                elapsed_str = self._format_time(elapsed)
                pct = (processed / self._total_entries) * 100
                logger.info(
                    f"[PROGRESS] {processed}/{self._total_entries} ({pct:.1f}%) | "
                    f"OK: {self._completed} Failed: {self._failed} | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                )

    async def download_all(self):
        logger.info("=" * 60)
        logger.info("THETADATA ASYNC BATCH DOWNLOADER")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Output: {self.data_dir}")
        logger.info(f"Concurrency: {self.concurrency}")
        logger.info(f"Timeout: {self.client.timeout}s")
        logger.info("=" * 60)

        if not await self.client.verify_connection():
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

        # Pre-filter
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
            await self.manifest.save()
            logger.info(f"Skipped {skipped} already-downloaded partitions")

        pending = filtered
        logger.info(f"To process: {len(pending)}")

        if not pending:
            logger.info("Nothing to download!")
            await self.client.close()
            return

        # Interleave symbols for balanced load
        entries_by_symbol = defaultdict(list)
        for entry in pending:
            entries_by_symbol[entry.symbol].append(entry)
        for symbol in entries_by_symbol:
            entries_by_symbol[symbol].sort(key=lambda e: (e.year, e.month), reverse=True)

        interleaved = []
        symbol_iterators = {sym: iter(entries) for sym, entries in entries_by_symbol.items()}
        symbols = list(symbol_iterators.keys())
        while symbol_iterators:
            for sym in list(symbols):
                try:
                    interleaved.append(next(symbol_iterators[sym]))
                except StopIteration:
                    del symbol_iterators[sym]
                    symbols.remove(sym)

        self._total_entries = len(interleaved)
        self._start_time = time.time()

        logger.info(f"Processing {self._total_entries} entries with concurrency={self.concurrency}")

        # Start progress monitor
        monitor_task = asyncio.create_task(self._progress_monitor())

        # Process entries with bounded concurrency via semaphore
        # Use asyncio.Semaphore at the month level too to avoid queueing
        # hundreds of tasks that all compete for the API semaphore
        month_semaphore = asyncio.Semaphore(self.concurrency * 2)

        async def bounded_process(entry: DownloadEntry):
            async with month_semaphore:
                if not self._shutdown_requested:
                    await self._process_entry(entry)

        tasks = [bounded_process(entry) for entry in interleaved]
        await asyncio.gather(*tasks)

        # Stop monitor
        self._shutdown_requested = True
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        await self.manifest.save()
        await self.client.close()

        elapsed = time.time() - self._start_time

        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"Completed: {self._completed}, Failed: {self._failed}")
        logger.info(f"Total rows: {self._total_rows:,}")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download combined options data from ThetaData using async streaming batch API calls"
    )
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: LIQUID_UNIVERSE)")
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE,
                        help=f"Start date (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (default: today)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max concurrent requests (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
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

    # Handle Ctrl+C gracefully on Windows
    def _signal_handler(sig, frame):
        logger.warning("Shutdown requested (Ctrl+C)...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    downloader = AsyncBatchDownloader(
        output_dir=output_dir,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retry_failed=args.retry_failed,
    )

    asyncio.run(downloader.download_all())


if __name__ == "__main__":
    main()
