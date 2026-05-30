"""Base downloader with shared infrastructure for all data acquisition plugins."""

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import pandas as pd
from alpaca.common.exceptions import APIError

from src.data.acquisition.manifest import DownloadManifest
from src.data.acquisition.schemas import SchemaValidationError, validate_schema
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_thread_local = threading.local()


@dataclass
class DownloadResult:
    """Result of a batch download operation."""

    total_symbols: int
    succeeded: int
    failed: int
    total_rows: int
    elapsed_seconds: float
    failed_symbols: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_symbols == 0:
            return 0.0
        return (self.succeeded / self.total_symbols) * 100


class BaseDownloader(ABC):
    """Shared infrastructure for all data acquisition plugins.

    Subclasses implement only the API-specific methods:
    - _create_client() -> Any
    - _fetch_symbol_data(client, symbol, start, end) -> pd.DataFrame
    - _get_schema() -> list[str]
    - _get_storage_subdir() -> str
    - _normalize_symbol(symbol) -> str
    """

    DEFAULT_NUM_THREADS = 6
    MAX_RETRIES = 3
    END_RETRY_ROUNDS = 3
    RETRY_DELAY = 5.0  # seconds
    MANIFEST_FLUSH_EVERY = 25  # completions
    MANIFEST_FLUSH_INTERVAL_SEC = 60.0

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        num_threads: int = DEFAULT_NUM_THREADS,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY,
    ):
        self.base_output_dir = output_dir or get_local_storage_dir()
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.manifest = DownloadManifest(
            self.base_output_dir, self._get_storage_subdir()
        )

    @abstractmethod
    def _create_client(self) -> Any:
        """Create an API client instance (called once per thread)."""
        pass

    @abstractmethod
    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch data for a single symbol from the API."""
        pass

    @abstractmethod
    def _get_schema(self) -> list[str]:
        """Return the expected column names for validation."""
        pass

    @abstractmethod
    def _get_storage_subdir(self) -> str:
        """Return subdirectory name, e.g. 'equities_1min'."""
        pass

    @abstractmethod
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to filesystem-safe format."""
        pass

    def _get_output_dir(self) -> Path:
        return self.base_output_dir / self._get_storage_subdir()

    def _get_client(self) -> Any:
        attr = f"_client_{self._get_storage_subdir()}"
        if not hasattr(_thread_local, attr):
            setattr(_thread_local, attr, self._create_client())
        return getattr(_thread_local, attr)

    def _emit_event(self, event: str, **fields: Any) -> None:
        """Append one JSON event line to <flattened-subdir>.progress.jsonl. Crash-safe."""
        log_dir = self.base_output_dir / "_manifests"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Flatten nested subdir into a single filename to match manifest layout
        flat_name = self._get_storage_subdir().replace("/", "_")
        log_path = log_dir / f"{flat_name}.progress.jsonl"
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def get_existing_symbols(self) -> Set[str]:
        output_dir = self._get_output_dir()
        existing = set()
        if output_dir.exists():
            for symbol_dir in output_dir.glob("symbol=*"):
                symbol = symbol_dir.name.replace("symbol=", "")
                if list(symbol_dir.glob("**/data.parquet")):
                    existing.add(symbol)
        return existing

    def _save_partitioned(self, df: pd.DataFrame, symbol: str) -> int:
        """Save DataFrame in hive-partitioned format. Returns row count.

        Writes via .tmp + os.replace for atomicity -- crash mid-write
        leaves no partial data.parquet behind.
        """
        output_dir = self._get_output_dir()
        fs_symbol = self._normalize_symbol(symbol)

        df = df.copy()
        ts_col = pd.to_datetime(df["timestamp"])
        df["_year"] = ts_col.dt.year
        df["_month"] = ts_col.dt.month

        rows_saved = 0
        for (year, month), group in df.groupby(["_year", "_month"]):
            partition_dir = (
                output_dir
                / f"symbol={fs_symbol}"
                / f"year={year}"
                / f"month={month}"
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            final_path = partition_dir / "data.parquet"
            tmp_path = partition_dir / "data.parquet.tmp"
            data_to_save = group.drop(columns=["_year", "_month"])
            data_to_save.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, final_path)
            rows_saved += len(data_to_save)

        return rows_saved

    def _download_single(
        self, symbol: str, start_date: str, end_date: str
    ) -> Tuple[str, bool, int, Optional[str]]:
        """Download a single symbol with retry logic."""
        tid = threading.current_thread().name
        self._emit_event("download_start", symbol=symbol, thread=tid)

        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    logger.info(
                        f"[{tid}] {symbol}: Retry {attempt}/{self.max_retries}"
                    )
                    self._emit_event(
                        "retry",
                        symbol=symbol,
                        attempt=attempt,
                    )

                self.manifest.set_entry(symbol, status="in_progress")
                client = self._get_client()
                df = self._fetch_symbol_data(client, symbol, start_date, end_date)

                if df.empty:
                    logger.warning(f"[{tid}] {symbol}: No data returned")
                    self.manifest.set_entry(symbol, status="complete", rows=0)
                    self._emit_event("download_complete", symbol=symbol, rows=0)
                    return (symbol, True, 0, None)

                validate_schema(df, self._get_schema())
                rows = self._save_partitioned(df, symbol)
                self.manifest.set_entry(symbol, status="complete", rows=rows)
                logger.info(f"[{tid}] {symbol}: Saved {rows:,} rows")
                self._emit_event("download_complete", symbol=symbol, rows=rows)
                return (symbol, True, rows, None)

            except SchemaValidationError as e:
                error_msg = str(e)
                logger.error(f"[{tid}] {symbol}: Schema error - {error_msg}")
                self.manifest.set_entry(symbol, status="failed", error=error_msg)
                self._emit_event(
                    "download_failed", symbol=symbol, error=error_msg
                )
                return (symbol, False, 0, error_msg)

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"[{tid}] {symbol}: Attempt {attempt} failed - {error_msg}"
                )
                if attempt < self.max_retries:
                    base_delay = self.retry_delay * attempt
                    if (
                        isinstance(e, APIError)
                        and getattr(e, "status_code", None) in (429, 500, 502, 503, 504)
                    ):
                        time.sleep(base_delay + 10.0)
                    else:
                        time.sleep(base_delay)
                else:
                    self.manifest.set_entry(
                        symbol, status="failed", error=error_msg
                    )
                    self._emit_event(
                        "download_failed", symbol=symbol, error=error_msg
                    )
                    return (symbol, False, 0, error_msg)

        return (symbol, False, 0, "Max retries exceeded")

    def download(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        skip_existing: bool = False,
    ) -> DownloadResult:
        """Download data for a list of symbols."""
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        output_dir = self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._emit_event(
            "pass_start",
            subdir=self._get_storage_subdir(),
            symbol_count=len(symbols),
        )

        if skip_existing:
            existing = self.get_existing_symbols()
            symbols = [
                s for s in symbols if self._normalize_symbol(s) not in existing
            ]

        if not symbols:
            return DownloadResult(
                total_symbols=0,
                succeeded=0,
                failed=0,
                total_rows=0,
                elapsed_seconds=0.0,
            )

        logger.info(f"Downloading {len(symbols)} symbols to {output_dir}")
        start_time = time.time()
        succeeded = 0
        failed_list: List[Tuple[str, str]] = []
        total_rows = 0

        last_flush_time = time.time()
        completions_since_flush = 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(
                    self._download_single, sym, start_date, end_date
                ): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                symbol, ok, rows, error = future.result()
                if ok:
                    succeeded += 1
                    total_rows += rows
                else:
                    failed_list.append((symbol, error or "Unknown"))

                completions_since_flush += 1
                now = time.time()
                if (
                    completions_since_flush >= self.MANIFEST_FLUSH_EVERY
                    or now - last_flush_time >= self.MANIFEST_FLUSH_INTERVAL_SEC
                ):
                    self.manifest.save()
                    completions_since_flush = 0
                    last_flush_time = now

        # End-of-run retry rounds
        for retry_round in range(1, self.END_RETRY_ROUNDS + 1):
            if not failed_list:
                break
            logger.info(f"Retry round {retry_round}: {len(failed_list)} symbols")
            time.sleep(self.retry_delay * retry_round)
            retry_symbols = [s for s, _ in failed_list]
            failed_list = []
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(
                        self._download_single, sym, start_date, end_date
                    ): sym
                    for sym in retry_symbols
                }
                for future in as_completed(futures):
                    symbol, ok, rows, error = future.result()
                    if ok:
                        succeeded += 1
                        total_rows += rows
                    else:
                        failed_list.append((symbol, error or "Unknown"))

                    completions_since_flush += 1
                    now = time.time()
                    if (
                        completions_since_flush >= self.MANIFEST_FLUSH_EVERY
                        or now - last_flush_time
                        >= self.MANIFEST_FLUSH_INTERVAL_SEC
                    ):
                        self.manifest.save()
                        completions_since_flush = 0
                        last_flush_time = now

        elapsed = time.time() - start_time
        self.manifest.save()

        self._emit_event(
            "pass_end",
            subdir=self._get_storage_subdir(),
            succeeded=succeeded,
            failed=len(failed_list),
            total_rows=total_rows,
        )

        return DownloadResult(
            total_symbols=len(symbols),
            succeeded=succeeded,
            failed=len(failed_list),
            total_rows=total_rows,
            elapsed_seconds=elapsed,
            failed_symbols=failed_list,
        )
