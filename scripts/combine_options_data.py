"""
Combine 1-min options data with EOD gamma + open interest.

Joins intraday data (options_1min/) with EOD data (options_eod/)
and saves to options_combined/.

Features:
    - Multi-threaded processing with work queue
    - Progress tracking with ETA
    - Resume capability (skips existing files)

Usage:
    # Combine all available data
    python scripts/combine_options_data.py

    # Combine specific symbols
    python scripts/combine_options_data.py --symbols SPY,QQQ

    # Combine specific date range
    python scripts/combine_options_data.py --start 2024-01-01 --end 2024-12-31

    # Dry run (show what would be combined)
    python scripts/combine_options_data.py --dry-run

    # Control thread count
    python scripts/combine_options_data.py --threads 8
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import logging

import polars as pl

from src.settings import get_options_data_dir

DEFAULT_NUM_THREADS = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def discover_partitions(data_dir: Path) -> List[Tuple[str, int, int]]:
    """
    Discover all (symbol, year, month) partitions in a Hive-partitioned directory.

    Returns list of (symbol, year, month) tuples.
    """
    partitions = []

    if not data_dir.exists():
        return partitions

    for root_dir in data_dir.glob("root=*"):
        symbol = root_dir.name.replace("root=", "")

        for year_dir in root_dir.glob("year=*"):
            year = int(year_dir.name.replace("year=", ""))

            for month_dir in year_dir.glob("month=*"):
                month = int(month_dir.name.replace("month=", ""))

                # Check if there's actual data
                parquet_files = list(month_dir.glob("*.parquet"))
                if parquet_files:
                    partitions.append((symbol, year, month))

    return partitions


def get_partition_path(base_dir: Path, symbol: str, year: int, month: int) -> Path:
    """Get the path for a partition."""
    return base_dir / f"root={symbol}" / f"year={year}" / f"month={month:02d}"


def load_partition(partition_dir: Path) -> Optional[pl.DataFrame]:
    """Load all parquet files in a partition directory, normalizing schemas."""
    parquet_files = list(partition_dir.glob("*.parquet"))

    if not parquet_files:
        return None

    dfs = []
    for f in parquet_files:
        try:
            df = pl.read_parquet(f)

            # Normalize numeric types to Float64 to avoid Float32/Float64 mismatches
            float_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
            if float_cols:
                df = df.with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])

            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return None

    if len(dfs) == 1:
        return dfs[0]

    # Use diagonal_relaxed to handle different column sets
    try:
        return pl.concat(dfs, how="diagonal_relaxed")
    except Exception:
        # Fallback: find common columns and concat
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        common_cols = sorted(common_cols)

        if not common_cols:
            return None

        normalized = [df.select(common_cols) for df in dfs]
        return pl.concat(normalized)


def combine_partition(
    intraday_dir: Path,
    eod_dir: Path,
    output_dir: Path,
    symbol: str,
    year: int,
    month: int,
    eod_partitions: Set[Tuple[str, int, int]],
) -> Tuple[int, int, bool]:
    """
    Combine intraday and EOD data for a single partition.

    Returns (rows_written, bytes_written, has_eod).
    """
    intraday_path = get_partition_path(intraday_dir, symbol, year, month)
    eod_path = get_partition_path(eod_dir, symbol, year, month)
    output_path = get_partition_path(output_dir, symbol, year, month)

    has_eod = (symbol, year, month) in eod_partitions

    # Load intraday data
    intraday_df = load_partition(intraday_path)
    if intraday_df is None or intraday_df.is_empty():
        return 0, 0, has_eod

    # Load EOD data (optional - may not exist yet)
    eod_df = load_partition(eod_path)

    # Drop intraday gamma/OI columns - we'll use EOD values instead
    # (intraday values are snapshots at each minute, EOD is the official end-of-day value)
    cols_to_drop = [c for c in ["gamma", "open_interest"] if c in intraday_df.columns]
    if cols_to_drop:
        intraday_df = intraday_df.drop(cols_to_drop)

    if eod_df is None or eod_df.is_empty():
        # No EOD data - add null gamma/OI columns
        combined = intraday_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("gamma_eod"),
            pl.lit(None).cast(pl.Int64).alias("open_interest_eod"),
        ])
    else:
        # Extract date from timestamp for joining
        if "timestamp" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("timestamp").str.slice(0, 10).alias("_date")
            )

        # Normalize join column types (1-min may have date/categorical, EOD has str)
        # expiration: date -> str
        if "expiration" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("expiration").cast(pl.Utf8).alias("expiration")
            )
        if "expiration" in eod_df.columns:
            eod_df = eod_df.with_columns(
                pl.col("expiration").cast(pl.Utf8).alias("expiration")
            )
        # right: categorical -> str
        if "right" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("right").cast(pl.Utf8).alias("right")
            )
        if "right" in eod_df.columns:
            eod_df = eod_df.with_columns(
                pl.col("right").cast(pl.Utf8).alias("right")
            )
        # strike: ensure same numeric type
        if "strike" in intraday_df.columns:
            intraday_df = intraday_df.with_columns(
                pl.col("strike").cast(pl.Float64).alias("strike")
            )
        if "strike" in eod_df.columns:
            eod_df = eod_df.with_columns(
                pl.col("strike").cast(pl.Float64).alias("strike")
            )

        # Prepare EOD data for join
        eod_cols = ["expiration", "strike", "right"]
        if "date" in eod_df.columns:
            eod_df = eod_df.rename({"date": "_date"})
            eod_cols = ["_date"] + eod_cols

        # Rename EOD gamma/OI columns to distinguish them
        if "gamma" in eod_df.columns:
            eod_df = eod_df.rename({"gamma": "gamma_eod"})
        if "open_interest" in eod_df.columns:
            eod_df = eod_df.rename({"open_interest": "open_interest_eod"})

        # Select only needed columns from EOD
        eod_select = eod_cols.copy()
        if "gamma_eod" in eod_df.columns:
            eod_select.append("gamma_eod")
        if "open_interest_eod" in eod_df.columns:
            eod_select.append("open_interest_eod")

        eod_df = eod_df.select([c for c in eod_select if c in eod_df.columns])

        # Deduplicate EOD data (one row per contract per day)
        eod_df = eod_df.unique(subset=[c for c in eod_cols if c in eod_df.columns])

        # Join
        join_cols = [c for c in eod_cols if c in intraday_df.columns and c in eod_df.columns]

        if join_cols:
            combined = intraday_df.join(eod_df, on=join_cols, how="left")
        else:
            combined = intraday_df.with_columns([
                pl.lit(None).cast(pl.Float64).alias("gamma_eod"),
                pl.lit(None).cast(pl.Int64).alias("open_interest_eod"),
            ])

        # Drop temporary date column
        if "_date" in combined.columns:
            combined = combined.drop("_date")

    # Ensure gamma_eod and open_interest_eod columns exist
    if "gamma_eod" not in combined.columns:
        combined = combined.with_columns(pl.lit(None).cast(pl.Float64).alias("gamma_eod"))
    if "open_interest_eod" not in combined.columns:
        combined = combined.with_columns(pl.lit(None).cast(pl.Int64).alias("open_interest_eod"))

    # Save atomically (write to temp, then move)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "data.parquet"
    temp_file = output_path / "data.parquet.tmp"

    # Write to temp file first
    combined.write_parquet(temp_file, compression="zstd")

    # Atomic move (replace if exists)
    temp_file.replace(output_file)

    rows = len(combined)
    bytes_written = output_file.stat().st_size

    return rows, bytes_written, has_eod


class OptionsCombiner:
    """Multi-threaded options data combiner."""

    def __init__(
        self,
        intraday_dir: Path,
        eod_dir: Path,
        output_dir: Path,
        eod_partitions: Set[Tuple[str, int, int]],
        num_threads: int = DEFAULT_NUM_THREADS,
        overwrite: bool = False,
    ):
        self.intraday_dir = intraday_dir
        self.eod_dir = eod_dir
        self.output_dir = output_dir
        self.eod_partitions = eod_partitions
        self.num_threads = num_threads
        self.overwrite = overwrite

        # Work queue
        self._work_queue: queue.Queue = queue.Queue()
        self._shutdown_requested = False

        # Stats tracking
        self._stats_lock = threading.Lock()
        self._completed = 0
        self._skipped = 0
        self._failed = 0
        self._total_rows = 0
        self._total_bytes = 0

        # In-flight tracking
        self._in_flight: Set[str] = set()
        self._in_flight_lock = threading.Lock()

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

    def _worker(self, worker_id: int) -> None:
        """Worker thread that processes partitions from the queue."""
        tid = f"W{worker_id}"

        while not self._shutdown_requested:
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                if self._work_queue.empty():
                    break
                continue

            if item is None:
                self._work_queue.task_done()
                break

            symbol, year, month = item
            key = f"{symbol}_{year}_{month:02d}"

            # Duplicate prevention
            with self._in_flight_lock:
                if key in self._in_flight:
                    self._work_queue.task_done()
                    continue
                self._in_flight.add(key)

            try:
                # Check if output exists
                output_path = get_partition_path(self.output_dir, symbol, year, month) / "data.parquet"
                if output_path.exists() and not self.overwrite:
                    with self._stats_lock:
                        self._skipped += 1
                    continue  # finally block will call task_done()

                # Process partition
                rows, bytes_written, has_eod = combine_partition(
                    self.intraday_dir,
                    self.eod_dir,
                    self.output_dir,
                    symbol,
                    year,
                    month,
                    self.eod_partitions,
                )

                if rows > 0:
                    with self._stats_lock:
                        self._completed += 1
                        self._total_rows += rows
                        self._total_bytes += bytes_written

                    eod_status = "+EOD" if has_eod else "no EOD"
                    logger.info(
                        f"[{tid}] [+] {symbol} {year}-{month:02d}: "
                        f"{rows:,} rows, {bytes_written / 1024 / 1024:.1f} MB ({eod_status})"
                    )
                else:
                    with self._stats_lock:
                        self._skipped += 1

            except Exception as e:
                logger.error(f"[{tid}] [-] {symbol} {year}-{month:02d}: {e}")
                with self._stats_lock:
                    self._failed += 1

            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(key)
                self._work_queue.task_done()

    def combine_all(self, partitions: List[Tuple[str, int, int]]) -> Dict[str, int]:
        """Combine all partitions using work queue."""
        if not partitions:
            logger.info("No partitions to combine")
            return {"completed": 0, "skipped": 0, "failed": 0, "rows": 0, "bytes": 0}

        # Populate queue (round-robin by symbol for better distribution)
        by_symbol = defaultdict(list)
        for p in partitions:
            by_symbol[p[0]].append(p)

        # Sort each symbol's partitions by date (newest first)
        for sym in by_symbol:
            by_symbol[sym].sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Interleave symbols
        symbol_iters = {s: iter(ps) for s, ps in by_symbol.items()}
        symbols = list(symbol_iters.keys())
        total_added = 0

        while symbol_iters:
            for sym in list(symbols):
                try:
                    p = next(symbol_iters[sym])
                    self._work_queue.put(p)
                    total_added += 1
                except StopIteration:
                    del symbol_iters[sym]
                    symbols.remove(sym)

        logger.info(f"Work queue populated: {total_added} partitions")

        # Start workers
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            threads.append(t)
            t.start()

        logger.info(f"Started {len(threads)} workers")

        # Monitor progress
        start_time = time.time()
        while not self._work_queue.empty() and not self._shutdown_requested:
            time.sleep(2.0)
            remaining = self._work_queue.qsize()
            if remaining > 0:
                with self._stats_lock:
                    processed = self._completed + self._skipped + self._failed
                elapsed = time.time() - start_time
                if processed > 0:
                    rate = processed / elapsed
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_str = self._format_time(eta_seconds)
                    elapsed_str = self._format_time(elapsed)
                    pct = (processed / total_added) * 100
                    logger.info(
                        f"[PROGRESS] {processed}/{total_added} ({pct:.1f}%) | "
                        f"Remaining: {remaining} | Elapsed: {elapsed_str} | ETA: {eta_str}"
                    )

        # Wait for threads
        for t in threads:
            t.join(timeout=30.0)

        return {
            "completed": self._completed,
            "skipped": self._skipped,
            "failed": self._failed,
            "rows": self._total_rows,
            "bytes": self._total_bytes,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Combine 1-min options data with EOD gamma + open interest"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to combine (default: all)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: all)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be combined without writing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing combined files",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f"Number of worker threads (default: {DEFAULT_NUM_THREADS})",
    )

    args = parser.parse_args()

    # Parse arguments
    symbols_filter = None
    if args.symbols:
        symbols_filter = [s.strip().upper() for s in args.symbols.split(",")]

    start_filter = None
    end_filter = None
    if args.start:
        start_filter = datetime.strptime(args.start, "%Y-%m-%d")
    if args.end:
        end_filter = datetime.strptime(args.end, "%Y-%m-%d")

    # Set up directories
    base_dir = get_options_data_dir()
    intraday_dir = base_dir / "options_1min"
    eod_dir = base_dir / "options_eod"
    output_dir = base_dir / "options_combined"

    logger.info("=" * 60)
    logger.info("OPTIONS DATA COMBINER")
    logger.info("=" * 60)
    logger.info(f"Intraday: {intraday_dir}")
    logger.info(f"EOD:      {eod_dir}")
    logger.info(f"Output:   {output_dir}")
    logger.info(f"Threads:  {args.threads}")
    logger.info("=" * 60)

    # Discover partitions
    intraday_partitions = set(discover_partitions(intraday_dir))
    eod_partitions = set(discover_partitions(eod_dir))

    logger.info(f"Found {len(intraday_partitions)} intraday partitions")
    logger.info(f"Found {len(eod_partitions)} EOD partitions")

    # Use intraday as the base (EOD may not exist for all)
    partitions = sorted(intraday_partitions)

    # Apply filters
    if symbols_filter:
        partitions = [(s, y, m) for s, y, m in partitions if s in symbols_filter]

    if start_filter:
        partitions = [
            (s, y, m) for s, y, m in partitions
            if datetime(y, m, 1) >= start_filter.replace(day=1)
        ]

    if end_filter:
        partitions = [
            (s, y, m) for s, y, m in partitions
            if datetime(y, m, 1) <= end_filter.replace(day=1)
        ]

    logger.info(f"Partitions to process: {len(partitions)}")

    if args.dry_run:
        logger.info("\n[DRY RUN] Would combine:")
        for symbol, year, month in partitions[:20]:
            has_eod = (symbol, year, month) in eod_partitions
            eod_status = "[+] has EOD" if has_eod else "[-] no EOD"
            print(f"  {symbol} {year}-{month:02d} {eod_status}")
        if len(partitions) > 20:
            print(f"  ... and {len(partitions) - 20} more")
        return

    # Cleanup stale temp files from interrupted runs
    stale_temps = list(output_dir.rglob("*.parquet.tmp"))
    if stale_temps:
        logger.info(f"Cleaning up {len(stale_temps)} stale temp files from interrupted runs...")
        for tmp in stale_temps:
            try:
                tmp.unlink()
            except Exception:
                pass

    # Run combiner
    start_time = time.time()
    combiner = OptionsCombiner(
        intraday_dir=intraday_dir,
        eod_dir=eod_dir,
        output_dir=output_dir,
        eod_partitions=eod_partitions,
        num_threads=args.threads,
        overwrite=args.overwrite,
    )

    stats = combiner.combine_all(partitions)
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"Processed: {stats['completed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    logger.info(f"Total rows: {stats['rows']:,}")
    logger.info(f"Total size: {stats['bytes'] / 1024 / 1024 / 1024:.2f} GB")
    logger.info(f"Time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
