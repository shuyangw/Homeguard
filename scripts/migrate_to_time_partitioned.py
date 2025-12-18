"""
Migrate hive-partitioned data from symbol-first to time-first structure.

Current structure (symbol-first):
    equities_1min/symbol=AAPL/year=2024/month=1/data.parquet

New structure (time-first):
    equities_1min_by_date/2024-01-02.parquet  (all symbols for this day)

This makes walk-forward backtests ~100x more efficient for I/O.

Usage:
    python scripts/migrate_to_time_partitioned.py --dry-run
    python scripts/migrate_to_time_partitioned.py
    python scripts/migrate_to_time_partitioned.py --resume
    python scripts/migrate_to_time_partitioned.py --workers 8  # Parallel with 8 workers
"""

import argparse
import json
import polars as pl
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple
import sys
import os
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import get_local_storage_dir
from src.utils.logger import logger


# Configuration
SOURCE_SUBDIR = "equities_1min"
TARGET_SUBDIR = "equities_1min_by_date"
PROGRESS_FILE = "migration_progress.json"
BATCH_SIZE = 50  # Symbols to process at once (memory management)


def get_all_symbols(source_path: Path) -> List[str]:
    """Get all available symbols from source directory."""
    symbols = []
    for d in source_path.iterdir():
        if d.is_dir() and d.name.startswith("symbol="):
            symbol = d.name.replace("symbol=", "")
            symbols.append(symbol)
    return sorted(symbols)


def get_all_trading_days(source_path: Path, symbols: List[str]) -> Set[str]:
    """Scan all symbols to find all unique trading days."""
    logger.info("Scanning for all trading days (this may take a while)...")

    all_dates = set()

    for i, symbol in enumerate(symbols):
        if i % 100 == 0:
            logger.info(f"  Scanned {i}/{len(symbols)} symbols, found {len(all_dates)} unique dates")

        symbol_path = source_path / f"symbol={symbol}"
        if not symbol_path.exists():
            continue

        for year_dir in symbol_path.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue

                parquet_file = month_dir / "data.parquet"
                if not parquet_file.exists():
                    continue

                try:
                    # Use lazy scan to just get unique dates
                    lf = pl.scan_parquet(parquet_file)
                    dates = (
                        lf.select(pl.col("timestamp").dt.date().alias("date"))
                        .unique()
                        .collect()
                    )
                    for date in dates["date"].to_list():
                        all_dates.add(date.strftime("%Y-%m-%d"))
                except Exception as e:
                    logger.warning(f"Error scanning {parquet_file}: {e}")

    logger.info(f"Found {len(all_dates)} unique trading days")
    return all_dates


def load_progress(target_path: Path) -> Dict:
    """Load migration progress from file."""
    progress_file = target_path / PROGRESS_FILE
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"completed_dates": [], "failed_dates": []}


def save_progress(target_path: Path, progress: Dict) -> None:
    """Save migration progress to file."""
    progress_file = target_path / PROGRESS_FILE
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def load_symbol_data_for_date(
    source_path: Path,
    symbol: str,
    target_date: str
) -> pl.DataFrame:
    """Load data for a single symbol on a specific date."""
    # Canonical schema (from CLAUDE.md)
    CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]

    # Parse date to find correct partition
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    year = dt.year
    month = dt.month

    parquet_file = (
        source_path / f"symbol={symbol}" / f"year={year}" / f"month={month}" / "data.parquet"
    )

    if not parquet_file.exists():
        return pl.DataFrame()

    try:
        # Load and filter to specific date
        df = pl.read_parquet(parquet_file)

        # Filter to target date
        df = df.filter(
            pl.col("timestamp").dt.date() == pl.lit(target_date).str.to_date()
        )

        if df.is_empty():
            return pl.DataFrame()

        # Enforce canonical schema - select only expected columns
        available_cols = set(df.columns)
        select_cols = [c for c in CANONICAL_COLUMNS if c in available_cols]

        # Add missing columns as null
        df = df.select(select_cols)
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # Reorder to canonical order
        df = df.select(CANONICAL_COLUMNS)

        # Add symbol column
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        return df

    except Exception as e:
        logger.warning(f"Error loading {symbol} for {target_date}: {e}")
        return pl.DataFrame()


def migrate_single_date(
    source_path: Path,
    target_path: Path,
    symbols: List[str],
    target_date: str,
    dry_run: bool = False
) -> Tuple[bool, int, int]:
    """
    Migrate all symbol data for a single date into one parquet file.

    Returns:
        (success, rows_written, symbols_included)
    """
    output_file = target_path / f"{target_date}.parquet"

    if output_file.exists():
        # Already migrated
        return True, 0, 0

    # Collect data from all symbols for this date
    all_data = []
    symbols_with_data = 0

    for symbol in symbols:
        df = load_symbol_data_for_date(source_path, symbol, target_date)
        if not df.is_empty():
            all_data.append(df)
            symbols_with_data += 1

    if not all_data:
        # No data for this date (weekend/holiday)
        return True, 0, 0

    # Concatenate all symbols
    combined = pl.concat(all_data)

    # Sort by timestamp, then symbol
    combined = combined.sort(["timestamp", "symbol"])

    total_rows = len(combined)

    if dry_run:
        logger.info(f"  [DRY RUN] Would write {target_date}: {total_rows:,} rows, {symbols_with_data} symbols")
        return True, total_rows, symbols_with_data

    # Write combined file
    try:
        combined.write_parquet(
            output_file,
            compression="zstd",
            compression_level=3
        )
        return True, total_rows, symbols_with_data
    except Exception as e:
        logger.error(f"Failed to write {output_file}: {e}")
        return False, 0, 0


def migrate_date_worker(args: Tuple) -> Tuple[str, bool, int, int]:
    """Worker function for parallel migration.

    Args:
        args: Tuple of (source_path, target_path, symbols, date, dry_run)

    Returns:
        (date, success, rows, symbols_count)
    """
    source_path, target_path, symbols, date, dry_run = args
    success, rows, syms = migrate_single_date(source_path, target_path, symbols, date, dry_run)
    return date, success, rows, syms


def migrate_batch(
    source_path: Path,
    target_path: Path,
    symbols: List[str],
    dates: List[str],
    progress: Dict,
    dry_run: bool = False,
    num_workers: int = 1
) -> None:
    """Migrate a batch of dates, optionally in parallel."""
    completed = set(progress["completed_dates"])
    failed = set(progress["failed_dates"])

    # Filter to dates not yet processed
    remaining = [d for d in dates if d not in completed]

    logger.info(f"Processing {len(remaining)} dates ({len(completed)} already completed)")
    if num_workers > 1:
        logger.info(f"Using {num_workers} parallel workers")

    total_rows = 0
    total_symbols = 0
    progress_lock = threading.Lock()
    counter = [0]  # Mutable counter for progress tracking

    if num_workers == 1:
        # Sequential processing (original behavior)
        for i, date in enumerate(remaining):
            success, rows, syms = migrate_single_date(
                source_path, target_path, symbols, date, dry_run
            )

            if success:
                if rows > 0:
                    total_rows += rows
                    total_symbols += syms
                    logger.info(f"  [{i+1}/{len(remaining)}] {date}: {rows:,} rows, {syms} symbols")
                completed.add(date)
            else:
                failed.add(date)
                logger.error(f"  [{i+1}/{len(remaining)}] {date}: FAILED")

            # Save progress periodically
            if (i + 1) % 10 == 0 and not dry_run:
                progress["completed_dates"] = sorted(list(completed))
                progress["failed_dates"] = sorted(list(failed))
                save_progress(target_path, progress)
    else:
        # Parallel processing
        work_items = [
            (source_path, target_path, symbols, date, dry_run)
            for date in remaining
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(migrate_date_worker, item): item[3] for item in work_items}

            for future in as_completed(futures):
                date, success, rows, syms = future.result()

                with progress_lock:
                    counter[0] += 1
                    current = counter[0]

                    if success:
                        if rows > 0:
                            total_rows += rows
                            total_symbols += syms
                            logger.info(f"  [{current}/{len(remaining)}] {date}: {rows:,} rows, {syms} symbols")
                        completed.add(date)
                    else:
                        failed.add(date)
                        logger.error(f"  [{current}/{len(remaining)}] {date}: FAILED")

                    # Save progress periodically
                    if current % 50 == 0 and not dry_run:
                        progress["completed_dates"] = sorted(list(completed))
                        progress["failed_dates"] = sorted(list(failed))
                        save_progress(target_path, progress)

    # Final progress save
    if not dry_run:
        progress["completed_dates"] = sorted(list(completed))
        progress["failed_dates"] = sorted(list(failed))
        save_progress(target_path, progress)

    logger.success(f"Migration complete: {total_rows:,} total rows across {len(completed)} dates")


def estimate_size(source_path: Path, symbols: List[str]) -> None:
    """Estimate total data size and migration time."""
    logger.info("Estimating migration scope...")

    total_files = 0
    total_size_mb = 0

    for symbol in symbols[:50]:  # Sample first 50 symbols
        symbol_path = source_path / f"symbol={symbol}"
        if not symbol_path.exists():
            continue

        for pq_file in symbol_path.rglob("*.parquet"):
            total_files += 1
            total_size_mb += pq_file.stat().st_size / (1024 * 1024)

    # Extrapolate
    scale = len(symbols) / 50
    est_files = int(total_files * scale)
    est_size_gb = (total_size_mb * scale) / 1024

    logger.info(f"Estimated source: {est_files:,} files, {est_size_gb:.1f} GB")
    logger.info(f"Estimated output: ~2,000 files (one per trading day)")


def main():
    parser = argparse.ArgumentParser(description="Migrate to time-partitioned data structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--estimate", action="store_true", help="Only estimate size, don't migrate")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    # Get paths
    base_path = get_local_storage_dir()
    source_path = base_path / SOURCE_SUBDIR
    target_path = base_path / TARGET_SUBDIR

    logger.info(f"Source: {source_path}")
    logger.info(f"Target: {target_path}")

    if not source_path.exists():
        logger.error(f"Source path does not exist: {source_path}")
        return 1

    # Get all symbols
    symbols = get_all_symbols(source_path)
    logger.info(f"Found {len(symbols)} symbols")

    if args.estimate:
        estimate_size(source_path, symbols)
        return 0

    # Create target directory
    if not args.dry_run:
        target_path.mkdir(parents=True, exist_ok=True)

    # Load or initialize progress
    if args.resume:
        progress = load_progress(target_path)
        logger.info(f"Resuming: {len(progress['completed_dates'])} dates already completed")
    else:
        progress = {"completed_dates": [], "failed_dates": []}

    # Get all trading days
    all_dates = get_all_trading_days(source_path, symbols)
    sorted_dates = sorted(list(all_dates))

    # Migrate
    migrate_batch(source_path, target_path, symbols, sorted_dates, progress, args.dry_run, args.workers)

    return 0


if __name__ == "__main__":
    sys.exit(main())
