"""Bulk-pull FX-adjacent equity ETFs into equities_1min/ via AlpacaEquitiesPlugin.

Reads config/universes/fx_adjacent_equity-2026.csv; for each row honors
effective_start_date; routes through existing AlpacaEquitiesPlugin pipeline.

Uses BaseDownloader.download() which supports batch downloads with
threaded parallelism and retry logic.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.alpaca_equities import AlpacaEquitiesPlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=PROJECT_ROOT / "config" / "universes" / "fx_adjacent_equity-2026.csv",
        help="Path to universe CSV file",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="End date (ISO format YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip symbols that already exist in storage",
    )
    args = parser.parse_args()

    if not args.universe.exists():
        logger.error(f"Universe file not found: {args.universe}")
        return 1

    rows = list(csv.DictReader(args.universe.open(encoding="utf-8")))
    logger.info(f"Loading {len(rows)} ETFs from {args.universe}")

    plugin = AlpacaEquitiesPlugin()

    symbols_by_start = {}
    for row in rows:
        sym = row["symbol"]
        start = date.fromisoformat(row["effective_start_date"])
        if start not in symbols_by_start:
            symbols_by_start[start] = []
        symbols_by_start[start].append(sym)

    total_succeeded = 0
    total_failed = 0
    total_rows = 0

    for start_date in sorted(symbols_by_start.keys()):
        symbols = symbols_by_start[start_date]
        logger.info(
            f"Downloading {len(symbols)} symbols from {start_date} to {args.end}"
        )

        start_iso = start_date.isoformat()
        end_iso = args.end.isoformat()

        result = plugin.download(
            symbols=symbols,
            start_date=start_iso,
            end_date=end_iso,
            skip_existing=args.skip_existing,
        )

        logger.info(
            f"  Success: {result.succeeded}/{result.total_symbols}, "
            f"Rows: {result.total_rows:,}, "
            f"Elapsed: {result.elapsed_seconds:.1f}s"
        )
        if result.failed_symbols:
            for sym, error in result.failed_symbols:
                logger.warning(f"    FAILED {sym}: {error}")

        total_succeeded += result.succeeded
        total_failed += result.failed
        total_rows += result.total_rows

    logger.info(
        f"Total: {total_succeeded} succeeded, {total_failed} failed, "
        f"{total_rows:,} rows saved"
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
