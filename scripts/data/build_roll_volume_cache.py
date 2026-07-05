"""Precompute per-(root, year) daily symbol-volume aggregates used by roll detection.

For each root and year in [start_year, end_year]:
    ContinuousContractDataLoader()._year_daily_symbol_volume(root, year)
writes futures/roll_volume/{root}/{year}.parquet as a side effect. This
eliminates the second per-contract 1-min scan that `detect_roll_dates` /
`_active_contract_per_day` would otherwise trigger during a walk-forward.
Mirrors build_daily_raw_cache.py.

Usage:
    python scripts/data/build_roll_volume_cache.py --roots GC CL ES --jobs 4
"""
from __future__ import annotations

import argparse

from src.backtesting.parallel import parallel_map
from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.data.futures.paths import roll_volume_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_one(args: tuple[str, int, int]) -> str | None:
    root, start_year, end_year = args
    ld = ContinuousContractDataLoader()
    total_rows = 0
    for year in range(start_year, end_year + 1):
        daily = ld._year_daily_symbol_volume(root, year)
        total_rows += daily.height
    if total_rows == 0:
        logger.warning(f"[build_roll_volume_cache] {root}: no daily rows across {start_year}-{end_year}, skipping")
        return None
    logger.info(f"[build_roll_volume_cache] {root}: {total_rows} rows across {start_year}-{end_year}")
    return root


def build_roll_volume_cache(
    roots: list[str], start_year: int = 2010, end_year: int = 2026, max_workers: int | None = None,
) -> list[str]:
    roll_volume_dir().mkdir(parents=True, exist_ok=True)
    tasks = [(root, start_year, end_year) for root in roots]
    written = parallel_map(_build_one, tasks, max_workers=max_workers)
    return [r for r in written if r is not None]


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-(root, year) daily symbol-volume cache for roll detection")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--jobs", type=int, default=None)
    args = p.parse_args()
    written = build_roll_volume_cache(
        args.roots, start_year=args.start_year, end_year=args.end_year, max_workers=args.jobs,
    )
    logger.info(f"[build_roll_volume_cache] wrote {len(written)}/{len(args.roots)} roots to {roll_volume_dir()}")


if __name__ == "__main__":
    main()
