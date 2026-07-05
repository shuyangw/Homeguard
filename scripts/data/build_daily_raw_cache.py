"""Precompute per-root daily-raw OHLCV series to futures/daily_raw/{root}.parquet.

For each root: ContinuousContractDataLoader().aggregate_to_daily(root, method="raw")
over the FULL available range -> daily_raw_dir()/{root}.parquet
[timestamp, open, high, low, close, volume]. Mirrors build_carry_cache.py.

Usage:
    python scripts/data/build_daily_raw_cache.py --roots GC CL ES --jobs 4
"""
from __future__ import annotations

import argparse

from src.backtesting.parallel import parallel_map
from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.data.futures.paths import daily_raw_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_one(root: str) -> str | None:
    df = ContinuousContractDataLoader().aggregate_to_daily(root, method="raw")
    if df.height == 0:
        logger.warning(f"[build_daily_raw_cache] {root}: no daily rows, skipping")
        return None
    df.write_parquet(daily_raw_dir() / f"{root}.parquet")
    logger.info(f"[build_daily_raw_cache] {root}: {df.height} rows")
    return root


def build_daily_raw_cache(roots: list[str], max_workers: int | None = None) -> list[str]:
    daily_raw_dir().mkdir(parents=True, exist_ok=True)
    written = parallel_map(_build_one, roots, max_workers=max_workers)
    return [r for r in written if r is not None]


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-root daily-raw OHLCV cache")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=None)
    args = p.parse_args()
    written = build_daily_raw_cache(args.roots, max_workers=args.jobs)
    logger.info(f"[build_daily_raw_cache] wrote {len(written)}/{len(args.roots)} roots to {daily_raw_dir()}")


if __name__ == "__main__":
    main()
