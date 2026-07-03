"""Precompute per-root carry series to futures/carry/{root}.parquet.

For each root: CarryCalculator.compute_history(root, asset_class_for(root), start, end)
-> carry_dir()/{root}.parquet [date, carry]. Mirrors build_roll_calendar.py.

Usage:
    python scripts/data/build_carry_cache.py --roots GC CL ES --start 2010-06-07 --end 2026-02-20 --jobs 4
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

from src.backtesting.parallel import parallel_map
from src.data.carry_calculator import CarryCalculator
from src.data.futures.asset_class import asset_class_for
from src.data.futures.paths import carry_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_one(spec: dict) -> str | None:
    root = spec["root"]
    start = spec["start"]
    end = spec["end"]
    ac = asset_class_for(root)
    hist = CarryCalculator().compute_history(root, ac, start, end)
    if hist.height == 0:
        logger.warning(f"[build_carry_cache] {root}: no carry rows, skipping")
        return None
    hist.write_parquet(carry_dir() / f"{root}.parquet")
    logger.info(f"[build_carry_cache] {root} ({ac}): {hist.height} rows")
    return root


def build_carry_cache(roots: list[str], start: date, end: date, max_workers: int | None = None) -> list[str]:
    carry_dir().mkdir(parents=True, exist_ok=True)
    specs = [{"root": r, "start": start, "end": end} for r in roots]
    written = parallel_map(_build_one, specs, max_workers=max_workers)
    return [r for r in written if r is not None]


def _as_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-root carry cache")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--jobs", type=int, default=None)
    args = p.parse_args()
    written = build_carry_cache(args.roots, _as_date(args.start), _as_date(args.end), max_workers=args.jobs)
    logger.info(f"[build_carry_cache] wrote {len(written)}/{len(args.roots)} roots to {carry_dir()}")


if __name__ == "__main__":
    main()
