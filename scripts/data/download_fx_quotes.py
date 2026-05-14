"""Bulk-acquire FX quote/BBO data from Massive flat-file S3 bucket."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.massive_fx_quotes_flat import (
    TargetPair, download_pairs,
)
from src.utils.logger import get_logger
logger = get_logger(__name__)


UNIVERSE_PATH = PROJECT_ROOT / "config" / "universes" / "fx_quotes_tier1-2026.csv"
DEFAULT_START = date(2010, 1, 1)


def load_universe(path: Path) -> list[TargetPair]:
    out: list[TargetPair] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eff = (
                date.fromisoformat(row["effective_start_date"])
                if row.get("effective_start_date") else DEFAULT_START
            )
            out.append(TargetPair(
                hg_symbol=row["symbol"],
                massive_ticker=row["massive_ticker"],
                effective_start=eff,
            ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=UNIVERSE_PATH)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel S3 downloads (quote files are LARGE; keep low)")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    pairs = load_universe(args.universe)
    if args.symbol:
        pairs = [p for p in pairs if p.hg_symbol == args.symbol]
    if not pairs:
        logger.error("no pairs match filters")
        return 1

    logger.info(f"=== FX quotes bulk download ===")
    logger.info(f"  universe: {args.universe}")
    logger.info(f"  pairs: {[p.hg_symbol for p in pairs]}")
    logger.info(f"  range: {args.start} -> {args.end}")

    t0 = time.time()
    summary = download_pairs(
        pairs, args.start, args.end,
        concurrency=args.concurrency,
        skip_existing=not args.no_skip_existing,
    )
    elapsed = time.time() - t0

    logger.info(f"=== Summary (elapsed {elapsed:.1f}s = {elapsed/60:.1f} min) ===")
    logger.info(f"  days attempted: {summary['total_days_attempted']:,}")
    logger.info(f"  days present: {summary['total_days_present']:,}")
    logger.info(f"  days missing: {summary['total_days_missing']:,}")
    logger.info(f"  months written: {summary['months_written']}")
    logger.info(f"  months skipped (existing): {summary['months_skipped_existing']}")
    logger.info("  rows per symbol:")
    for sym, n in summary["rows_per_symbol"].items():
        logger.info(f"    {sym}: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
