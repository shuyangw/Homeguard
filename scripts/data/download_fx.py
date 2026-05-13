"""Bulk-acquire FX pairs from the Massive flat-file S3 bucket.

Reads the universe CSV at config/universes/fx-2026.csv, downloads each pair's
daily aggregates from 2010-01-01 (or per-pair effective_start) through today,
writes per-symbol-per-month parquet under fx_1min/.

Usage:
    python scripts/data/download_fx.py                         # all 7 new pairs
    python scripts/data/download_fx.py --tier 1                # only Tier 1
    python scripts/data/download_fx.py --symbol USDNOK         # single symbol
    python scripts/data/download_fx.py --dry-run               # plan only

Uses MASSIVE_S3_* credentials from .env. See plan file
C:\\Users\\qwqw1\\.claude\\plans\\approved-continue-on-for-encapsulated-hopcroft.md
for context.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.massive_fx_flat import (
    TargetPair, download_pairs,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


UNIVERSE_PATH = PROJECT_ROOT / "config" / "universes" / "fx-2026.csv"
DEFAULT_START = date(2010, 1, 1)


def load_universe(path: Path) -> list[TargetPair]:
    """Parse fx-2026.csv into TargetPair instances."""
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
    parser.add_argument("--universe", type=Path, default=UNIVERSE_PATH,
                        help="Path to universe CSV (default: config/universes/fx-2026.csv)")
    parser.add_argument("--tier", type=int, default=None,
                        help="Restrict to one tier (1 or 2)")
    parser.add_argument("--symbol", default=None,
                        help="Restrict to one symbol (e.g. USDNOK)")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START,
                        help="Start date (YYYY-MM-DD; default 2010-01-01)")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(),
                        help="End date (YYYY-MM-DD; default today)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel S3 downloads per month (default 8)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Overwrite existing monthly parquet (default: skip)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit")
    args = parser.parse_args()

    pairs = load_universe(args.universe)
    # Filter
    if args.tier is not None:
        # Need to re-parse to get tier (TargetPair doesn't carry it). Re-read.
        with args.universe.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            wanted_syms = {r["symbol"] for r in reader if int(r["tier"]) == args.tier}
        pairs = [p for p in pairs if p.hg_symbol in wanted_syms]
    if args.symbol:
        pairs = [p for p in pairs if p.hg_symbol == args.symbol]

    if not pairs:
        logger.error("no pairs match filters")
        return 1

    logger.info(f"=== FX bulk download ===")
    logger.info(f"universe: {args.universe}")
    logger.info(f"date range: {args.start} -> {args.end} ({(args.end - args.start).days + 1} days)")
    logger.info(f"target pairs ({len(pairs)}):")
    for p in pairs:
        logger.info(f"  {p.hg_symbol:<10} ticker={p.massive_ticker:<14} eff_start={p.effective_start}")

    if args.dry_run:
        logger.info("DRY RUN: no downloads performed")
        return 0

    t0 = time.time()
    summary = download_pairs(
        pairs, args.start, args.end,
        concurrency=args.concurrency,
        skip_existing=not args.no_skip_existing,
    )
    elapsed = time.time() - t0

    logger.info("=== Summary ===")
    logger.info(f"elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"days attempted: {summary['total_days_attempted']:,}")
    logger.info(f"days present: {summary['total_days_present']:,}")
    logger.info(f"days missing (weekends/pre-archive): {summary['total_days_missing']:,}")
    logger.info(f"months written: {summary['months_written']:,}")
    logger.info(f"months skipped (existing): {summary['months_skipped_existing']:,}")
    logger.info("rows per symbol:")
    for sym, n in summary["rows_per_symbol"].items():
        logger.info(f"  {sym:<10} {n:>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
