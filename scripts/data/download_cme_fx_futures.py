"""Bulk-pull CME FX futures per config/universes/cme_fx_futures-2026.csv.

Tier 1 contracts: OHLCV-1m AND MBP-1
Tier 2 contracts: OHLCV-1m only (E-micros)
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.databento_futures import DatabentoFuturesPlugin
from src.utils.logger import get_logger
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path,
                        default=PROJECT_ROOT / "config" / "universes" / "cme_fx_futures-2026.csv")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--schemas", nargs="+", default=["ohlcv-1m", "mbp-1"],
                        help="Subset: ohlcv-1m, mbp-1, or both")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.universe.open(encoding="utf-8")))
    tier1_symbols = [r["symbol"] for r in rows if int(r["tier"]) == 1]
    tier2_symbols = [r["symbol"] for r in rows if int(r["tier"]) == 2]
    earliest_start = min(date.fromisoformat(r["effective_start_date"]) for r in rows)

    logger.info(f"Tier 1 symbols ({len(tier1_symbols)}): ohlcv-1m + mbp-1 (if in schemas)")
    logger.info(f"Tier 2 symbols ({len(tier2_symbols)}): ohlcv-1m only")

    for schema in args.schemas:
        # All tier 1+2 get ohlcv-1m; only tier 1 gets mbp-1
        symbols = tier1_symbols if schema == "mbp-1" else (tier1_symbols + tier2_symbols)
        if not symbols:
            logger.info(f"  no symbols for schema={schema}; skipping")
            continue
        logger.info(f"  schema={schema}: {len(symbols)} symbols, {earliest_start} -> {args.end}")
        plugin = DatabentoFuturesPlugin(schema=schema)
        # Per-symbol start dates would be cleaner, but BaseDownloader uses one
        # start for all. Use earliest_start; Databento returns nothing for pre-listing
        # dates and the plugin handles that gracefully (empty df).
        try:
            result = plugin.download(
                symbols, start_date=earliest_start.isoformat(),
                end_date=args.end.isoformat(), skip_existing=True,
            )
            logger.info(f"  {schema} done: {result.succeeded}/{result.total_symbols} symbols, {result.total_rows:,} rows")
        except Exception as e:
            logger.error(f"  {schema} FAILED: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
