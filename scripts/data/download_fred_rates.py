"""Bulk-pull FRED series listed in config/universes/fred_series-2026.csv."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.fred_rates import FREDRatesPlugin
from src.utils.logger import get_logger
logger = get_logger(__name__)


DEFAULT_START = date(1995, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path,
                        default=PROJECT_ROOT / "config" / "universes" / "fred_series-2026.csv")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    plugin = FREDRatesPlugin()
    skip_existing = not args.no_skip_existing

    rows = list(csv.DictReader(args.universe.open(encoding="utf-8")))
    logger.info(f"=== FRED rates bulk pull ===")
    logger.info(f"  {len(rows)} series, {args.start} -> {args.end}")

    summary = {"fetched": 0, "skipped": 0, "errored": 0}
    for r in rows:
        result = plugin.fetch_series(r["series_id"], args.start, args.end,
                                     skip_existing=skip_existing)
        if result.get("error"):
            summary["errored"] += 1
        elif result.get("skipped"):
            summary["skipped"] += 1
        else:
            summary["fetched"] += 1

    logger.info(f"=== Summary: fetched={summary['fetched']}, "
                f"skipped={summary['skipped']}, errored={summary['errored']} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
