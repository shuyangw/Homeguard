"""Bulk-pull CFTC TFF for all FX-relevant futures contracts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.cftc_cot import CFTCCOTPlugin
from src.utils.logger import get_logger
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    plugin = CFTCCOTPlugin()
    summary = plugin.fetch_all_instruments(
        start_year=args.start_year, end_year=args.end_year,
        skip_existing=not args.no_skip_existing,
    )
    logger.info(f"=== COT Summary ===")
    for inst, status in summary.items():
        logger.info(f"  {inst}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
