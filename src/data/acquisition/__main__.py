"""CLI entry point: python -m src.data.acquisition"""

import argparse
import sys
from pathlib import Path

from src.data.acquisition.manager import DataAcquisitionManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Unified data acquisition CLI",
        prog="python -m src.data.acquisition",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Data source (equities, crypto, futures, news)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols (e.g., ES,NQ,CL)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with Symbol column",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip symbols with existing data",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed symbols",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status for all sources",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Number of download threads",
    )

    args = parser.parse_args()
    manager = DataAcquisitionManager()

    if args.status:
        for source in manager.list_sources():
            try:
                status = manager.get_status(source)
                logger.info(
                    f"{source}: {status['complete']} complete, "
                    f"{status['failed']} failed, "
                    f"{status['pending']} pending "
                    f"(total: {status['total']})"
                )
            except Exception:
                logger.info(f"{source}: no manifest found")
        return

    if not args.source:
        parser.error("--source is required (or use --status)")

    # Build symbol list
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.csv:
        import pandas as pd

        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        col = next(
            (c for c in df.columns if c.lower() in ("symbol", "ticker")),
            None,
        )
        if col is None:
            logger.error("CSV must have a 'Symbol' or 'Ticker' column")
            sys.exit(1)
        symbols = df[col].dropna().str.strip().tolist()

    result = manager.download(
        source=args.source,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        skip_existing=args.skip_existing,
        num_threads=args.threads,
    )

    logger.info(
        f"Result: {result.succeeded} succeeded, "
        f"{result.failed} failed, {result.total_rows:,} rows"
    )

    if result.failed_symbols:
        logger.warning("Failed symbols:")
        for sym, err in result.failed_symbols:
            logger.warning(f"  {sym}: {err}")


if __name__ == "__main__":
    main()
