#!/usr/bin/env python
"""
Download all crypto USD pairs from Alpaca.

Out-of-the-box script - just run: python scripts/download_crypto.py

All 18 USD pairs are hardcoded (no stablecoins). Data is saved to the
local_storage_dir specified in settings.ini in hive-partitioned format.

Uses the unified data acquisition module (src.data.acquisition).

Usage:
    python scripts/download_crypto.py                    # Download all 18 pairs, minute data
    python scripts/download_crypto.py --skip-existing    # Skip already downloaded
    python scripts/download_crypto.py --start 2022-01-01 # Custom start date
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.acquisition import DataAcquisitionManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# All 18 USD pairs (no stablecoins) - runs without any arguments
DEFAULT_CRYPTO_PAIRS = [
    "AAVE/USD",
    "AVAX/USD",
    "BAT/USD",
    "BCH/USD",
    "BTC/USD",
    "CRV/USD",
    "DOGE/USD",
    "DOT/USD",
    "ETH/USD",
    "GRT/USD",
    "LINK/USD",
    "LTC/USD",
    "MKR/USD",
    "SHIB/USD",
    "SUSHI/USD",
    "UNI/USD",
    "XTZ/USD",
    "YFI/USD",
]


def main():
    parser = argparse.ArgumentParser(
        description='Download crypto USD pairs from Alpaca API.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_crypto.py                    # Download all pairs
    python scripts/download_crypto.py --skip-existing    # Skip already downloaded
    python scripts/download_crypto.py --start 2022-01-01 # Custom start date
        """
    )

    parser.add_argument(
        '--skip-existing', '-s',
        action='store_true',
        help='Skip symbols that already have data'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date in YYYY-MM-DD format (default: 2020-01-01)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (default: today)'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=6,
        help='Number of download threads (default: 6)'
    )

    args = parser.parse_args()

    # Download using the unified acquisition module
    manager = DataAcquisitionManager()

    logger.info(f"Downloading {len(DEFAULT_CRYPTO_PAIRS)} crypto USD pairs...")
    logger.info(f"Pairs: {', '.join(DEFAULT_CRYPTO_PAIRS)}")

    result = manager.download(
        source="crypto",
        symbols=DEFAULT_CRYPTO_PAIRS,
        start_date=args.start,
        end_date=args.end,
        skip_existing=args.skip_existing,
        num_threads=args.threads,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Succeeded: {result.succeeded}/{result.total_symbols}")
    logger.info(f"Failed: {result.failed}")
    logger.info(f"Total rows: {result.total_rows:,}")
    logger.info(f"Success rate: {result.success_rate:.1f}%")
    logger.info(f"Time elapsed: {result.elapsed_seconds:.1f}s")

    if result.failed > 0:
        logger.info("Failed symbols:")
        for sym, err in result.failed_symbols:
            logger.info(f"  {sym}: {err}")
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
