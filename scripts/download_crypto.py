#!/usr/bin/env python
"""
Download all crypto USD pairs from Alpaca.

Out-of-the-box script - just run: python scripts/download_crypto.py

All 18 USD pairs are hardcoded (no stablecoins). Data is saved to the
local_storage_dir specified in settings.ini in hive-partitioned format.

Usage:
    python scripts/download_crypto.py                    # Download all 18 pairs, minute data
    python scripts/download_crypto.py --skip-existing    # Skip already downloaded
    python scripts/download_crypto.py --timeframe hour   # Hourly data instead
    python scripts/download_crypto.py --timeframe day    # Daily data instead
    python scripts/download_crypto.py --start 2022-01-01 # Custom start date
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.crypto_downloader import CryptoDownloader, Timeframe

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


def parse_timeframe(value: str) -> Timeframe:
    """Parse timeframe string to Timeframe enum."""
    value = value.lower().strip()

    minute_aliases = ['minute', 'min', '1min', '1m', 'm']
    hour_aliases = ['hour', 'hourly', '1hour', '1h', 'h']
    day_aliases = ['day', 'daily', '1day', '1d', 'd']

    if value in minute_aliases:
        return Timeframe.MINUTE
    elif value in hour_aliases:
        return Timeframe.HOUR
    elif value in day_aliases:
        return Timeframe.DAY
    else:
        raise ValueError(
            f"Invalid timeframe: '{value}'. "
            f"Use: minute, hour, or day"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Download crypto USD pairs from Alpaca API.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_crypto.py                    # Download all pairs, minute data
    python scripts/download_crypto.py --skip-existing    # Skip already downloaded
    python scripts/download_crypto.py --timeframe hour   # Hourly data
    python scripts/download_crypto.py --start 2022-01-01 # Custom start date
        """
    )

    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        default='minute',
        help='Data timeframe: minute (default), hour, or day'
    )

    parser.add_argument(
        '--skip-existing', '-s',
        action='store_true',
        help='Skip symbols that already have data'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=None,
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

    # Parse timeframe
    try:
        timeframe = parse_timeframe(args.timeframe)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create downloader
    downloader = CryptoDownloader(
        start_date=args.start,
        end_date=args.end,
        num_threads=args.threads,
    )

    # Download all pairs
    print(f"\nDownloading {len(DEFAULT_CRYPTO_PAIRS)} crypto USD pairs...")
    print(f"Pairs: {', '.join(DEFAULT_CRYPTO_PAIRS)}\n")

    result = downloader.download_symbols(
        symbols=DEFAULT_CRYPTO_PAIRS,
        timeframe=timeframe,
        skip_existing=args.skip_existing,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Succeeded: {result.succeeded}/{result.total_symbols}")
    print(f"Failed: {result.failed}")
    print(f"Total bars: {result.total_bars:,}")
    print(f"Success rate: {result.success_rate:.1f}%")
    print(f"Time elapsed: {result.elapsed_seconds:.1f}s")

    if result.failed > 0:
        print("\nFailed symbols:")
        for sym, err in result.failed_symbols:
            print(f"  {sym}: {err}")
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
