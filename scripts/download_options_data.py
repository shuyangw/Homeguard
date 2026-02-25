"""
Download historical options data from ThetaData.

Usage:
    python scripts/download_options_data.py --symbols SPY,QQQ,IWM --start 2020-01-01 --end 2024-12-31

    # Download for OpEx strategy (core universe)
    python scripts/download_options_data.py --preset opex

    # Check what data is available
    python scripts/download_options_data.py --status

Requirements:
    - Theta Terminal must be running locally
    - THETADATA environment variables configured (optional)
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from typing import List, Optional

from src.utils.logger import logger
from src.data.options.thetadata_client import (
    ThetaDataClient,
    ThetaTerminalNotRunning,
    ThetaDataError,
)
from src.data.options.options_store import OptionsDataStore


# Predefined symbol universes
PRESETS = {
    "opex": ["SPY", "QQQ", "IWM"],
    "extended": ["SPY", "QQQ", "IWM", "TQQQ", "SOXL", "UPRO"],
    "test": ["SPY"],
}


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download historical options data from ThetaData",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download SPY, QQQ, IWM for 2020-2024
    python scripts/download_options_data.py --symbols SPY,QQQ,IWM --start 2020-01-01 --end 2024-12-31

    # Use preset (opex = SPY, QQQ, IWM)
    python scripts/download_options_data.py --preset opex --start 2020-01-01

    # Check terminal connection
    python scripts/download_options_data.py --check

    # Show storage status
    python scripts/download_options_data.py --status
        """,
    )

    # Symbol selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., SPY,QQQ,IWM)",
    )
    group.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help=f"Use predefined symbol list: {list(PRESETS.keys())}",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD), default: 2020-01-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), default: today",
    )

    # Options
    parser.add_argument(
        "--monthly-only",
        action="store_true",
        default=True,
        help="Only download monthly expirations (default: True)",
    )
    parser.add_argument(
        "--strike-range",
        type=float,
        default=0.15,
        help="Strike range as fraction of spot (default: 0.15 = +/- 15%%)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates that already have data",
    )

    # Actions
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Theta Terminal connection and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show storage status and exit",
    )

    # Connection settings
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Theta Terminal host (default: localhost)",
    )

    return parser.parse_args()


def check_connection(client: ThetaDataClient) -> bool:
    """Check if Theta Terminal is running."""
    if client.check_connection():
        logger.info("[+] Theta Terminal is running and responsive")
        return True
    else:
        logger.error("[-] Cannot connect to Theta Terminal")
        logger.error("    Make sure Theta Terminal is running")
        return False


def show_status(store: OptionsDataStore):
    """Show storage status."""
    stats = store.get_stats()
    logger.info("Options Data Storage Status")
    logger.info("-" * 40)
    logger.info(f"  Location: {stats['chains_dir']}")
    logger.info(f"  Symbols:  {stats['symbols']}")
    logger.info(f"  Files:    {stats['files']}")
    logger.info(f"  Size:     {stats['size_mb']:.1f} MB")

    symbols = store.get_available_symbols()
    if symbols:
        logger.info("")
        logger.info("Available Symbols:")
        for symbol in symbols:
            min_date, max_date = store.get_date_range(symbol)
            if min_date and max_date:
                logger.info(f"  {symbol}: {min_date} to {max_date}")


def download_symbol(
    symbol: str,
    client: ThetaDataClient,
    store: OptionsDataStore,
    start_date: date,
    end_date: date,
    monthly_only: bool = True,
    strike_range: float = 0.15,
    skip_existing: bool = False,
) -> int:
    """
    Download options data for a single symbol.

    Returns:
        Number of chains saved
    """
    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

    # Check existing data if skip_existing
    existing_min, existing_max = None, None
    if skip_existing:
        existing_min, existing_max = store.get_date_range(symbol)
        if existing_min and existing_max:
            logger.info(f"  Existing data: {existing_min} to {existing_max}")

    # Get expirations for date range
    try:
        expirations = client.list_expirations(symbol, start_date, end_date)
    except ThetaDataError as e:
        logger.error(f"  Failed to list expirations: {e}")
        return 0

    # Filter to monthly if requested
    if monthly_only:
        expirations = [
            exp for exp in expirations
            if _is_monthly_expiration(exp)
        ]

    logger.info(f"  Found {len(expirations)} {'monthly ' if monthly_only else ''}expirations")

    chains_saved = 0

    for i, expiry in enumerate(expirations):
        logger.info(f"  Processing expiry {i+1}/{len(expirations)}: {expiry}")

        # Determine date range for this expiry
        # We want data from T-10 to T+5 around each OpEx
        chain_start = max(start_date, expiry - timedelta(days=10))
        chain_end = min(end_date, expiry + timedelta(days=5))

        current = chain_start
        while current <= chain_end:
            # Skip if we already have data
            if skip_existing and existing_min and existing_max:
                if existing_min <= current <= existing_max:
                    current += timedelta(days=1)
                    continue

            try:
                chain = client.get_options_chain(
                    symbol=symbol,
                    snapshot_date=current,
                    expiry=expiry,
                    strike_range_pct=strike_range,
                )

                if chain.contracts:
                    store.save_chain(chain)
                    chains_saved += 1
                    logger.debug(f"    {current}: {len(chain.contracts)} contracts")

            except ThetaDataError as e:
                logger.debug(f"    {current}: No data - {e}")

            current += timedelta(days=1)

    logger.info(f"  Saved {chains_saved} chains for {symbol}")
    return chains_saved


def _is_monthly_expiration(exp_date: date) -> bool:
    """Check if date is a monthly expiration (third Friday)."""
    if exp_date.weekday() != 4:  # Not Friday
        return False

    first_of_month = exp_date.replace(day=1)
    days_to_friday = (4 - first_of_month.weekday()) % 7
    first_friday = first_of_month + timedelta(days=days_to_friday)
    third_friday = first_friday + timedelta(days=14)

    return exp_date == third_friday


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize client and store
    client = ThetaDataClient(host=args.host)
    store = OptionsDataStore()

    # Handle status action
    if args.status:
        show_status(store)
        return 0

    # Handle check action
    if args.check:
        if check_connection(client):
            return 0
        return 1

    # Verify connection
    if not check_connection(client):
        logger.error("Cannot proceed without Theta Terminal connection")
        return 1

    # Determine symbols
    if args.preset:
        symbols = PRESETS[args.preset]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        logger.error("Must specify --symbols or --preset")
        return 1

    # Parse dates
    start_date = parse_date(args.start)
    end_date = parse_date(args.end) if args.end else date.today()

    logger.info("=" * 60)
    logger.info("ThetaData Options Downloader")
    logger.info("=" * 60)
    logger.info(f"Symbols:    {', '.join(symbols)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Monthly only: {args.monthly_only}")
    logger.info(f"Strike range: +/- {args.strike_range * 100:.0f}%")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 60)

    total_chains = 0

    for symbol in symbols:
        try:
            chains = download_symbol(
                symbol=symbol,
                client=client,
                store=store,
                start_date=start_date,
                end_date=end_date,
                monthly_only=args.monthly_only,
                strike_range=args.strike_range,
                skip_existing=args.skip_existing,
            )
            total_chains += chains
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            continue

    logger.info("=" * 60)
    logger.info(f"Download complete: {total_chains} chains saved")
    show_status(store)

    return 0


if __name__ == "__main__":
    sys.exit(main())
