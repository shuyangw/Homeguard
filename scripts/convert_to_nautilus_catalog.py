"""
Convert Homeguard parquet data to NautilusTrader ParquetDataCatalog.

This script copies existing Hive-partitioned parquet data to NautilusTrader's
catalog format. The original data remains unchanged.

Usage:
    # Convert S&P 500 symbols
    python scripts/convert_to_nautilus_catalog.py --csv backtest_lists/sp500-2025.csv

    # Convert specific symbols
    python scripts/convert_to_nautilus_catalog.py --symbols AAPL,MSFT,GOOGL

    # Convert with date range
    python scripts/convert_to_nautilus_catalog.py --csv backtest_lists/sp500-2025.csv \\
        --start 2020-01-01 --end 2024-12-31

    # Convert only daily data
    python scripts/convert_to_nautilus_catalog.py --symbols AAPL --timeframe 1day

    # Custom output directory
    python scripts/convert_to_nautilus_catalog.py --symbols AAPL --output F:/nautilus_data
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.settings import get_local_storage_dir


def load_symbols_from_csv(csv_path: str) -> list:
    """Load symbols from a CSV file."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Try common column names
    for col in ['Symbol', 'symbol', 'SYMBOL', 'ticker', 'Ticker']:
        if col in df.columns:
            return df[col].dropna().tolist()

    # If no header, assume first column is symbols
    return df.iloc[:, 0].dropna().tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Homeguard parquet data to NautilusTrader catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv backtest_lists/sp500-2025.csv
  %(prog)s --symbols AAPL,MSFT,GOOGL
  %(prog)s --csv backtest_lists/sp500-2025.csv --start 2020-01-01
  %(prog)s --symbols AAPL --timeframe 1day
        """
    )

    parser.add_argument(
        '--csv',
        help='Path to CSV file containing symbols'
    )
    parser.add_argument(
        '--symbols',
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--timeframe',
        default='1min',
        choices=['1min', '1hour', '1day'],
        help='Timeframe to convert (default: 1min)'
    )
    parser.add_argument(
        '--start',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        help='Output catalog path (default: {data_dir}/nautilus_catalog)'
    )
    parser.add_argument(
        '--all-timeframes',
        action='store_true',
        help='Convert all timeframes (1min, 1hour, 1day)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.csv and not args.symbols:
        parser.error("Must specify either --csv or --symbols")

    # Load symbols
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
        symbols = load_symbols_from_csv(str(csv_path))
        logger.info(f"Loaded {len(symbols)} symbols from {csv_path}")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        logger.info(f"Using {len(symbols)} symbols from command line")

    if not symbols:
        logger.error("No symbols to convert")
        sys.exit(1)

    # Get source directory
    source_dir = get_local_storage_dir()
    logger.info(f"Source data directory: {source_dir}")

    # Set output path
    if args.output:
        catalog_path = Path(args.output)
    else:
        catalog_path = source_dir / 'nautilus_catalog'

    # Determine timeframes
    if args.all_timeframes:
        timeframes = ['1min', '1hour', '1day']
    else:
        timeframes = [args.timeframe]

    # Check if NautilusTrader is available
    try:
        from src.backtesting.adapters.nautilus import NAUTILUS_AVAILABLE
        if not NAUTILUS_AVAILABLE:
            raise ImportError("NautilusTrader not available")
    except ImportError:
        logger.error("NautilusTrader is not installed.")
        logger.error("Install with: pip install nautilus_trader")
        sys.exit(1)

    # Import conversion modules
    from src.backtesting.adapters.nautilus.config import DataConversionConfig
    from src.backtesting.adapters.nautilus.data_converter import homeguard_to_nautilus_catalog

    # Create config
    try:
        config = DataConversionConfig(
            source_dir=source_dir,
            symbols=symbols,
            catalog_path=catalog_path,
            start_date=args.start,
            end_date=args.end,
            timeframes=timeframes,
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Run conversion
    logger.info("=" * 60)
    logger.info("Starting NautilusTrader catalog conversion")
    logger.info("=" * 60)
    logger.info(f"  Symbols: {len(symbols)}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Start date: {args.start or 'All'}")
    logger.info(f"  End date: {args.end or 'All'}")
    logger.info(f"  Output: {catalog_path}")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        catalog = homeguard_to_nautilus_catalog(config)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Conversion complete!")
        logger.info(f"  Elapsed time: {elapsed:.1f} seconds")
        logger.info(f"  Catalog path: {catalog_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
