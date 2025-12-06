"""
Generic Stock Data Downloader CLI.

Download minute/hourly/daily OHLCV data from Alpaca API for any list of symbols.
Supports multiple input methods and enforces canonical schema.

Usage:
    # Download specific symbols
    python scripts/download_symbols.py --symbols AAPL,MSFT,GOOGL

    # Download from CSV file (uses 'Symbol' column)
    python scripts/download_symbols.py --csv backtest_lists/sp500-2025.csv

    # Download from text file (one symbol per line)
    python scripts/download_symbols.py --file my_symbols.txt

    # Skip existing symbols
    python scripts/download_symbols.py --csv etfs.csv --skip-existing

    # Custom date range
    python scripts/download_symbols.py --symbols SPY,QQQ --start 2020-01-01 --end 2024-12-31

    # Download hourly data
    python scripts/download_symbols.py --csv etfs.csv --timeframe hour

    # Download daily data
    python scripts/download_symbols.py --csv sp500.csv --timeframe day --skip-existing
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader import AlpacaDownloader, Timeframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_symbols_arg(symbols_str: str) -> list:
    """Parse comma-separated symbols string."""
    return [s.strip().upper() for s in symbols_str.split(',') if s.strip()]


def load_symbols_from_csv(csv_path: Path) -> list:
    """Load symbols from CSV file with 'Symbol' column."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Try common column names
    symbol_col = None
    for col in ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'TICKER']:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        raise ValueError(f"No symbol column found in {csv_path}. Expected 'Symbol' or 'Ticker' column.")

    symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
    # Filter out empty strings and 'NAN' strings (from NaN conversion)
    return [s for s in symbols if s and s != 'NAN']


def load_symbols_from_file(file_path: Path) -> list:
    """Load symbols from text file (one per line). Lines starting with # are ignored."""
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    with open(file_path, 'r') as f:
        symbols = []
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                symbols.append(line.upper())

    return symbols


def parse_timeframe(timeframe_str: str) -> Timeframe:
    """Parse timeframe string to Timeframe enum."""
    timeframe_map = {
        'minute': Timeframe.MINUTE,
        'min': Timeframe.MINUTE,
        '1min': Timeframe.MINUTE,
        'hour': Timeframe.HOUR,
        'hourly': Timeframe.HOUR,
        '1hour': Timeframe.HOUR,
        'day': Timeframe.DAY,
        'daily': Timeframe.DAY,
        '1day': Timeframe.DAY,
    }
    key = timeframe_str.lower()
    if key not in timeframe_map:
        valid = ', '.join(sorted(set(timeframe_map.keys())))
        raise ValueError(f"Invalid timeframe '{timeframe_str}'. Valid options: {valid}")
    return timeframe_map[key]


def main():
    parser = argparse.ArgumentParser(
        description='Download historical OHLCV data from Alpaca API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_symbols.py --symbols AAPL,MSFT,GOOGL
  python scripts/download_symbols.py --csv backtest_lists/sp500-2025.csv --skip-existing
  python scripts/download_symbols.py --file symbols.txt --timeframe hour
  python scripts/download_symbols.py --symbols SPY,QQQ --start 2020-01-01 --end 2024-12-31
        """
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--symbols', '-s',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )
    input_group.add_argument(
        '--csv', '-c',
        type=str,
        help='Path to CSV file with Symbol column'
    )
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to text file with one symbol per line'
    )

    # Optional arguments
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        default='minute',
        help='Data timeframe: minute (default), hour, or day'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip symbols that already have data downloaded'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2017-01-01',
        help='Start date in YYYY-MM-DD format (default: 2017-01-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=6,
        help='Number of download threads (default: 6)'
    )

    args = parser.parse_args()

    # Load symbols from the specified source
    try:
        if args.symbols:
            symbols = parse_symbols_arg(args.symbols)
            logger.info(f"Loaded {len(symbols)} symbols from command line")
        elif args.csv:
            csv_path = Path(args.csv)
            if not csv_path.is_absolute():
                csv_path = PROJECT_ROOT / csv_path
            symbols = load_symbols_from_csv(csv_path)
            logger.info(f"Loaded {len(symbols)} symbols from {csv_path}")
        elif args.file:
            file_path = Path(args.file)
            if not file_path.is_absolute():
                file_path = PROJECT_ROOT / file_path
            symbols = load_symbols_from_file(file_path)
            logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
        else:
            parser.error("One of --symbols, --csv, or --file is required")
            return
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    if not symbols:
        logger.error("No symbols to download")
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    symbols = unique_symbols

    # Parse timeframe
    try:
        timeframe = parse_timeframe(args.timeframe)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Validate dates
    try:
        start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        if start_dt >= end_dt:
            logger.error(f"Start date ({args.start}) must be before end date ({args.end})")
            sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    # Create downloader and run
    downloader = AlpacaDownloader(
        start_date=args.start,
        end_date=args.end,
        num_threads=args.threads,
    )

    result = downloader.download_symbols(
        symbols=symbols,
        timeframe=timeframe,
        skip_existing=args.skip_existing,
    )

    # Exit with error code if there were failures
    if result.failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
