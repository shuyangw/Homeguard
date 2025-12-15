"""
News Data Downloader CLI.

Download historical news data from Alpaca API for any list of symbols.
Supports multiple input methods and stores in Hive-partitioned Parquet format.

Usage:
    # Download specific symbols
    python scripts/download_news.py --symbols AAPL,MSFT,GOOGL

    # Download from CSV file (uses 'Symbol' column)
    python scripts/download_news.py --csv backtest_lists/sp500-2025.csv

    # Download from text file (one symbol per line)
    python scripts/download_news.py --file my_symbols.txt

    # Skip existing symbols
    python scripts/download_news.py --csv etfs.csv --skip-existing

    # Custom date range
    python scripts/download_news.py --symbols SPY,QQQ --start 2024-01-01 --end 2024-12-31

    # Include full article content (larger storage)
    python scripts/download_news.py --symbols AAPL --include-content
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.news import NewsDownloader
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
        raise ValueError(f"No symbol column found in {csv_path}. Expected 'Symbol' or 'Ticker'")

    symbols = df[symbol_col].dropna().str.strip().str.upper().tolist()
    return symbols


def load_symbols_from_file(file_path: Path) -> list:
    """Load symbols from text file (one per line)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Symbol file not found: {file_path}")

    with open(file_path, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]

    return symbols


def main():
    parser = argparse.ArgumentParser(
        description='Download news data from Alpaca API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download news for specific symbols
    python scripts/download_news.py --symbols AAPL,MSFT,GOOGL

    # Download from S&P 500 list
    python scripts/download_news.py --csv backtest_lists/sp500-2025.csv --skip-existing

    # Download with custom date range
    python scripts/download_news.py --symbols SPY --start 2024-01-01 --end 2024-12-31
        """
    )

    # Symbol input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )
    input_group.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with Symbol column'
    )
    input_group.add_argument(
        '--file',
        type=str,
        help='Path to text file with one symbol per line'
    )

    # Date range
    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD). Default: 2020-01-01'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD). Default: today'
    )

    # Options
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip symbols that already have news data'
    )
    parser.add_argument(
        '--include-content',
        action='store_true',
        help='Include full article content (increases storage)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel download threads. Default: 4'
    )

    args = parser.parse_args()

    # Load symbols from appropriate source
    try:
        if args.symbols:
            symbols = parse_symbols_arg(args.symbols)
            logger.info(f"Loaded {len(symbols)} symbols from command line")
        elif args.csv:
            csv_path = Path(args.csv)
            if not csv_path.is_absolute():
                csv_path = PROJECT_ROOT / csv_path
            symbols = load_symbols_from_csv(csv_path)
            logger.info(f"Loaded {len(symbols)} symbols from {args.csv}")
        elif args.file:
            file_path = Path(args.file)
            if not file_path.is_absolute():
                file_path = PROJECT_ROOT / file_path
            symbols = load_symbols_from_file(file_path)
            logger.info(f"Loaded {len(symbols)} symbols from {args.file}")
        else:
            parser.error("Must specify symbols via --symbols, --csv, or --file")
            return

        if not symbols:
            logger.error("No symbols to download")
            return

    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Create downloader and run
    downloader = NewsDownloader(
        start_date=args.start,
        end_date=args.end,
        max_workers=args.workers,
        include_content=args.include_content,
    )

    result = downloader.download_symbols(
        symbols=symbols,
        skip_existing=args.skip_existing,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("NEWS DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total symbols:    {result.total_symbols}")
    print(f"Succeeded:        {result.succeeded}")
    print(f"Failed:           {result.failed}")
    print(f"Success rate:     {result.success_rate:.1f}%")
    print(f"Total articles:   {result.total_articles}")
    print(f"Elapsed time:     {result.elapsed_seconds:.1f}s")
    print("=" * 60)

    if result.failed_symbols:
        print("\nFailed symbols:")
        for symbol, error in result.failed_symbols[:20]:
            print(f"  {symbol}: {error}")
        if len(result.failed_symbols) > 20:
            print(f"  ... and {len(result.failed_symbols) - 20} more")

    # Exit with error code if significant failures
    if result.failed > result.succeeded:
        sys.exit(1)


if __name__ == '__main__':
    main()
