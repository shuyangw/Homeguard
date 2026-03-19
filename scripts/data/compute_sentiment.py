"""
Batch Sentiment Computation CLI.

Compute FinBERT sentiment analysis for downloaded news data and cache results.

Usage:
    # Compute sentiment for all news without sentiment
    python scripts/compute_sentiment.py --all

    # Compute for specific symbols
    python scripts/compute_sentiment.py --symbols AAPL,MSFT,GOOGL

    # Compute for specific year range
    python scripts/compute_sentiment.py --symbols AAPL --start-year 2023 --end-year 2024

    # List news data status
    python scripts/compute_sentiment.py --list

    # Force recompute (overwrite existing)
    python scripts/compute_sentiment.py --symbols AAPL --overwrite
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.news.sentiment_cache import SentimentCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_symbols_arg(symbols_str: str) -> list:
    """Parse comma-separated symbols string."""
    return [s.strip().upper() for s in symbols_str.split(',') if s.strip()]


def list_status(cache: SentimentCache) -> None:
    """List status of news and sentiment data."""
    available = cache.list_available_news()

    if not available:
        print("No news data found.")
        return

    print("\n" + "=" * 70)
    print("NEWS & SENTIMENT DATA STATUS")
    print("=" * 70)
    print(f"{'Symbol':<10} {'Year':<8} {'Has Sentiment':<15}")
    print("-" * 70)

    total_news = 0
    total_sentiment = 0

    for item in available:
        total_news += 1
        has_sent = "[+]" if item['has_sentiment'] else "[-]"
        if item['has_sentiment']:
            total_sentiment += 1
        print(f"{item['symbol']:<10} {item['year']:<8} {has_sent:<15}")

    print("-" * 70)
    print(f"Total news files: {total_news}")
    print(f"With sentiment:   {total_sentiment}")
    print(f"Missing:          {total_news - total_sentiment}")
    print("=" * 70)


def compute_all_missing(cache: SentimentCache, overwrite: bool = False) -> None:
    """Compute sentiment for all news data missing sentiment."""
    missing = cache.list_missing_sentiment()

    if not missing and not overwrite:
        print("All news data already has sentiment computed.")
        return

    if overwrite:
        # Get all news data if overwriting
        targets = cache.list_available_news()
    else:
        targets = missing

    total_articles = 0
    processed_count = 0

    print(f"\nProcessing {len(targets)} symbol/year combinations...")
    print("-" * 50)

    for item in targets:
        symbol = item['symbol']
        year = item['year']

        try:
            count = cache.compute_and_save(symbol, year, overwrite=overwrite)
            total_articles += count
            processed_count += 1
            status = f"{count} articles" if count > 0 else "skipped"
            print(f"  {symbol}/{year}: {status}")
        except Exception as e:
            logger.error(f"Failed to process {symbol}/{year}: {e}")

    print("-" * 50)
    print(f"Processed: {processed_count} combinations")
    print(f"Total articles: {total_articles}")


def compute_symbols(
    cache: SentimentCache,
    symbols: list,
    start_year: int,
    end_year: int,
    overwrite: bool = False,
) -> None:
    """Compute sentiment for specific symbols."""
    total_articles = 0

    print(f"\nProcessing {len(symbols)} symbols for years {start_year}-{end_year}...")
    print("-" * 50)

    for symbol in symbols:
        for year in range(start_year, end_year + 1):
            if not cache.has_news(symbol, year):
                logger.debug(f"No news data for {symbol}/{year}")
                continue

            try:
                count = cache.compute_and_save(symbol, year, overwrite=overwrite)
                total_articles += count
                status = f"{count} articles" if count > 0 else "skipped (exists)"
                print(f"  {symbol}/{year}: {status}")
            except Exception as e:
                logger.error(f"Failed to process {symbol}/{year}: {e}")

    print("-" * 50)
    print(f"Total articles processed: {total_articles}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute FinBERT sentiment for news data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List current status
    python scripts/compute_sentiment.py --list

    # Compute all missing sentiment
    python scripts/compute_sentiment.py --all

    # Compute for specific symbols
    python scripts/compute_sentiment.py --symbols AAPL,MSFT --start-year 2024

    # Force recompute
    python scripts/compute_sentiment.py --symbols AAPL --overwrite
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--list',
        action='store_true',
        help='List news/sentiment data status'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Compute sentiment for all news data missing sentiment'
    )
    mode_group.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to process'
    )

    # Options
    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Start year for processing (default: 2020)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year,
        help='End year for processing (default: current year)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing sentiment data'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for inference (default: auto)'
    )

    args = parser.parse_args()

    # Create cache
    cache = SentimentCache()

    # Execute based on mode
    if args.list:
        list_status(cache)
    elif args.all:
        print("Computing sentiment for all missing news data...")
        print(f"Using device: {args.device}")
        compute_all_missing(cache, overwrite=args.overwrite)
    elif args.symbols:
        symbols = parse_symbols_arg(args.symbols)
        print(f"Computing sentiment for: {symbols}")
        print(f"Year range: {args.start_year}-{args.end_year}")
        print(f"Using device: {args.device}")
        compute_symbols(
            cache,
            symbols,
            args.start_year,
            args.end_year,
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()
