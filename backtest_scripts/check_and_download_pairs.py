"""
Check availability and download missing pairs trading symbols via Alpaca API.

This script checks if the symbols needed for pairs trading validation are:
1. Available via Alpaca API
2. Already stored locally
3. Downloads them if available but not stored

Usage:
    conda activate fintech
    python backtest_scripts/check_and_download_pairs.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_engine.api.alpaca_client import AlpacaClient
from src.data_engine.orchestration.ingestion_pipeline import IngestionPipeline
from src.utils.logger import Logger
from alpaca.data import TimeFrame

logger = Logger()


def check_symbol_availability(symbols):
    """
    Check which symbols are available via Alpaca API.

    Args:
        symbols: List of symbol strings to check

    Returns:
        Dict with keys 'available' and 'unavailable'
    """
    logger.header("Checking Symbol Availability via Alpaca API")

    client = AlpacaClient()

    available = []
    unavailable = []

    for symbol in symbols:
        logger.info(f"Checking {symbol}...")
        try:
            # Try to fetch 1 day of data as a test (use a known trading day)
            data = client.fetch_bars(
                symbol=symbol,
                start_date_str='2024-11-01',
                end_date_str='2024-11-04',
                timeframe=TimeFrame.Minute
            )

            if data is not None and len(data) > 0:
                available.append(symbol)
                logger.success(f"{symbol}: Available ({len(data)} bars found)")
            else:
                unavailable.append(symbol)
                logger.error(f"{symbol}: No data returned")

        except Exception as e:
            unavailable.append(symbol)
            logger.error(f"{symbol}: API Error - {str(e)}")

    return {
        'available': available,
        'unavailable': unavailable
    }


def check_local_storage(symbols):
    """
    Check which symbols are already stored locally.

    Args:
        symbols: List of symbol strings to check

    Returns:
        Dict with keys 'stored' and 'missing'
    """
    logger.header("Checking Local Storage")

    from src.data_engine.storage.parquet_storage import ParquetStorage

    storage = ParquetStorage()

    stored = []
    missing = []

    for symbol in symbols:
        # Check if parquet file exists
        file_path = storage.base_path / f"{symbol}.parquet"

        if file_path.exists():
            stored.append(symbol)
            logger.success(f"{symbol}: Stored locally")
            logger.info(f"  File: {file_path}")

            # Try to get file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Size: {size_mb:.2f} MB")
        else:
            missing.append(symbol)
            logger.warning(f"{symbol}: Not found locally")

    return {
        'stored': stored,
        'missing': missing
    }


def download_symbols(symbols, start_date='2017-01-01', end_date=None):
    """
    Download symbols using the ingestion pipeline.

    Args:
        symbols: List of symbols to download
        start_date: Start date for historical data
        end_date: End date (defaults to today)
    """
    if not symbols:
        logger.warning("No symbols to download")
        return

    logger.header(f"Downloading {len(symbols)} Symbols")
    logger.info(f"Date range: {start_date} to {end_date or 'today'}")

    pipeline = IngestionPipeline(max_workers=4)

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        result = pipeline.ingest_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=TimeFrame.Minute
        )

        logger.success(f"Download complete!")
        logger.info(f"Successful: {len(result['successful'])}")
        logger.info(f"Failed: {len(result['failed'])}")

        if result['failed']:
            logger.warning("Failed symbols:")
            for symbol, error in result['failed']:
                logger.error(f"  {symbol}: {error}")

    except Exception as e:
        logger.error(f"Exception during download: {str(e)}")


def main():
    """Main execution."""
    logger.header("PAIRS TRADING SYMBOL AVAILABILITY CHECK")

    # Define symbols needed for pairs trading validation
    pairs_symbols = [
        'SPY',   # S&P 500 ETF
        'IWM',   # Russell 2000 ETF
        'GLD',   # Gold ETF
        'GDX',   # Gold Miners ETF
        'XLE',   # Energy Sector ETF
        'XLU',   # Utilities Sector ETF
        'EWA',   # Australia ETF
        'EWC'    # Canada ETF
    ]

    logger.info(f"Checking {len(pairs_symbols)} symbols for pairs trading:")
    for symbol in pairs_symbols:
        logger.info(f"  - {symbol}")

    print()

    # Check local storage first
    local_status = check_local_storage(pairs_symbols)
    print()

    # Check API availability for missing symbols
    if local_status['missing']:
        api_status = check_symbol_availability(local_status['missing'])
        print()

        # Summary
        logger.header("SUMMARY")
        logger.success(f"Already stored locally: {len(local_status['stored'])}")
        if local_status['stored']:
            for symbol in local_status['stored']:
                logger.info(f"  + {symbol}")

        logger.info(f"Available via Alpaca API: {len(api_status['available'])}")
        if api_status['available']:
            for symbol in api_status['available']:
                logger.info(f"  + {symbol}")

        logger.warning(f"Not available: {len(api_status['unavailable'])}")
        if api_status['unavailable']:
            for symbol in api_status['unavailable']:
                logger.info(f"  - {symbol}")

        # Ask to download
        if api_status['available']:
            print("\n" + "="*80)
            response = input(f"\nDownload {len(api_status['available'])} available symbols? (y/n): ").strip().lower()

            if response == 'y':
                print()
                download_symbols(
                    api_status['available'],
                    start_date='2017-01-01',
                    end_date=None
                )

                logger.success("Download complete!")
                logger.info("You can now run the pairs trading validation")
            else:
                logger.info("Skipping download")
    else:
        logger.success("All symbols already stored locally!")
        logger.info("You can run the pairs trading validation now")

    logger.header("COMPLETE")


if __name__ == '__main__':
    main()
