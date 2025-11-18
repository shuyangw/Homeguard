"""
Download ETF Universe Data

Downloads all available ETFs from the defined universe via Alpaca API.
Starts from 2017-01-01 to present.

Usage:
    conda activate fintech
    python backtest_scripts/download_etf_universe.py
"""

import sys
from pathlib import Path
from datetime import datetime

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path

from src.data_engine.api.alpaca_client import AlpacaClient
from src.data_engine.orchestration.ingestion_pipeline import IngestionPipeline
from src.utils.logger import Logger
from alpaca.data import TimeFrame

logger = Logger()


def get_etf_universe():
    """
    Define a comprehensive universe of liquid ETFs.

    Returns:
        List of all ETF symbols
    """
    etf_universe = {
        'broad_market': ['SPY', 'IWM', 'QQQ', 'DIA', 'VTI', 'VOO'],
        'sectors': ['XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLU', 'XLP', 'XLY', 'XLB'],
        'international': ['EWA', 'EWC', 'EWG', 'EWJ', 'EWU', 'EWZ', 'FXI'],
        'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'GDX', 'SIL'],
        'fixed_income': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'AGG'],
        'volatility': ['VXX', 'UVXY', 'SVXY']
    }

    # Flatten all categories
    all_symbols = []
    for category, symbols in etf_universe.items():
        all_symbols.extend(symbols)

    return all_symbols, etf_universe


def check_availability(symbols):
    """
    Check which symbols are available via Alpaca API.

    Args:
        symbols: List of symbols to check

    Returns:
        Dict with 'available' and 'unavailable' lists
    """
    logger.header("Checking Symbol Availability")
    logger.info(f"Testing {len(symbols)} symbols via Alpaca API")

    client = AlpacaClient()
    available = []
    unavailable = []

    for symbol in symbols:
        logger.info(f"Checking {symbol}...")
        try:
            # Test fetch with recent trading day
            data = client.fetch_bars(
                symbol=symbol,
                start_date_str='2024-11-01',
                end_date_str='2024-11-04',
                timeframe=TimeFrame.Minute
            )

            if data is not None and len(data) > 0:
                available.append(symbol)
                logger.success(f"  {symbol}: Available ({len(data)} bars)")
            else:
                unavailable.append(symbol)
                logger.warning(f"  {symbol}: No data returned")

        except Exception as e:
            unavailable.append(symbol)
            logger.error(f"  {symbol}: {str(e)[:60]}")

    return {
        'available': available,
        'unavailable': unavailable
    }


def download_symbols(symbols, start_date='2017-01-01', end_date=None):
    """
    Download symbols using ingestion pipeline.

    Args:
        symbols: List of symbols to download
        start_date: Start date (default: 2017-01-01)
        end_date: End date (default: today)
    """
    if not symbols:
        logger.warning("No symbols to download")
        return

    logger.header(f"Downloading {len(symbols)} Symbols")
    logger.info(f"Date range: {start_date} to {end_date or 'today'}")
    logger.info(f"Symbols: {', '.join(symbols)}")

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

        logger.success("Download Complete!")
        logger.info(f"Successful: {len(result['successful'])}")
        logger.info(f"Failed: {len(result['failed'])}")

        if result['successful']:
            logger.info("\nSuccessfully downloaded:")
            for symbol in result['successful']:
                logger.info(f"  + {symbol}")

        if result['failed']:
            logger.warning("\nFailed downloads:")
            for symbol, error in result['failed']:
                logger.error(f"  - {symbol}: {error[:60]}")

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")


def main():
    """Main execution."""
    logger.header("ETF UNIVERSE DATA DOWNLOAD")
    logger.info("Downloading comprehensive ETF universe from Alpaca")

    # Get universe
    all_symbols, categories = get_etf_universe()

    logger.info(f"\nTotal symbols in universe: {len(all_symbols)}")
    for category, symbols in categories.items():
        logger.info(f"  {category}: {', '.join(symbols)}")

    print("\n" + "="*80)

    # Check availability
    availability = check_availability(all_symbols)

    print("\n" + "="*80)

    # Summary
    logger.header("AVAILABILITY SUMMARY")
    logger.success(f"Available: {len(availability['available'])}/{len(all_symbols)}")

    if availability['available']:
        logger.info("\nAvailable symbols:")
        for symbol in availability['available']:
            logger.info(f"  + {symbol}")

    if availability['unavailable']:
        logger.warning(f"\nUnavailable: {len(availability['unavailable'])}")
        for symbol in availability['unavailable']:
            logger.info(f"  - {symbol}")

    # Download
    if availability['available']:
        print("\n" + "="*80)
        logger.info(f"\nReady to download {len(availability['available'])} symbols from 2017-01-01")

        # Auto-download (user already confirmed)
        logger.info("Starting download...")
        print()

        download_symbols(
            availability['available'],
            start_date='2017-01-01',
            end_date=None
        )

        logger.success("\nAll downloads complete!")
        logger.info("You can now run pair discovery with full ETF universe")
    else:
        logger.error("No symbols available for download!")

    logger.header("COMPLETE")


if __name__ == '__main__':
    main()
