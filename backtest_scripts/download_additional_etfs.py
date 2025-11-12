"""
Download additional leveraged ETF data for optimization.

This script downloads daily OHLCV data for additional leveraged ETFs
to expand the symbol universe for overnight mean reversion strategy optimization.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

DATA_DIR = Path('data/leveraged_etfs')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Additional symbols to download
ADDITIONAL_SYMBOLS = {
    '3x Healthcare': ['CURE', 'CUT'],
    '3x Energy': ['ERX', 'ERY'],
    '3x Retail': ['RETL'],  # RETS may not exist
    '3x Internet': ['WEBL'],  # WEBS may not exist
    '3x Banking': ['DPST'],
    '3x Defense': ['DFEN'],
    '3x Construction': ['NAIL'],
    '3x Miners': ['DUST'],
    '2x Real Estate': ['UWM', 'TWM'],
    '2x Gold': ['UGL', 'GLL'],
    '2x Financials': ['UYG', 'SKF'],
    '2x Semiconductors': ['USD'],  # BIS may not exist
    '2x Energy': ['UCO', 'SCO'],  # Additional 2x
    '2x Biotech': ['BIB', 'BIS'],  # Additional 2x
    'Single Stock': ['TSLL', 'NVDL']  # Tesla 2x, Nvidia 2x
}

# Flatten into single list
ALL_ADDITIONAL = []
for category, symbols in ADDITIONAL_SYMBOLS.items():
    ALL_ADDITIONAL.extend(symbols)

# Remove duplicates
ALL_ADDITIONAL = sorted(list(set(ALL_ADDITIONAL)))

START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
MIN_BARS = 100  # Minimum bars to be considered valid


def download_symbol(symbol, start_date, end_date):
    """Download data for a single symbol using yfinance."""
    try:
        logger.info(f"  Downloading {symbol}...")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if df.empty:
            logger.warning(f"    {symbol}: No data available")
            return None

        if len(df) < MIN_BARS:
            logger.warning(f"    {symbol}: Insufficient data ({len(df)} bars < {MIN_BARS})")
            return None

        # Standardize column names
        df.index.name = 'timestamp'
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })

        # Keep only OHLCV
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Reset index to make timestamp a column
        df = df.reset_index()

        # Save to parquet
        output_path = DATA_DIR / f'{symbol}_1d.parquet'
        df.to_parquet(output_path, index=False)

        logger.success(f"    {symbol}: Downloaded {len(df)} bars ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")

        return {
            'symbol': symbol,
            'bars': len(df),
            'start_date': df['timestamp'].min(),
            'end_date': df['timestamp'].max(),
            'success': True
        }

    except Exception as e:
        logger.error(f"    {symbol}: Error - {str(e)}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


def main():
    logger.info("\n" + "="*80)
    logger.info("DOWNLOADING ADDITIONAL LEVERAGED ETF DATA")
    logger.info("="*80)
    logger.info(f"\nTarget symbols: {len(ALL_ADDITIONAL)}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Minimum bars: {MIN_BARS}")
    logger.info(f"Output directory: {DATA_DIR}")

    # Check which symbols already exist
    existing = []
    for symbol in ALL_ADDITIONAL:
        if (DATA_DIR / f'{symbol}_1d.parquet').exists():
            existing.append(symbol)

    if existing:
        logger.info(f"\nAlready downloaded: {len(existing)} symbols")
        logger.info(f"  {', '.join(existing)}")

    to_download = [s for s in ALL_ADDITIONAL if s not in existing]

    if not to_download:
        logger.success("\nAll symbols already downloaded!")
        return

    logger.info(f"\nWill download: {len(to_download)} symbols")
    logger.info(f"  {', '.join(to_download)}")

    # Download each symbol
    results = []

    logger.info("\n" + "="*80)
    logger.info("DOWNLOADING...")
    logger.info("="*80)

    for i, symbol in enumerate(to_download, 1):
        logger.info(f"\n[{i}/{len(to_download)}] Processing {symbol}...")
        result = download_symbol(symbol, START_DATE, END_DATE)
        if result:
            results.append(result)

    # Summary
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    logger.info(f"  Total attempted: {len(results)}")
    logger.success(f"  Successful: {len(successful)}")
    if failed:
        logger.error(f"  Failed: {len(failed)}")
        logger.info("\nFailed symbols:")
        for r in failed:
            logger.error(f"  - {r['symbol']}: {r.get('error', 'Unknown error')}")

    if successful:
        logger.info("\nSuccessful downloads:")
        for r in successful:
            logger.success(f"  - {r['symbol']}: {r['bars']} bars")

    # Count total symbols now available
    total_files = len(list(DATA_DIR.glob('*_1d.parquet')))
    logger.info(f"\nTotal symbols available: {total_files}")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = Path('reports/20251112_additional_etf_download_summary.csv')
    summary_path.parent.mkdir(exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.success(f"\nSaved download summary to {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
