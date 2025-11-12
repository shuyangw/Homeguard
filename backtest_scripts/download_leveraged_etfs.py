"""
Download historical data for leveraged ETFs for overnight mean reversion strategy.

This script downloads 10 years of daily data for leveraged ETFs
to be used in the regime-based overnight mean reversion strategy.
Daily data is sufficient for calculating overnight returns (close to next open).
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# Define leveraged ETF universe for overnight mean reversion
LEVERAGED_ETF_UNIVERSE = {
    # 3x Leveraged Long ETFs
    'TQQQ': {'leverage': 3, 'underlying': 'QQQ', 'direction': 'long', 'name': 'ProShares UltraPro QQQ'},
    'UPRO': {'leverage': 3, 'underlying': 'SPY', 'direction': 'long', 'name': 'ProShares UltraPro S&P500'},
    'UDOW': {'leverage': 3, 'underlying': 'DIA', 'direction': 'long', 'name': 'ProShares UltraPro Dow30'},
    'TNA': {'leverage': 3, 'underlying': 'IWM', 'direction': 'long', 'name': 'Direxion Daily Small Cap Bull 3X'},
    'SOXL': {'leverage': 3, 'underlying': 'SOXX', 'direction': 'long', 'name': 'Direxion Daily Semiconductor Bull 3X'},
    'FAS': {'leverage': 3, 'underlying': 'XLF', 'direction': 'long', 'name': 'Direxion Daily Financial Bull 3X'},
    'LABU': {'leverage': 3, 'underlying': 'XBI', 'direction': 'long', 'name': 'Direxion Daily S&P Biotech Bull 3X'},
    'TECL': {'leverage': 3, 'underlying': 'XLK', 'direction': 'long', 'name': 'Direxion Daily Technology Bull 3X'},

    # 3x Leveraged Short ETFs
    'SQQQ': {'leverage': -3, 'underlying': 'QQQ', 'direction': 'short', 'name': 'ProShares UltraPro Short QQQ'},
    'SPXU': {'leverage': -3, 'underlying': 'SPY', 'direction': 'short', 'name': 'ProShares UltraPro Short S&P500'},
    'SDOW': {'leverage': -3, 'underlying': 'DIA', 'direction': 'short', 'name': 'ProShares UltraPro Short Dow30'},
    'TZA': {'leverage': -3, 'underlying': 'IWM', 'direction': 'short', 'name': 'Direxion Daily Small Cap Bear 3X'},
    'SOXS': {'leverage': -3, 'underlying': 'SOXX', 'direction': 'short', 'name': 'Direxion Daily Semiconductor Bear 3X'},
    'FAZ': {'leverage': -3, 'underlying': 'XLF', 'direction': 'short', 'name': 'Direxion Daily Financial Bear 3X'},
    'LABD': {'leverage': -3, 'underlying': 'XBI', 'direction': 'short', 'name': 'Direxion Daily S&P Biotech Bear 3X'},
    'TECS': {'leverage': -3, 'underlying': 'XLK', 'direction': 'short', 'name': 'Direxion Daily Technology Bear 3X'},

    # 2x Leveraged ETFs
    'QLD': {'leverage': 2, 'underlying': 'QQQ', 'direction': 'long', 'name': 'ProShares Ultra QQQ'},
    'SSO': {'leverage': 2, 'underlying': 'SPY', 'direction': 'long', 'name': 'ProShares Ultra S&P500'},
    'QID': {'leverage': -2, 'underlying': 'QQQ', 'direction': 'short', 'name': 'ProShares UltraShort QQQ'},
    'SDS': {'leverage': -2, 'underlying': 'SPY', 'direction': 'short', 'name': 'ProShares UltraShort S&P500'},

    # Volatility ETFs
    'UVXY': {'leverage': 1.5, 'underlying': 'VIX', 'direction': 'long', 'name': 'ProShares Ultra VIX Short-Term Futures'},
    'SVXY': {'leverage': -0.5, 'underlying': 'VIX', 'direction': 'short', 'name': 'ProShares Short VIX Short-Term Futures'},
    'VIXY': {'leverage': 1, 'underlying': 'VIX', 'direction': 'long', 'name': 'ProShares VIX Short-Term Futures'},
}

# Add market indicators for regime detection
MARKET_INDICATORS = {
    'SPY': {'name': 'SPDR S&P 500 ETF', 'use': 'Market regime detection'},
    '^VIX': {'name': 'CBOE Volatility Index', 'use': 'Volatility regime detection'}
}

def download_data(symbols, start_date, end_date, data_dir='data/leveraged_etfs'):
    """
    Download historical data for specified symbols.

    Args:
        symbols: List of symbols to download
        start_date: Start date for data
        end_date: End date for data
        data_dir: Directory to save data files
    """
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading data for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date}")

    results = {}
    failed = []

    for symbol in symbols:
        try:
            logger.info(f"Downloading {symbol}...")

            # Check if data already exists
            file_path = data_path / f"{symbol}_1d.parquet"
            if file_path.exists():
                logger.warning(f"  Data already exists for {symbol}, skipping...")
                continue

            # Download daily data using yfinance
            # Daily data is sufficient for overnight mean reversion
            # We use close-to-open returns, which are available in daily data
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval='1d',
                progress=False
            )

            if df is None or df.empty:
                logger.error(f"  No data returned for {symbol}")
                failed.append(symbol)
                continue

            # Save to parquet file
            df.to_parquet(file_path, engine='pyarrow')

            results[symbol] = {
                'rows': len(df),
                'start': df.index.min(),
                'end': df.index.max(),
                'file': str(file_path)
            }

            logger.success(f"  Saved {len(df):,} bars for {symbol} to {file_path}")

        except Exception as e:
            logger.error(f"  Error downloading {symbol}: {e}")
            failed.append(symbol)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    logger.success(f"Successfully downloaded: {len(results)}/{len(symbols)} symbols")

    if failed:
        logger.error(f"Failed symbols: {', '.join(failed)}")

    # Display results table
    if results:
        logger.info("\nDownloaded Data Summary:")
        logger.info(f"{'Symbol':<10} {'Bars':<15} {'Start Date':<12} {'End Date':<12}")
        logger.info("-"*60)

        for symbol, info in results.items():
            logger.info(
                f"{symbol:<10} {info['rows']:<15,} "
                f"{info['start'].strftime('%Y-%m-%d'):<12} "
                f"{info['end'].strftime('%Y-%m-%d'):<12}"
            )

    return results, failed

def validate_data(data_dir='data/leveraged_etfs'):
    """
    Validate downloaded data for completeness and quality.
    """
    data_path = Path(data_dir)

    logger.info("\n" + "="*80)
    logger.info("VALIDATING DOWNLOADED DATA")
    logger.info("="*80)

    all_symbols = list(LEVERAGED_ETF_UNIVERSE.keys()) + list(MARKET_INDICATORS.keys())

    validation_results = {}

    for symbol in all_symbols:
        file_path = data_path / f"{symbol}_1d.parquet"

        if not file_path.exists():
            logger.warning(f"{symbol}: No data file found")
            validation_results[symbol] = {'status': 'missing'}
            continue

        try:
            # Load data
            df = pd.read_parquet(file_path)

            # Basic validation
            issues = []

            # Check for sufficient data (for daily data, ~250 days/year * 10 years = 2500)
            if len(df) < 1000:
                issues.append(f"Insufficient data ({len(df)} bars)")

            # Check for gaps
            df.index = pd.to_datetime(df.index)
            time_diffs = df.index.to_series().diff()

            # Find gaps larger than 7 days (weekends/holidays are OK)
            large_gaps = time_diffs[time_diffs > pd.Timedelta(days=7)]
            if len(large_gaps) > 10:
                issues.append(f"Many gaps detected ({len(large_gaps)} gaps > 7 days)")

            # Check for null values
            null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            if null_pct > 1:
                issues.append(f"High null percentage ({null_pct:.2f}%)")

            if issues:
                logger.warning(f"{symbol}: {', '.join(issues)}")
                validation_results[symbol] = {'status': 'issues', 'issues': issues}
            else:
                logger.success(f"{symbol}: [OK] Valid ({len(df):,} bars)")
                validation_results[symbol] = {'status': 'valid', 'bars': len(df)}

        except Exception as e:
            logger.error(f"{symbol}: Error validating - {e}")
            validation_results[symbol] = {'status': 'error', 'error': str(e)}

    # Summary
    valid_count = sum(1 for v in validation_results.values() if v['status'] == 'valid')
    logger.info("\n" + "="*80)
    logger.info(f"Validation Complete: {valid_count}/{len(all_symbols)} symbols valid")

    return validation_results

def main():
    """Main function to download leveraged ETF data."""

    logger.info("\n" + "="*80)
    logger.info("LEVERAGED ETF DATA DOWNLOAD FOR OVERNIGHT MEAN REVERSION")
    logger.info("="*80)

    # Define date range (10 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)

    logger.info(f"\nData Collection Parameters:")
    logger.info(f"  Start Date: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"  End Date: {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Leveraged ETFs: {len(LEVERAGED_ETF_UNIVERSE)}")
    logger.info(f"  Market Indicators: {len(MARKET_INDICATORS)}")

    # Combine all symbols
    all_symbols = list(LEVERAGED_ETF_UNIVERSE.keys()) + list(MARKET_INDICATORS.keys())

    # For initial testing, start with a subset
    test_symbols = ['TQQQ', 'SQQQ', 'SPY', '^VIX', 'UPRO', 'SPXU']

    logger.info("\n[Phase 1] Starting with test symbols for validation...")
    logger.info(f"Test symbols: {', '.join(test_symbols)}")

    # Download test data first
    results, failed = download_data(
        test_symbols,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    # Validate downloaded data
    validation_results = validate_data()

    # Check if we should continue with full download
    valid_test = all(
        validation_results.get(s, {}).get('status') == 'valid'
        for s in test_symbols if s in validation_results
    )

    if valid_test:
        logger.success("\n[OK] Test symbols downloaded successfully!")

        user_input = input("\nDownload remaining symbols? (y/n): ")
        if user_input.lower() == 'y':
            remaining_symbols = [s for s in all_symbols if s not in test_symbols]
            logger.info(f"\n[Phase 2] Downloading {len(remaining_symbols)} remaining symbols...")

            results2, failed2 = download_data(
                remaining_symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            results.update(results2)
            failed.extend(failed2)

            # Final validation
            final_validation = validate_data()
    else:
        logger.error("\n[FAIL] Test symbols had issues, please check before continuing")

    logger.info("\n" + "="*80)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Build market regime detector")
    logger.info("2. Train Bayesian probability model")
    logger.info("3. Implement signal generation")
    logger.info("4. Backtest overnight mean reversion strategy")

if __name__ == "__main__":
    main()