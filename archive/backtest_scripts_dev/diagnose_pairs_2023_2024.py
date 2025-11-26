"""
Diagnostic script for pairs trading - testing 2023-2024 period with minute data.

This script will test if traditional pairs have regained cointegration
after COVID market disruptions stabilized.

Tests:
1. Check cointegration test results on MINUTE data (not daily)
2. Analyze spread characteristics
3. Examine z-score distribution
4. Test signal generation
"""

import sys
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
import pandas as pd
import numpy as np

# Add src to path

from src.backtesting.engine.data_loader import DataLoader
from src.backtesting.utils.pairs import PairsUtils
from src.utils.logger import Logger

logger = Logger()


def diagnose_pair(symbol1: str, symbol2: str, start_date: str, end_date: str):
    """
    Diagnose why a pair isn't generating trades using MINUTE data.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        start_date: Start date for analysis
        end_date: End date for analysis
    """
    logger.header(f"DIAGNOSING PAIR: {symbol1} / {symbol2}")
    logger.info(f"Period: {start_date} to {end_date}")

    # Load data
    loader = DataLoader()

    logger.info("\n[Step 1] Loading minute data...")
    # Load both symbols together
    df = loader.load_symbols([symbol1, symbol2], start_date, end_date)

    # Split by symbol
    data1 = df.xs(symbol1, level='symbol')
    data2 = df.xs(symbol2, level='symbol')

    logger.success(f"Loaded {len(data1):,} minute bars for {symbol1}")
    logger.success(f"Loaded {len(data2):,} minute bars for {symbol2}")

    # Already synchronized since loaded together
    common_index = data1.index.intersection(data2.index)
    data1 = data1.loc[common_index]
    data2 = data2.loc[common_index]

    logger.info(f"Synchronized: {len(common_index):,} common timestamps")

    # Use MINUTE data for cointegration (resample to hourly to reduce noise)
    logger.info("\n[Step 2] Resampling to hourly bars for cointegration test...")
    data1_hourly = data1.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    data2_hourly = data2.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.success(f"Hourly bars - {symbol1}: {len(data1_hourly)} hours")
    logger.success(f"Hourly bars - {symbol2}: {len(data2_hourly)} hours")

    close1 = data1_hourly['close']
    close2 = data2_hourly['close']

    # Test cointegration on full period
    logger.info("\n[Step 3] Testing cointegration (full period - HOURLY data)...")
    try:
        is_coint, p_value, test_stat = PairsUtils.test_cointegration(
            close1, close2, significance_level=0.05
        )

        if is_coint:
            logger.success(f"Cointegrated: p-value = {p_value:.6f} (< 0.05)")
        else:
            logger.error(f"NOT cointegrated: p-value = {p_value:.6f} (>= 0.05)")
            logger.info(f"Test statistic: {test_stat:.6f}")

            # Try with more lenient threshold
            logger.info("\n[Step 3b] Testing with p-value < 0.10 threshold...")
            if p_value < 0.10:
                logger.warning(f"Marginally cointegrated: p-value = {p_value:.6f} (< 0.10)")
            else:
                logger.error(f"Still not cointegrated with p < 0.10")
                return False

    except Exception as e:
        logger.error(f"Cointegration test failed: {e}")
        return False

    # Test cointegration on first ~1000 hours (strategy uses 252 days = ~1800 trading hours)
    logger.info("\n[Step 4] Testing cointegration (first 1000 hours - training window)...")
    if len(close1) >= 1000:
        close1_train = close1.iloc[:1000]
        close2_train = close2.iloc[:1000]

        is_coint_train, p_value_train, test_stat_train = PairsUtils.test_cointegration(
            close1_train, close2_train, significance_level=0.05
        )

        if is_coint_train:
            logger.success(f"Cointegrated (training): p-value = {p_value_train:.6f}")
        else:
            logger.error(f"NOT cointegrated (training): p-value = {p_value_train:.6f}")
            logger.warning("Strategy won't generate signals if training period fails cointegration!")
            if p_value_train >= 0.10:
                return False
    else:
        logger.warning(f"Not enough data for 1000-hour window (have {len(close1)} hours)")

    # Calculate hedge ratio
    logger.info("\n[Step 5] Calculating hedge ratio...")
    hedge_ratio = PairsUtils.calculate_hedge_ratio(close1, close2, method='ols')
    logger.success(f"Hedge ratio: {hedge_ratio:.6f}")

    # Calculate spread
    logger.info("\n[Step 6] Calculating spread...")
    spread = PairsUtils.calculate_spread(close1, close2, hedge_ratio)
    logger.success(f"Spread statistics:")
    logger.info(f"  Mean: {spread.mean():.4f}")
    logger.info(f"  Std:  {spread.std():.4f}")
    logger.info(f"  Min:  {spread.min():.4f}")
    logger.info(f"  Max:  {spread.max():.4f}")

    # Calculate z-score
    logger.info("\n[Step 7] Calculating z-score (window=20)...")
    zscore = PairsUtils.spread_zscore(spread, window=20)

    # Remove NaN values
    zscore_clean = zscore.dropna()

    if len(zscore_clean) == 0:
        logger.error("All z-scores are NaN!")
        return False

    logger.success(f"Z-score statistics (non-NaN values):")
    logger.info(f"  Count: {len(zscore_clean):,}")
    logger.info(f"  Mean:  {zscore_clean.mean():.4f}")
    logger.info(f"  Std:   {zscore_clean.std():.4f}")
    logger.info(f"  Min:   {zscore_clean.min():.4f}")
    logger.info(f"  Max:   {zscore_clean.max():.4f}")

    # Check threshold crossings with default parameters
    logger.info("\n[Step 8] Checking signal thresholds...")

    entry_threshold = 2.0
    exit_threshold = 0.5
    stop_loss_threshold = 3.5

    logger.info(f"Entry threshold: +/- {entry_threshold}")
    logger.info(f"Exit threshold: +/- {exit_threshold}")
    logger.info(f"Stop loss threshold: +/- {stop_loss_threshold}")

    # Count threshold crossings
    long_entries = (zscore_clean < -entry_threshold).sum()
    short_entries = (zscore_clean > entry_threshold).sum()
    extreme_zscores = (abs(zscore_clean) > entry_threshold).sum()

    logger.info(f"\nThreshold crossings:")
    logger.info(f"  Long entries (z < -2.0): {long_entries}")
    logger.info(f"  Short entries (z > 2.0): {short_entries}")
    logger.info(f"  Total extreme z-scores: {extreme_zscores}")

    if extreme_zscores == 0:
        logger.warning("\nNO THRESHOLD CROSSINGS with entry_zscore=2.0!")
        logger.info("Z-score never exceeded +/- 2.0")

        # Test with lower thresholds
        logger.info("\n[Step 9] Testing with lower thresholds...")
        for test_threshold in [1.5, 1.0, 0.5]:
            crossings = (abs(zscore_clean) > test_threshold).sum()
            logger.info(f"  Crossings with threshold={test_threshold}: {crossings}")

        return False
    else:
        logger.success(f"\nZ-score exceeded threshold {extreme_zscores} times")

        # Generate actual signals
        logger.info("\n[Step 9] Generating signals...")
        long_e, long_x, short_e, short_x = PairsUtils.generate_pairs_signals(
            zscore,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss_threshold
        )

        long_entries_count = long_e.sum()
        short_entries_count = short_e.sum()
        long_exits_count = long_x.sum()
        short_exits_count = short_x.sum()

        logger.info(f"Signal counts:")
        logger.info(f"  Long entries: {long_entries_count}")
        logger.info(f"  Long exits: {long_exits_count}")
        logger.info(f"  Short entries: {short_entries_count}")
        logger.info(f"  Short exits: {short_exits_count}")

        if long_entries_count > 0 or short_entries_count > 0:
            logger.success("Signals generated successfully!")
            return True
        else:
            logger.error("No entry signals generated despite threshold crossings!")
            return False


def main():
    """Main diagnostic routine for 2023-2024 period."""
    logger.header("PAIRS TRADING DIAGNOSTIC - 2023-2024 PERIOD")
    logger.info("Testing if pairs have regained cointegration post-COVID")

    # Test pairs
    test_pairs = [
        ('SPY', 'IWM'),
        ('GLD', 'GDX'),
        ('XLE', 'XLU'),
        ('EWA', 'EWC')
    ]

    # Test recent period (post-COVID stabilization)
    start_date = '2023-01-01'
    end_date = '2024-11-11'

    logger.info(f"Test period: {start_date} to {end_date}")
    logger.info(f"Using HOURLY bars for cointegration test\n")

    results = []
    for symbol1, symbol2 in test_pairs:
        print("\n" + "="*80)
        success = diagnose_pair(symbol1, symbol2, start_date, end_date)
        results.append((f"{symbol1}/{symbol2}", success))
        print()

    # Summary
    logger.header("DIAGNOSTIC SUMMARY - 2023-2024")
    for pair, success in results:
        status = "[OK] Working" if success else "[X] Issue Found"
        logger.info(f"{status}: {pair}")

    successful = sum(1 for _, s in results if s)
    logger.info(f"\nPairs working: {successful}/{len(results)}")

    if successful > 0:
        logger.success(f"\n{successful} pairs are cointegrated in 2023-2024!")
        logger.info("These pairs can be used for validation backtests")
    else:
        logger.warning("\nNo pairs cointegrated in 2023-2024 either")
        logger.info("May need to search for different pair combinations")


if __name__ == '__main__':
    main()
