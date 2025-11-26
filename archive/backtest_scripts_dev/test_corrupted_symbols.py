"""
Test script to verify all 9 previously corrupted symbols are now working.

Tests:
1. File exists
2. Has DatetimeIndex (not RangeIndex)
3. Timezone-naive (not timezone-aware)
4. Has required OHLCV columns
5. Can filter by date range
6. Has sufficient data (>500 bars)
"""

import sys
import os

import pandas as pd
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
from src.utils.logger import logger

DATA_DIR = Path('data/leveraged_etfs')

# The 9 symbols that were corrupted
CORRUPTED_SYMBOLS = [
    'USD', 'UYG', 'DFEN', 'WEBL', 'UCO',
    'NAIL', 'ERX', 'RETL', 'CUT'
]

# Also test a few known-good symbols for comparison
CONTROL_SYMBOLS = ['TQQQ', 'UPRO', 'SOXL']


def test_symbol(symbol, is_control=False):
    """Test a single symbol."""
    file_path = DATA_DIR / f'{symbol}_1d.parquet'

    results = {
        'symbol': symbol,
        'type': 'CONTROL' if is_control else 'FIXED',
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }

    # Test 1: File exists
    if not file_path.exists():
        results['errors'].append('File not found')
        results['tests_failed'] += 1
        return results
    results['tests_passed'] += 1

    try:
        # Load data
        df = pd.read_parquet(file_path)

        # Test 2: Has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            results['errors'].append(f'Wrong index type: {type(df.index).__name__}')
            results['tests_failed'] += 1
        else:
            results['tests_passed'] += 1

        # Test 3: Timezone-naive
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                results['errors'].append(f'Timezone-aware: {df.index.tz}')
                results['tests_failed'] += 1
            else:
                results['tests_passed'] += 1

        # Test 4: Has required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['errors'].append(f'Missing columns: {missing_cols}')
            results['tests_failed'] += 1
        else:
            results['tests_passed'] += 1

        # Test 5: Can filter by date range
        try:
            train_start = pd.Timestamp('2015-11-16')
            train_end = pd.Timestamp('2017-11-16')
            filtered = df.loc[train_start:train_end]
            results['tests_passed'] += 1
        except Exception as e:
            results['errors'].append(f'Date filtering failed: {str(e)}')
            results['tests_failed'] += 1

        # Test 6: Sufficient data
        if len(df) < 500:
            results['errors'].append(f'Insufficient data: {len(df)} bars')
            results['tests_failed'] += 1
        else:
            results['tests_passed'] += 1

        # Additional info
        results['bars'] = len(df)
        results['date_range'] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"

    except Exception as e:
        results['errors'].append(f'Load error: {str(e)}')
        results['tests_failed'] += 1

    return results


def main():
    """Test all symbols."""
    logger.info("="*80)
    logger.info("CORRUPTED SYMBOLS RE-TEST")
    logger.info("="*80)
    logger.info(f"\nTesting 9 previously corrupted symbols + 3 control symbols")
    logger.info(f"Tests per symbol: 6 (existence, index type, timezone, columns, filtering, data size)")

    # Test corrupted symbols
    logger.info("\n" + "="*80)
    logger.info("TESTING PREVIOUSLY CORRUPTED SYMBOLS (9)")
    logger.info("="*80)

    corrupted_results = []
    for symbol in CORRUPTED_SYMBOLS:
        result = test_symbol(symbol, is_control=False)
        corrupted_results.append(result)

        if result['tests_failed'] == 0:
            logger.success(f"[OK] {symbol}: ALL TESTS PASSED ({result['tests_passed']}/6)")
            logger.info(f"    {result['bars']} bars, {result['date_range']}")
        else:
            logger.error(f"[FAIL] {symbol}: {result['tests_failed']} TESTS FAILED")
            for error in result['errors']:
                logger.error(f"    - {error}")

    # Test control symbols
    logger.info("\n" + "="*80)
    logger.info("TESTING CONTROL SYMBOLS (3)")
    logger.info("="*80)

    control_results = []
    for symbol in CONTROL_SYMBOLS:
        result = test_symbol(symbol, is_control=True)
        control_results.append(result)

        if result['tests_failed'] == 0:
            logger.success(f"[OK] {symbol}: ALL TESTS PASSED ({result['tests_passed']}/6)")
            logger.info(f"    {result['bars']} bars, {result['date_range']}")
        else:
            logger.error(f"[FAIL] {symbol}: {result['tests_failed']} TESTS FAILED")
            for error in result['errors']:
                logger.error(f"    - {error}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    corrupted_passed = sum(1 for r in corrupted_results if r['tests_failed'] == 0)
    control_passed = sum(1 for r in control_results if r['tests_failed'] == 0)

    logger.info(f"\nPreviously Corrupted Symbols: {corrupted_passed}/{len(CORRUPTED_SYMBOLS)} passing all tests")
    logger.info(f"Control Symbols: {control_passed}/{len(CONTROL_SYMBOLS)} passing all tests")

    total_tests = sum(r['tests_passed'] for r in corrupted_results + control_results)
    total_failed = sum(r['tests_failed'] for r in corrupted_results + control_results)

    logger.info(f"\nTotal Tests Run: {total_tests + total_failed}")
    logger.info(f"Tests Passed: {total_tests}")
    logger.info(f"Tests Failed: {total_failed}")

    if corrupted_passed == len(CORRUPTED_SYMBOLS):
        logger.success(f"\n[SUCCESS] All 9 corrupted symbols are now FIXED and working!")
        logger.success("Data quality issue RESOLVED")
    else:
        logger.error(f"\n[WARNING] {len(CORRUPTED_SYMBOLS) - corrupted_passed} symbols still have issues")
        logger.error("Additional fixes may be needed")

    # Detailed comparison
    logger.info("\n" + "="*80)
    logger.info("DETAILED COMPARISON")
    logger.info("="*80)

    logger.info("\nPreviously Corrupted Symbols:")
    logger.info(f"{'Symbol':<10} {'Status':<10} {'Tests':<15} {'Data':<20} {'Date Range':<30}")
    logger.info("-" * 85)
    for r in corrupted_results:
        status = "PASS" if r['tests_failed'] == 0 else "FAIL"
        tests = f"{r['tests_passed']}/{r['tests_passed'] + r['tests_failed']}"
        bars = r.get('bars', 'N/A')
        date_range = r.get('date_range', 'N/A')
        logger.info(f"{r['symbol']:<10} {status:<10} {tests:<15} {bars:<20} {date_range:<30}")

    logger.info("\nControl Symbols (Should All Pass):")
    logger.info(f"{'Symbol':<10} {'Status':<10} {'Tests':<15} {'Data':<20} {'Date Range':<30}")
    logger.info("-" * 85)
    for r in control_results:
        status = "PASS" if r['tests_failed'] == 0 else "FAIL"
        tests = f"{r['tests_passed']}/{r['tests_passed'] + r['tests_failed']}"
        bars = r.get('bars', 'N/A')
        date_range = r.get('date_range', 'N/A')
        logger.info(f"{r['symbol']:<10} {status:<10} {tests:<15} {bars:<20} {date_range:<30}")

    return corrupted_passed == len(CORRUPTED_SYMBOLS) and control_passed == len(CONTROL_SYMBOLS)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
