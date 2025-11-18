"""
Pair Discovery Script - Find Cointegrated Pairs in Recent Data

This script scans through available symbols to find pairs that are
actually cointegrated in 2023-2024 market conditions.

Methodology:
1. Load universe of liquid ETFs
2. Test all pair combinations for cointegration
3. Rank by cointegration strength (p-value)
4. Output top cointegrated pairs for validation

Usage:
    conda activate fintech
    python backtest_scripts/discover_cointegrated_pairs.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path

from src.backtesting.engine.data_loader import DataLoader
from src.backtesting.utils.pairs import PairsUtils
from src.utils.logger import Logger
from src.config import get_backtest_results_dir

logger = Logger()


def get_etf_universe():
    """
    Define a universe of liquid ETFs to test for cointegration.

    Returns:
        List of ETF symbols grouped by category
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


def load_symbol_data(symbols, start_date, end_date, max_workers=4):
    """
    Load data for multiple symbols and cache in memory.

    Args:
        symbols: List of symbols to load
        start_date: Start date string
        end_date: End date string
        max_workers: Number of parallel workers

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV data
    """
    logger.header("Loading Symbol Data")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Period: {start_date} to {end_date}")

    loader = DataLoader()
    symbol_data = {}
    failed = []

    # Load symbols in batches
    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"\nLoading batch {i//batch_size + 1} ({len(batch)} symbols)...")

        for symbol in batch:
            try:
                # Load single symbol
                df = loader.load_symbols([symbol], start_date, end_date)

                if len(df) > 0:
                    # Extract data for this symbol
                    data = df.xs(symbol, level='symbol')

                    # Resample to daily for cointegration testing
                    data_daily = data.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    symbol_data[symbol] = data_daily
                    logger.success(f"  {symbol}: {len(data_daily)} days")
                else:
                    failed.append(symbol)
                    logger.warning(f"  {symbol}: No data")

            except Exception as e:
                failed.append(symbol)
                logger.error(f"  {symbol}: {str(e)[:50]}")

    logger.info(f"\nLoaded: {len(symbol_data)}/{len(symbols)} symbols")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")

    return symbol_data


def test_pair_cointegration(symbol1, symbol2, data1, data2, significance=0.05):
    """
    Test if a pair is cointegrated.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        data1: DataFrame for symbol1 (daily OHLCV)
        data2: DataFrame for symbol2 (daily OHLCV)
        significance: P-value threshold for cointegration

    Returns:
        Dict with test results or None if test fails
    """
    try:
        # Align data on common dates
        common_index = data1.index.intersection(data2.index)

        if len(common_index) < 252:  # Need at least 1 year
            return None

        close1 = data1.loc[common_index, 'close']
        close2 = data2.loc[common_index, 'close']

        # Test cointegration
        is_coint, p_value, test_stat = PairsUtils.test_cointegration(
            close1, close2, significance_level=significance
        )

        # Calculate correlation for reference
        correlation = close1.corr(close2)

        # Calculate hedge ratio
        hedge_ratio = PairsUtils.calculate_hedge_ratio(close1, close2, method='ols')

        return {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'test_stat': test_stat,
            'correlation': correlation,
            'hedge_ratio': hedge_ratio,
            'days': len(common_index)
        }

    except Exception as e:
        logger.warning(f"Error testing {symbol1}/{symbol2}: {str(e)[:50]}")
        return None


def discover_pairs(symbol_data, significance=0.05, min_pairs=10):
    """
    Test all pair combinations for cointegration.

    Args:
        symbol_data: Dict mapping symbol -> DataFrame
        significance: P-value threshold
        min_pairs: Minimum number of pairs to find

    Returns:
        DataFrame with cointegration test results
    """
    logger.header("Testing Pair Combinations")

    symbols = list(symbol_data.keys())
    total_pairs = len(list(combinations(symbols, 2)))

    logger.info(f"Testing {total_pairs:,} pair combinations")
    logger.info(f"Significance level: p < {significance}")

    results = []
    tested = 0

    for symbol1, symbol2 in combinations(symbols, 2):
        tested += 1

        if tested % 50 == 0:
            logger.info(f"Progress: {tested}/{total_pairs} pairs tested ({tested/total_pairs*100:.1f}%)")

        result = test_pair_cointegration(
            symbol1, symbol2,
            symbol_data[symbol1],
            symbol_data[symbol2],
            significance=significance
        )

        if result:
            results.append(result)

    logger.info(f"\nCompleted: {tested} pairs tested")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        logger.error("No valid pair tests!")
        return df

    # Sort by p-value (lower is better)
    df = df.sort_values('p_value')

    # Count cointegrated pairs
    cointegrated = df[df['is_cointegrated'] == True]

    logger.success(f"\nCointegrated pairs found: {len(cointegrated)} (p < {significance})")

    if len(cointegrated) < min_pairs:
        logger.warning(f"Only found {len(cointegrated)} pairs (wanted {min_pairs})")
        logger.info(f"Showing top {min_pairs} pairs by p-value (may include non-cointegrated)")

    return df


def display_top_pairs(df, top_n=20):
    """
    Display top cointegrated pairs.

    Args:
        df: DataFrame with pair test results
        top_n: Number of top pairs to display
    """
    logger.header(f"TOP {top_n} PAIRS BY COINTEGRATION STRENGTH")

    if len(df) == 0:
        logger.error("No pairs to display!")
        return

    top_pairs = df.head(top_n)

    logger.info(f"{'Rank':<6} {'Pair':<15} {'P-Value':<12} {'Coint?':<8} {'Corr':<8} {'Hedge':<8} {'Days':<6}")
    logger.info("=" * 80)

    for idx, row in enumerate(top_pairs.itertuples(), 1):
        pair_str = f"{row.symbol1}/{row.symbol2}"
        coint_str = "[OK]" if row.is_cointegrated else "[NO]"

        logger.info(
            f"{idx:<6} {pair_str:<15} {row.p_value:<12.6f} {coint_str:<8} "
            f"{row.correlation:<8.3f} {row.hedge_ratio:<8.3f} {row.days:<6}"
        )

    # Summary statistics
    logger.info("\n" + "=" * 80)
    cointegrated = df[df['is_cointegrated'] == True]

    if len(cointegrated) > 0:
        logger.success(f"\nCointegrated Pairs: {len(cointegrated)}")
        logger.info(f"Best p-value: {cointegrated['p_value'].min():.6f}")
        logger.info(f"Average p-value: {cointegrated['p_value'].mean():.6f}")
        logger.info(f"Average correlation: {cointegrated['correlation'].mean():.3f}")
    else:
        logger.warning("\nNo cointegrated pairs found at p < 0.05")
        logger.info("Consider using p < 0.10 threshold or different time period")


def save_results(df, output_path):
    """
    Save pair discovery results to CSV.

    Args:
        df: DataFrame with results
        output_path: Path to output CSV file
    """
    if len(df) == 0:
        logger.warning("No results to save")
        return

    df.to_csv(output_path, index=False)
    logger.success(f"\nResults saved to: {output_path}")
    logger.info(f"Total pairs: {len(df)}")


def main():
    """Main pair discovery routine."""
    logger.header("COINTEGRATED PAIR DISCOVERY")
    logger.info("Scanning for cointegrated pairs in 2023-2024 data")

    # Configuration
    start_date = '2023-01-01'
    end_date = '2024-11-11'
    significance = 0.05

    # Get ETF universe
    logger.info("\n[Step 1] Defining ETF universe...")
    all_symbols, etf_categories = get_etf_universe()

    logger.info(f"Total symbols: {len(all_symbols)}")
    for category, symbols in etf_categories.items():
        logger.info(f"  {category}: {len(symbols)} symbols")

    # Load data
    logger.info("\n[Step 2] Loading historical data...")
    symbol_data = load_symbol_data(all_symbols, start_date, end_date)

    if len(symbol_data) < 2:
        logger.error("Not enough symbols loaded!")
        return

    # Discover pairs
    logger.info("\n[Step 3] Testing pair combinations...")
    results_df = discover_pairs(symbol_data, significance=significance)

    if len(results_df) == 0:
        logger.error("No valid pairs found!")
        return

    # Display results
    logger.info("\n[Step 4] Analyzing results...")
    display_top_pairs(results_df, top_n=20)

    # Save results
    output_path = get_backtest_results_dir() / 'cointegrated_pairs_discovery.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results_df, output_path)

    # Recommendations
    logger.header("RECOMMENDATIONS")

    cointegrated = results_df[results_df['is_cointegrated'] == True]

    if len(cointegrated) >= 4:
        logger.success(f"Found {len(cointegrated)} cointegrated pairs!")
        logger.info("\nTop pairs for validation:")

        for idx, row in enumerate(cointegrated.head(4).itertuples(), 1):
            logger.info(f"  {idx}. {row.symbol1}/{row.symbol2} (p={row.p_value:.6f})")

        logger.info("\nNext steps:")
        logger.info("  1. Update real_pairs_validation.py with these pairs")
        logger.info("  2. Run backtest validation")
        logger.info("  3. Optimize parameters if needed")

    elif len(cointegrated) > 0:
        logger.warning(f"Only found {len(cointegrated)} cointegrated pairs")
        logger.info("\nConsider:")
        logger.info("  - Using p < 0.10 threshold (less strict)")
        logger.info("  - Testing different time period (2017-2019)")
        logger.info("  - Adding more symbols to universe")

    else:
        logger.error("No cointegrated pairs found!")
        logger.info("\nOptions:")
        logger.info("  1. Relax significance to p < 0.10")
        logger.info("  2. Test pre-COVID period (2017-2019)")
        logger.info("  3. Use synthetic cointegrated data for validation")
        logger.info("  4. Switch to correlation-based pair selection")

    logger.header("DISCOVERY COMPLETE")


if __name__ == '__main__':
    main()
