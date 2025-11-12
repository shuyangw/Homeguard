"""
Stable Pair Discovery - Find Cointegrated Pairs with Subperiod Stability

This script addresses the walk-forward validation failure by requiring pairs to
be cointegrated in MULTIPLE time periods, not just the combined window.

Key Improvements:
1. Tests cointegration on FULL period (2023-2024)
2. Tests cointegration on 2023 ONLY
3. Tests cointegration on 2024 ONLY
4. Only returns pairs that pass ALL THREE tests
5. This filters out unstable relationships that only appear cointegrated by chance

Methodology:
- Prevents lookahead bias by validating stability
- Identifies regime-stable pairs suitable for production
- Filters out UVXY pairs that are regime-dependent

Usage:
    conda activate fintech
    python backtest_scripts/discover_stable_pairs.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine.data_loader import DataLoader
from src.backtesting.utils.pairs import PairsUtils
from src.utils.logger import Logger

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


def load_symbol_data_multi_period(symbols, start_date, end_date):
    """
    Load data for multiple symbols in multiple time periods.

    Args:
        symbols: List of symbols to load
        start_date: Start date string
        end_date: End date string

    Returns:
        Dict with keys 'full', '2023', '2024' -> Dict[symbol -> DataFrame]
    """
    logger.header("Loading Symbol Data for Multiple Periods")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Full period: {start_date} to {end_date}")
    logger.info(f"Subperiod 1: 2023-01-01 to 2023-12-31")
    logger.info(f"Subperiod 2: 2024-01-01 to {end_date}")

    loader = DataLoader()

    # Define time periods
    periods = {
        'full': (start_date, end_date),
        '2023': ('2023-01-01', '2023-12-31'),
        '2024': ('2024-01-01', end_date)
    }

    all_data = {}

    for period_name, (period_start, period_end) in periods.items():
        logger.info(f"\n[Loading {period_name.upper()} Period: {period_start} to {period_end}]")

        symbol_data = {}
        failed = []

        for symbol in symbols:
            try:
                # Load single symbol
                df = loader.load_symbols([symbol], period_start, period_end)

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

        logger.info(f"\nLoaded {period_name}: {len(symbol_data)}/{len(symbols)} symbols")
        if failed:
            logger.warning(f"Failed: {', '.join(failed)}")

        all_data[period_name] = symbol_data

    return all_data


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

        if len(common_index) < 100:  # Need at least 100 days
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


def discover_stable_pairs(all_period_data, significance=0.05):
    """
    Test all pair combinations for cointegration STABILITY.

    A pair is considered stable if it passes cointegration test on:
    - Full period (2023-2024)
    - 2023 only
    - 2024 only

    Args:
        all_period_data: Dict with 'full', '2023', '2024' -> symbol_data
        significance: P-value threshold

    Returns:
        DataFrame with stable pairs only
    """
    logger.header("Testing Pair Combinations for STABILITY")

    # Get common symbols across all periods
    symbols_full = set(all_period_data['full'].keys())
    symbols_2023 = set(all_period_data['2023'].keys())
    symbols_2024 = set(all_period_data['2024'].keys())

    common_symbols = list(symbols_full & symbols_2023 & symbols_2024)

    logger.info(f"Symbols available in all periods: {len(common_symbols)}")

    total_pairs = len(list(combinations(common_symbols, 2)))

    logger.info(f"Testing {total_pairs:,} pair combinations across 3 periods")
    logger.info(f"Significance level: p < {significance}")
    logger.info(f"Requirement: MUST pass cointegration in ALL 3 periods\n")

    results = []
    tested = 0
    stable_count = 0

    for symbol1, symbol2 in combinations(common_symbols, 2):
        tested += 1

        if tested % 50 == 0:
            logger.info(f"Progress: {tested}/{total_pairs} pairs tested ({tested/total_pairs*100:.1f}%) | Stable: {stable_count}")

        # Test on all three periods
        test_results = {}

        for period_name in ['full', '2023', '2024']:
            result = test_pair_cointegration(
                symbol1, symbol2,
                all_period_data[period_name][symbol1],
                all_period_data[period_name][symbol2],
                significance=significance
            )

            if result is None:
                break  # Skip this pair if any period fails

            test_results[period_name] = result

        # Check if we have results for all periods
        if len(test_results) != 3:
            continue

        # Check if cointegrated in ALL periods
        is_stable = (
            test_results['full']['is_cointegrated'] and
            test_results['2023']['is_cointegrated'] and
            test_results['2024']['is_cointegrated']
        )

        if is_stable:
            stable_count += 1
            logger.success(f"  STABLE PAIR FOUND: {symbol1}/{symbol2} "
                         f"(p_full={test_results['full']['p_value']:.4f}, "
                         f"p_2023={test_results['2023']['p_value']:.4f}, "
                         f"p_2024={test_results['2024']['p_value']:.4f})")

        # Store comprehensive results
        results.append({
            'symbol1': symbol1,
            'symbol2': symbol2,
            'is_stable': is_stable,

            # Full period
            'full_cointegrated': test_results['full']['is_cointegrated'],
            'full_p_value': test_results['full']['p_value'],
            'full_test_stat': test_results['full']['test_stat'],
            'full_correlation': test_results['full']['correlation'],
            'full_hedge_ratio': test_results['full']['hedge_ratio'],
            'full_days': test_results['full']['days'],

            # 2023
            '2023_cointegrated': test_results['2023']['is_cointegrated'],
            '2023_p_value': test_results['2023']['p_value'],
            '2023_test_stat': test_results['2023']['test_stat'],
            '2023_correlation': test_results['2023']['correlation'],
            '2023_hedge_ratio': test_results['2023']['hedge_ratio'],
            '2023_days': test_results['2023']['days'],

            # 2024
            '2024_cointegrated': test_results['2024']['is_cointegrated'],
            '2024_p_value': test_results['2024']['p_value'],
            '2024_test_stat': test_results['2024']['test_stat'],
            '2024_correlation': test_results['2024']['correlation'],
            '2024_hedge_ratio': test_results['2024']['hedge_ratio'],
            '2024_days': test_results['2024']['days'],

            # Stability metrics
            'avg_p_value': np.mean([test_results[p]['p_value'] for p in ['full', '2023', '2024']]),
            'max_p_value': np.max([test_results[p]['p_value'] for p in ['full', '2023', '2024']]),
            'min_p_value': np.min([test_results[p]['p_value'] for p in ['full', '2023', '2024']]),
        })

    logger.info(f"\nCompleted: {tested} pairs tested across 3 periods")
    logger.success(f"STABLE PAIRS FOUND: {stable_count}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        logger.error("No valid pair tests!")
        return df

    # Sort by average p-value (lower is better)
    df = df.sort_values('avg_p_value')

    return df


def display_stable_pairs(df, top_n=20):
    """
    Display stable cointegrated pairs.

    Args:
        df: DataFrame with pair test results
        top_n: Number of top pairs to display
    """
    logger.header("STABLE COINTEGRATED PAIRS")

    if len(df) == 0:
        logger.error("No pairs to display!")
        return

    stable_pairs = df[df['is_stable'] == True]

    if len(stable_pairs) == 0:
        logger.error("No stable pairs found!")
        logger.info("\nTesting pairs that were cointegrated in FULL period:")
        full_coint = df[df['full_cointegrated'] == True].head(10)

        if len(full_coint) > 0:
            logger.info(f"\n{'Pair':<15} {'Full':<8} {'2023':<8} {'2024':<8} {'Avg P-Value':<12}")
            logger.info("=" * 70)

            for row in full_coint.itertuples():
                pair_str = f"{row.symbol1}/{row.symbol2}"
                full_str = "✓" if row.full_cointegrated else "✗"
                y2023_str = "✓" if getattr(row, '_2023_cointegrated') else "✗"
                y2024_str = "✓" if getattr(row, '_2024_cointegrated') else "✗"

                logger.warning(
                    f"{pair_str:<15} {full_str:<8} {y2023_str:<8} {y2024_str:<8} {row.avg_p_value:<12.6f}"
                )

            logger.info("\nThese pairs are NOT stable (failed subperiod validation)")

        return

    # Display stable pairs
    logger.success(f"Found {len(stable_pairs)} stable pairs!")
    logger.info(f"\n{'Rank':<6} {'Pair':<15} {'Avg P':<10} {'Full P':<10} {'2023 P':<10} {'2024 P':<10} {'Corr':<8}")
    logger.info("=" * 90)

    display_count = min(top_n, len(stable_pairs))

    for idx, row in enumerate(stable_pairs.head(display_count).itertuples(), 1):
        pair_str = f"{row.symbol1}/{row.symbol2}"

        logger.info(
            f"{idx:<6} {pair_str:<15} {row.avg_p_value:<10.6f} "
            f"{row.full_p_value:<10.6f} {getattr(row, '_2023_p_value'):<10.6f} "
            f"{getattr(row, '_2024_p_value'):<10.6f} {row.full_correlation:<8.3f}"
        )

    # Summary statistics
    logger.info("\n" + "=" * 90)
    logger.success(f"\nTotal Stable Pairs: {len(stable_pairs)}")
    logger.info(f"Average P-Value: {stable_pairs['avg_p_value'].mean():.6f}")
    logger.info(f"Best P-Value: {stable_pairs['min_p_value'].min():.6f}")
    logger.info(f"Worst P-Value (but still stable): {stable_pairs['max_p_value'].max():.6f}")
    logger.info(f"Average Correlation: {stable_pairs['full_correlation'].mean():.3f}")

    # Analyze by category
    analyze_pair_categories(stable_pairs)


def analyze_pair_categories(df):
    """
    Analyze stable pairs by category (sector, commodity, etc.).

    Args:
        df: DataFrame with stable pairs
    """
    logger.header("PAIR CATEGORY ANALYSIS")

    # Define categories
    volatility_symbols = {'VXX', 'UVXY', 'SVXY'}
    commodity_symbols = {'GLD', 'SLV', 'USO', 'UNG', 'GDX', 'SIL'}
    sector_symbols = {'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLU', 'XLP', 'XLY', 'XLB'}
    international_symbols = {'EWA', 'EWC', 'EWG', 'EWJ', 'EWU', 'EWZ', 'FXI'}

    # Categorize pairs
    categories = {
        'Volatility Pairs': [],
        'Commodity Pairs': [],
        'Sector Pairs': [],
        'International Pairs': [],
        'Cross-Category': []
    }

    for row in df.itertuples():
        s1, s2 = row.symbol1, row.symbol2
        pair = f"{s1}/{s2}"

        # Check categories
        has_vol = s1 in volatility_symbols or s2 in volatility_symbols
        has_commodity = s1 in commodity_symbols or s2 in commodity_symbols
        has_sector = s1 in sector_symbols or s2 in sector_symbols
        has_intl = s1 in international_symbols or s2 in international_symbols

        if has_vol:
            categories['Volatility Pairs'].append(pair)
        elif s1 in commodity_symbols and s2 in commodity_symbols:
            categories['Commodity Pairs'].append(pair)
        elif s1 in sector_symbols and s2 in sector_symbols:
            categories['Sector Pairs'].append(pair)
        elif s1 in international_symbols and s2 in international_symbols:
            categories['International Pairs'].append(pair)
        else:
            categories['Cross-Category'].append(pair)

    # Display categories
    for category, pairs in categories.items():
        if len(pairs) > 0:
            logger.info(f"\n{category}: {len(pairs)}")
            for pair in pairs:
                logger.info(f"  - {pair}")

    # Warning about volatility pairs
    if len(categories['Volatility Pairs']) > 0:
        logger.warning(f"\n⚠️  WARNING: {len(categories['Volatility Pairs'])} pairs contain volatility ETFs (VXX/UVXY/SVXY)")
        logger.warning("   These passed stability tests but may still be regime-dependent")
        logger.warning("   Consider implementing regime detection for these pairs")


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
    logger.info(f"Total pairs tested: {len(df)}")

    stable = df[df['is_stable'] == True]
    logger.info(f"Stable pairs: {len(stable)}")


def main():
    """Main stable pair discovery routine."""
    logger.header("STABLE PAIR DISCOVERY")
    logger.info("Finding pairs cointegrated across MULTIPLE time periods")
    logger.info("This addresses walk-forward validation failure\n")

    # Configuration
    start_date = '2023-01-01'
    end_date = '2024-11-11'
    significance = 0.05

    # Get ETF universe
    logger.info("[Step 1] Defining ETF universe...")
    all_symbols, etf_categories = get_etf_universe()

    logger.info(f"Total symbols: {len(all_symbols)}")
    for category, symbols in etf_categories.items():
        logger.info(f"  {category}: {len(symbols)} symbols")

    # Load data for multiple periods
    logger.info("\n[Step 2] Loading historical data for 3 periods...")
    all_period_data = load_symbol_data_multi_period(all_symbols, start_date, end_date)

    if len(all_period_data['full']) < 2:
        logger.error("Not enough symbols loaded!")
        return

    # Discover stable pairs
    logger.info("\n[Step 3] Testing pair stability across periods...")
    results_df = discover_stable_pairs(all_period_data, significance=significance)

    if len(results_df) == 0:
        logger.error("No valid pairs found!")
        return

    # Display results
    logger.info("\n[Step 4] Analyzing stable pairs...")
    display_stable_pairs(results_df, top_n=20)

    # Save results
    output_path = Path(__file__).parent.parent / 'output' / 'stable_pairs_discovery.csv'
    output_path.parent.mkdir(exist_ok=True)
    save_results(results_df, output_path)

    # Recommendations
    logger.header("RECOMMENDATIONS")

    stable_pairs = results_df[results_df['is_stable'] == True]

    if len(stable_pairs) >= 5:
        logger.success(f"✅ Found {len(stable_pairs)} stable pairs!")
        logger.info("\nTop stable pairs for production:")

        for idx, row in enumerate(stable_pairs.head(5).itertuples(), 1):
            logger.info(f"  {idx}. {row.symbol1}/{row.symbol2} "
                       f"(avg_p={row.avg_p_value:.4f}, max_p={row.max_p_value:.4f})")

        logger.info("\n✅ Next steps:")
        logger.info("  1. Run backtest validation on these stable pairs")
        logger.info("  2. Parameter optimization")
        logger.info("  3. Build multi-pair portfolio")
        logger.info("  4. Walk-forward validation will likely PASS for these pairs")

    elif len(stable_pairs) > 0:
        logger.warning(f"⚠️  Only found {len(stable_pairs)} stable pairs")
        logger.info("\nStable pairs:")
        for idx, row in enumerate(stable_pairs.itertuples(), 1):
            logger.info(f"  {idx}. {row.symbol1}/{row.symbol2}")

        logger.info("\n⚠️  Consider:")
        logger.info("  - Using p < 0.10 threshold (less strict)")
        logger.info("  - Testing longer time periods (2020-2024)")
        logger.info("  - Implementing Kalman filter for unstable pairs")

    else:
        logger.error("❌ No stable pairs found!")
        logger.info("\nThis confirms the walk-forward validation failure:")
        logger.info("  - Pairs that appeared cointegrated on full period")
        logger.info("  - Failed cointegration on subperiods")
        logger.info("  - This indicates regime-dependent relationships")

        logger.info("\n⚠️  Options:")
        logger.info("  1. Relax significance to p < 0.10")
        logger.info("  2. Use Kalman filter for dynamic hedge ratios")
        logger.info("  3. Implement regime detection")
        logger.info("  4. Test pre-COVID period (2017-2019)")

    logger.header("DISCOVERY COMPLETE")


if __name__ == '__main__':
    main()
