"""
Validate All 50 Cointegrated Pairs

Tests all 50 cointegrated pairs discovered from comprehensive screening
using the FIXED pairs trading strategy (daily signals, not minute signals).
"""

import pandas as pd
import sys
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add project root to path
project_root = Path(__file__).parent.parent

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.utils import logger


def load_discovered_pairs():
    """Load the 50 cointegrated pairs from discovery results."""
    df = pd.read_csv(project_root / "output" / "cointegrated_pairs_discovery.csv")

    # Filter only cointegrated pairs
    cointegrated = df[df['is_cointegrated'] == True].copy()

    logger.info(f"Loaded {len(cointegrated)} cointegrated pairs")

    return cointegrated


def validate_pair(symbol1, symbol2, p_value, start_date, end_date):
    """
    Run backtest on a single pair.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        p_value: Cointegration p-value
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Dictionary with results
    """
    # Create strategy with fixed parameters
    strategy = PairsTrading(
        pair_selection_window=252,
        cointegration_pvalue=0.05,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.5,
        zscore_window=20
    )

    # Create backtest engine with realistic costs
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.0001,  # 0.01%
        slippage=0.001  # 0.10%
    )

    try:
        # Run backtest
        portfolio = engine.run(
            strategy=strategy,
            symbols=[symbol1, symbol2],
            start_date=start_date,
            end_date=end_date
        )

        # Get statistics
        stats = portfolio.stats()

        if stats is None:
            return {
                'pair': f"{symbol1}/{symbol2}",
                'symbol1': symbol1,
                'symbol2': symbol2,
                'p_value': p_value,
                'sharpe': 0,
                'return': 0,
                'annual_return': 0,
                'max_dd': 0,
                'win_rate': 0,
                'trades': 0,
                'final_equity': 0,
                'success': False,
                'error': 'No stats generated'
            }

        return {
            'pair': f"{symbol1}/{symbol2}",
            'symbol1': symbol1,
            'symbol2': symbol2,
            'p_value': p_value,
            'sharpe': stats['Sharpe Ratio'],
            'return': stats['Total Return [%]'],
            'annual_return': stats['Annual Return [%]'],
            'max_dd': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'trades': stats['Total Trades'],
            'final_equity': stats['End Value'],
            'success': True,
            'error': ''
        }

    except Exception as e:
        logger.error(f"Error validating {symbol1}/{symbol2}: {str(e)}")
        return {
            'pair': f"{symbol1}/{symbol2}",
            'symbol1': symbol1,
            'symbol2': symbol2,
            'p_value': p_value,
            'sharpe': 0,
            'return': 0,
            'annual_return': 0,
            'max_dd': 0,
            'win_rate': 0,
            'trades': 0,
            'final_equity': 0,
            'success': False,
            'error': str(e)
        }


def main():
    """Main validation routine."""
    logger.info("ALL 50 PAIRS VALIDATION")
    logger.info(" Testing all 50 cointegrated pairs from comprehensive discovery")
    logger.info(" Strategy: FIXED pairs trading (daily signals)")
    logger.info(" Transaction costs: 0.01% fees + 0.1% slippage")
    logger.info("")

    # Test period
    start_date = "2023-01-01"
    end_date = "2024-11-11"
    logger.info(f"Test Period: {start_date} to {end_date}")

    # Load pairs
    pairs_df = load_discovered_pairs()
    logger.info(f" Pairs to validate: {len(pairs_df)}\n")

    # Validate all pairs
    results = []
    for idx, row in pairs_df.iterrows():
        logger.info(f"\n[{idx+1}/{len(pairs_df)}] Testing {row['symbol1']}/{row['symbol2']} (p={row['p_value']:.6f})")
        result = validate_pair(row['symbol1'], row['symbol2'], row['p_value'], start_date, end_date)
        results.append(result)

        # Log result
        if result['success']:
            logger.info(f"  Return: {result['return']:.2f}%, Sharpe: {result['sharpe']:.3f}, Trades: {result['trades']}")
        else:
            logger.error(f"  FAILED: {result['error']}")

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    df = df.sort_values('sharpe', ascending=False)

    # Save results
    output_path = project_root / "output" / "all_50_pairs_validation.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n[+] Results saved to: {output_path}")

    # Print summary
    logger.info("\nVALIDATION SUMMARY")
    logger.info(f" Total Pairs: {len(df)}")
    logger.info(f" Successful Tests: {df['success'].sum()}/{len(df)}")
    logger.info(f" Failed Tests: {(~df['success']).sum()}/{len(df)}")
    logger.info("")

    # Filter successful pairs
    successful = df[df['success']]

    if len(successful) > 0:
        # Categorize by performance
        profitable = successful[successful['return'] > 0]
        production_ready = successful[successful['sharpe'] >= 0.8]
        needs_optimization = successful[(successful['sharpe'] >= 0.3) & (successful['sharpe'] < 0.8)]
        marginally_profitable = successful[(successful['return'] > 0) & (successful['sharpe'] < 0.3)]
        not_viable = successful[successful['return'] <= 0]

        logger.info("PERFORMANCE CATEGORIES")
        logger.info(f" Production Ready (Sharpe >= 0.8): {len(production_ready)}")
        logger.info(f" Needs Optimization (0.3 <= Sharpe < 0.8): {len(needs_optimization)}")
        logger.info(f" Marginally Profitable (Return > 0, Sharpe < 0.3): {len(marginally_profitable)}")
        logger.info(f" Not Profitable (Return <= 0): {len(not_viable)}")
        logger.info("")

        # Show profitable pairs
        logger.info(f"PROFITABLE PAIRS ({len(profitable)}):")
        for _, row in profitable.head(20).iterrows():
            logger.info(f"  {row['pair']}: Return {row['return']:.2f}%, Sharpe {row['sharpe']:.3f}, p-value {row['p_value']:.6f}")

        if len(production_ready) > 0:
            logger.info("\n[+] PRODUCTION READY PAIRS:")
            for _, row in production_ready.iterrows():
                logger.info(f"   - {row['pair']}: Sharpe {row['sharpe']:.3f}, Return {row['return']:.2f}%")
        else:
            logger.info("\n[~] No production-ready pairs (Sharpe >= 0.8)")
            logger.info(" Pairs need parameter optimization")

        # Statistics
        logger.info(f"\nAVERAGE METRICS (Successful Pairs):")
        logger.info(f"  Sharpe Ratio: {successful['sharpe'].mean():.3f}")
        logger.info(f"  Total Return: {successful['return'].mean():.2f}%")
        logger.info(f"  Annual Return: {successful['annual_return'].mean():.2f}%")
        logger.info(f"  Max Drawdown: {successful['max_dd'].mean():.2f}%")
        logger.info(f"  Win Rate: {successful['win_rate'].mean():.2f}%")
        logger.info(f"  Trades per Pair: {successful['trades'].mean():.1f}")

    else:
        logger.info("[X] No successful backtests")

    logger.info("\n[+] VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
