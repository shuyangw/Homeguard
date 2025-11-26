"""
Validate Top 10 Cointegrated Pairs

Tests the 10 strongest cointegrated pairs discovered from comprehensive screening
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

# Top 10 pairs from comprehensive discovery (2023-2024)
TOP_PAIRS = [
    ("SPY", "VOO", "S&P 500 / S&P 500 VOO", 0.000000),
    ("XLV", "EWJ", "Healthcare / Japan", 0.000813),
    ("XLV", "EWG", "Healthcare / Germany", 0.001774),
    ("XLP", "EWU", "Consumer Staples / UK", 0.003232),
    ("XLP", "EWG", "Consumer Staples / Germany", 0.005046),
    ("XLV", "EWC", "Healthcare / Canada", 0.007018),
    ("XLV", "XLB", "Healthcare / Materials", 0.007029),
    ("XLP", "EWA", "Consumer Staples / Australia", 0.007444),
    ("XLU", "GDX", "Utilities / Gold Miners", 0.009218),
    ("QQQ", "XLK", "Nasdaq 100 / Tech Sector", 0.009558),
]


def validate_pair(symbol1, symbol2, description, p_value, start_date, end_date):
    """
    Run backtest on a single pair.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        description: Pair description
        p_value: Cointegration p-value
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Validating: {symbol1} / {symbol2}")
    logger.info(f" Description: {description}")
    logger.info(f" Cointegration: p-value = {p_value:.6f}")
    logger.info(f" Period: {start_date} to {end_date}")
    logger.info(f" Fees: 0.01%, Slippage: 0.10%")

    # Create strategy with fixed parameters
    strategy = PairsTrading(
        pair_selection_window=252,      # Use 252 days for cointegration test
        cointegration_pvalue=0.05,      # 5% significance level
        entry_zscore=2.0,                # Enter when spread is 2 std devs away
        exit_zscore=0.5,                 # Exit when spread normalizes to 0.5 std devs
        stop_loss_zscore=3.5,            # Stop loss at 3.5 std devs
        zscore_window=20                 # 20-day rolling window for Z-score
    )

    # Create backtest engine with realistic costs
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.0001,        # 0.01% (IB Pro rates)
        slippage=0.001      # 0.10% slippage
    )

    try:
        # Run backtest
        logger.info(" \nRunning backtest...")
        portfolio = engine.run(
            strategy=strategy,
            symbols=[symbol1, symbol2],
            start_date=start_date,
            end_date=end_date
        )

        # Get statistics
        stats = portfolio.stats()

        if stats is None:
            logger.error(f"[X] Failed to generate stats for {symbol1}/{symbol2}")
            return {
                'pair': f"{symbol1}/{symbol2}",
                'description': description,
                'p_value': p_value,
                'sharpe': 0,
                'return': 0,
                'max_dd': 0,
                'win_rate': 0,
                'trades': 0,
                'final_equity': 0,
                'success': False,
                'error': 'No stats generated'
            }

        # Print results
        logger.info(f"\n{'='*80}")
        logger.info("BACKTEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f" Total Return:       {stats['Total Return [%]']:.2f}%")
        logger.info(f" Sharpe Ratio:       {stats['Sharpe Ratio']:.3f}")
        logger.info(f" Max Drawdown:       {stats['Max Drawdown [%]']:.2f}%")
        logger.info(f" Win Rate:           {stats['Win Rate [%]']:.2f}%")
        logger.info(f" Total Trades:       {stats['Total Trades']}")
        logger.info(f" Final Value:        ${stats['End Value']:.2f}")
        logger.info(f"{'='*80}\n")

        # Assess quality
        sharpe = stats['Sharpe Ratio']
        total_return = stats['Total Return [%]']

        if sharpe >= 0.8 and total_return > 0:
            assessment = "[+] PRODUCTION READY"
        elif sharpe >= 0.3 and total_return > 0:
            assessment = "[^] NEEDS OPTIMIZATION"
        elif total_return > 0:
            assessment = "[~] MARGINALLY PROFITABLE"
        else:
            assessment = "[X] NOT VIABLE"

        logger.info(f" \nPerformance Metrics:")
        logger.info(f"   Total Return: {total_return:.2f}%")
        logger.info(f"   Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"   Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        logger.info(f"   Win Rate: {stats['Win Rate [%]']:.2f}%")
        logger.info(f"   Total Trades: {stats['Total Trades']}")
        logger.info(f"   Final Equity: ${stats['End Value']:.2f}")
        logger.info(f" \nAssessment:")
        logger.info(f"   {assessment}\n")

        return {
            'pair': f"{symbol1}/{symbol2}",
            'description': description,
            'p_value': p_value,
            'sharpe': sharpe,
            'return': total_return,
            'max_dd': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'trades': stats['Total Trades'],
            'final_equity': stats['End Value'],
            'success': True,
            'error': ''
        }

    except Exception as e:
        logger.error(f"[X] Error validating {symbol1}/{symbol2}: {str(e)}")
        return {
            'pair': f"{symbol1}/{symbol2}",
            'description': description,
            'p_value': p_value,
            'sharpe': 0,
            'return': 0,
            'max_dd': 0,
            'win_rate': 0,
            'trades': 0,
            'final_equity': 0,
            'success': False,
            'error': str(e)
        }


def main():
    """Main validation routine."""
    logger.info("TOP 10 PAIRS VALIDATION")
    logger.info(" Testing top 10 cointegrated pairs from comprehensive discovery")
    logger.info(" Strategy: FIXED pairs trading (daily signals)")
    logger.info(" Transaction costs: 0.01% fees + 0.1% slippage (IB Pro rates)")
    logger.info(" ")

    # Test period
    start_date = "2023-01-01"
    end_date = "2024-11-11"
    logger.info(f"Test Period: {start_date} to {end_date}")
    logger.info(f" Pairs to validate: {len(TOP_PAIRS)}\n")

    # Validate all pairs
    results = []
    for symbol1, symbol2, description, p_value in TOP_PAIRS:
        result = validate_pair(symbol1, symbol2, description, p_value, start_date, end_date)
        results.append(result)

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    df = df.sort_values('sharpe', ascending=False)

    # Save results
    output_path = project_root / "output" / "top_10_pairs_validation.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n[+] \nResults saved to: {output_path}")

    # Print summary
    logger.info("\nVALIDATION SUMMARY")
    logger.info(f" Successful Tests: {df['success'].sum()}/{len(df)}")
    logger.info(f" Failed Tests: {(~df['success']).sum()}/{len(df)}")
    logger.info(" ")

    # Filter successful pairs
    successful = df[df['success']]

    if len(successful) > 0:
        logger.info("Pair Performance (sorted by Sharpe):")
        logger.info(" ")
        for _, row in successful.iterrows():
            logger.info(f"{row['pair']} ({row['description']}):")
            logger.info(f"   Cointegration: p={row['p_value']:.6f}")
            logger.info(f"   Sharpe: {row['sharpe']:.3f}")
            logger.info(f"   Return: {row['return']:.2f}%")
            logger.info(f"   Max DD: {row['max_dd']:.2f}%")
            logger.info(f"   Win Rate: {row['win_rate']:.2f}%")
            logger.info(f"   Trades: {int(row['trades'])}")
            logger.info(" ")

        # Production readiness
        production_ready = successful[successful['sharpe'] >= 0.8]
        needs_optimization = successful[(successful['sharpe'] >= 0.3) & (successful['sharpe'] < 0.8)]
        not_viable = successful[successful['sharpe'] < 0.3]

        logger.info("PRODUCTION READINESS")
        logger.info(f" Production Ready (Sharpe >= 0.8): {len(production_ready)}")
        logger.info(f" Needs Enhancement (0.3 <= Sharpe < 0.8): {len(needs_optimization)}")
        logger.info(f" Not Viable (Sharpe < 0.3): {len(not_viable)}")

        if len(production_ready) > 0:
            logger.info("\n[+] PRODUCTION READY PAIRS:")
            for _, row in production_ready.iterrows():
                logger.info(f"   - {row['pair']}: Sharpe {row['sharpe']:.3f}, Return {row['return']:.2f}%")
        else:
            logger.info("\n[X] \n[X] NO PRODUCTION READY PAIRS FOUND")
            logger.info(" All pairs need parameter optimization or are not viable")
    else:
        logger.info("[X] No successful backtests")

    logger.info("\n[+] \nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()
