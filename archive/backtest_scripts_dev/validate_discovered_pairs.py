"""
Validate Discovered Cointegrated Pairs

Tests the cointegrated pairs discovered via systematic screening:
1. XLU/GDX (Utilities / Gold Miners) - p-value: 0.009
2. XLU/EWC (Utilities / Canada ETF) - p-value: 0.043

Runs full backtests with realistic transaction costs to validate profitability.

Usage:
    conda activate fintech
    python backtest_scripts/validate_discovered_pairs.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.utils.logger import Logger

logger = Logger()


def validate_pair(
    symbol1: str,
    symbol2: str,
    description: str,
    p_value: float,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    fees: float = 0.0001,
    slippage: float = 0.001
):
    """
    Validate a cointegrated pair with full backtest.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        description: Pair description
        p_value: Cointegration p-value
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        fees: Commission rate
        slippage: Slippage rate

    Returns:
        Dict with backtest results
    """
    logger.header(f"Validating: {symbol1} / {symbol2}")
    logger.info(f"Description: {description}")
    logger.info(f"Cointegration: p-value = {p_value:.6f}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Fees: {fees*100:.2f}%, Slippage: {slippage*100:.2f}%")

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage
    )

    # Create strategy with default parameters
    strategy = PairsTrading(
        pair_selection_window=252,
        cointegration_pvalue=0.05,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.5,
        zscore_window=20
    )

    try:
        # Run backtest
        logger.info("\nRunning backtest...")
        portfolio = engine.run(
            strategy=strategy,
            symbols=[symbol1, symbol2],
            start_date=start_date,
            end_date=end_date
        )

        # Get statistics
        stats = portfolio.stats()

        # Display results
        logger.success("Backtest Complete!")
        logger.info("\nPerformance Metrics:")
        logger.info(f"  Total Return: {stats.get('Total Return [%]', 0):.2f}%")
        logger.info(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.3f}")
        logger.info(f"  Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%")
        logger.info(f"  Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
        logger.info(f"  Total Trades: {stats.get('# Trades', 0)}")
        logger.info(f"  Final Equity: ${stats.get('Equity Final [$]', 0):,.2f}")

        # Assess viability
        sharpe = stats.get('Sharpe Ratio', 0)
        trades = stats.get('# Trades', 0)

        logger.info("\nAssessment:")
        if sharpe >= 0.8 and trades > 0:
            logger.success(f"  [OK] Production Ready (Sharpe: {sharpe:.3f})")
        elif sharpe >= 0.3 and trades > 0:
            logger.warning(f"  [!] Needs Enhancement (Sharpe: {sharpe:.3f})")
        elif trades == 0:
            logger.error("  [X] No Trades Generated")
        else:
            logger.error(f"  [X] Not Viable (Sharpe: {sharpe:.3f})")

        return {
            'pair': f"{symbol1}/{symbol2}",
            'description': description,
            'p_value': p_value,
            'sharpe': sharpe,
            'return': stats.get('Total Return [%]', 0),
            'max_dd': stats.get('Max Drawdown [%]', 0),
            'win_rate': stats.get('Win Rate [%]', 0),
            'trades': trades,
            'final_equity': stats.get('Equity Final [$]', 0),
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'pair': f"{symbol1}/{symbol2}",
            'description': description,
            'p_value': p_value,
            'sharpe': 0,
            'return': 0,
            'max_dd': 0,
            'win_rate': 0,
            'trades': 0,
            'final_equity': initial_capital,
            'success': False,
            'error': str(e)
        }


def main():
    """Main validation routine."""
    logger.header("DISCOVERED PAIRS VALIDATION")
    logger.info("Testing cointegrated pairs found via systematic screening")
    logger.info("Transaction costs: 0.01% fees + 0.1% slippage (IB Pro rates)")

    # Define discovered pairs
    discovered_pairs = [
        {
            'symbol1': 'XLU',
            'symbol2': 'GDX',
            'description': 'Utilities / Gold Miners',
            'p_value': 0.009218
        },
        {
            'symbol1': 'XLU',
            'symbol2': 'EWC',
            'description': 'Utilities / Canada ETF',
            'p_value': 0.042649
        }
    ]

    # Test period (same as discovery period)
    start_date = '2023-01-01'
    end_date = '2024-11-11'

    logger.info(f"\nTest Period: {start_date} to {end_date}")
    logger.info(f"Pairs to validate: {len(discovered_pairs)}\n")

    # Validate each pair
    results = []
    for pair_info in discovered_pairs:
        print("\n" + "="*80)
        result = validate_pair(
            symbol1=pair_info['symbol1'],
            symbol2=pair_info['symbol2'],
            description=pair_info['description'],
            p_value=pair_info['p_value'],
            start_date=start_date,
            end_date=end_date
        )
        results.append(result)
        print()

    # Summary
    logger.header("VALIDATION SUMMARY")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    logger.info(f"Successful Tests: {len(successful)}/{len(results)}")
    logger.info(f"Failed Tests: {len(failed)}/{len(results)}")

    if successful:
        logger.info("\nPair Performance:")
        df = pd.DataFrame(successful)
        df = df.sort_values('sharpe', ascending=False)

        for _, row in df.iterrows():
            logger.info(f"\n{row['pair']} ({row['description']}):")
            logger.info(f"  Cointegration: p={row['p_value']:.6f}")
            logger.info(f"  Sharpe: {row['sharpe']:.3f}")
            logger.info(f"  Return: {row['return']:.2f}%")
            logger.info(f"  Max DD: {row['max_dd']:.2f}%")
            logger.info(f"  Win Rate: {row['win_rate']:.2f}%")
            logger.info(f"  Trades: {row['trades']}")

        # Production readiness
        logger.header("PRODUCTION READINESS")

        production_ready = df[df['sharpe'] >= 0.8]
        needs_work = df[(df['sharpe'] >= 0.3) & (df['sharpe'] < 0.8)]
        not_viable = df[df['sharpe'] < 0.3]

        logger.info(f"Production Ready (Sharpe >= 0.8): {len(production_ready)}")
        logger.info(f"Needs Enhancement (0.3 <= Sharpe < 0.8): {len(needs_work)}")
        logger.info(f"Not Viable (Sharpe < 0.3): {len(not_viable)}")

        if len(production_ready) > 0:
            logger.success("\n[OK] PAIRS READY FOR PRODUCTION")
            for _, row in production_ready.iterrows():
                logger.info(f"  + {row['pair']} (Sharpe: {row['sharpe']:.3f})")
        elif len(needs_work) > 0:
            logger.warning("\n[!] PAIRS NEED PARAMETER OPTIMIZATION")
            for _, row in needs_work.iterrows():
                logger.info(f"  ~ {row['pair']} (Sharpe: {row['sharpe']:.3f})")
        else:
            logger.error("\n[X] NO VIABLE PAIRS FOUND")

        # Save results
        output_path = Path(__file__).parent.parent / 'output' / 'pair_validation_results.csv'
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"\nResults saved to: {output_path}")

    if failed:
        logger.warning("\nFailed Pairs:")
        for row in failed:
            logger.error(f"  {row['pair']}: {row['error']}")

    logger.header("VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
