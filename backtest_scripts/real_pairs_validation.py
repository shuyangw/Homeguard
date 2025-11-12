"""
Real Market Pairs Trading Validation

Tests pairs trading framework on real market symbols with realistic transaction costs.

Common Cointegrated Pairs:
- SPY/IWM (S&P 500 / Russell 2000)
- GLD/GDX (Gold / Gold Miners)
- XLE/XLU (Energy / Utilities)
- EWA/EWC (Australia / Canada ETFs)

Usage:
    conda activate fintech
    python backtest_scripts/real_pairs_validation.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.backtesting.optimization.grid_search import GridSearchOptimizer
from src.utils.logger import Logger

logger = Logger()


def test_pair(
    symbol1: str,
    symbol2: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    fees: float = 0.0001,  # 0.01% (IB Pro rates)
    slippage: float = 0.001  # 0.1%
):
    """
    Test a single pair with default parameters.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        fees: Commission rate
        slippage: Slippage rate
    """
    logger.header(f"Testing Pair: {symbol1} / {symbol2}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Fees: {fees*100:.2f}%, Slippage: {slippage*100:.2f}%")

    # Initialize engine with realistic costs
    engine = BacktestEngine(
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage
    )

    # Create strategy with conservative parameters
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
        portfolio = engine.run(
            strategy=strategy,
            symbols=[symbol1, symbol2],
            start_date=start_date,
            end_date=end_date
        )

        # Get statistics
        stats = portfolio.stats()

        # Display results
        logger.success("Backtest Complete")
        logger.info(f"Total Return: {stats.get('Total Return [%]', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.3f}")
        logger.info(f"Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%")
        logger.info(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
        logger.info(f"Total Trades: {stats.get('# Trades', 0)}")
        logger.info(f"Final Equity: ${stats.get('Equity Final [$]', 0):,.2f}")

        return {
            'pair': f"{symbol1}/{symbol2}",
            'sharpe': stats.get('Sharpe Ratio', 0),
            'return': stats.get('Total Return [%]', 0),
            'max_dd': stats.get('Max Drawdown [%]', 0),
            'win_rate': stats.get('Win Rate [%]', 0),
            'trades': stats.get('# Trades', 0),
            'final_equity': stats.get('Equity Final [$]', 0),
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return {
            'pair': f"{symbol1}/{symbol2}",
            'sharpe': 0,
            'return': 0,
            'max_dd': 0,
            'win_rate': 0,
            'trades': 0,
            'final_equity': initial_capital,
            'success': False,
            'error': str(e)
        }


def optimize_pair(
    symbol1: str,
    symbol2: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    fees: float = 0.0001,
    slippage: float = 0.001
):
    """
    Optimize parameters for a single pair.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        fees: Commission rate
        slippage: Slippage rate
    """
    logger.header(f"Optimizing Pair: {symbol1} / {symbol2}")

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage
    )

    # Initialize optimizer
    optimizer = GridSearchOptimizer(engine)

    # Define parameter grid
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.25, 0.5, 0.75],
        'zscore_window': [15, 20, 30]
    }

    logger.info(f"Parameter combinations: {3 * 3 * 3} = 27")

    try:
        # Run optimization
        result = optimizer.optimize_parallel(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=[symbol1, symbol2],
            start_date=start_date,
            end_date=end_date,
            metric='sharpe_ratio',
            max_workers=4,
            use_cache=True
        )

        logger.success("Optimization Complete")
        logger.info(f"Best Sharpe: {result['best_value']:.3f}")
        logger.info(f"Best Parameters: {result['best_params']}")
        logger.info(f"Time: {result['total_time']:.1f}s")

        return {
            'pair': f"{symbol1}/{symbol2}",
            'best_sharpe': result['best_value'],
            'best_params': result['best_params'],
            'time': result['total_time'],
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return {
            'pair': f"{symbol1}/{symbol2}",
            'best_sharpe': 0,
            'best_params': None,
            'time': 0,
            'success': False,
            'error': str(e)
        }


def main():
    """Main validation routine."""
    logger.header("REAL MARKET PAIRS TRADING VALIDATION")
    logger.info("Testing pairs trading framework on real market symbols")
    logger.info("Transaction costs: 0.01% fees + 0.1% slippage (IB Pro rates)")

    # Define test pairs
    test_pairs = [
        ('SPY', 'IWM', 'S&P 500 / Russell 2000'),
        ('GLD', 'GDX', 'Gold / Gold Miners'),
        ('XLE', 'XLU', 'Energy / Utilities'),
        ('EWA', 'EWC', 'Australia / Canada ETFs')
    ]

    # Test period
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    logger.info(f"Test Period: {start_date} to {end_date}")
    logger.info(f"Pairs to test: {len(test_pairs)}\n")

    # Test each pair
    results = []
    for symbol1, symbol2, description in test_pairs:
        logger.info(f"Description: {description}")
        result = test_pair(
            symbol1=symbol1,
            symbol2=symbol2,
            start_date=start_date,
            end_date=end_date
        )
        results.append(result)
        print()  # Blank line between tests

    # Summary
    logger.header("VALIDATION SUMMARY")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    logger.info(f"Successful Tests: {len(successful)}/{len(results)}")
    logger.info(f"Failed Tests: {len(failed)}/{len(results)}")

    if successful:
        logger.info("\nSuccessful Pairs Performance:")
        df = pd.DataFrame(successful)
        df = df.sort_values('sharpe', ascending=False)

        for _, row in df.iterrows():
            logger.info(f"\n{row['pair']}:")
            logger.info(f"  Sharpe: {row['sharpe']:.3f}")
            logger.info(f"  Return: {row['return']:.2f}%")
            logger.info(f"  Max DD: {row['max_dd']:.2f}%")
            logger.info(f"  Win Rate: {row['win_rate']:.2f}%")
            logger.info(f"  Trades: {row['trades']}")

        # Best pair
        best = df.iloc[0]
        logger.success(f"\nBest Pair: {best['pair']} (Sharpe: {best['sharpe']:.3f})")

        # Average performance
        avg_sharpe = df['sharpe'].mean()
        avg_return = df['return'].mean()
        logger.info(f"\nAverage Sharpe: {avg_sharpe:.3f}")
        logger.info(f"Average Return: {avg_return:.2f}%")

        # Production readiness assessment
        logger.header("PRODUCTION READINESS ASSESSMENT")

        production_ready = df[df['sharpe'] >= 0.8]
        needs_work = df[(df['sharpe'] >= 0.3) & (df['sharpe'] < 0.8)]
        not_viable = df[df['sharpe'] < 0.3]

        logger.info(f"Production Ready (Sharpe >= 0.8): {len(production_ready)}")
        logger.info(f"Needs Enhancement (0.3 <= Sharpe < 0.8): {len(needs_work)}")
        logger.info(f"Not Viable (Sharpe < 0.3): {len(not_viable)}")

        if len(production_ready) > 0:
            logger.success("\n[OK] READY FOR PRODUCTION")
            logger.info("The following pairs meet production standards:")
            for _, row in production_ready.iterrows():
                logger.info(f"  - {row['pair']} (Sharpe: {row['sharpe']:.3f})")
        elif len(needs_work) > 0:
            logger.warning("\n[!] NEEDS ENHANCEMENT")
            logger.info("Consider implementing:")
            logger.info("  - Kalman filter for dynamic hedge ratios")
            logger.info("  - Multi-pair portfolio diversification")
            logger.info("  - Regime detection (VIX-based)")
            logger.info("  - Advanced entry/exit timing")
        else:
            logger.error("\n[X] NOT PRODUCTION READY")
            logger.info("Core strategy requires fundamental improvements")

    if failed:
        logger.warning("\nFailed Pairs:")
        for row in failed:
            logger.error(f"{row['pair']}: {row['error']}")

    # Ask about optimization
    print("\n" + "="*80)
    response = input("\nRun parameter optimization on successful pairs? (y/n): ").strip().lower()

    if response == 'y' and successful:
        logger.header("PARAMETER OPTIMIZATION")

        opt_results = []
        for result in successful[:2]:  # Optimize top 2 pairs
            symbol1, symbol2 = result['pair'].split('/')
            opt_result = optimize_pair(
                symbol1=symbol1,
                symbol2=symbol2,
                start_date=start_date,
                end_date=end_date
            )
            opt_results.append(opt_result)
            print()

        # Optimization summary
        logger.header("OPTIMIZATION SUMMARY")
        for opt in opt_results:
            if opt['success']:
                logger.info(f"\n{opt['pair']}:")
                logger.info(f"  Best Sharpe: {opt['best_sharpe']:.3f}")
                logger.info(f"  Best Params: {opt['best_params']}")
    else:
        logger.info("Skipping optimization")

    logger.header("VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
