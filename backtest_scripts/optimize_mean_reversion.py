"""
Optimization script for MeanReversion (Bollinger Bands) strategy.

Tests Bollinger Bands mean reversion on liquid ETFs and large caps
with comprehensive parameter grid for band width and exit conditions.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.mean_reversion import MeanReversion
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def optimize_mean_reversion():
    """Optimize MeanReversion strategy on liquid instruments."""
    logger.blank()
    logger.separator()
    logger.header("MEAN REVERSION (BOLLINGER BANDS) OPTIMIZATION")
    logger.separator()
    logger.blank()

    # Symbols: Liquid ETFs and large caps
    symbols = ['AAPL', 'SPY', 'QQQ']

    # Parameter grid
    param_grid = {
        'window': [10, 15, 20, 25, 30, 40, 50],
        'num_std': [1.5, 2.0, 2.5, 3.0],
        'exit_at_middle': [True, False]
    }

    # Calculate total combinations
    total_combos = (len(param_grid['window']) *
                   len(param_grid['num_std']) *
                   len(param_grid['exit_at_middle']))

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Total combinations: {total_combos} per symbol")
    logger.info(f"Total tests: {total_combos * len(symbols)}")
    logger.blank()

    # Create engine with moderate risk (10% position sizing)
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,  # 0.1% trading fees
        slippage=0.0005  # 0.05% slippage
    )

    # Set risk config explicitly to moderate (10% position sizing)
    engine.risk_config = RiskConfig.moderate()

    logger.info(f"Risk Profile: Moderate (10% position sizing)")
    logger.info(f"Initial Capital: ${engine.initial_capital:,.0f}")
    logger.info(f"Trading Fees: {engine.fees*100:.2f}%")
    logger.info(f"Slippage: {engine.slippage*100:.3f}%")
    logger.blank()

    # Create optimizer
    optimizer = GridSearchOptimizer(engine)

    # Run optimization for each symbol
    results = {}
    start_time = time.time()

    for symbol in symbols:
        logger.separator()
        logger.header(f"OPTIMIZING {symbol}")
        logger.separator()
        logger.blank()

        symbol_start = time.time()

        try:
            # Run optimization (2022-2024 for more data)
            result = optimizer.optimize_parallel(
                strategy_class=MeanReversion,
                param_grid=param_grid,
                symbols=symbol,
                start_date='2022-01-01',
                end_date='2024-01-01',
                metric='sharpe_ratio',
                max_workers=4,
                export_results=True
            )

            symbol_elapsed = time.time() - symbol_start

            results[symbol] = {
                'best_params': result['best_params'],
                'best_sharpe': result['best_value'],
                'time_taken': symbol_elapsed,
                'total_tested': result.get('total_time', 0)
            }

            logger.blank()
            logger.success(f"[SUCCESS] {symbol} optimization complete!")
            logger.profit(f"Best Sharpe Ratio: {result['best_value']:.4f}")
            logger.metric(f"Best Parameters: {result['best_params']}")
            logger.info(f"Time taken: {symbol_elapsed/60:.2f} minutes")
            logger.blank()

        except Exception as e:
            logger.error(f"[FAILED] {symbol} optimization failed: {e}")
            results[symbol] = {
                'error': str(e)
            }

    # Summary
    total_elapsed = time.time() - start_time

    logger.blank()
    logger.separator()
    logger.header("OPTIMIZATION SUMMARY")
    logger.separator()
    logger.blank()

    logger.info(f"Total execution time: {total_elapsed/60:.2f} minutes")
    logger.blank()

    # Print results table
    logger.header("RESULTS BY SYMBOL")
    logger.blank()

    for symbol, result in results.items():
        if 'error' in result:
            logger.error(f"{symbol}: FAILED - {result['error']}")
        else:
            logger.success(f"{symbol}:")
            logger.profit(f"  Sharpe Ratio: {result['best_sharpe']:.4f}")
            logger.metric(f"  Parameters: {result['best_params']}")
            logger.info(f"  Time: {result['time_taken']/60:.2f} min")
        logger.blank()

    # Find best overall
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_symbol = max(valid_results.keys(), key=lambda k: valid_results[k]['best_sharpe'])
        logger.blank()
        logger.separator()
        logger.header("BEST CONFIGURATION")
        logger.separator()
        logger.blank()
        logger.success(f"Symbol: {best_symbol}")
        logger.profit(f"Sharpe Ratio: {valid_results[best_symbol]['best_sharpe']:.4f}")
        logger.metric(f"Parameters: {valid_results[best_symbol]['best_params']}")
        logger.blank()

    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_mean_reversion()
        logger.success("[SUCCESS] Script completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FAILED] Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
