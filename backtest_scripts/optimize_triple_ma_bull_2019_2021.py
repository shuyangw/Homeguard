"""
Optimization script for TripleMovingAverage strategy on BULL MARKET period (2019-2021).

UNTESTED STRATEGY with 2040+ combinations - Using BAYESIAN OPTIMIZATION for efficiency.
Tests trend-aligned entry (fast > medium > slow MA).
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import BayesianOptimizer

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
from strategies.base_strategies.moving_average import TripleMovingAverage
from backtesting.utils.risk_config import RiskConfig
from utils import logger

# Check if scikit-optimize is available
try:
    from skopt.space import Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger.error("scikit-optimize not available - falling back to Random Search")


def optimize_triple_ma_bull():
    """Optimize TripleMovingAverage strategy on bull market using Bayesian optimization."""
    logger.blank()
    logger.separator()
    logger.header("TRIPLE MA OPTIMIZATION - BULL MARKET (2019-2021) - BAYESIAN")
    logger.separator()
    logger.blank()

    if not BAYESIAN_AVAILABLE:
        logger.error("Cannot run: scikit-optimize not installed")
        logger.info("Install with: conda install -c conda-forge scikit-optimize")
        return None

    # Symbols: Large cap stocks
    symbols = ['AAPL', 'MSFT']

    # Parameter space for Bayesian optimization
    param_space = [
        Integer(5, 30, name='fast_window'),
        Integer(15, 60, name='medium_window'),
        Integer(40, 200, name='slow_window'),
        Categorical(['sma', 'ema'], name='ma_type')
    ]

    logger.info(f"Test Period: 2019-01-01 to 2021-12-31 (BULL MARKET)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Parameter space:")
    logger.info(f"  - fast_window: [5, 30]")
    logger.info(f"  - medium_window: [15, 60]")
    logger.info(f"  - slow_window: [40, 200]")
    logger.info(f"  - ma_type: ['sma', 'ema']")
    logger.info(f"Grid Search would require: ~2040 combinations")
    logger.info(f"Bayesian Search will test: 100 iterations (~20x speedup)")
    logger.blank()
    logger.warning("UNTESTED STRATEGY - First comprehensive test")
    logger.blank()

    # Create engine with moderate risk (10% position sizing)
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )

    engine.risk_config = RiskConfig.moderate()

    logger.info(f"Risk Profile: Moderate (10% position sizing)")
    logger.info(f"Initial Capital: ${engine.initial_capital:,.0f}")
    logger.info(f"Trading Fees: {engine.fees*100:.2f}%")
    logger.info(f"Slippage: {engine.slippage*100:.3f}%")
    logger.blank()

    # Create optimizer
    optimizer = BayesianOptimizer(engine)

    # Run optimization for each symbol
    results = {}
    start_time = time.time()

    for symbol in symbols:
        logger.separator()
        logger.header(f"OPTIMIZING {symbol} - BAYESIAN SEARCH")
        logger.separator()
        logger.blank()

        symbol_start = time.time()

        try:
            result = optimizer.optimize(
                strategy_class=TripleMovingAverage,
                param_space=param_space,
                symbols=symbol,
                start_date='2019-01-01',
                end_date='2021-12-31',
                metric='sharpe_ratio',
                n_iterations=100,
                n_initial_points=20,
                acquisition_func='EI',  # Expected Improvement
                convergence_tolerance=0.01,
                convergence_patience=10,
                enable_plots=False,  # Set to True if you want convergence plots
                export_results=True
            )

            symbol_elapsed = time.time() - symbol_start

            results[symbol] = {
                'best_params': result['best_params'],
                'best_sharpe': result['best_value'],
                'time_taken': symbol_elapsed,
                'iterations': result.get('n_iterations', 100),
                'converged': result.get('converged', False)
            }

            logger.blank()
            logger.success(f"[SUCCESS] {symbol} optimization complete!")
            logger.profit(f"Best Sharpe Ratio: {result['best_value']:.4f}")
            logger.metric(f"Best Parameters: {result['best_params']}")
            logger.info(f"Iterations completed: {result.get('n_iterations', 100)}")
            logger.info(f"Converged: {result.get('converged', False)}")
            logger.info(f"Time taken: {symbol_elapsed/60:.2f} minutes")
            logger.blank()

            if result['best_value'] > 1.5:
                logger.success(f"EXCELLENT: Sharpe > 1.5 - Triple MA filter works great!")
            elif result['best_value'] > 1.0:
                logger.success(f"GREAT: Sharpe > 1.0")
            elif result['best_value'] > 0.5:
                logger.info(f"GOOD: Sharpe > 0.5")
            elif result['best_value'] > 0.0:
                logger.warning(f"MARGINAL: Sharpe > 0 but < 0.5")
            else:
                logger.error(f"NEGATIVE: Sharpe < 0")

        except Exception as e:
            logger.error(f"[FAILED] {symbol} optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = {
                'error': str(e)
            }

    # Summary
    total_elapsed = time.time() - start_time

    logger.blank()
    logger.separator()
    logger.header("OPTIMIZATION SUMMARY - TRIPLE MA BULL MARKET (2019-2021)")
    logger.separator()
    logger.blank()

    logger.info(f"Total execution time: {total_elapsed/60:.2f} minutes")
    logger.info(f"Method: Bayesian Optimization (Expected Improvement)")
    logger.blank()

    logger.header("RESULTS BY SYMBOL")
    logger.blank()

    for symbol, result in results.items():
        if 'error' in result:
            logger.error(f"{symbol}: FAILED - {result['error']}")
        else:
            logger.success(f"{symbol}:")
            logger.profit(f"  Sharpe Ratio: {result['best_sharpe']:.4f}")
            logger.metric(f"  Parameters: {result['best_params']}")
            logger.info(f"  Iterations: {result['iterations']}")
            logger.info(f"  Converged: {result['converged']}")
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
        logger.header("COMPARISON WITH DUAL MA CROSSOVER")
        logger.separator()
        logger.blank()
        logger.info("Triple MA should have FEWER but HIGHER QUALITY signals")
        logger.info("Expecting: Lower trade count, higher win rate, similar/better Sharpe")
        logger.blank()

    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_triple_ma_bull()
        if results is not None:
            logger.success("[SUCCESS] Script completed successfully!")
            sys.exit(0)
        else:
            logger.error("[FAILED] Script failed - Bayesian optimizer not available")
            sys.exit(1)
    except Exception as e:
        logger.error(f"[FAILED] Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
