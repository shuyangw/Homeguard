"""
Optimization script for VolatilityTargetedMomentum strategy on FULL PERIOD (2019-2024).

UNTESTED ADVANCED STRATEGY with position sizing based on volatility targeting.
Using BAYESIAN OPTIMIZATION due to 3456 possible combinations.

Tests across both bull (2019-2021) and bear (2022-2024) periods to find robust parameters.
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
from strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
from backtesting.utils.risk_config import RiskConfig
from utils import logger

# Check if scikit-optimize is available
try:
    from skopt.space import Integer, Real, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger.error("scikit-optimize not available")


def optimize_vol_targeted_momentum():
    """Optimize VolatilityTargetedMomentum on full period using Bayesian optimization."""
    logger.blank()
    logger.separator()
    logger.header("VOLATILITY TARGETED MOMENTUM OPTIMIZATION - FULL PERIOD (2019-2024)")
    logger.separator()
    logger.blank()

    if not BAYESIAN_AVAILABLE:
        logger.error("Cannot run: scikit-optimize not installed")
        logger.info("Install with: conda install -c conda-forge scikit-optimize")
        return None

    # Symbols: Large cap stocks
    symbols = ['AAPL', 'SPY']

    # Parameter space for Bayesian optimization
    param_space = [
        Integer(100, 250, name='lookback_period'),
        Integer(150, 250, name='ma_window'),
        Integer(10, 30, name='vol_window'),
        Real(0.10, 0.25, name='target_vol'),
        Real(1.5, 2.5, name='max_leverage'),
        Categorical([True, False], name='use_ma_filter'),
        Categorical([True, False], name='use_return_filter'),
        Categorical(['and', 'or'], name='combine_filters')
    ]

    logger.info(f"Test Period: 2019-01-01 to 2024-01-01 (FULL PERIOD - Bull + Bear)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Parameter space:")
    logger.info(f"  - lookback_period: [100, 250]")
    logger.info(f"  - ma_window: [150, 250]")
    logger.info(f"  - vol_window: [10, 30]")
    logger.info(f"  - target_vol: [0.10, 0.25]")
    logger.info(f"  - max_leverage: [1.5, 2.5]")
    logger.info(f"  - use_ma_filter: [True, False]")
    logger.info(f"  - use_return_filter: [True, False]")
    logger.info(f"  - combine_filters: ['and', 'or']")
    logger.info(f"Grid Search would require: ~3456 combinations")
    logger.info(f"Bayesian Search will test: 100 iterations (~35x speedup)")
    logger.blank()
    logger.warning("UNTESTED ADVANCED STRATEGY - Position sizing is KEY")
    logger.blank()

    # Create engine with moderate risk (10% position sizing)
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )

    engine.risk_config = RiskConfig.moderate()

    logger.info(f"Risk Profile: Moderate (10% base position sizing)")
    logger.info(f"Note: Strategy applies additional volatility-based scaling")
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
        logger.header(f"OPTIMIZING {symbol} - BAYESIAN SEARCH (FULL PERIOD)")
        logger.separator()
        logger.blank()

        symbol_start = time.time()

        try:
            result = optimizer.optimize(
                strategy_class=VolatilityTargetedMomentum,
                param_space=param_space,
                symbols=symbol,
                start_date='2019-01-01',
                end_date='2024-01-01',
                metric='sharpe_ratio',
                n_iterations=100,
                n_initial_points=20,
                acquisition_func='EI',  # Expected Improvement
                convergence_tolerance=0.01,
                convergence_patience=10,
                enable_plots=False,
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
                logger.success(f"EXCELLENT: Sharpe > 1.5 - Volatility targeting works!")
            elif result['best_value'] > 1.0:
                logger.success(f"GREAT: Sharpe > 1.0")
            elif result['best_value'] > 0.5:
                logger.info(f"GOOD: Sharpe > 0.5")
            elif result['best_value'] > 0.0:
                logger.warning(f"MARGINAL: Sharpe > 0 but < 0.5")
            else:
                logger.error(f"NEGATIVE: Sharpe < 0")

            # Highlight key parameters
            logger.blank()
            logger.header("KEY PARAMETER INSIGHTS")
            target_vol = result['best_params'].get('target_vol', 0)
            max_lev = result['best_params'].get('max_leverage', 0)
            logger.info(f"Target Volatility: {target_vol:.2%} annualized")
            logger.info(f"Max Leverage: {max_lev:.2f}x")
            logger.info(f"MA Filter: {result['best_params'].get('use_ma_filter', False)}")
            logger.info(f"Return Filter: {result['best_params'].get('use_return_filter', False)}")
            logger.blank()

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
    logger.header("OPTIMIZATION SUMMARY - VOL TARGETED MOMENTUM FULL PERIOD")
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
        logger.header("STRATEGY EVALUATION")
        logger.separator()
        logger.blank()
        logger.info("Volatility targeting should provide:")
        logger.info("  ✓ Consistent risk across different market regimes")
        logger.info("  ✓ Automatic leverage reduction in high-vol periods")
        logger.info("  ✓ Automatic leverage increase in low-vol periods")
        logger.info("  ✓ Better risk-adjusted returns than fixed position sizing")
        logger.blank()

        if valid_results[best_symbol]['best_sharpe'] > 0.5:
            logger.success("VALIDATION: Volatility targeting adds value!")
        else:
            logger.warning("CONCERN: Volatility targeting may not be sufficient alone")
        logger.blank()

    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_vol_targeted_momentum()
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
