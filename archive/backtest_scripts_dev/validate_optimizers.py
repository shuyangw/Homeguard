"""
Validation script for Random Search and Bayesian Optimization.

Compares performance, quality, and convergence of optimization methods.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import (

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BAYESIAN_AVAILABLE
)
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger

# Conditional imports for Bayesian optimization
BayesianOptimizer = None
Integer = None
try:
    from backtesting.optimization import BayesianOptimizer
    from skopt.space import Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


def validate_random_search():
    """Validate Random Search optimizer."""
    logger.blank()
    logger.separator()
    logger.header("VALIDATING RANDOM SEARCH OPTIMIZER")
    logger.separator()
    logger.blank()

    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0001
    )

    # Create optimizer
    optimizer = RandomSearchOptimizer(engine)

    # Define parameter ranges
    param_ranges = {
        'fast_window': (5, 30),
        'slow_window': (40, 120)
    }

    logger.info("Testing Random Search with 20 iterations...")
    logger.info(f"Parameter ranges: {param_ranges}")

    start_time = time.time()

    try:
        # Run optimization
        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-01',
            metric='sharpe_ratio',
            n_iterations=20,
            max_workers=2,
            use_cache=True,
            export_results=True,
            random_seed=42  # For reproducibility
        )

        elapsed = time.time() - start_time

        # Validate results
        logger.blank()
        logger.success("✅ Random Search completed successfully!")
        logger.blank()
        logger.metric(f"Best parameters: {result['best_params']}")
        logger.profit(f"Best Sharpe Ratio: {result['best_value']:.4f}")
        logger.info(f"Total iterations: {result['n_iterations']}")
        logger.info(f"Cache hits: {result['cache_hits']}")
        logger.info(f"Cache misses: {result['cache_misses']}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average time per iteration: {result['avg_time_per_test']:.2f}s")

        # Validate result structure
        assert 'best_params' in result, "Missing best_params"
        assert 'best_value' in result, "Missing best_value"
        assert 'best_portfolio' in result, "Missing best_portfolio"
        assert 'all_results' in result, "Missing all_results"
        assert len(result['all_results']) == 20, f"Expected 20 results, got {len(result['all_results'])}"

        # Validate parameter bounds
        assert 5 <= result['best_params']['fast_window'] <= 30, "fast_window out of bounds"
        assert 40 <= result['best_params']['slow_window'] <= 120, "slow_window out of bounds"

        logger.blank()
        logger.success("✅ All Random Search validations passed!")
        logger.separator()

        return result

    except Exception as e:
        logger.error(f"❌ Random Search validation failed: {e}")
        logger.separator()
        raise


def validate_bayesian():
    """Validate Bayesian Optimizer."""
    if not BAYESIAN_AVAILABLE:
        logger.blank()
        logger.separator()
        logger.warning("⚠️  BAYESIAN OPTIMIZATION NOT AVAILABLE")
        logger.warning("scikit-optimize is not installed")
        logger.info("Install with: pip install scikit-optimize")
        logger.separator()
        logger.blank()
        return None

    logger.blank()
    logger.separator()
    logger.header("VALIDATING BAYESIAN OPTIMIZER")
    logger.separator()
    logger.blank()

    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0001
    )

    # Create optimizer
    optimizer = BayesianOptimizer(engine)

    # Define parameter space
    param_space = [
        Integer(5, 30, name='fast_window'),
        Integer(40, 120, name='slow_window')
    ]

    logger.info("Testing Bayesian Optimization with 20 iterations...")
    logger.info(f"Parameter space: {[str(dim) for dim in param_space]}")

    start_time = time.time()

    try:
        # Run optimization
        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-01',
            metric='sharpe_ratio',
            n_iterations=20,
            n_initial_points=5,
            acquisition_func='EI',
            max_workers=2,
            use_cache=True,
            export_results=True,
            enable_plots=True,
            random_seed=42  # For reproducibility
        )

        elapsed = time.time() - start_time

        # Validate results
        logger.blank()
        logger.success("✅ Bayesian Optimization completed successfully!")
        logger.blank()
        logger.metric(f"Best parameters: {result['best_params']}")
        logger.profit(f"Best Sharpe Ratio: {result['best_value']:.4f}")
        logger.info(f"Total iterations: {result['n_iterations']}")
        logger.info(f"Early stopped: {result.get('early_stopped', False)}")
        logger.info(f"Cache hits: {result['cache_hits']}")
        logger.info(f"Cache misses: {result['cache_misses']}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average time per iteration: {result['avg_time_per_iteration']:.2f}s")

        # Validate result structure
        assert 'best_params' in result, "Missing best_params"
        assert 'best_value' in result, "Missing best_value"
        assert 'best_portfolio' in result, "Missing best_portfolio"
        assert 'all_results' in result, "Missing all_results"
        assert 'convergence_data' in result, "Missing convergence_data"
        assert len(result['all_results']) <= 20, f"Expected ≤20 results, got {len(result['all_results'])}"

        # Validate parameter bounds
        assert 5 <= result['best_params']['fast_window'] <= 30, "fast_window out of bounds"
        assert 40 <= result['best_params']['slow_window'] <= 120, "slow_window out of bounds"

        # Validate convergence data
        conv_data = result['convergence_data']
        assert 'iterations' in conv_data, "Missing iterations in convergence_data"
        assert 'best_values' in conv_data, "Missing best_values in convergence_data"
        assert len(conv_data['best_values']) == result['n_iterations'], "Convergence data length mismatch"

        logger.blank()
        logger.success("✅ All Bayesian Optimization validations passed!")
        logger.separator()

        return result

    except Exception as e:
        logger.error(f"❌ Bayesian Optimization validation failed: {e}")
        logger.separator()
        raise


def compare_optimizers(random_result, bayesian_result):
    """Compare Random Search and Bayesian results."""
    if bayesian_result is None:
        logger.warning("⚠️  Cannot compare - Bayesian not available")
        return

    logger.blank()
    logger.separator()
    logger.header("OPTIMIZER COMPARISON")
    logger.separator()
    logger.blank()

    # Compare best values
    random_best = random_result['best_value']
    bayesian_best = bayesian_result['best_value']

    logger.metric(f"Random Search best Sharpe:   {random_best:.4f}")
    logger.metric(f"Bayesian Opt best Sharpe:    {bayesian_best:.4f}")

    if bayesian_best > random_best:
        improvement = ((bayesian_best - random_best) / abs(random_best)) * 100
        logger.success(f"✅ Bayesian is better by {improvement:.1f}%")
    elif random_best > bayesian_best:
        improvement = ((random_best - bayesian_best) / abs(bayesian_best)) * 100
        logger.warning(f"⚠️  Random Search is better by {improvement:.1f}%")
    else:
        logger.info("Both methods found same optimum")

    logger.blank()

    # Compare efficiency
    random_time = random_result['avg_time_per_test']
    bayesian_time = bayesian_result['avg_time_per_iteration']

    logger.metric(f"Random Search avg time:      {random_time:.2f}s per iteration")
    logger.metric(f"Bayesian Opt avg time:       {bayesian_time:.2f}s per iteration")

    if bayesian_time < random_time:
        speedup = random_time / bayesian_time
        logger.success(f"✅ Bayesian is {speedup:.1f}x faster per iteration")
    else:
        slowdown = bayesian_time / random_time
        logger.warning(f"⚠️  Bayesian is {slowdown:.1f}x slower per iteration")

    logger.blank()

    # Compare cache efficiency
    random_cache_rate = (random_result['cache_hits'] / 20) * 100 if random_result['cache_hits'] > 0 else 0
    bayesian_cache_rate = (bayesian_result['cache_hits'] / 20) * 100 if bayesian_result['cache_hits'] > 0 else 0

    logger.metric(f"Random Search cache hit rate:  {random_cache_rate:.1f}%")
    logger.metric(f"Bayesian Opt cache hit rate:   {bayesian_cache_rate:.1f}%")

    logger.blank()
    logger.separator()


def main():
    """Run validation for all optimizers."""
    logger.blank()
    logger.separator()
    logger.header("OPTIMIZER VALIDATION SUITE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.separator()
    logger.blank()

    results = {}

    # Test 1: Validate Random Search
    try:
        results['random'] = validate_random_search()
    except Exception as e:
        logger.error(f"Random Search validation failed: {e}")
        return 1

    # Test 2: Validate Bayesian (if available)
    try:
        results['bayesian'] = validate_bayesian()
    except Exception as e:
        logger.error(f"Bayesian Optimization validation failed: {e}")
        # Don't fail if Bayesian is not available
        if BAYESIAN_AVAILABLE:
            return 1

    # Test 3: Compare optimizers
    if results.get('random') and results.get('bayesian'):
        compare_optimizers(results['random'], results['bayesian'])

    # Final summary
    logger.blank()
    logger.separator()
    logger.header("VALIDATION SUMMARY")
    logger.separator()
    logger.blank()

    if results.get('random'):
        logger.success("✅ Random Search: PASSED")
    else:
        logger.error("❌ Random Search: FAILED")

    if BAYESIAN_AVAILABLE:
        if results.get('bayesian'):
            logger.success("✅ Bayesian Optimization: PASSED")
        else:
            logger.error("❌ Bayesian Optimization: FAILED")
    else:
        logger.warning("⚠️  Bayesian Optimization: NOT AVAILABLE (install scikit-optimize)")

    logger.blank()
    logger.separator()

    return 0


if __name__ == '__main__':
    sys.exit(main())
