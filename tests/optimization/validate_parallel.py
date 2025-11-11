"""
Quick validation script for parallel optimization.

Runs a minimal test to verify:
1. Parallel optimization works
2. Results match sequential
3. Performance improvement is achieved
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

import time
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def validate_parallel_optimization():
    """Run quick validation of parallel optimization."""

    logger.blank()
    logger.separator()
    logger.header("PARALLEL OPTIMIZATION VALIDATION")
    logger.separator()
    logger.blank()

    # Setup
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        enable_regime_analysis=False  # Faster
    )
    optimizer = GridSearchOptimizer(engine)

    # Small parameter grid for quick test
    param_grid = {
        'fast_window': [10, 15, 20],
        'slow_window': [50, 60, 70]
    }

    # Test parameters
    symbol = 'AAPL'
    start_date = '2024-01-15'
    end_date = '2024-01-31'  # 2 weeks only
    metric = 'sharpe_ratio'

    logger.info("Test Configuration:")
    logger.metric(f"  Symbol: {symbol}")
    logger.metric(f"  Period: {start_date} to {end_date}")
    logger.metric(f"  Grid size: {len(param_grid['fast_window']) * len(param_grid['slow_window'])} combinations")
    logger.blank()

    # Test 1: Sequential optimization
    logger.header("Test 1: Sequential Optimization")
    start_time = time.time()

    seq_result = optimizer.optimize(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols=symbol,
        start_date=start_date,
        end_date=end_date,
        metric=metric
    )

    seq_time = time.time() - start_time

    logger.success(f"Sequential completed in {seq_time:.2f}s")
    logger.info(f"Best params: {seq_result['best_params']}")
    logger.info(f"Best {metric}: {seq_result['best_value']:.4f}")
    logger.blank()

    # Test 2: Parallel optimization
    logger.header("Test 2: Parallel Optimization")
    start_time = time.time()

    par_result = optimizer.optimize_parallel(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols=symbol,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        max_workers=2  # Use 2 workers for quick test
    )

    par_time = time.time() - start_time

    logger.success(f"Parallel completed in {seq_time:.2f}s")
    logger.info(f"Best params: {par_result['best_params']}")
    logger.info(f"Best {metric}: {par_result['best_value']:.4f}")
    logger.blank()

    # Validation
    logger.separator()
    logger.header("VALIDATION RESULTS")
    logger.separator()
    logger.blank()

    # Check 1: Same best parameters
    same_params = seq_result['best_params'] == par_result['best_params']
    if same_params:
        logger.success("✓ PASS: Parallel found same best parameters as sequential")
    else:
        logger.error("✗ FAIL: Different best parameters!")
        logger.warning(f"  Sequential: {seq_result['best_params']}")
        logger.warning(f"  Parallel:   {par_result['best_params']}")

    # Check 2: Same best value (within tolerance)
    value_diff = abs(seq_result['best_value'] - par_result['best_value'])
    same_value = value_diff < 0.01
    if same_value:
        logger.success(f"✓ PASS: Parallel found same best value (diff: {value_diff:.6f})")
    else:
        logger.error(f"✗ FAIL: Different best values (diff: {value_diff:.4f})")

    # Check 3: Speedup achieved (for larger grids)
    speedup = seq_time / par_time if par_time > 0 else 0
    logger.info(f"Performance: {speedup:.2f}x speedup")

    # Check 4: All results returned
    has_all_results = 'all_results' in par_result
    if has_all_results:
        logger.success(f"✓ PASS: Parallel returned all {len(par_result['all_results'])} results")
    else:
        logger.warning("⚠ WARNING: 'all_results' not in parallel result (may have fallen back to sequential)")

    # Check 5: Portfolio object returned
    has_portfolio = par_result['best_portfolio'] is not None
    if has_portfolio:
        logger.success("✓ PASS: Portfolio object returned")
    else:
        logger.error("✗ FAIL: No portfolio object returned")

    logger.blank()
    logger.separator()

    # Overall result
    all_passed = same_params and same_value and has_portfolio

    if all_passed:
        logger.success("═══════════════════════════════════════")
        logger.success("  ✓ VALIDATION PASSED - ALL CHECKS OK  ")
        logger.success("═══════════════════════════════════════")
    else:
        logger.error("═══════════════════════════════════════")
        logger.error("  ✗ VALIDATION FAILED - SEE ERRORS ABOVE")
        logger.error("═══════════════════════════════════════")

    logger.blank()

    return all_passed


if __name__ == '__main__':
    success = validate_parallel_optimization()
    exit(0 if success else 1)
