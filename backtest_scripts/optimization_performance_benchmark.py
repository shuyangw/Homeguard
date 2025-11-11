"""
Benchmark script to compare sequential vs parallel optimization performance.

This script measures the speedup achieved by parallel parameter optimization
for different grid sizes and worker counts.
"""

import time
from datetime import datetime
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def benchmark_optimization(param_grid, max_workers=None):
    """
    Benchmark sequential vs parallel optimization.

    Args:
        param_grid: Parameter grid to test
        max_workers: Number of workers for parallel (None = auto)

    Returns:
        dict with timing results
    """
    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        enable_regime_analysis=False  # Disable for faster benchmarking
    )
    optimizer = GridSearchOptimizer(engine)

    # Test dates (short period for faster benchmarking)
    start_date = '2024-01-01'
    end_date = '2024-02-01'
    symbol = 'AAPL'
    metric = 'sharpe_ratio'

    # Calculate grid size
    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)

    logger.blank()
    logger.separator()
    logger.header(f"BENCHMARKING: Grid size = {grid_size} combinations")
    logger.separator()
    logger.blank()

    # Sequential optimization
    logger.info("Running SEQUENTIAL optimization...")
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

    logger.blank()
    logger.success(f"Sequential completed in {seq_time:.2f} seconds")
    logger.info(f"Best params: {seq_result['best_params']}")
    logger.info(f"Best {metric}: {seq_result['best_value']:.4f}")
    logger.blank()

    # Parallel optimization
    logger.info("Running PARALLEL optimization...")
    start_time = time.time()

    par_result = optimizer.optimize_parallel(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols=symbol,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        max_workers=max_workers
    )

    par_time = time.time() - start_time

    logger.blank()
    logger.success(f"Parallel completed in {par_time:.2f} seconds")
    logger.info(f"Best params: {par_result['best_params']}")
    logger.info(f"Best {metric}: {par_result['best_value']:.4f}")
    logger.blank()

    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0

    logger.separator()
    logger.header("BENCHMARK RESULTS")
    logger.separator()
    logger.metric(f"Grid size: {grid_size} combinations")
    logger.metric(f"Sequential time: {seq_time:.2f} seconds")
    logger.metric(f"Parallel time: {par_time:.2f} seconds")
    logger.profit(f"Speedup: {speedup:.2f}x")
    logger.separator()
    logger.blank()

    return {
        'grid_size': grid_size,
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': speedup,
        'workers': max_workers or 'auto',
        'seq_result': seq_result,
        'par_result': par_result
    }


if __name__ == '__main__':
    logger.blank()
    logger.separator()
    logger.header("GRID SEARCH OPTIMIZATION BENCHMARK")
    logger.separator()
    logger.blank()

    # Test 1: Small grid (should use sequential fallback)
    logger.header("Test 1: Small Grid (4 combinations)")
    small_grid = {
        'fast_window': [10, 20],
        'slow_window': [50, 100]
    }
    result1 = benchmark_optimization(small_grid, max_workers=4)

    # Test 2: Medium grid
    logger.header("Test 2: Medium Grid (16 combinations)")
    medium_grid = {
        'fast_window': [10, 15, 20, 25],
        'slow_window': [50, 60, 70, 80]
    }
    result2 = benchmark_optimization(medium_grid, max_workers=4)

    # Test 3: Large grid
    logger.header("Test 3: Large Grid (36 combinations)")
    large_grid = {
        'fast_window': [10, 15, 20, 25, 30, 35],
        'slow_window': [50, 60, 70, 80, 90, 100]
    }
    result3 = benchmark_optimization(large_grid, max_workers=4)

    # Test 4: Very large grid (with 2 workers)
    logger.header("Test 4: Large Grid with 2 Workers (36 combinations)")
    result4 = benchmark_optimization(large_grid, max_workers=2)

    # Summary
    logger.blank()
    logger.separator()
    logger.header("BENCHMARK SUMMARY")
    logger.separator()
    logger.blank()

    results = [result1, result2, result3, result4]
    test_names = [
        "Small Grid (4 combos)",
        "Medium Grid (16 combos)",
        "Large Grid - 4 workers (36 combos)",
        "Large Grid - 2 workers (36 combos)"
    ]

    for i, (name, result) in enumerate(zip(test_names, results), 1):
        logger.info(f"{name}:")
        logger.metric(f"  Sequential: {result['sequential_time']:.2f}s")
        logger.metric(f"  Parallel:   {result['parallel_time']:.2f}s")
        logger.profit(f"  Speedup:    {result['speedup']:.2f}x")
        logger.blank()

    logger.separator()
    logger.success("Benchmark complete!")
    logger.separator()
    logger.blank()

    # Performance analysis
    logger.header("ANALYSIS")
    logger.blank()

    # Average speedup for large grids
    avg_speedup = (result2['speedup'] + result3['speedup']) / 2
    logger.info(f"Average speedup for medium/large grids: {avg_speedup:.2f}x")

    # Worker efficiency
    worker_efficiency = result3['speedup'] / 4  # 4 workers
    logger.info(f"Worker efficiency (4 workers): {worker_efficiency*100:.1f}%")
    logger.info("  (100% = perfect scaling, >70% = excellent)")

    logger.blank()
    logger.separator()
