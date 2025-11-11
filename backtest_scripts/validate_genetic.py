"""
Genetic Algorithm validation script.

Tests the GeneticOptimizer end-to-end and compares with Random Search.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GeneticOptimizer, RandomSearchOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def validate_genetic_algorithm():
    """Validate Genetic Algorithm optimizer."""
    logger.blank()
    logger.separator()
    logger.header("VALIDATING GENETIC ALGORITHM OPTIMIZER")
    logger.separator()
    logger.blank()

    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0001
    )

    # Create optimizer
    optimizer = GeneticOptimizer(engine)

    # Define parameter ranges
    param_ranges = {
        'fast_window': (5, 30),
        'slow_window': (40, 120)
    }

    logger.info("Testing Genetic Algorithm with small population...")
    logger.info(f"Parameter ranges: {param_ranges}")
    logger.info(f"Population size: 20")
    logger.info(f"Generations: 5")

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
            population_size=20,
            n_generations=5,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.2,
            max_workers=2,
            use_cache=True,
            export_results=True,
            enable_plots=True,
            random_seed=42  # For reproducibility
        )

        elapsed = time.time() - start_time

        # Validate results
        logger.blank()
        logger.success("✅ Genetic Algorithm completed successfully!")
        logger.blank()
        logger.metric(f"Best parameters: {result['best_params']}")
        logger.profit(f"Best Sharpe Ratio: {result['best_value']:.4f}")
        logger.info(f"Generations: {result['n_generations']}")
        logger.info(f"Total evaluations: {result['total_evaluations']}")
        logger.info(f"Early stopped: {result.get('early_stopped', False)}")
        logger.info(f"Cache hits: {result['cache_hits']}")
        logger.info(f"Cache misses: {result['cache_misses']}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average time per evaluation: {result['avg_time_per_evaluation']:.2f}s")

        # Validate result structure
        assert 'best_params' in result, "Missing best_params"
        assert 'best_value' in result, "Missing best_value"
        assert 'best_portfolio' in result, "Missing best_portfolio"
        assert 'convergence_data' in result, "Missing convergence_data"
        assert 'total_evaluations' in result, "Missing total_evaluations"

        # Validate convergence data
        conv_data = result['convergence_data']
        assert 'diversity' in conv_data, "Missing diversity in convergence_data"
        assert 'best_fitness' in conv_data, "Missing best_fitness"
        assert 'avg_fitness' in conv_data, "Missing avg_fitness"
        assert len(conv_data['diversity']) == result['n_generations'] + 1, "Diversity length mismatch"

        # Validate parameter bounds
        assert 5 <= result['best_params']['fast_window'] <= 30, "fast_window out of bounds"
        assert 40 <= result['best_params']['slow_window'] <= 120, "slow_window out of bounds"

        # Check plots and CSV
        if result.get('plots_path'):
            logger.info(f"Evolution plots: {result['plots_path']}")
            assert result['plots_path'].exists(), "Plots file not created"

        if result.get('csv_path'):
            logger.info(f"Results CSV: {result['csv_path']}")
            assert result['csv_path'].exists(), "CSV file not created"

        logger.blank()
        logger.success("✅ All Genetic Algorithm validations passed!")
        logger.separator()

        return result

    except Exception as e:
        logger.error(f"❌ Genetic Algorithm validation failed: {e}")
        logger.separator()
        raise


def compare_genetic_vs_random():
    """Compare Genetic Algorithm vs Random Search."""
    logger.blank()
    logger.separator()
    logger.header("COMPARING GENETIC vs RANDOM SEARCH")
    logger.separator()
    logger.blank()

    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0001
    )

    param_ranges = {
        'fast_window': (5, 30),
        'slow_window': (40, 120)
    }

    # Test Genetic Algorithm
    logger.info("Running Genetic Algorithm (20 individuals × 5 generations)...")
    genetic_optimizer = GeneticOptimizer(engine)

    genetic_start = time.time()
    genetic_result = genetic_optimizer.optimize(
        strategy_class=MovingAverageCrossover,
        param_ranges=param_ranges,
        symbols='AAPL',
        start_date='2023-01-01',
        end_date='2023-06-01',
        metric='sharpe_ratio',
        population_size=20,
        n_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        max_workers=2,
        use_cache=True,
        export_results=False,
        enable_plots=False,
        random_seed=42
    )
    genetic_time = time.time() - genetic_start

    # Test Random Search (same number of evaluations)
    logger.blank()
    logger.info("Running Random Search (100 iterations)...")
    random_optimizer = RandomSearchOptimizer(engine)

    random_start = time.time()
    random_result = random_optimizer.optimize(
        strategy_class=MovingAverageCrossover,
        param_ranges=param_ranges,
        symbols='AAPL',
        start_date='2023-01-01',
        end_date='2023-06-01',
        metric='sharpe_ratio',
        n_iterations=100,
        max_workers=2,
        use_cache=True,
        export_results=False,
        random_seed=42
    )
    random_time = time.time() - random_start

    # Compare results
    logger.blank()
    logger.separator()
    logger.header("COMPARISON RESULTS")
    logger.separator()
    logger.blank()

    # Best values
    genetic_best = genetic_result['best_value']
    random_best = random_result['best_value']

    logger.metric(f"Genetic Algorithm best Sharpe:   {genetic_best:.4f}")
    logger.metric(f"Random Search best Sharpe:        {random_best:.4f}")

    if genetic_best > random_best:
        improvement = ((genetic_best - random_best) / abs(random_best)) * 100
        logger.success(f"✅ Genetic is better by {improvement:.1f}%")
    elif random_best > genetic_best:
        improvement = ((random_best - genetic_best) / abs(genetic_best)) * 100
        logger.warning(f"⚠️  Random Search is better by {improvement:.1f}%")
    else:
        logger.info("Both methods found same optimum")

    logger.blank()

    # Efficiency
    genetic_evals = genetic_result['total_evaluations']
    random_evals = random_result['n_iterations']

    logger.metric(f"Genetic evaluations:              {genetic_evals}")
    logger.metric(f"Random evaluations:               {random_evals}")
    logger.blank()

    logger.metric(f"Genetic total time:               {genetic_time:.2f}s")
    logger.metric(f"Random total time:                {random_time:.2f}s")
    logger.blank()

    genetic_per_eval = genetic_result['avg_time_per_evaluation']
    random_per_eval = random_result['avg_time_per_test']

    logger.metric(f"Genetic time per evaluation:      {genetic_per_eval:.2f}s")
    logger.metric(f"Random time per evaluation:       {random_per_eval:.2f}s")
    logger.blank()

    # Cache efficiency
    genetic_cache_rate = (genetic_result['cache_hits'] / genetic_evals * 100) if genetic_evals > 0 else 0
    random_cache_rate = (random_result['cache_hits'] / random_evals * 100) if random_evals > 0 else 0

    logger.metric(f"Genetic cache hit rate:           {genetic_cache_rate:.1f}%")
    logger.metric(f"Random cache hit rate:            {random_cache_rate:.1f}%")
    logger.blank()

    # Diversity (Genetic only)
    if 'convergence_data' in genetic_result:
        final_diversity = genetic_result['convergence_data']['diversity'][-1]
        logger.metric(f"Genetic final diversity:          {final_diversity:.4f}")
        logger.info("(Higher diversity = more exploration)")

    logger.blank()
    logger.separator()

    return {
        'genetic': genetic_result,
        'random': random_result,
        'genetic_time': genetic_time,
        'random_time': random_time
    }


def main():
    """Run Genetic Algorithm validation."""
    logger.blank()
    logger.separator()
    logger.header("GENETIC ALGORITHM VALIDATION SUITE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.separator()
    logger.blank()

    results = {}

    # Test 1: Validate Genetic Algorithm
    try:
        results['genetic'] = validate_genetic_algorithm()
    except Exception as e:
        logger.error(f"Genetic Algorithm validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 2: Compare with Random Search
    try:
        comparison = compare_genetic_vs_random()
        results.update(comparison)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail overall if comparison fails
        pass

    # Final summary
    logger.blank()
    logger.separator()
    logger.header("VALIDATION SUMMARY")
    logger.separator()
    logger.blank()

    if results.get('genetic'):
        logger.success("✅ Genetic Algorithm: PASSED")
    else:
        logger.error("❌ Genetic Algorithm: FAILED")

    logger.blank()
    logger.separator()

    return 0


if __name__ == '__main__':
    sys.exit(main())
