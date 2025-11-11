"""
Unit tests for GeneticOptimizer class.

Tests Genetic Algorithm optimization with evolutionary principles.
"""

import pytest
import pandas as pd
from datetime import datetime

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GeneticOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover


class TestGeneticOptimizer:
    """Test suite for Genetic Algorithm optimization."""

    @pytest.fixture
    def engine(self):
        """Create BacktestEngine for testing."""
        return BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.0001
        )

    @pytest.fixture
    def param_ranges(self):
        """Define parameter ranges for testing."""
        return {
            'fast_window': (5, 20),
            'slow_window': (30, 60)
        }

    def test_genetic_optimizer_initialization(self, engine):
        """Test that GeneticOptimizer can be initialized."""
        optimizer = GeneticOptimizer(engine)
        assert optimizer is not None
        assert optimizer.engine == engine

    def test_basic_optimization(self, engine, param_ranges):
        """Test basic Genetic Algorithm optimization works."""
        optimizer = GeneticOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-03-01',
            metric='sharpe_ratio',
            population_size=10,  # Small for fast testing
            n_generations=3,
            mutation_rate=0.1,
            crossover_rate=0.7,
            random_seed=42
        )

        # Check result structure
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'convergence_data' in result
        assert 'total_evaluations' in result

        # Check best params
        assert 'fast_window' in result['best_params']
        assert 'slow_window' in result['best_params']

        # Check parameter bounds
        assert 5 <= result['best_params']['fast_window'] <= 20
        assert 30 <= result['best_params']['slow_window'] <= 60

    def test_population_initialization(self, engine, param_ranges):
        """Test that population is initialized correctly."""
        optimizer = GeneticOptimizer(engine)

        population = optimizer._initialize_population(param_ranges, 20)

        assert len(population) == 20
        for individual in population:
            assert 'fast_window' in individual.params
            assert 'slow_window' in individual.params
            assert 5 <= individual.params['fast_window'] <= 20
            assert 30 <= individual.params['slow_window'] <= 60

    def test_diversity_tracking(self, engine, param_ranges):
        """Test that diversity is tracked correctly."""
        optimizer = GeneticOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=3,
            random_seed=42
        )

        # Check convergence data
        conv_data = result['convergence_data']
        assert 'diversity' in conv_data
        assert 'best_fitness' in conv_data
        assert 'avg_fitness' in conv_data

        # Diversity history should have n_generations + 1 entries (initial + generations)
        assert len(conv_data['diversity']) == result['n_generations'] + 1

    def test_different_population_sizes(self, engine, param_ranges):
        """Test optimization with different population sizes."""
        optimizer = GeneticOptimizer(engine)

        for pop_size in [10, 20]:
            result = optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-02-01',
                metric='sharpe_ratio',
                population_size=pop_size,
                n_generations=2,
                random_seed=42
            )

            assert result['best_params'] is not None
            assert result['total_evaluations'] > 0

    def test_mutation_and_crossover_rates(self, engine, param_ranges):
        """Test different mutation and crossover rates."""
        optimizer = GeneticOptimizer(engine)

        # High mutation rate
        result1 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            mutation_rate=0.3,  # High
            crossover_rate=0.7,
            random_seed=42
        )

        # Low mutation rate
        result2 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            mutation_rate=0.01,  # Low
            crossover_rate=0.7,
            random_seed=43  # Different seed
        )

        assert result1['best_params'] is not None
        assert result2['best_params'] is not None

    def test_cache_integration(self, engine, param_ranges):
        """Test that caching works with Genetic Algorithm."""
        optimizer = GeneticOptimizer(engine)

        # First run
        result1 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            use_cache=True,
            random_seed=42
        )

        # Second run (should use cache)
        result2 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            use_cache=True,
            random_seed=42
        )

        # Should have cache hits in second run
        assert result2['cache_hits'] > 0

    def test_early_stopping(self, engine, param_ranges):
        """Test convergence-based early stopping."""
        optimizer = GeneticOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=10,
            convergence_patience=2,  # Stop after 2 gens without improvement
            random_seed=42
        )

        # May or may not early stop, but structure should be correct
        assert 'early_stopped' in result
        if result['early_stopped']:
            assert result['n_generations'] < 10

    def test_discrete_parameters(self, engine):
        """Test optimization with discrete parameters."""
        optimizer = GeneticOptimizer(engine)

        param_ranges = {
            'fast_window': [5, 10, 15, 20],  # Discrete list
            'slow_window': (30, 60)  # Continuous range
        }

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            random_seed=42
        )

        # fast_window should be one of the discrete values
        assert result['best_params']['fast_window'] in [5, 10, 15, 20]
        # slow_window should be in continuous range
        assert 30 <= result['best_params']['slow_window'] <= 60

    def test_reproducibility(self, engine, param_ranges):
        """Test that same random seed gives same results."""
        optimizer1 = GeneticOptimizer(engine)
        optimizer2 = GeneticOptimizer(engine)

        result1 = optimizer1.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            use_cache=False,  # Disable cache for pure reproducibility
            random_seed=42
        )

        result2 = optimizer2.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            use_cache=False,
            random_seed=42
        )

        # With same seed, should get same results
        assert result1['best_params'] == result2['best_params']
        assert result1['best_value'] == result2['best_value']

    def test_different_metrics(self, engine, param_ranges):
        """Test optimization with different metrics."""
        optimizer = GeneticOptimizer(engine)

        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown']:
            result = optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-02-01',
                metric=metric,
                population_size=10,
                n_generations=2,
                random_seed=42
            )

            assert result['best_params'] is not None
            assert result['metric'] == metric

    def test_csv_export(self, engine, param_ranges):
        """Test that results are exported to CSV."""
        optimizer = GeneticOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            population_size=10,
            n_generations=2,
            export_results=True,
            random_seed=42
        )

        assert 'csv_path' in result
        if result['csv_path']:
            assert result['csv_path'].exists()

    def test_invalid_parameters_raise_error(self, engine, param_ranges):
        """Test that invalid parameters raise errors."""
        optimizer = GeneticOptimizer(engine)

        # Invalid mutation rate
        with pytest.raises(ValueError):
            optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-02-01',
                metric='sharpe_ratio',
                population_size=10,
                n_generations=2,
                mutation_rate=1.5  # Invalid (>1)
            )

        # Invalid crossover rate
        with pytest.raises(ValueError):
            optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-02-01',
                metric='sharpe_ratio',
                population_size=10,
                n_generations=2,
                crossover_rate=-0.5  # Invalid (<0)
            )

        # Invalid metric
        with pytest.raises(ValueError):
            optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-02-01',
                metric='invalid_metric',
                population_size=10,
                n_generations=2
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
