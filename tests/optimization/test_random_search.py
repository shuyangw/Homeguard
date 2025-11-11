"""
Tests for RandomSearchOptimizer (Phase 4a).

Tests random search parameter optimization with various parameter types,
caching integration, and result quality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import RandomSearchOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover


class TestRandomSearchOptimizer:
    """Test suite for random search optimization."""

    @pytest.fixture
    def engine(self):
        """Create a basic backtest engine for testing."""
        return BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.0
        )

    @pytest.fixture
    def param_ranges(self):
        """Standard parameter ranges for testing."""
        return {
            'fast_window': (5, 30),      # Uniform int range
            'slow_window': (40, 120)     # Uniform int range
        }

    def test_basic_optimization(self, engine, param_ranges):
        """Test that basic random search optimization works."""
        optimizer = RandomSearchOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # Check result structure
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'metric' in result
        assert 'all_results' in result
        assert 'method' in result

        # Check method identifier
        assert result['method'] == 'random_search'

        # Check that parameters are in valid ranges
        assert 5 <= result['best_params']['fast_window'] <= 30
        assert 40 <= result['best_params']['slow_window'] <= 120

        # Check that we tested the right number of iterations
        assert len(result['all_results']) == 10

    def test_finds_reasonable_solution(self, engine, param_ranges):
        """Test that random search finds a reasonable solution."""
        optimizer = RandomSearchOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            n_iterations=20,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # Should find a valid solution
        assert result['best_params'] is not None
        assert result['best_value'] is not None
        assert not np.isnan(result['best_value'])
        assert not np.isinf(result['best_value'])

    def test_uniform_sampling(self, engine):
        """Test uniform integer sampling."""
        optimizer = RandomSearchOptimizer(engine)

        param_ranges = {
            'fast_window': (10, 20),     # Small range
            'slow_window': (50, 60)
        }

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=15,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # All parameters should be in range
        for res in result['all_results']:
            assert 10 <= res['params']['fast_window'] <= 20
            assert 50 <= res['params']['slow_window'] <= 60

    def test_discrete_choice_sampling(self, engine):
        """Test discrete parameter choices."""
        optimizer = RandomSearchOptimizer(engine)

        param_ranges = {
            'fast_window': [10, 15, 20],         # Discrete choices
            'slow_window': [50, 75, 100]
        }

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # All parameters should be from the discrete choices
        valid_fast = {10, 15, 20}
        valid_slow = {50, 75, 100}

        for res in result['all_results']:
            assert res['params']['fast_window'] in valid_fast
            assert res['params']['slow_window'] in valid_slow

    def test_reproducibility_with_seed(self, engine, param_ranges):
        """Test that random seed makes results reproducible."""
        optimizer = RandomSearchOptimizer(engine)

        # Run twice with same seed
        result1 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=1,  # Use 1 worker for determinism
            export_results=False,
            use_cache=False,
            random_seed=42
        )

        result2 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=1,
            export_results=False,
            use_cache=False,
            random_seed=42
        )

        # Results should be identical
        assert result1['best_params'] == result2['best_params']
        assert abs(result1['best_value'] - result2['best_value']) < 0.01

    def test_caching_integration(self, engine, param_ranges, tmp_path):
        """Test that caching works with random search."""
        from backtesting.optimization import CacheConfig

        cache_config = CacheConfig(
            enabled=True,
            cache_dir=tmp_path
        )

        optimizer = RandomSearchOptimizer(engine)

        # First run - should have 0 cache hits
        result1 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=True,
            cache_config=cache_config,
            random_seed=42  # Ensure same samples
        )

        assert result1['cache_hits'] == 0
        assert result1['cache_misses'] == 10

        # Second run with same seed - should have cache hits
        result2 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=True,
            cache_config=cache_config,
            random_seed=42  # Same seed = same samples
        )

        # All should be cached
        assert result2['cache_hits'] == 10
        assert result2['cache_misses'] == 0

    def test_different_metrics(self, engine, param_ranges):
        """Test optimization with different metrics."""
        optimizer = RandomSearchOptimizer(engine)

        metrics_to_test = ['sharpe_ratio', 'total_return', 'max_drawdown']

        for metric in metrics_to_test:
            result = optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2024-01-15',
                end_date='2024-01-31',
                metric=metric,
                n_iterations=10,
                max_workers=2,
                export_results=False,
                use_cache=False
            )

            assert result['metric'] == metric
            assert result['best_params'] is not None

    def test_csv_export(self, engine, param_ranges, tmp_path):
        """Test CSV export functionality."""
        optimizer = RandomSearchOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=True,
            output_dir=tmp_path,
            use_cache=False
        )

        # Find the optimization directory
        opt_dirs = list(tmp_path.glob('*_MovingAverageCrossover_AAPL_RandomSearch'))
        assert len(opt_dirs) == 1

        opt_dir = opt_dirs[0]

        # Check that CSV files were created
        csv_file = opt_dir / 'optimization_results.csv'
        summary_file = opt_dir / 'optimization_summary.txt'

        assert csv_file.exists()
        assert summary_file.exists()

        # Check CSV contents
        df = pd.read_csv(csv_file)
        assert len(df) == 10
        assert 'method' in df.columns
        assert 'param_fast_window' in df.columns
        assert 'param_slow_window' in df.columns
        assert 'sharpe_ratio' in df.columns
        assert df['method'].iloc[0] == 'RandomSearch'

    def test_timing_statistics(self, engine, param_ranges):
        """Test that timing statistics are returned."""
        optimizer = RandomSearchOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        assert 'total_time' in result
        assert 'avg_time_per_test' in result
        assert 'n_iterations' in result

        assert result['total_time'] > 0
        assert result['avg_time_per_test'] > 0
        assert result['n_iterations'] == 10

    def test_invalid_metric_raises_error(self, engine, param_ranges):
        """Test that invalid metric raises ValueError."""
        optimizer = RandomSearchOptimizer(engine)

        with pytest.raises(ValueError, match="Unknown metric"):
            optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                symbols='AAPL',
                start_date='2024-01-15',
                end_date='2024-01-31',
                metric='invalid_metric',
                n_iterations=5
            )

    def test_portfolio_object_returned(self, engine, param_ranges):
        """Test that portfolio object is returned for best params."""
        optimizer = RandomSearchOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        assert result['best_portfolio'] is not None
        # Portfolio should have stats method
        assert hasattr(result['best_portfolio'], 'stats')

    def test_cache_disabled(self, engine, param_ranges):
        """Test that caching can be disabled."""
        optimizer = RandomSearchOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            n_iterations=10,
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # Should have 0 cache hits/misses when disabled
        assert result['cache_hits'] == 0
        assert result['cache_misses'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
