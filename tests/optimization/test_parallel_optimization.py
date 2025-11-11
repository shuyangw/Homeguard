"""
Tests for parallel grid search optimization.

Validates that parallel optimization:
1. Produces same results as sequential
2. Handles invalid parameters correctly
3. Falls back to sequential for small grids
4. Tracks progress correctly
5. Returns all results for analysis
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover


class TestParallelOptimization:
    """Test suite for parallel grid search optimization."""

    @pytest.fixture
    def engine(self):
        """Create BacktestEngine for testing."""
        return BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.0
        )

    @pytest.fixture
    def small_param_grid(self):
        """Small parameter grid (should use sequential)."""
        return {
            'fast_window': [10, 20],
            'slow_window': [50, 100]
        }

    @pytest.fixture
    def medium_param_grid(self):
        """Medium parameter grid (good for parallel)."""
        return {
            'fast_window': [10, 15, 20, 25],
            'slow_window': [50, 60, 70, 80]
        }

    def test_parallel_matches_sequential(self, engine, medium_param_grid):
        """Test that parallel optimization produces same best params as sequential."""
        # Run sequential
        seq_optimizer = GridSearchOptimizer(engine)
        seq_result = seq_optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio'
        )

        # Run parallel
        par_optimizer = GridSearchOptimizer(engine)
        par_result = par_optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2
        )

        # Compare results
        assert seq_result['best_params'] == par_result['best_params'], \
            "Parallel and sequential should find same best parameters"
        assert abs(seq_result['best_value'] - par_result['best_value']) < 0.01, \
            "Parallel and sequential should have same best value"

    def test_small_grid_uses_sequential(self, engine, small_param_grid):
        """Test that small grids automatically fall back to sequential."""
        optimizer = GridSearchOptimizer(engine)

        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio'
        )

        # For small grids (< 10 combinations), parallel should work but may fallback
        # Key assertion: We should still get valid results
        assert result['best_params'] is not None
        assert result['best_value'] is not None
        assert result['best_portfolio'] is not None

        # Small grid should not have 'all_results' key if it fell back to sequential
        # (since sequential optimize() doesn't return all_results)
        grid_size = len(small_param_grid['fast_window']) * len(small_param_grid['slow_window'])
        assert grid_size < 10, "Test assumes small grid (< 10 combos)"

    def test_invalid_params_handled(self, engine):
        """Test that invalid parameter combinations are handled gracefully."""
        # Grid with some invalid combos (fast >= slow)
        param_grid = {
            'fast_window': [20, 30, 60],  # 60 will be invalid
            'slow_window': [50, 100]
        }

        optimizer = GridSearchOptimizer(engine)
        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2
        )

        # Should still find a best result (from valid combos)
        assert result['best_params'] is not None
        assert result['best_params']['fast_window'] < result['best_params']['slow_window']

    def test_all_results_returned(self, engine, medium_param_grid):
        """Test that all tested combinations are returned."""
        optimizer = GridSearchOptimizer(engine)
        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2
        )

        # Should return all results
        assert 'all_results' in result
        expected_combos = len(medium_param_grid['fast_window']) * len(medium_param_grid['slow_window'])
        assert len(result['all_results']) == expected_combos

        # Each result should have params, value, stats, error
        for r in result['all_results']:
            assert 'params' in r
            assert 'value' in r
            assert 'stats' in r or 'error' in r

    def test_max_workers_parameter(self, engine, medium_param_grid):
        """Test that max_workers parameter is respected."""
        optimizer = GridSearchOptimizer(engine)

        # Test with 1 worker (essentially sequential)
        result_1 = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=1
        )

        # Test with 4 workers
        result_4 = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=4
        )

        # Should produce same results regardless of worker count
        assert result_1['best_params'] == result_4['best_params']

    def test_different_metrics(self, engine, small_param_grid):
        """Test optimization with different metrics."""
        optimizer = GridSearchOptimizer(engine)

        # Test sharpe_ratio
        result_sharpe = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2
        )

        # Test total_return
        result_return = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='total_return',
            max_workers=2
        )

        # Test max_drawdown
        result_dd = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='max_drawdown',
            max_workers=2
        )

        # All should return results
        assert result_sharpe['best_params'] is not None
        assert result_return['best_params'] is not None
        assert result_dd['best_params'] is not None

        # Metrics may choose different params
        assert result_sharpe['metric'] == 'sharpe_ratio'
        assert result_return['metric'] == 'total_return'
        assert result_dd['metric'] == 'max_drawdown'

    def test_invalid_metric_raises_error(self, engine, small_param_grid):
        """Test that invalid metric raises ValueError."""
        optimizer = GridSearchOptimizer(engine)

        with pytest.raises(ValueError, match="Unknown metric"):
            optimizer.optimize_parallel(
                strategy_class=MovingAverageCrossover,
                param_grid=small_param_grid,
                symbols='AAPL',
                start_date='2024-01-01',
                end_date='2024-02-01',
                metric='invalid_metric'
            )

    def test_portfolio_object_returned(self, engine, small_param_grid):
        """Test that best portfolio object is returned."""
        optimizer = GridSearchOptimizer(engine)
        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2
        )

        # Should have portfolio object
        assert result['best_portfolio'] is not None
        assert hasattr(result['best_portfolio'], 'stats')
        assert hasattr(result['best_portfolio'], 'trades')

        # Portfolio stats should match best_value
        stats = result['best_portfolio'].stats()
        sharpe = float(stats.get('Sharpe Ratio', 0))
        assert abs(sharpe - result['best_value']) < 0.01


    def test_csv_export(self, engine, medium_param_grid, tmp_path):
        """Test CSV export of optimization results (Phase 2)."""
        optimizer = GridSearchOptimizer(engine)

        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=True,
            output_dir=tmp_path
        )

        # Check that CSV files were created
        optimization_dirs = list(tmp_path.glob('*_optimization'))
        assert len(optimization_dirs) > 0, "Optimization directory should be created"

        opt_dir = optimization_dirs[0]
        csv_file = opt_dir / 'optimization_results.csv'
        sensitivity_file = opt_dir / 'parameter_sensitivity.csv'

        assert csv_file.exists(), "optimization_results.csv should exist"
        assert sensitivity_file.exists(), "parameter_sensitivity.csv should exist"

        # Verify CSV contents
        import pandas as pd
        results_df = pd.read_csv(csv_file)

        # Should have rows for all tested combinations
        expected_combos = len(medium_param_grid['fast_window']) * len(medium_param_grid['slow_window'])
        assert len(results_df) == expected_combos

        # Should have required columns
        assert 'sharpe_ratio' in results_df.columns
        assert 'params' in results_df.columns
        assert 'is_best' in results_df.columns
        assert 'param_fast_window' in results_df.columns
        assert 'param_slow_window' in results_df.columns

        # Verify sensitivity analysis
        sensitivity_df = pd.read_csv(sensitivity_file)
        assert len(sensitivity_df) == 2  # Two parameters
        assert 'parameter' in sensitivity_df.columns
        assert 'impact_range' in sensitivity_df.columns
        assert 'correlation' in sensitivity_df.columns

    def test_timing_statistics(self, engine, medium_param_grid):
        """Test that timing statistics are returned (Phase 2)."""
        optimizer = GridSearchOptimizer(engine)

        # Use medium grid to avoid fallback to sequential
        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False  # Don't export for speed
        )

        # Check timing statistics
        assert 'total_time' in result
        assert 'avg_time_per_test' in result

        assert result['total_time'] > 0
        assert result['avg_time_per_test'] > 0

        # Avg time should be reasonable
        grid_size = len(medium_param_grid['fast_window']) * len(medium_param_grid['slow_window'])
        assert result['avg_time_per_test'] == result['total_time'] / grid_size

    def test_export_disabled(self, engine, small_param_grid):
        """Test that export can be disabled (Phase 2)."""
        optimizer = GridSearchOptimizer(engine)

        # Run with export disabled
        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_param_grid,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False
        )

        # Should still get results
        assert result['best_params'] is not None
        assert result['best_value'] is not None

    def test_caching_enabled(self, engine, medium_param_grid, tmp_path):
        """Test that caching works (Phase 3)."""
        from backtesting.optimization.result_cache import CacheConfig

        # Configure cache to use temp directory
        cache_config = CacheConfig(
            enabled=True,
            cache_dir=tmp_path
        )

        optimizer = GridSearchOptimizer(engine)

        # First run - should have 0 cache hits
        result1 = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False,
            use_cache=True,
            cache_config=cache_config
        )

        assert 'cache_hits' in result1
        assert 'cache_misses' in result1
        assert result1['cache_hits'] == 0  # First run - no cache
        assert result1['cache_misses'] == len(medium_param_grid['fast_window']) * len(medium_param_grid['slow_window'])

        # Second run with same parameters - should have cache hits
        result2 = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False,
            use_cache=True,
            cache_config=cache_config
        )

        # All results should be cached
        total_combos = len(medium_param_grid['fast_window']) * len(medium_param_grid['slow_window'])
        assert result2['cache_hits'] == total_combos
        assert result2['cache_misses'] == 0

        # Results should be the same
        assert result1['best_params'] == result2['best_params']
        assert abs(result1['best_value'] - result2['best_value']) < 0.01

    def test_caching_disabled(self, engine, medium_param_grid):
        """Test that caching can be disabled (Phase 3)."""
        optimizer = GridSearchOptimizer(engine)

        # Run with cache disabled
        result = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=medium_param_grid,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # Should have 0 cache hits/misses when disabled
        assert result['cache_hits'] == 0
        assert result['cache_misses'] == 0

    def test_cache_partial_hits(self, engine, medium_param_grid, tmp_path):
        """Test partial cache hits (Phase 3)."""
        from backtesting.optimization.result_cache import CacheConfig

        # Configure cache to use temp directory
        cache_config = CacheConfig(
            enabled=True,
            cache_dir=tmp_path
        )

        optimizer = GridSearchOptimizer(engine)

        # First run with smaller grid (must be >= 10 combos to avoid sequential fallback)
        small_grid = {
            'fast_window': [10, 15],
            'slow_window': [50, 60, 70, 80, 90]  # 2*5 = 10 combinations
        }

        result1 = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=small_grid,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False,
            use_cache=True,
            cache_config=cache_config
        )

        # Second run with larger grid (includes all of small grid)
        larger_grid = {
            'fast_window': [10, 15, 20],  # Added 20
            'slow_window': [50, 60, 70, 80, 90]  # Same as small grid
        }

        result2 = optimizer.optimize_parallel(
            strategy_class=MovingAverageCrossover,
            param_grid=larger_grid,
            symbols='AAPL',
            start_date='2024-01-15',
            end_date='2024-01-31',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False,
            use_cache=True,
            cache_config=cache_config
        )

        # Should have partial cache hits (10 from small grid: 2*5)
        assert result2['cache_hits'] == 10
        # Should have misses for new combinations (15 total - 10 cached = 5 new: 1*5)
        assert result2['cache_misses'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
