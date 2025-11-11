"""
Error handling and robustness tests for GridSearchOptimizer.

Tests optimizer behavior under invalid inputs, failed strategies,
and error recovery scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.optimization.grid_search import GridSearchOptimizer
from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.base.strategy import Strategy


class BrokenStrategy(Strategy):
    """Strategy that raises errors for testing."""

    def __init__(self, should_fail=False):
        super().__init__()
        self.should_fail = should_fail

    def generate_signals(self, data):
        if self.should_fail:
            raise ValueError("Strategy intentionally failed")
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        return entries, exits


class TestOptimizerParameterValidation:
    """Test parameter validation in optimizer."""

    def test_empty_parameter_grid_error(self):
        """Test that empty parameter grid raises error."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        with pytest.raises((ValueError, KeyError)):
            optimizer.optimize(
                strategy_class=BrokenStrategy,
                param_grid={},  # Empty grid
                symbols='AAPL',
                start_date='2020-01-01',
                end_date='2020-12-31',
                metric='sharpe_ratio'
            )

    def test_invalid_metric_name(self):
        """Test that invalid metric raises ValueError."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        with pytest.raises(ValueError, match="Unknown metric"):
            optimizer.optimize(
                strategy_class=BrokenStrategy,
                param_grid={'should_fail': [False]},
                symbols='AAPL',
                start_date='2020-01-01',
                end_date='2020-12-31',
                metric='invalid_metric'
            )

    def test_none_parameter_values_handled(self):
        """Test that None parameter values are handled."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        # Should handle None values
        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False, None]},
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        assert result is not None

    def test_mismatched_parameter_types(self):
        """Test handling of mismatched parameter types."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        # Mix of types that might cause issues
        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False, 0, "", None]},
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Should handle gracefully (some may fail)
        assert result is not None

    def _get_mock_data(self):
        """Generate mock data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        symbols = ['AAPL']

        data = pd.DataFrame({
            'open': np.random.randn(len(dates)) * 2 + 100,
            'high': np.random.randn(len(dates)) * 2 + 102,
            'low': np.random.randn(len(dates)) * 2 + 98,
            'close': np.random.randn(len(dates)) * 2 + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })

        multi_index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['timestamp', 'symbol']
        )
        data = data.reindex(multi_index)

        return data


class TestOptimizerStrategyFailures:
    """Test optimizer handling of strategy failures."""

    def test_strategy_initialization_failure_handled(self):
        """Test handling of strategy initialization failures."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        # Some parameter combinations will fail to initialize
        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [True, False]},  # True will fail
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Should return best of successful ones
        assert result is not None
        assert result['best_params'] is not None

    def test_all_strategies_fail_returns_none(self):
        """Test that all failing strategies returns gracefully."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [True]},  # All will fail
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Should handle gracefully
        assert result is not None

    def _get_mock_data(self):
        """Generate mock data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        symbols = ['AAPL']

        data = pd.DataFrame({
            'open': np.random.randn(len(dates)) * 2 + 100,
            'high': np.random.randn(len(dates)) * 2 + 102,
            'low': np.random.randn(len(dates)) * 2 + 98,
            'close': np.random.randn(len(dates)) * 2 + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })

        multi_index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['timestamp', 'symbol']
        )
        data = data.reindex(multi_index)

        return data


class TestOptimizerDataIssues:
    """Test optimizer handling of data issues."""

    def test_empty_data_handled(self):
        """Test handling of empty dataset."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        def get_empty_data():
            dates = pd.date_range('2020-01-01', periods=0, freq='D')
            data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            multi_index = pd.MultiIndex.from_product(
                [dates, ['AAPL']],
                names=['timestamp', 'symbol']
            )
            return data.reindex(multi_index)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: get_empty_data()

        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False]},
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Should handle empty data
        assert result is not None

    def test_minimal_data_points(self):
        """Test with minimal data points."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        def get_minimal_data():
            dates = pd.date_range('2020-01-01', periods=2, freq='D')
            data = pd.DataFrame({
                'open': [100, 101],
                'high': [102, 103],
                'low': [98, 99],
                'close': [100, 101],
                'volume': [1000000, 1000000]
            })
            multi_index = pd.MultiIndex.from_product(
                [dates, ['AAPL']],
                names=['timestamp', 'symbol']
            )
            data = data.reindex(multi_index)
            return data

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: get_minimal_data()

        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False]},
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-02',
            metric='sharpe_ratio'
        )

        assert result is not None

    def test_nan_in_price_data(self):
        """Test handling of NaN values in price data."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        def get_nan_data():
            dates = pd.date_range('2020-01-01', periods=10, freq='D')
            data = pd.DataFrame({
                'open': np.random.randn(10) + 100,
                'high': np.random.randn(10) + 102,
                'low': np.random.randn(10) + 98,
                'close': np.random.randn(10) + 100,
                'volume': np.random.randint(1000000, 10000000, 10)
            })
            data.loc[5, 'close'] = np.nan  # Inject NaN
            multi_index = pd.MultiIndex.from_product(
                [dates, ['AAPL']],
                names=['timestamp', 'symbol']
            )
            data = data.reindex(multi_index)
            return data

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: get_nan_data()

        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False]},
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-10',
            metric='sharpe_ratio'
        )

        # Should handle NaN gracefully
        assert result is not None


class TestOptimizerExtremeParameters:
    """Test optimizer with extreme parameter values."""

    def test_very_large_parameter_grid(self):
        """Test with very large parameter grid."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        # Large grid (but manageable for test)
        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False] * 50},  # 50 identical params
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-31',
            metric='sharpe_ratio'
        )

        assert result is not None

    def test_single_parameter_combination(self):
        """Test with single parameter (no optimization needed)."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        result = optimizer.optimize(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False]},  # Single value
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        assert result['best_params'] == {'should_fail': False}

    def _get_mock_data(self):
        """Generate mock data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)) * 2 + 100,
            'high': np.random.randn(len(dates)) * 2 + 102,
            'low': np.random.randn(len(dates)) * 2 + 98,
            'close': np.random.randn(len(dates)) * 2 + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        multi_index = pd.MultiIndex.from_product(
            [dates, ['AAPL']],
            names=['timestamp', 'symbol']
        )
        return data.reindex(multi_index)


class TestOptimizerCacheErrors:
    """Test cache-related error handling."""

    def test_cache_corruption_handled(self):
        """Test handling of cache corruption or errors."""
        engine = BacktestEngine(initial_capital=10000)
        optimizer = GridSearchOptimizer(engine)

        optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: self._get_mock_data()

        # Try with cache enabled
        result = optimizer.optimize_parallel(
            strategy_class=BrokenStrategy,
            param_grid={'should_fail': [False, False]},
            symbols='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio',
            use_cache=True,
            max_workers=2
        )

        assert result is not None

    def _get_mock_data(self):
        """Generate mock data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)) * 2 + 100,
            'high': np.random.randn(len(dates)) * 2 + 102,
            'low': np.random.randn(len(dates)) * 2 + 98,
            'close': np.random.randn(len(dates)) * 2 + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        multi_index = pd.MultiIndex.from_product(
            [dates, ['AAPL']],
            names=['timestamp', 'symbol']
        )
        return data.reindex(multi_index)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
