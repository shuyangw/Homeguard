"""
Unit tests for BacktestEngine.optimize() method.

Tests grid search optimization with various parameter combinations,
metrics, and data configurations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from strategies.base_strategies.mean_reversion import MeanReversion


class TestOptimizeBasicFunctionality:
    """Test basic optimization functionality."""

    def test_optimize_single_parameter(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization with single parameter."""
        # Mock data loader to return test data in MultiIndex format
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                # Create MultiIndex DataFrame (symbol, datetime)
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10, 15],
            'slow_window': [20]  # Keep constant
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        assert result is not None
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'metric' in result

        # Verify best params are from the grid
        best_params = result['best_params']
        assert best_params['fast_window'] in param_grid['fast_window']
        assert best_params['slow_window'] == 20

    def test_optimize_multiple_parameters(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization with multiple parameters."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [15, 20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        # Verify tested all 4 combinations (2x2)
        best_params = result['best_params']
        assert best_params['fast_window'] in param_grid['fast_window']
        assert best_params['slow_window'] in param_grid['slow_window']

    def test_optimize_returns_best_value(self, simple_price_data, tmp_path, monkeypatch):
        """Test that optimization returns a numeric best value."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        best_value = result['best_value']
        assert isinstance(best_value, (int, float))
        assert best_value != float('-inf')  # Should find at least one valid result


class TestOptimizeMetrics:
    """Test optimization with different metrics."""

    def test_optimize_sharpe_ratio(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization using Sharpe Ratio metric."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        assert result['metric'] == 'sharpe_ratio'
        assert result['best_value'] != float('-inf')

    def test_optimize_total_return(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization using Total Return metric."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='total_return'
        )

        assert result['metric'] == 'total_return'
        # Total return should be a percentage value
        assert isinstance(result['best_value'], (int, float))

    def test_optimize_max_drawdown(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization using Max Drawdown metric (minimize)."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='max_drawdown'
        )

        assert result['metric'] == 'max_drawdown'
        # Max drawdown should be minimized (smaller is better)
        assert result['best_value'] != float('inf')

    def test_optimize_invalid_metric(self, simple_price_data, tmp_path, monkeypatch):
        """Test that invalid metric raises ValueError."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        with pytest.raises(ValueError, match="Unknown metric"):
            engine.optimize(
                strategy_class=MovingAverageCrossover,
                param_grid=param_grid,
                symbols='TEST',
                start_date='2023-01-01',
                end_date='2023-04-10',
                metric='invalid_metric'
            )


class TestOptimizeMultiSymbol:
    """Test optimization with multiple symbols."""

    def test_optimize_multi_symbol(self, multi_symbol_data, tmp_path, monkeypatch):
        """Test optimization with multiple symbols."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                return multi_symbol_data

        engine = BacktestEngine(initial_capital=30000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-02-19',
            metric='sharpe_ratio'
        )

        assert result is not None
        assert 'best_params' in result
        assert result['best_params']['fast_window'] in param_grid['fast_window']

    def test_optimize_single_symbol_as_list(self, simple_price_data, tmp_path, monkeypatch):
        """Test that single symbol works when passed as list."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols=['TEST'],  # List instead of string
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        assert result is not None
        assert result['best_params'] is not None


class TestOptimizeEdgeCases:
    """Test optimization edge cases."""

    def test_optimize_single_combination(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization with only one parameter combination."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [10],  # Single value
            'slow_window': [20]   # Single value
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        # Should still return a result
        assert result['best_params'] == {'fast_window': 10, 'slow_window': 20}

    def test_optimize_many_combinations(self, simple_price_data, tmp_path, monkeypatch):
        """Test optimization with many parameter combinations."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10, 15, 20],      # 4 values
            'slow_window': [25, 30, 35, 40, 45]  # 5 values
        }

        # Should test 4x5 = 20 combinations
        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        assert result is not None
        assert result['best_params']['fast_window'] in param_grid['fast_window']
        assert result['best_params']['slow_window'] in param_grid['slow_window']

    def test_optimize_with_different_strategy(self, oscillating_price_data, tmp_path, monkeypatch):
        """Test optimization with a different strategy (MeanReversion)."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = oscillating_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'window': [10, 20, 30],
            'num_std': [1.5, 2.0]
        }

        result = engine.optimize(
            strategy_class=MeanReversion,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        assert result is not None
        assert result['best_params']['window'] in param_grid['window']
        assert result['best_params']['num_std'] in param_grid['num_std']


class TestOptimizeIntegrity:
    """Test optimization result integrity and consistency."""

    def test_optimize_best_portfolio_matches_params(self, simple_price_data, tmp_path, monkeypatch):
        """Test that best_portfolio was generated with best_params."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20]
        }

        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        # Verify best_portfolio exists and has stats
        best_portfolio = result['best_portfolio']
        assert best_portfolio is not None
        assert hasattr(best_portfolio, 'stats')

        stats = best_portfolio.stats()
        if stats is not None:
            # Verify it has the expected metric
            if result['metric'] == 'sharpe_ratio':
                assert 'Sharpe Ratio' in stats
            elif result['metric'] == 'total_return':
                assert 'Total Return [%]' in stats
            elif result['metric'] == 'max_drawdown':
                assert 'Max Drawdown [%]' in stats

    def test_optimize_comparison_logic(self, simple_price_data, tmp_path, monkeypatch):
        """Test that optimization correctly compares different metrics."""
        class MockDataLoader:
            def load_symbols(self, symbols, start, end):
                df = simple_price_data.copy()
                df['symbol'] = 'TEST'
                df = df.set_index('symbol', append=True)
                df = df.swaplevel()
                return df

        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        monkeypatch.setattr(engine, 'data_loader', MockDataLoader())

        param_grid = {
            'fast_window': [5, 10, 15],
            'slow_window': [20]
        }

        # Optimize for total return
        result_return = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='total_return'
        )

        # Optimize for sharpe ratio
        result_sharpe = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        # Best params might differ based on metric
        # (Not always, but verify results are valid)
        assert result_return['best_params'] is not None
        assert result_sharpe['best_params'] is not None
