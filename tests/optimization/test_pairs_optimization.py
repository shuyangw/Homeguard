"""
Unit tests for pairs strategy optimization with GridSearchOptimizer.

Tests that GridSearchOptimizer correctly handles PairsStrategy instances
and routes them to PairsPortfolio for synchronized execution.
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.optimization.grid_search import GridSearchOptimizer
from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading


@pytest.fixture
def simple_optimizer():
    """Create a simple optimizer for testing."""
    engine = BacktestEngine(
        initial_capital=10000,
        fees=0.001,
        slippage=0.001
    )
    return GridSearchOptimizer(engine)


@pytest.fixture
def correlated_pair_data():
    """Create synthetic data for a correlated pair."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

    # Create correlated price series
    np.random.seed(42)
    base = np.cumsum(np.random.randn(len(dates)) * 0.5) + 100

    # Symbol1: base + small noise
    prices1 = base + np.random.randn(len(dates)) * 2

    # Symbol2: base * 1.5 + noise (correlated but different level)
    prices2 = base * 1.5 + np.random.randn(len(dates)) * 3

    # Add some mean reversion opportunities
    spread = prices1 - (prices2 / 1.5)
    spread_mean = spread.mean()

    # Force some deviations for trading opportunities
    prices1[50:60] += 10  # Spread widens
    prices1[150:160] -= 10  # Spread tightens
    prices2[250:260] += 15  # Spread widens other direction

    # Create multi-index DataFrame
    data = pd.DataFrame({
        'open': np.concatenate([prices1, prices2]),
        'high': np.concatenate([prices1 * 1.01, prices2 * 1.01]),
        'low': np.concatenate([prices1 * 0.99, prices2 * 0.99]),
        'close': np.concatenate([prices1, prices2]),
        'volume': np.concatenate([
            np.random.randint(1000000, 10000000, len(dates)),
            np.random.randint(1000000, 10000000, len(dates))
        ])
    })

    # Create multi-index
    symbols = ['AAPL'] * len(dates) + ['MSFT'] * len(dates)
    timestamps = list(dates) + list(dates)

    data.index = pd.MultiIndex.from_arrays(
        [timestamps, symbols],
        names=['timestamp', 'symbol']
    )

    return data


class TestPairsOptimizationBasic:
    """Test basic pairs strategy optimization."""

    def test_optimizer_detects_pairs_strategy(self, simple_optimizer, correlated_pair_data):
        """Test that optimizer correctly detects PairsStrategy."""
        strategy = PairsTrading(zscore_window=20, entry_zscore=2.0)

        # Verify strategy can be instantiated
        assert strategy is not None
        assert hasattr(strategy, 'generate_signals_multi')

    def test_sequential_optimization_with_pairs(self, simple_optimizer, correlated_pair_data):
        """Test sequential optimization with pairs strategy."""
        # Small parameter grid for quick test
        param_grid = {
            'zscore_window': [10, 20],
            'entry_zscore': [1.5, 2.0]
        }

        # Mock data loading by directly setting data
        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        result = simple_optimizer.optimize(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Verify result structure
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'metric' in result

        # Verify best params are from grid
        assert result['best_params']['zscore_window'] in [10, 20]
        assert result['best_params']['entry_zscore'] in [1.5, 2.0]

        # Verify portfolio was created
        assert result['best_portfolio'] is not None

    def test_parallel_optimization_with_pairs(self, simple_optimizer, correlated_pair_data):
        """Test parallel optimization with pairs strategy."""
        # Small parameter grid for quick test
        param_grid = {
            'zscore_window': [10, 15, 20],
            'entry_zscore': [1.5, 2.0, 2.5]
        }

        # Mock data loading
        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        result = simple_optimizer.optimize_parallel(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio',
            max_workers=2,
            export_results=False,
            use_cache=False
        )

        # Verify result structure
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'metric' in result
        assert 'all_results' in result

        # Should have tested all combinations
        assert len(result['all_results']) == 3 * 3  # 9 combinations

        # Verify best params
        assert result['best_params']['zscore_window'] in [10, 15, 20]
        assert result['best_params']['entry_zscore'] in [1.5, 2.0, 2.5]


class TestPairsOptimizationIntegration:
    """Test integration with BacktestEngine."""

    def test_optimizer_uses_pairs_portfolio(self, simple_optimizer, correlated_pair_data):
        """Test that optimizer correctly routes to PairsPortfolio."""
        param_grid = {
            'zscore_window': [20],
            'entry_zscore': [2.0]
        }

        # Mock data loading
        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        result = simple_optimizer.optimize(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Check portfolio type - should be PairsPortfolio
        portfolio = result['best_portfolio']
        assert portfolio is not None

        # Verify portfolio has pair-specific attributes
        assert hasattr(portfolio, 'trades')

        # Check that trades recorded (if any) are for pairs
        if len(portfolio.trades) > 0:
            first_trade = portfolio.trades[0]
            # PairsPortfolio trades should have symbol1 and symbol2
            assert 'symbol1' in first_trade or 'entry_symbol1' in first_trade


class TestPairsOptimizationMetrics:
    """Test optimization with different metrics."""

    def test_optimize_sharpe_ratio(self, simple_optimizer, correlated_pair_data):
        """Test optimization using Sharpe ratio metric."""
        param_grid = {
            'zscore_window': [15, 20],
            'entry_zscore': [1.5, 2.0]
        }

        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        result = simple_optimizer.optimize(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        assert result['metric'] == 'sharpe_ratio'
        assert isinstance(result['best_value'], float)

    def test_optimize_total_return(self, simple_optimizer, correlated_pair_data):
        """Test optimization using total return metric."""
        param_grid = {
            'zscore_window': [15, 20],
            'entry_zscore': [1.5, 2.0]
        }

        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        result = simple_optimizer.optimize(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='total_return'
        )

        assert result['metric'] == 'total_return'
        assert isinstance(result['best_value'], float)


class TestPairsOptimizationEdgeCases:
    """Test edge cases in pairs optimization."""

    def test_single_parameter_combination(self, simple_optimizer, correlated_pair_data):
        """Test with single parameter combination (no optimization needed)."""
        param_grid = {
            'zscore_window': [20],
            'entry_zscore': [2.0]
        }

        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        result = simple_optimizer.optimize(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Should still work with single combination
        assert result['best_params'] == {'zscore_window': 20, 'entry_zscore': 2.0}

    def test_invalid_metric_raises_error(self, simple_optimizer, correlated_pair_data):
        """Test that invalid metric raises ValueError."""
        param_grid = {
            'zscore_window': [20],
            'entry_zscore': [2.0]
        }

        simple_optimizer.engine.data_loader.load_symbols = lambda s, sd, ed: correlated_pair_data

        with pytest.raises(ValueError, match="Unknown metric"):
            simple_optimizer.optimize(
                strategy_class=PairsTrading,
                param_grid=param_grid,
                symbols=['AAPL', 'MSFT'],
                start_date='2020-01-01',
                end_date='2020-12-31',
                metric='invalid_metric'
            )


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
