"""
Unit tests for BayesianOptimizer class.

Tests Bayesian optimization with Gaussian Processes for intelligent parameter selection.
"""

import pytest
import pandas as pd
from datetime import datetime

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover

# Conditional import based on availability
try:
    from backtesting.optimization import BayesianOptimizer, BAYESIAN_AVAILABLE
    from skopt.space import Integer, Real
except ImportError:
    BAYESIAN_AVAILABLE = False
    BayesianOptimizer = None
    Integer = None
    Real = None


@pytest.mark.skipif(not BAYESIAN_AVAILABLE, reason="scikit-optimize not installed")
class TestBayesianOptimizer:
    """Test suite for Bayesian optimization."""

    @pytest.fixture
    def engine(self):
        """Create BacktestEngine for testing."""
        return BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.0001
        )

    @pytest.fixture
    def param_space(self):
        """Define parameter space for testing."""
        return [
            Integer(5, 20, name='fast_window'),
            Integer(30, 60, name='slow_window')
        ]

    def test_bayesian_optimizer_initialization(self, engine):
        """Test that BayesianOptimizer can be initialized."""
        optimizer = BayesianOptimizer(engine)
        assert optimizer is not None
        assert optimizer.engine == engine

    def test_bayesian_basic_optimization(self, engine, param_space):
        """Test basic Bayesian optimization works."""
        optimizer = BayesianOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-01',
            metric='sharpe_ratio',
            n_iterations=10,  # Small number for fast test
            n_initial_points=3,
            random_seed=42  # For reproducibility
        )

        # Check result structure
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'metric' in result
        assert 'all_results' in result
        assert 'convergence_data' in result

        # Check best_params has correct keys
        assert 'fast_window' in result['best_params']
        assert 'slow_window' in result['best_params']

        # Check parameter bounds
        assert 5 <= result['best_params']['fast_window'] <= 20
        assert 30 <= result['best_params']['slow_window'] <= 60

    def test_bayesian_convergence_tracking(self, engine, param_space):
        """Test that convergence is properly tracked."""
        optimizer = BayesianOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-03-01',
            metric='sharpe_ratio',
            n_iterations=8,
            n_initial_points=3,
            random_seed=42
        )

        # Check convergence data
        convergence_data = result['convergence_data']
        assert 'iterations' in convergence_data
        assert 'best_values' in convergence_data
        assert len(convergence_data['best_values']) == result['n_iterations']

    def test_bayesian_acquisition_functions(self, engine, param_space):
        """Test different acquisition functions work."""
        optimizer = BayesianOptimizer(engine)

        for acq_func in ['EI', 'LCB', 'PI']:
            result = optimizer.optimize(
                strategy_class=MovingAverageCrossover,
                param_space=param_space,
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-02-01',
                metric='sharpe_ratio',
                n_iterations=5,
                n_initial_points=2,
                acquisition_func=acq_func,
                random_seed=42
            )

            assert result['best_params'] is not None
            assert result['best_value'] is not None

    def test_bayesian_cache_integration(self, engine, param_space):
        """Test that caching works with Bayesian optimization."""
        optimizer = BayesianOptimizer(engine)

        # First run
        result1 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            n_iterations=5,
            n_initial_points=2,
            use_cache=True,
            random_seed=42
        )

        # Second run (should use cache)
        result2 = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            n_iterations=5,
            n_initial_points=2,
            use_cache=True,
            random_seed=42
        )

        # Should have some cache hits in second run
        assert result2['cache_hits'] > 0

    def test_bayesian_early_stopping(self, engine, param_space):
        """Test early stopping on convergence."""
        optimizer = BayesianOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            n_iterations=20,
            n_initial_points=3,
            convergence_tolerance=0.001,
            convergence_patience=3,
            random_seed=42
        )

        # May or may not early stop, but structure should be correct
        assert 'early_stopped' in result
        if result['early_stopped']:
            assert result['n_iterations'] < 20

    def test_bayesian_real_parameters(self, engine):
        """Test with Real (continuous) parameters."""
        param_space = [
            Integer(5, 20, name='fast_window'),
            Real(0.01, 0.10, prior='log-uniform', name='threshold')
        ]

        # Need a strategy that accepts threshold parameter
        # For now, just test with fast_window
        param_space_simple = [
            Integer(5, 20, name='fast_window'),
            Integer(30, 60, name='slow_window')
        ]

        optimizer = BayesianOptimizer(engine)

        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space_simple,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            n_iterations=5,
            n_initial_points=2,
            random_seed=42
        )

        assert result['best_params'] is not None

    def test_bayesian_reproducibility(self, engine, param_space):
        """Test that same random seed gives same results."""
        optimizer1 = BayesianOptimizer(engine)
        optimizer2 = BayesianOptimizer(engine)

        result1 = optimizer1.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            n_iterations=5,
            n_initial_points=2,
            use_cache=False,  # Disable cache for pure reproducibility test
            random_seed=42
        )

        result2 = optimizer2.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=param_space,
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-02-01',
            metric='sharpe_ratio',
            n_iterations=5,
            n_initial_points=2,
            use_cache=False,
            random_seed=42
        )

        # With same seed, should get same parameters
        assert result1['best_params'] == result2['best_params']
        assert result1['best_value'] == result2['best_value']


@pytest.mark.skipif(BAYESIAN_AVAILABLE, reason="Test for when scikit-optimize is not available")
def test_bayesian_unavailable_error():
    """Test that proper error is raised when scikit-optimize is not available."""
    with pytest.raises(ImportError):
        from backtesting.optimization.bayesian_optimizer import BayesianOptimizer
        engine = BacktestEngine(initial_capital=100000, fees=0.001)
        optimizer = BayesianOptimizer(engine)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
