"""
Unit tests for run_generic_parameter_sweep function.
"""

import pytest
from typing import Dict, Any

from src.backtesting.optimization.sweep_runner import run_generic_parameter_sweep


def simple_backtest_fn(a: int, b: float, shared_value: int = 0) -> Dict[str, float]:
    """Simple mock backtest function for testing."""
    return {
        'sharpe': a * b + shared_value,
        'cagr': a + b,
        'max_dd': -a * 0.1
    }


def failing_backtest_fn(**kwargs) -> Dict[str, float]:
    """Always fails."""
    raise ValueError("Simulated failure")


def sometimes_failing_fn(a: int, **kwargs) -> Dict[str, float]:
    """Fails when a > 2."""
    if a > 2:
        raise ValueError(f"Failed for a={a}")
    return {'sharpe': a * 10, 'cagr': a}


class TestRunGenericParameterSweep:
    """Tests for run_generic_parameter_sweep function."""

    def test_basic_sweep(self):
        """Should run basic parameter sweep."""
        param_space = {
            'a': [1, 2],
            'b': [0.5, 1.0]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space,
            n_workers=2
        )

        assert 'best_params' in result
        assert 'best_metrics' in result
        assert 'all_results' in result

    def test_finds_best_params(self):
        """Should find parameters that maximize the metric."""
        param_space = {
            'a': [1, 2, 3],
            'b': [1.0, 2.0]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space,
            metric='sharpe',
            n_workers=2
        )

        # Best sharpe should be a=3, b=2.0 -> 3*2.0 = 6.0
        assert result['best_params']['a'] == 3
        assert result['best_params']['b'] == 2.0
        assert result['best_metrics']['sharpe'] == 6.0

    def test_minimize_mode(self):
        """Should find parameters that minimize metric when minimize=True."""
        param_space = {
            'a': [1, 2, 3],
            'b': [1.0]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space,
            metric='max_dd',
            minimize=True,
            n_workers=2
        )

        # Most negative max_dd is a=3 -> -0.3
        assert result['best_params']['a'] == 3
        assert result['best_metrics']['max_dd'] == pytest.approx(-0.3)

    def test_shared_data_passed(self):
        """Should pass shared_data to backtest function."""
        param_space = {
            'a': [1],
            'b': [1.0]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space,
            shared_data={'shared_value': 100},
            n_workers=1
        )

        # sharpe = a*b + shared_value = 1*1.0 + 100 = 101
        assert result['best_metrics']['sharpe'] == 101.0

    def test_all_results_tracked(self):
        """Should track all parameter combinations."""
        param_space = {
            'a': [1, 2, 3],
            'b': [0.5, 1.0]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space,
            n_workers=2
        )

        # Should have 3 * 2 = 6 results
        assert len(result['all_results']) == 6

    def test_handles_failures_gracefully(self):
        """Should continue sweep even if some trials fail."""
        param_space = {
            'a': [1, 2, 3, 4]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=sometimes_failing_fn,
            param_space=param_space,
            n_workers=2
        )

        # Should still find best among successful trials
        assert result['best_params'] is not None
        assert result['best_params']['a'] <= 2  # Only a=1,2 succeed

        # Should have all 4 in results
        assert len(result['all_results']) == 4

        # Check error tracking
        errors = [r for r in result['all_results'] if r['error'] is not None]
        assert len(errors) == 2  # a=3 and a=4 failed

    def test_all_failures_returns_none(self):
        """Should return None for best when all trials fail."""
        param_space = {'a': [1]}

        result = run_generic_parameter_sweep(
            backtest_fn=failing_backtest_fn,
            param_space=param_space,
            n_workers=1
        )

        assert result['best_params'] is None
        assert result['best_metrics'] is None
        assert len(result['all_results']) == 1

    def test_default_workers(self):
        """Should use default workers when not specified."""
        param_space = {'a': [1], 'b': [1.0]}

        # Should not raise
        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space
            # n_workers not specified
        )

        assert result['best_params'] is not None

    def test_different_metric(self):
        """Should optimize for specified metric."""
        param_space = {
            'a': [1, 2, 3],
            'b': [1.0]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest_fn,
            param_space=param_space,
            metric='cagr',
            n_workers=2
        )

        # Best cagr should be a=3, b=1.0 -> 3+1.0 = 4.0
        assert result['best_params']['a'] == 3
        assert result['best_metrics']['cagr'] == 4.0


class TestThreadSafety:
    """Tests for thread safety of the sweep runner."""

    def test_concurrent_execution(self):
        """Should handle concurrent execution correctly."""
        import time

        def slow_backtest(delay: float, value: int) -> Dict[str, float]:
            """Backtest that takes some time."""
            time.sleep(delay)
            return {'sharpe': value}

        param_space = {
            'delay': [0.01, 0.02],
            'value': [1, 2, 3, 4]
        }

        start = time.time()
        result = run_generic_parameter_sweep(
            backtest_fn=slow_backtest,
            param_space=param_space,
            n_workers=4  # Use 4 workers
        )
        elapsed = time.time() - start

        # With 4 workers and 8 trials, should be faster than sequential
        # Sequential would be ~0.12s (8 * 0.015 avg)
        # Parallel should be ~0.03-0.05s
        assert elapsed < 0.10  # Should be much faster than sequential

        # Should still get correct best
        assert result['best_params']['value'] == 4

    def test_shared_data_not_modified(self):
        """Shared data should not be modified by any trial."""
        shared = {'counter': 0, 'data': [1, 2, 3]}

        def check_shared(a: int, counter: int, data: list) -> Dict[str, float]:
            # Should see original values
            assert counter == 0
            assert data == [1, 2, 3]
            return {'sharpe': a}

        param_space = {'a': [1, 2, 3, 4]}

        result = run_generic_parameter_sweep(
            backtest_fn=check_shared,
            param_space=param_space,
            shared_data=shared,
            n_workers=4
        )

        # Original shared data should be unchanged
        assert shared['counter'] == 0
        assert shared['data'] == [1, 2, 3]
