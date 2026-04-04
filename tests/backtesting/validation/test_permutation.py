"""Tests for Permutation test module."""

import numpy as np
import pytest

from src.backtesting.validation.permutation import (
    PermutationResult,
    run_permutation_test,
)


class TestRunPermutationTest:

    def test_returns_result(self):
        def shuffled(i):
            return np.random.default_rng(i).normal(0, 0.3)

        result = run_permutation_test(0.7, shuffled, n_iterations=50)
        assert isinstance(result, PermutationResult)
        assert result.n_iterations == 50
        assert len(result.null_distribution) == 50

    def test_strong_signal_passes(self):
        """Observed Sharpe far above null distribution should pass."""
        def null_signal(i):
            return np.random.default_rng(i).normal(0.0, 0.1)

        result = run_permutation_test(
            observed_sharpe=1.5,
            run_shuffled_fn=null_signal,
            n_iterations=100,
        )
        assert result.p_value < 0.05
        assert result.passed is True

    def test_weak_signal_fails(self):
        """Observed Sharpe within null distribution should fail."""
        def null_signal(i):
            return np.random.default_rng(i).normal(0.5, 0.3)

        result = run_permutation_test(
            observed_sharpe=0.5,  # right at null mean
            run_shuffled_fn=null_signal,
            n_iterations=100,
        )
        assert result.p_value > 0.05
        assert result.passed is False

    def test_p_value_formula_correct(self):
        """p = (count(null >= observed) + 1) / (n + 1)"""
        # All null values below observed
        def always_below(i):
            return -1.0

        result = run_permutation_test(0.5, always_below, n_iterations=100)
        # count_ge = 0, so p = (0 + 1) / (100 + 1) = 0.0099
        assert abs(result.p_value - 1 / 101) < 1e-6

    def test_all_null_above_gives_p_near_one(self):
        def always_above(i):
            return 10.0

        result = run_permutation_test(0.5, always_above, n_iterations=100)
        # count_ge = 100, so p = (100 + 1) / (100 + 1) = 1.0
        assert abs(result.p_value - 1.0) < 1e-6
        assert result.passed is False

    def test_percentile_rank(self):
        def null_signal(i):
            return float(i) / 100  # 0.0 to 0.99

        result = run_permutation_test(0.5, null_signal, n_iterations=100)
        # 50 values below 0.5
        assert abs(result.percentile_rank - 50.0) < 2.0

    def test_progress_callback_called(self):
        calls = []

        def track(idx, total, sharpe):
            calls.append(idx)

        def null_fn(i):
            return 0.0

        run_permutation_test(
            0.5, null_fn, n_iterations=20,
            progress_callback=track,
        )
        assert len(calls) == 20

    def test_exception_in_shuffled_handled(self):
        def failing(i):
            if i == 5:
                raise RuntimeError("test")
            return 0.0

        result = run_permutation_test(0.5, failing, n_iterations=20)
        assert len(result.null_distribution) == 20
