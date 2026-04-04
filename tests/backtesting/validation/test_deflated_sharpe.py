"""Tests for Deflated Sharpe Ratio module."""

import numpy as np
import pytest

from src.backtesting.validation.deflated_sharpe import (
    DSRResult,
    compute_deflated_sharpe,
)


class TestComputeDeflatedSharpe:

    def test_returns_dsr_result(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, size=500)
        result = compute_deflated_sharpe(returns, n_trials=10)
        assert isinstance(result, DSRResult)
        assert result.n_observations == 500
        assert result.n_trials == 10

    def test_positive_sharpe_with_few_trials_passes(self):
        """A clearly positive Sharpe with few trials should pass."""
        rng = np.random.default_rng(42)
        # Strong positive drift
        returns = rng.normal(0.002, 0.01, size=1000)
        result = compute_deflated_sharpe(returns, n_trials=2)
        assert result.observed_sharpe > 1.0
        assert result.passed is True
        assert result.p_value < 0.05

    def test_zero_sharpe_fails(self):
        """Zero-drift returns should fail DSR."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, size=1000)
        result = compute_deflated_sharpe(returns, n_trials=10)
        assert result.passed is False
        assert result.p_value > 0.05

    def test_many_trials_makes_it_harder(self):
        """More trials -> higher expected max -> harder to pass."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, size=500)

        result_few = compute_deflated_sharpe(returns, n_trials=2)
        result_many = compute_deflated_sharpe(returns, n_trials=100)

        assert result_many.expected_max_sharpe > result_few.expected_max_sharpe
        assert result_many.p_value >= result_few.p_value

    def test_short_returns_fails_gracefully(self):
        """Very short return series should return failed result."""
        returns = np.array([0.01, -0.01])
        result = compute_deflated_sharpe(returns, n_trials=10)
        assert result.passed is False
        assert result.n_observations == 2

    def test_single_trial(self):
        """With 1 trial, expected max = 0, so positive Sharpe should pass."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=500)
        result = compute_deflated_sharpe(returns, n_trials=1)
        assert result.expected_max_sharpe == 0.0
        assert result.observed_sharpe > 0

    def test_skewness_and_kurtosis_computed(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, size=1000)
        result = compute_deflated_sharpe(returns, n_trials=5)
        # Normal dist should have skew near 0, excess kurtosis near 0
        assert abs(result.skewness) < 0.5
        assert abs(result.kurtosis) < 1.0

    def test_handles_nan_and_inf(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=100)
        returns[10] = np.nan
        returns[20] = np.inf
        result = compute_deflated_sharpe(returns, n_trials=5)
        assert result.n_observations == 98  # 2 removed
        assert np.isfinite(result.p_value)
