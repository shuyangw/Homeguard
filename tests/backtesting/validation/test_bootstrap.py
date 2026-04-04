"""Tests for Bootstrap confidence interval module."""

import numpy as np
import pytest

from src.backtesting.validation.bootstrap import (
    BootstrapCIResult,
    BootstrapSuiteResult,
    bootstrap_metric,
    bootstrap_strategy_suite,
)


class TestBootstrapMetric:

    def test_returns_ci_result(self):
        values = np.random.default_rng(42).normal(0.05, 0.2, size=100)
        result = bootstrap_metric(
            values, np.mean, n_iterations=1000, metric_name="mean"
        )
        assert isinstance(result, BootstrapCIResult)
        assert result.metric_name == "mean"

    def test_ci_contains_true_mean(self):
        """95% CI from N(0.05, 0.2) should contain 0.05 most of the time."""
        rng = np.random.default_rng(42)
        values = rng.normal(0.05, 0.2, size=200)
        result = bootstrap_metric(
            values, np.mean, n_iterations=5000, seed=42, metric_name="mean"
        )
        # CI should contain sample mean
        assert result.ci_lower <= result.observed <= result.ci_upper
        # CI bounds should be reasonable
        assert result.ci_lower > -0.5
        assert result.ci_upper < 0.5

    def test_p_positive_near_one_for_positive_distribution(self):
        values = np.random.default_rng(42).normal(1.0, 0.1, size=100)
        result = bootstrap_metric(
            values, np.mean, n_iterations=1000, metric_name="mean"
        )
        assert result.p_positive > 0.99

    def test_p_positive_near_zero_for_negative_distribution(self):
        values = np.random.default_rng(42).normal(-1.0, 0.1, size=100)
        result = bootstrap_metric(
            values, np.mean, n_iterations=1000, metric_name="mean"
        )
        assert result.p_positive < 0.01

    def test_reproducible_with_same_seed(self):
        values = np.random.default_rng(42).normal(0.05, 0.2, size=50)
        r1 = bootstrap_metric(values, np.mean, n_iterations=500, seed=99)
        r2 = bootstrap_metric(values, np.mean, n_iterations=500, seed=99)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


class TestBootstrapStrategySuite:

    def test_returns_suite_result(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.3, size=88)
        holding = rng.integers(5, 30, size=88).astype(float)
        result = bootstrap_strategy_suite(returns, holding, n_iterations=500)
        assert isinstance(result, BootstrapSuiteResult)
        assert result.n_trades == 88

    def test_passed_when_sharpe_ci_lower_positive(self):
        rng = np.random.default_rng(42)
        # Strong positive returns -> Sharpe CI lower should be > 0
        returns = rng.normal(0.15, 0.2, size=200)
        holding = np.full(200, 15.0)
        result = bootstrap_strategy_suite(returns, holding, n_iterations=2000)
        assert result.sharpe.ci_lower > 0
        assert result.passed is True

    def test_fails_when_sharpe_near_zero(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.3, size=50)
        holding = np.full(50, 15.0)
        result = bootstrap_strategy_suite(returns, holding, n_iterations=2000)
        # With zero-mean returns, CI should include 0
        assert result.sharpe.ci_lower <= 0
        assert result.passed is False

    def test_few_trades_returns_not_passed(self):
        returns = np.array([0.1, -0.05, 0.2])
        holding = np.array([10.0, 15.0, 12.0])
        result = bootstrap_strategy_suite(returns, holding)
        assert result.passed is False
        assert result.n_trades == 3

    def test_profit_factor_computed(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.2, size=100)
        holding = np.full(100, 15.0)
        result = bootstrap_strategy_suite(returns, holding, n_iterations=500)
        assert result.profit_factor.observed > 0

    def test_win_rate_computed(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.2, size=100)
        holding = np.full(100, 15.0)
        result = bootstrap_strategy_suite(returns, holding, n_iterations=500)
        assert 0 < result.win_rate.observed < 1
