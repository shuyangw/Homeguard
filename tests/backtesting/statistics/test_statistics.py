"""Tests for the statistics module.

Properties verified rather than golden numbers from papers (a couple of
synthetic-data spot checks where I do have a closed-form reference).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.backtesting.statistics import dsr, expected_max_sharpe, pbo, psr, sharpe


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_sharpe_zero_for_constant_series() -> None:
    assert sharpe([0.01] * 250) == 0.0
    assert sharpe([]) == 0.0


def test_sharpe_matches_closed_form() -> None:
    # Synthetic daily returns with known mean and std.
    # mean = 0.001/day, std = 0.01/day -> annualized Sharpe ~ 0.001 / 0.01 * sqrt(252) ~ 1.587.
    n = 252 * 10
    r = np.full(n, 0.001) + np.zeros(n)  # std is 0 -> sharpe = 0
    assert sharpe(r) == 0.0

    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.001, scale=0.01, size=n)
    s = sharpe(r)
    # Expected ~ 1.587 in expectation; sample variance is wide so allow a
    # generous band. The point of this test is "directionally correct,
    # roughly matches sqrt(252) annualization", not "exact match".
    assert 0.8 < s < 2.5


def test_psr_above_05_when_observed_exceeds_benchmark() -> None:
    # SR_hat > SR_benchmark on a sample of n=252, gaussian moments -> PSR > 0.5.
    p = psr(sr_hat=1.0, sr_benchmark=0.0, n=252, skew=0.0, kurt=3.0)
    assert 0.5 < p <= 1.0


def test_psr_below_05_when_observed_below_benchmark() -> None:
    p = psr(sr_hat=-0.1, sr_benchmark=0.0, n=252, skew=0.0, kurt=3.0)
    assert 0.0 <= p < 0.5


def test_psr_decreases_with_negative_skew() -> None:
    # Same observed Sharpe but worse left-tail behavior should reduce PSR.
    # Use small n and marginal SR so both PSR values are mid-range, not
    # saturated near 1.0 where floating-point makes the comparison fragile.
    base = psr(sr_hat=0.1, sr_benchmark=0.0, n=30, skew=0.0, kurt=3.0)
    skewed = psr(sr_hat=0.1, sr_benchmark=0.0, n=30, skew=-1.0, kurt=6.0)
    assert skewed < base, f"skewed={skewed} should be < base={base}"


def test_psr_returns_05_when_sr_equals_benchmark() -> None:
    assert abs(psr(0.5, 0.5, 252, 0.0, 3.0) - 0.5) < 1e-9


def test_psr_handles_nonsense_inputs() -> None:
    assert np.isnan(psr(1.0, 0.0, 1, 0.0, 3.0))
    # Pathological denominator: SR very large with low skew/kurt combo.
    p = psr(sr_hat=10.0, sr_benchmark=0.0, n=252, skew=5.0, kurt=2.0)
    assert 0.0 <= p <= 1.0


def test_expected_max_sharpe_scales_with_trial_count() -> None:
    trials = np.linspace(0.2, 1.0, 50).tolist()
    # E[max] under N=1000 trials > E[max] under N=10 trials (more chances to be high).
    assert expected_max_sharpe(trials, 1000) > expected_max_sharpe(trials, 10)


def test_expected_max_sharpe_zero_for_no_variance() -> None:
    assert expected_max_sharpe([1.0, 1.0, 1.0], 100) == 0.0


def test_dsr_more_conservative_than_psr_when_trials_large() -> None:
    # The candidate Sharpe is 1.0; trial Sharpes are spread; with N=1000 trials
    # the DSR benchmark > 0 -> DSR < PSR(0).
    trial_sharpes = list(np.random.default_rng(0).normal(0.3, 0.5, 1000))
    p = psr(sr_hat=1.0, sr_benchmark=0.0, n=252, skew=0.0, kurt=3.0)
    d = dsr(
        sr_hat=1.0,
        trial_sharpes=trial_sharpes,
        n=252,
        skew=0.0,
        kurt=3.0,
        n_trials_project=1000,
    )
    assert 0.0 <= d <= p


def test_pbo_low_when_configs_are_consistent(rng) -> None:
    # Configs have stable column-mean differences (one is genuinely better
    # everywhere). PBO should be low -- the in-sample best is also OOS best.
    T, N = 800, 6
    means = np.linspace(0.0, 0.02, N)  # config N-1 is strictly best
    arr = rng.normal(loc=means, scale=0.01, size=(T, N))
    p = pbo(arr, s=16)
    assert p < 0.25, f"PBO {p:.3f} should be low for genuinely separated configs"


def test_pbo_around_05_for_pure_noise(rng) -> None:
    # All configs i.i.d. zero-mean -> the in-sample best is no better than
    # any other config OOS. PBO should be near 0.5.
    T, N = 1600, 10
    arr = rng.normal(loc=0.0, scale=0.01, size=(T, N))
    p = pbo(arr, s=16)
    # Allow generous tolerance; CSCV is itself a sample-based estimator and
    # one fixed seed can drift. Tighten the bound by running multiple seeds
    # if/when this proves flaky.
    assert 0.20 < p < 0.80, f"PBO {p:.3f} should be roughly central for pure noise"


def test_pbo_validates_inputs() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        pbo(np.zeros(10))
    with pytest.raises(ValueError, match="even"):
        pbo(np.zeros((100, 4)), s=7)
    # Too few rows -> NaN, not crash.
    assert np.isnan(pbo(np.zeros((4, 4)), s=16))


def test_combined_gate_acceptance_example() -> None:
    """End-to-end mini gate per Section 2.5 -- not a hard pass criterion,
    just exercises that the three statistics agree on a synthetic 'good'
    strategy. Strong mean signal + long series so single-seed sample noise
    doesn't push it negative.
    """
    rng = np.random.default_rng(7)
    n = 252 * 20
    # Daily expected mean 0.0015, daily vol 0.01 -> expected Sharpe ~ 2.38
    returns = rng.normal(0.0015, 0.01, size=n)
    sr = sharpe(returns)
    p = psr(
        sr_hat=sr,
        sr_benchmark=0.0,
        n=n,
        skew=stats.skew(returns),
        kurt=stats.kurtosis(returns, fisher=False),
    )
    assert sr > 1.0, f"sr={sr}"
    assert p > 0.95, f"psr={p}"
