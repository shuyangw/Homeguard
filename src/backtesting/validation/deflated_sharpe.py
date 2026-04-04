"""Deflated Sharpe Ratio (DSR) - Bailey & Lopez de Prado (2014).

Adjusts the observed Sharpe ratio for multiple testing bias by estimating
E[max(SR)] under the null hypothesis that all strategies have zero expected
Sharpe. Uses the Euler-Mascheroni approximation for the expected maximum
of N independent standard normals.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class DSRResult:
    observed_sharpe: float
    expected_max_sharpe: float
    dsr_statistic: float
    p_value: float
    skewness: float
    kurtosis: float  # excess kurtosis
    n_observations: int
    n_trials: int
    passed: bool  # p_value < significance_level


def compute_deflated_sharpe(
    daily_returns: np.ndarray,
    n_trials: int,
    significance_level: float = 0.05,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio.

    Args:
        daily_returns: Array of daily portfolio returns (not prices).
        n_trials: Number of strategy configurations tested (including
            the one being evaluated). For ramp-long-calls: 18 trials.
        significance_level: Threshold for statistical significance.

    Returns:
        DSRResult with test statistic, p-value, and pass/fail.
    """
    returns = np.asarray(daily_returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 10:
        return DSRResult(
            observed_sharpe=0.0,
            expected_max_sharpe=0.0,
            dsr_statistic=0.0,
            p_value=1.0,
            skewness=0.0,
            kurtosis=0.0,
            n_observations=n,
            n_trials=n_trials,
            passed=False,
        )

    # Observed annualized Sharpe
    sr_daily = returns.mean() / returns.std()
    sr_annual = sr_daily * np.sqrt(252)

    # Higher moments
    skew = float(np.mean(((returns - returns.mean()) / returns.std()) ** 3))
    excess_kurt = float(
        np.mean(((returns - returns.mean()) / returns.std()) ** 4) - 3.0
    )

    # E[max(SR)] under null: Euler-Mascheroni approximation
    # E[max] ~ Z_inv(1 - 1/N) * (1 - gamma/ln(N)) + gamma/ln(N) * Z_inv(1 - 1/(N*e))
    # Simplified: E[max] approx= sqrt(2 * ln(N)) - (gamma + ln(ln(N) + pi/2)) / (2 * sqrt(2 * ln(N)))
    # where gamma = 0.5772... (Euler-Mascheroni constant)
    if n_trials <= 1:
        expected_max_sr = 0.0
    else:
        gamma_em = 0.5772156649
        log_n = np.log(n_trials)
        expected_max_sr = (
            np.sqrt(2 * log_n)
            - (gamma_em + np.log(np.log(n_trials) + np.pi / 2))
            / (2 * np.sqrt(2 * log_n))
        )

    # Standard error of SR corrected for non-normality
    # SE(SR) = sqrt((1 - skew*SR + (kurt/4)*SR^2) / (n-1))
    # Bailey & Lopez de Prado (2014), eq. 4
    se_sr = np.sqrt(
        (1.0 - skew * sr_annual + (excess_kurt / 4.0) * sr_annual**2)
        / max(n - 1, 1)
    )

    if se_sr <= 0 or not np.isfinite(se_sr):
        se_sr = 1.0 / np.sqrt(max(n - 1, 1))

    # DSR test statistic
    dsr_stat = (sr_annual - expected_max_sr) / se_sr

    # One-sided p-value (null: true SR <= E[max])
    p_value = float(1.0 - norm.cdf(dsr_stat))

    return DSRResult(
        observed_sharpe=float(sr_annual),
        expected_max_sharpe=float(expected_max_sr),
        dsr_statistic=float(dsr_stat),
        p_value=p_value,
        skewness=skew,
        kurtosis=excess_kurt,
        n_observations=n,
        n_trials=n_trials,
        passed=p_value < significance_level,
    )
