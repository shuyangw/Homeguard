"""Deflated Sharpe Ratio convenience wrapper.

This module is a thin compatibility wrapper around the canonical
`src.backtesting.statistics.dsr` (methodology Section 2.3). It accepts a daily
returns array and returns the same `DSRResult` dataclass used historically.

Prior version of this file used an unscaled Euler-Mascheroni z-score for the
expected-max Sharpe benchmark AND passed annualized SR into a per-period
variance formula. Both have been corrected here by delegating to the canonical
implementation.

NOTE on multi-trial correction: the methodologically-honest DSR requires the
actual Sharpe distribution across the trials you ran. When the caller does not
pass `trial_sharpes_daily`, this wrapper falls back to a scale-aware spread
around the observed daily Sharpe (`[0.7 * sr_daily, sr_daily, 1.3 * sr_daily]`),
which produces a defensible-enough expected-max benchmark for sanity-check
usage but is NOT a substitute for real trial Sharpes from an optimization grid.
Use the canonical `src.backtesting.statistics.dsr.dsr` directly when you have
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from src.backtesting.statistics.dsr import dsr as _dsr_canonical
from src.backtesting.statistics.dsr import expected_max_sharpe as _expected_max_sharpe


@dataclass
class DSRResult:
    observed_sharpe: float
    expected_max_sharpe: float
    dsr_statistic: float
    p_value: float
    skewness: float
    kurtosis: float  # excess kurtosis (normal = 0)
    n_observations: int
    n_trials: int
    passed: bool


def compute_deflated_sharpe(
    daily_returns: np.ndarray,
    n_trials: int,
    significance_level: float = 0.05,
    trial_sharpes_daily: Optional[Sequence[float]] = None,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio with per-period units.

    Args:
        daily_returns: Array of daily portfolio returns (not prices).
        n_trials: Number of strategy configurations evaluated (project-wide
            cumulative trial count per methodology Section 2.3).
        significance_level: p-value threshold for the `passed` field.
        trial_sharpes_daily: Optional sequence of trial Sharpes IN DAILY UNITS.
            If your trial Sharpes are annualized, divide by sqrt(P) before
            passing. When None, a standardized [-1, 0, 1] spread is used as a
            scale-free placeholder (V[trial] = 1).

    Returns:
        DSRResult with annualized observed/benchmark Sharpe (for human
        narrative), the DSR z-statistic, the p-value under the null,
        excess kurtosis, and the pass/fail boolean.
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

    std = float(returns.std())
    if std <= 0.0:
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

    sr_daily = float(returns.mean() / std)
    sr_annual = sr_daily * float(np.sqrt(252))
    skew = float(np.mean(((returns - returns.mean()) / std) ** 3))
    pearson_kurt = float(np.mean(((returns - returns.mean()) / std) ** 4))
    excess_kurt = pearson_kurt - 3.0

    if trial_sharpes_daily is None:
        # Scale-aware placeholder; do not interpret as a real trial distribution.
        trial_sharpes_daily = [0.7 * sr_daily, sr_daily, 1.3 * sr_daily]
    else:
        trial_sharpes_daily = list(trial_sharpes_daily)

    sr_zero_daily = _expected_max_sharpe(trial_sharpes_daily, n_trials)
    sr_zero_annual = sr_zero_daily * float(np.sqrt(252))

    dsr_prob = _dsr_canonical(
        sr_hat=sr_daily,
        trial_sharpes=trial_sharpes_daily,
        n=n,
        skew=skew,
        kurt=pearson_kurt,
        n_trials_project=n_trials,
    )

    # Map the canonical probability (high = significant) onto the legacy
    # frequentist p-value (low = significant) used by callers and tests.
    p_value = float(1.0 - dsr_prob)
    # The z-statistic recoverable from the probability via inverse normal CDF.
    if 0.0 < dsr_prob < 1.0:
        dsr_stat = float(norm.ppf(dsr_prob))
    elif dsr_prob >= 1.0:
        dsr_stat = float('inf')
    else:
        dsr_stat = float('-inf')

    return DSRResult(
        observed_sharpe=sr_annual,
        expected_max_sharpe=sr_zero_annual,
        dsr_statistic=dsr_stat,
        p_value=p_value,
        skewness=skew,
        kurtosis=excess_kurt,
        n_observations=n,
        n_trials=n_trials,
        passed=p_value < significance_level,
    )
