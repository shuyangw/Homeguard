"""Probabilistic Sharpe Ratio per docs/methodology/backtesting.md Section 2.2.

PSR(SR*) = Phi((SR_hat - SR*) sqrt(n - 1) /
                sqrt(1 - gamma3 * SR_hat + ((gamma4 - 1)/4) * SR_hat^2))

where gamma3 is sample skewness and gamma4 is sample kurtosis (Pearson's,
normal = 3, not excess). SR_hat and SR* must be in the same units.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def psr(sr_hat: float, sr_benchmark: float, n: int, skew: float, kurt: float,
        periods_per_year: float = 1.0) -> float:
    """Probability that the true Sharpe exceeds the benchmark.

    Args:
        sr_hat: observed Sharpe (per-period or annualized; match sr_benchmark)
        sr_benchmark: benchmark Sharpe under the null (commonly 0)
        n: number of return observations
        skew: sample skewness of returns
        kurt: sample kurtosis (Pearson, normal = 3 -- NOT excess kurtosis)
        periods_per_year: set to 252 when sr_hat/sr_benchmark are ANNUALIZED
            and n counts DAILY observations; leave 1.0 when they are already
            per-observation. Mismatching these inflates PSR badly.

    Returns:
        Probability in [0, 1]. A PSR of 0.95 means "95% confident the true
        Sharpe exceeds the benchmark." Section 2.2 sets the pass threshold
        at 0.95.
    """
    if n <= 1:
        return float("nan")
    # UNITS (fixed 2026-07-25): the z below scales by sqrt(n-1) where n counts
    # RETURN OBSERVATIONS, so sr_hat must be the PER-OBSERVATION Sharpe. Every
    # walk-forward runner was passing an ANNUALIZED Sharpe against a DAILY n,
    # inflating z by ~sqrt(252) = 15.9 and making PSR wildly optimistic (a
    # corrected 0.57 was being reported as 0.998). Callers with annualized
    # inputs must now pass periods_per_year=252 so they are de-annualized here,
    # keeping the conversion in one place. Default 1.0 leaves any caller that
    # already passes per-period Sharpes bit-identical.
    scale = np.sqrt(periods_per_year)
    sr_hat = sr_hat / scale
    sr_benchmark = sr_benchmark / scale
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat
    if denom_sq <= 0:
        # Pathological tail behavior: the denominator becomes non-real.
        # Treat as zero confidence -- the observed Sharpe cannot be trusted.
        return 0.0
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1.0) / np.sqrt(denom_sq)
    return float(norm.cdf(z))
