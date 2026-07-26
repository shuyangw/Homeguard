"""Deflated Sharpe Ratio per docs/methodology/backtesting.md Section 2.3.

DSR = PSR(SR_0*) where SR_0* is the expected maximum Sharpe under the null
after N independent trials:

SR_0* = sqrt(V[trial_sharpes]) * [(1 - gamma_em) * Phi^{-1}(1 - 1/N)
                                  + gamma_em * Phi^{-1}(1 - 1/(N * e^-1))]

Critical: N is the project-wide cumulative trial count (Section 2.3 explicit
note), pulled from `output/experiments.duckdb` via
`src.experiments.n_trials_project_wide()` -- NOT the per-run config count.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm

from src.backtesting.statistics.psr import psr

_EULER_MASCHERONI = 0.5772156649


def expected_max_sharpe(trial_sharpes: Sequence[float], n_trials: int) -> float:
    """E[max Sharpe] across N trials under the null per Section 2.3.

    Args:
        trial_sharpes: observed Sharpe ratios across all evaluated configs
            (used for the variance term)
        n_trials: project-wide cumulative trial count

    Returns:
        The benchmark Sharpe to use in DSR. Larger when trial Sharpes spread
        widely or when many trials have been run.
    """
    if n_trials < 2:
        return 0.0
    arr = np.asarray(list(trial_sharpes), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    v = float(arr.var(ddof=1))
    if v <= 0:
        return 0.0
    em = _EULER_MASCHERONI
    return float(
        np.sqrt(v)
        * (
            (1.0 - em) * norm.ppf(1.0 - 1.0 / n_trials)
            + em * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    )


def dsr(
    sr_hat: float,
    trial_sharpes: Sequence[float],
    n: int,
    skew: float,
    kurt: float,
    n_trials_project: int,
    periods_per_year: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio.

    Args:
        sr_hat: observed Sharpe of the candidate config
        trial_sharpes: Sharpes of all trials evaluated (for the variance term)
        n: number of return observations for sr_hat
        skew: sample skewness of sr_hat's return series
        kurt: sample kurtosis (Pearson, normal = 3) of sr_hat's return series
        n_trials_project: cumulative project-wide trial count -- typically
            obtained via src.experiments.n_trials_project_wide()

    Returns:
        Probability in [0, 1] that the candidate's true Sharpe exceeds the
        expected maximum under the null. Section 2.5 sets the pass threshold
        at 0.95.
    """
    # sr_zero inherits the units of `trial_sharpes`, so sr_hat and sr_zero are
    # consistent; psr() de-annualizes BOTH via periods_per_year.
    sr_zero = expected_max_sharpe(trial_sharpes, n_trials_project)
    return psr(sr_hat, sr_zero, n, skew, kurt, periods_per_year=periods_per_year)
