"""Deflated Sharpe Ratio (DSR) -- Bailey & Lopez de Prado (2014).

THIS MODULE CONTAINS NO DSR FORMULA. It is a thin ergonomic adapter over
`src.backtesting.statistics.dsr`, which `src/backtesting/statistics/__init__.py`
declares to be "the project's single source of truth ... Callers must use them
rather than reimplementing." The adapter exists only because callers here hold
a raw return series rather than pre-computed moments.

History (consolidated 2026-07-30). This module used to carry a SECOND
implementation:

    expected_max_sr = sqrt(2*log_n) - (gamma_em + log(log(n) + pi/2))
                                      / (2*sqrt(2*log_n))
    dsr_stat = (sr_annual - expected_max_sr) / se_sr

That is a UNIT ERROR. `expected_max_sr` there is the expected maximum of a
STANDARD NORMAL -- a pure quantile in units of sigma_SR (1.97 at N=18) -- while
`sr_annual` is an annualized Sharpe. Subtracting one from the other deflated
every candidate by ~2.0 Sharpe points and made the gate nearly unpassable. The
Bailey-Lopez de Prado benchmark is `SR_0 = sigma_SR * E[max Z]`, where sigma_SR
is the cross-trial DISPERSION of Sharpes -- the factor this module omitted.

Evidence of the damage:
docs/reports/ramp-long-calls/20260402_statistical_validation.json records
`expected_max_sharpe: 1.9733` at `n_trials: 18` with `p_value: 1.0`.

Deriving sigma_SR
-----------------
`statistics.dsr.expected_max_sharpe()` takes sigma_SR from the empirical
variance of the trial Sharpes. When the caller has that distribution it should
pass `trial_sharpes`. When it does not (a single strategy's pooled returns),
we fall back to the asymptotic standard error of an annualized Sharpe estimate
under the null, `sigma_SR = 1 / sqrt(years)` -- the same theoretical prior the
options pre-registration chain uses (see
docs/strategies/research/options-slate/20260728_lifetime_trial_count.md S1.1).

The prior is injected THROUGH the shared function rather than recomputed here:
for a two-point sample {0, s*sqrt(2)} the ddof=1 variance is exactly s^2, so
`expected_max_sharpe([0, s*sqrt(2)], N)` returns the Bailey-LdP maximum at
sigma_SR = s. No formula is duplicated.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from src.backtesting.statistics.dsr import expected_max_sharpe
from src.backtesting.statistics.psr import psr

ANNUALIZATION_PERIODS = 252

# Clamp for reporting the DSR probability back as a z-score: the probability
# saturates at 0 and 1 in float64, and +/-8.2 sigma is the representable edge.
_Z_CLAMP = 8.2


@dataclass
class DSRResult:
    observed_sharpe: float          # annualized
    expected_max_sharpe: float      # SR_0 benchmark, ANNUALIZED SHARPE units
    dsr_statistic: float            # z-score equivalent of dsr_probability
    dsr_probability: float          # P(true SR > SR_0) -- the actual DSR, [0, 1]
    p_value: float                  # 1 - dsr_probability
    skewness: float
    kurtosis: float                 # excess kurtosis
    n_observations: int
    n_trials: int
    passed: bool                    # p_value < significance_level


def compute_deflated_sharpe(
    daily_returns: np.ndarray,
    n_trials: int,
    significance_level: float = 0.05,
    trial_sharpes: Optional[Sequence[float]] = None,
    periods_per_year: float = ANNUALIZATION_PERIODS,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio for one return series.

    Args:
        daily_returns: Array of periodic portfolio returns (not prices).
        n_trials: Project-wide cumulative trial count N. Obtain it from
            `src.experiments.n_trials_project_wide()` -- NOT a per-run config
            count (methodology Section 2.3).
        significance_level: `passed` requires p_value below this.
        trial_sharpes: ANNUALIZED Sharpes of the trials searched over, used for
            the sigma_SR dispersion term. Strongly preferred when available.
            When omitted, sigma_SR falls back to the prior 1/sqrt(years).
        periods_per_year: annualization factor for `daily_returns`.

    Returns:
        DSRResult. The headline number is `dsr_probability` (pass at >= 0.95
        per Section 2.5); `p_value` is its complement.
    """
    returns = np.asarray(daily_returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 10:
        return DSRResult(
            observed_sharpe=0.0,
            expected_max_sharpe=0.0,
            dsr_statistic=0.0,
            dsr_probability=0.0,
            p_value=1.0,
            skewness=0.0,
            kurtosis=0.0,
            n_observations=n,
            n_trials=n_trials,
            passed=False,
        )

    sd = float(returns.std())
    if sd == 0:
        sr_annual = skew = excess_kurt = 0.0
    else:
        z = (returns - returns.mean()) / sd
        sr_annual = float(returns.mean() / sd * np.sqrt(periods_per_year))
        skew = float(np.mean(z ** 3))
        excess_kurt = float(np.mean(z ** 4) - 3.0)

    sr_zero = _benchmark_sharpe(n, n_trials, trial_sharpes, periods_per_year)

    # psr() wants Pearson kurtosis (normal = 3) and de-annualizes BOTH Sharpes
    # itself via periods_per_year, so `n` stays a period count.
    dsr_probability = psr(
        sr_annual,
        sr_zero,
        n,
        skew,
        excess_kurt + 3.0,
        periods_per_year=periods_per_year,
    )
    if not np.isfinite(dsr_probability):
        dsr_probability = 0.0

    p_value = float(1.0 - dsr_probability)
    z_stat = float(np.clip(
        norm.ppf(float(np.clip(dsr_probability, 1e-16, 1.0 - 1e-16))),
        -_Z_CLAMP, _Z_CLAMP))

    return DSRResult(
        observed_sharpe=sr_annual,
        expected_max_sharpe=float(sr_zero),
        dsr_statistic=z_stat,
        dsr_probability=float(dsr_probability),
        p_value=p_value,
        skewness=skew,
        kurtosis=excess_kurt,
        n_observations=n,
        n_trials=n_trials,
        passed=bool(p_value < significance_level),
    )


def _benchmark_sharpe(
    n: int,
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]],
    periods_per_year: float,
) -> float:
    """SR_0: the expected maximum ANNUALIZED Sharpe under the null.

    Both branches delegate the formula to `statistics.dsr.expected_max_sharpe`.
    This function only decides where sigma_SR comes from.
    """
    if trial_sharpes is not None:
        arr = np.asarray(list(trial_sharpes), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2 and arr.var(ddof=1) > 0:
            return expected_max_sharpe(arr, n_trials)

    years = n / float(periods_per_year)
    if years <= 0:
        return 0.0
    sigma_sr = 1.0 / np.sqrt(years)
    return expected_max_sharpe([0.0, sigma_sr * np.sqrt(2.0)], n_trials)
