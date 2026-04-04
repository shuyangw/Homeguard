"""Bootstrap confidence intervals for strategy metrics.

Resamples trade-level returns with replacement to construct confidence
intervals for Sharpe ratio, profit factor, win rate, and mean return.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BootstrapCIResult:
    metric_name: str
    observed: float
    ci_lower: float
    ci_upper: float
    mean_bootstrap: float
    std_bootstrap: float
    p_positive: float  # P(metric > 0)


@dataclass
class BootstrapSuiteResult:
    sharpe: BootstrapCIResult
    profit_factor: BootstrapCIResult
    win_rate: BootstrapCIResult
    mean_return: BootstrapCIResult
    n_trades: int
    n_iterations: int
    passed: bool  # sharpe CI lower > 0


def bootstrap_metric(
    values: np.ndarray,
    metric_fn,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
    metric_name: str = "metric",
) -> BootstrapCIResult:
    """Bootstrap a single metric from an array of per-trade values.

    Args:
        values: 1D array of per-trade values (e.g., returns).
        metric_fn: Callable that takes a 1D array and returns a scalar.
        n_iterations: Number of bootstrap resamples.
        ci_level: Confidence level (0.95 = 95% CI).
        seed: Random seed for reproducibility.
        metric_name: Label for the result.

    Returns:
        BootstrapCIResult with observed value, CI bounds, and P(metric > 0).
    """
    rng = np.random.default_rng(seed)
    n = len(values)

    observed = float(metric_fn(values))

    bootstrap_stats = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_stats[i] = metric_fn(sample)

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.nanpercentile(bootstrap_stats, alpha * 100))
    ci_upper = float(np.nanpercentile(bootstrap_stats, (1 - alpha) * 100))
    p_positive = float(np.nanmean(bootstrap_stats > 0))

    return BootstrapCIResult(
        metric_name=metric_name,
        observed=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_bootstrap=float(np.nanmean(bootstrap_stats)),
        std_bootstrap=float(np.nanstd(bootstrap_stats)),
        p_positive=p_positive,
    )


def bootstrap_strategy_suite(
    trade_returns: np.ndarray,
    holding_days: np.ndarray,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapSuiteResult:
    """Run bootstrap CI for Sharpe, profit factor, win rate, mean return.

    Args:
        trade_returns: 1D array of per-trade return_on_premium values.
        holding_days: 1D array of per-trade holding days.
        n_iterations: Number of bootstrap resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        BootstrapSuiteResult with CI for all 4 metrics.
    """
    returns = np.asarray(trade_returns, dtype=np.float64)
    hold = np.asarray(holding_days, dtype=np.float64)
    n = len(returns)

    if n < 5:
        empty = BootstrapCIResult("", 0, 0, 0, 0, 0, 0)
        return BootstrapSuiteResult(
            sharpe=empty, profit_factor=empty, win_rate=empty,
            mean_return=empty, n_trades=n, n_iterations=0, passed=False,
        )

    avg_hold = float(np.mean(hold))
    annualization = np.sqrt(252.0 / max(avg_hold, 1.0))

    def sharpe_fn(r):
        std = r.std()
        if std == 0 or not np.isfinite(std):
            return 0.0
        return float(r.mean() / std * annualization)

    def profit_factor_fn(r):
        gains = r[r > 0].sum()
        losses = abs(r[r <= 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def win_rate_fn(r):
        return float(np.mean(r > 0))

    def mean_return_fn(r):
        return float(r.mean())

    sharpe_ci = bootstrap_metric(
        returns, sharpe_fn, n_iterations, ci_level, seed, "sharpe"
    )
    pf_ci = bootstrap_metric(
        returns, profit_factor_fn, n_iterations, ci_level, seed + 1,
        "profit_factor",
    )
    wr_ci = bootstrap_metric(
        returns, win_rate_fn, n_iterations, ci_level, seed + 2, "win_rate"
    )
    mr_ci = bootstrap_metric(
        returns, mean_return_fn, n_iterations, ci_level, seed + 3,
        "mean_return",
    )

    return BootstrapSuiteResult(
        sharpe=sharpe_ci,
        profit_factor=pf_ci,
        win_rate=wr_ci,
        mean_return=mr_ci,
        n_trades=n,
        n_iterations=n_iterations,
        passed=sharpe_ci.ci_lower > 0,
    )
