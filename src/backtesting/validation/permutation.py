"""Permutation test framework for signal significance.

Tests whether a strategy's observed performance is significantly better than
what would be achieved with random signal assignments. The null hypothesis
is that the signal has no predictive power.

Generic framework: accepts a callable that runs a backtest with a shuffled
signal and returns a Sharpe ratio.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class PermutationResult:
    observed_sharpe: float
    null_distribution: List[float] = field(default_factory=list)
    p_value: float = 1.0
    n_iterations: int = 0
    mean_null: float = 0.0
    std_null: float = 0.0
    passed: bool = False

    @property
    def percentile_rank(self) -> float:
        """Where the observed Sharpe falls in the null distribution (0-100)."""
        if not self.null_distribution:
            return 0.0
        return float(
            np.mean(np.array(self.null_distribution) < self.observed_sharpe)
            * 100
        )


def run_permutation_test(
    observed_sharpe: float,
    run_shuffled_fn: Callable[[int], float],
    n_iterations: int = 200,
    significance_level: float = 0.05,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> PermutationResult:
    """Run a permutation test comparing observed Sharpe to shuffled signals.

    Args:
        observed_sharpe: The Sharpe ratio from the real signal.
        run_shuffled_fn: Callable(iteration_index) -> Sharpe ratio.
            Each call should run the strategy with a randomly shuffled or
            randomized signal. The iteration index can be used as a seed.
        n_iterations: Number of permutation iterations.
        significance_level: Threshold for statistical significance.
        progress_callback: Optional fn(iter_idx, total, sharpe) called
            after each iteration.

    Returns:
        PermutationResult with null distribution and p-value.
    """
    null_sharpes: List[float] = []

    for i in range(n_iterations):
        try:
            sharpe = run_shuffled_fn(i)
            null_sharpes.append(sharpe)
        except Exception as e:
            logger.error(f"Permutation iteration {i} failed: {e}")
            null_sharpes.append(0.0)

        if progress_callback:
            progress_callback(i, n_iterations, sharpe)

        if (i + 1) % 20 == 0:
            current_p = (
                sum(1 for s in null_sharpes if s >= observed_sharpe) + 1
            ) / (len(null_sharpes) + 1)
            logger.info(
                f"Permutation progress: {i + 1}/{n_iterations}, "
                f"current p-value={current_p:.3f}"
            )

    # p-value: fraction of null >= observed, with continuity correction
    # (count(null >= observed) + 1) / (n_iterations + 1)
    null_arr = np.array(null_sharpes)
    count_ge = int(np.sum(null_arr >= observed_sharpe))
    p_value = (count_ge + 1) / (n_iterations + 1)

    result = PermutationResult(
        observed_sharpe=observed_sharpe,
        null_distribution=null_sharpes,
        p_value=float(p_value),
        n_iterations=n_iterations,
        mean_null=float(np.mean(null_arr)),
        std_null=float(np.std(null_arr)),
        passed=p_value < significance_level,
    )

    logger.info(
        f"Permutation test complete: observed={observed_sharpe:.3f}, "
        f"null mean={result.mean_null:.3f} +/- {result.std_null:.3f}, "
        f"p-value={result.p_value:.4f}, passed={result.passed}"
    )
    return result
