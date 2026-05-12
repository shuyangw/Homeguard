"""Probability of Backtest Overfitting per docs/methodology/backtesting.md Section 2.4.

Implements Combinatorially Symmetric Cross-Validation (CSCV) of Bailey,
Borwein, Lopez de Prado, Zhu (2017). Estimates the probability that the
best in-sample config underperforms the median out-of-sample.

Thresholds (Section 2.4):
- PBO < 0.25  acceptable
- 0.25 -- 0.50 concerning
- > 0.50 strong overfitting
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def pbo(returns_matrix: np.ndarray, s: int = 16) -> float:
    """Probability of Backtest Overfitting.

    Args:
        returns_matrix: T x N array. Rows are time, columns are configs.
            Cell [t, n] is the return of config n at time t.
        s: number of submatrices to split the rows into. Must be even.
            Section 2.4 typical value is 16. The number of CSCV folds is
            C(s, s/2).

    Returns:
        PBO in [0, 1]. Returns NaN if the input is too small to compute
        (need at least s rows and 2 columns) -- callers should propagate.
    """
    arr = np.asarray(returns_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("returns_matrix must be 2D (T x N)")
    if s % 2 != 0:
        raise ValueError(f"s must be even, got {s}")
    T, N = arr.shape
    if T < s or N < 2:
        return float("nan")

    rows_per_split = T // s
    if rows_per_split < 2:
        # Not enough data per submatrix to compute Sharpe meaningfully.
        return float("nan")

    # Split rows into s contiguous submatrices of equal size (drop remainder).
    submats = [arr[i * rows_per_split : (i + 1) * rows_per_split, :] for i in range(s)]

    indices = list(range(s))
    half = s // 2
    n_below_median = 0
    n_folds = 0

    for is_idx in combinations(indices, half):
        oos_idx = [i for i in indices if i not in is_idx]
        is_block = np.concatenate([submats[i] for i in is_idx], axis=0)
        oos_block = np.concatenate([submats[i] for i in oos_idx], axis=0)

        is_sharpes = _column_sharpes(is_block)
        oos_sharpes = _column_sharpes(oos_block)
        if np.all(np.isnan(is_sharpes)) or np.all(np.isnan(oos_sharpes)):
            continue

        best_in_is = int(np.nanargmax(is_sharpes))
        # Relative rank of best_in_is among OOS Sharpes (1.0 = best, 0.0 = worst).
        oos_valid = ~np.isnan(oos_sharpes)
        if not oos_valid.any():
            continue
        oos_only = oos_sharpes[oos_valid]
        rank = float((oos_only < oos_sharpes[best_in_is]).sum()) / float(oos_only.size)
        # Logit lambda; PBO is the fraction of folds with lambda < 0
        # (equivalently rank < 0.5).
        if rank < 0.5:
            n_below_median += 1
        n_folds += 1

    if n_folds == 0:
        return float("nan")
    return n_below_median / n_folds


def _column_sharpes(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (un-annualized). NaN for zero-variance columns."""
    mean = np.nanmean(block, axis=0)
    std = np.nanstd(block, axis=0, ddof=1)
    out = np.full_like(mean, np.nan, dtype=float)
    nz = std > 0
    out[nz] = mean[nz] / std[nz]
    return out
