"""Combined CPCV + DSR + PBO statistical gate (methodology Section 2.5).

Composes the real validation modules over a set of CPCV out-of-sample
return splits (see `src.backtesting.validation.cpcv`) into a single
pass/fail decision:

- DSR (`src.backtesting.validation.deflated_sharpe.compute_deflated_sharpe`)
  on the pooled OOS returns, adjusted for `n_trials`.
- PBO (`src.backtesting.statistics.pbo.pbo`) treating each CPCV split's OOS
  return path as one "config" column -- this is the standard CPCV+PBO
  construction (Bailey/Lopez de Prado): each combinatorial split produces a
  distinct backtest path, and PBO measures whether the path that looks best
  in-sample also looks best out-of-sample.
- Mean per-split annualized Sharpe as a plain magnitude check.

`pass` requires all three legs to clear their thresholds.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.backtesting.statistics.pbo import pbo
from src.backtesting.validation.deflated_sharpe import compute_deflated_sharpe

DSR_PASS_THRESHOLD = 0.95
MEAN_SHARPE_PASS_THRESHOLD = 0.5
PBO_PASS_THRESHOLD = 0.5
ANNUALIZATION_PERIODS = 252


def _annualized_sharpe(returns: np.ndarray, periods: int = ANNUALIZATION_PERIODS) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / sd)


def _pbo_via_splits_as_configs(oos_returns_by_split: Sequence[np.ndarray]) -> float:
    """Treat each CPCV split's OOS path as a config column and run real PBO.

    Falls back to a simple proxy (fraction of splits below the median split
    Sharpe) when there are too few observations/splits for the real CSCV
    machinery to produce a fold.
    """
    lengths = [len(np.asarray(r)) for r in oos_returns_by_split]
    n_splits = len(oos_returns_by_split)
    min_len = min(lengths) if lengths else 0
    matrix = np.column_stack([np.asarray(r, dtype=float)[:min_len] for r in oos_returns_by_split])

    for s in (16, 12, 8, 6, 4):
        if s % 2 != 0 or s > min_len or min_len // s < 2 or n_splits < 2:
            continue
        result = pbo(matrix, s=s)
        if np.isfinite(result):
            return float(result)

    sharpes = [_annualized_sharpe(r) for r in oos_returns_by_split]
    med = float(np.median(sharpes))
    return float(np.mean([sh < med for sh in sharpes]))


def combined_gate(oos_returns_by_split: Sequence[np.ndarray], n_trials: int) -> dict:
    """Run the combined CPCV + DSR + PBO statistical gate.

    Args:
        oos_returns_by_split: per-split out-of-sample return series
            (e.g. produced by `cpcv_splits` + a backtest run per split).
        n_trials: number of strategy configurations tested, for DSR's
            multiple-testing deflation.

    Returns:
        dict with `dsr` (1 - DSR p-value, i.e. the deflated Sharpe
        probability estimate bounded in [0, 1]), `pbo`, `mean_oos_sharpe`,
        and `pass` (bool).
    """
    per_split_sharpe = [_annualized_sharpe(r) for r in oos_returns_by_split]
    mean_oos_sharpe = float(np.mean(per_split_sharpe))

    pooled_returns = np.concatenate([np.asarray(r, dtype=float) for r in oos_returns_by_split])
    dsr_result = compute_deflated_sharpe(pooled_returns, n_trials=n_trials)
    dsr = float(1.0 - dsr_result.p_value)

    pbo_value = _pbo_via_splits_as_configs(oos_returns_by_split)

    passed = bool(
        dsr_result.passed
        and dsr > DSR_PASS_THRESHOLD
        and mean_oos_sharpe > MEAN_SHARPE_PASS_THRESHOLD
        and np.isfinite(pbo_value)
        and pbo_value < PBO_PASS_THRESHOLD
    )

    return {
        "dsr": dsr,
        "pbo": pbo_value,
        "mean_oos_sharpe": mean_oos_sharpe,
        "pass": passed,
    }
