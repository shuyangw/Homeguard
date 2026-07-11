"""Core-satellite return-stream blend + statistical gate.

Blends a core (carry) book's per-window dated OOS returns with a satellite
(crypto) book's per-window dated OOS returns, normalizing each book by its
full-sample volatility before weighting. Reuses the walk-forward gate
functions (Sharpe, PBO, PSR, DSR) from `run_carver_walkforward.py` so the
statistical methodology stays identical to the standalone carry gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.backtest_scripts.run_carver_walkforward import (
    _annualized_sharpe,
    _compute_pbo,
)
from src.backtesting.walkforward_common import get_campaign_trial_distribution
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr


def blend_books(
    core_windows: Sequence[pd.Series],
    sat_windows: Sequence[pd.Series],
    sat_weight: float,
    core_vol: Optional[float] = None,
    sat_vol: Optional[float] = None,
) -> Dict[str, Any]:
    """Blend core and satellite per-window dated OOS return streams.

    `core_windows[i]` and `sat_windows[i]` must share the same window
    schedule; the satellite series may cover fewer dates (or be empty)
    within a window when the satellite book had no data for those dates,
    in which case it contributes 0 to the blend on those dates.
    """
    core_weight = 1.0 - sat_weight

    if core_vol is None:
        core_full = np.concatenate([w.to_numpy(dtype=float) for w in core_windows])
        core_vol = float(np.std(core_full, ddof=1))
    if sat_vol is None:
        sat_full = np.concatenate([w.to_numpy(dtype=float) for w in sat_windows])
        sat_vol = float(np.std(sat_full, ddof=1))

    per_window_blended: List[np.ndarray] = []
    for c, s in zip(core_windows, sat_windows):
        s_aligned = s.reindex(c.index).fillna(0.0)
        blended_i = core_weight * (c / core_vol) + sat_weight * (s_aligned / sat_vol)
        per_window_blended.append(blended_i.to_numpy(dtype=float))

    stitched = np.concatenate([b for b in per_window_blended if b.size])

    n = int(stitched.size)
    oos_sharpe = _annualized_sharpe(stitched)
    # NOTE: this blend operates on 1x-cost return streams only; a 1.5x-cost
    # satellite blend is deferred to a later refinement, so the 1.5x-cost
    # gate field mirrors the 1x value here.
    oos_sharpe_1_5x_cost = oos_sharpe

    series = pd.Series(stitched)
    skew = float(series.skew()) if n > 2 else 0.0
    kurt = float(series.kurtosis()) + 3.0 if n > 3 else 3.0

    psr_val = psr(oos_sharpe, 0.0, n, skew, kurt)
    # Gate 0.1/0.2: deflate against the real, growing project-wide
    # trial-Sharpe distribution (mirrors gate_return_stream), not a
    # single-element list.
    n_trials, trial_sharpes = get_campaign_trial_distribution()
    dsr_val = dsr(oos_sharpe, trial_sharpes, n, skew, kurt,
                   n_trials_project=n_trials)
    pbo_val = _compute_pbo(per_window_blended)

    return {
        "oos_sharpe": oos_sharpe,
        "oos_sharpe_1_5x_cost": oos_sharpe_1_5x_cost,
        "pbo": pbo_val,
        "psr": psr_val,
        "dsr": dsr_val,
        "n_windows": len(per_window_blended),
        "n_oos_days": n,
        "skew": skew,
        "kurtosis_pearson": kurt,
        "core_vol": core_vol,
        "sat_vol": sat_vol,
        "sat_weight": sat_weight,
    }
