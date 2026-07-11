"""Aggregate per-root session return streams and gate via the shared walk-forward
PSR/DSR/PBO helpers (identical methodology to the VIX roll-down sleeve)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.backtesting.walkforward_common import (
    _annualized_sharpe, _compute_pbo, get_campaign_trial_distribution)
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr


def aggregate_returns(per_root: Dict[str, pd.Series]) -> pd.Series:
    """Vol-normalized equal-risk mean of per-root return streams on the union of dates.

    Missing dates for a given root contribute 0 (that root is flat, not absent)."""
    streams = {k: v for k, v in per_root.items() if v is not None and len(v)}
    if not streams:
        return pd.Series(dtype=float)
    all_dates = sorted(set().union(*[set(s.index) for s in streams.values()]))
    idx = pd.Index(all_dates)
    norm = []
    for s in streams.values():
        vol = float(s.std(ddof=1))
        aligned = s.reindex(idx).fillna(0.0)
        norm.append(aligned / vol if vol > 0 else aligned * 0.0)
    result = sum(norm) / float(len(norm))
    result.index = pd.DatetimeIndex(result.index)
    return result


def _oos_windows(returns: pd.Series, train_months: int, test_months: int,
                  step_months: int) -> List[pd.Series]:
    """Split a dated return series into walk-forward OOS (test) segments."""
    returns = returns.dropna()
    if returns.empty:
        return []
    start, end = returns.index.min(), returns.index.max()
    oos: List[pd.Series] = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        seg = returns[(returns.index >= train_end) & (returns.index < test_end)]
        if seg.size >= 10:
            oos.append(seg)
        if test_end > end:
            break
        cursor = cursor + pd.DateOffset(months=step_months)
    return oos


def gate_session_stream(returns: pd.Series, train_months: int = 36,
                         test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    """Walk-forward OOS Sharpe/PSR/DSR/PBO gate for an aggregated session return stream."""
    oos = _oos_windows(returns, train_months, test_months, step_months)
    per_window = [w.to_numpy(dtype=float) for w in oos]
    stitched = np.concatenate(per_window) if per_window else np.array([])
    n = int(stitched.size)
    sharpe = _annualized_sharpe(stitched) if n else float("nan")
    s = pd.Series(stitched)
    skew = float(s.skew()) if n > 2 else 0.0
    kurt = float(s.kurtosis()) + 3.0 if n > 3 else 3.0
    # Gate 0.1/0.2: deflate against the real, growing project-wide
    # trial-Sharpe distribution (mirrors gate_return_stream), not a
    # single-element list.
    n_trials, trial_sharpes = get_campaign_trial_distribution()
    return {
        "oos_sharpe": sharpe, "n_oos": n, "n_windows": len(oos),
        "psr": psr(sharpe, 0.0, n, skew, kurt) if n else float("nan"),
        "dsr": dsr(sharpe, trial_sharpes, n, skew, kurt, n_trials_project=n_trials) if n else float("nan"),
        "pbo": _compute_pbo(per_window) if len(per_window) > 1 else float("nan"),
        "skew": skew, "kurtosis": kurt,
        "trial_count": n_trials,
    }
