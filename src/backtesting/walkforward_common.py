"""Shared walk-forward helpers: window-building, OOS-slicing, and the
statistical gate. Pure functions that operate only on dates, equity curves,
and return arrays -- no asset-class-specific concepts. Used by both the
futures walk-forward (`scripts/backtest_scripts/run_carver_walkforward.py`)
and the FX walk-forward (`scripts/backtest_scripts/run_fx_walkforward.py`).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.pbo import pbo
from src.backtesting.statistics.psr import psr

# Honest cumulative count of distinct pre-registered trials run across the
# futures campaign (SP-A/E/B/C ledgers + the pre-campaign carry/crypto sweep),
# used to deflate the Sharpe for multiple testing (methodology Section 2). DSR
# grows only like sqrt(2 ln N), so this is a documented, defensible count, not
# a live query. Derivation:
#   SP-A ledger  (docs/strategies/research/20260707_FUTURES_SP_A_TRIALS.md): 7
#     (6 executed: #3,10,13,15,16,23 + 1 deferred pre-registered: #9)
#   SP-E ledger  (docs/strategies/research/20260707_FUTURES_SP_E_TRIALS.md): 4
#     (#49, #37, #26, #27 -- catalog numbers counted individually)
#   SP-B ledger  (docs/strategies/research/20260710_FUTURES_SP_B_TRIALS.md): 4
#     (#21, #25 overnight drift, #21 hour-slice variant, #39 pre-FOMC)
#   SP-C ledger  (docs/strategies/research/20260710_FUTURES_SP_C_TRIALS.md): 14
#     (5 continuous-engine rows + 9 convergence-engine rows)
#   Pre-campaign carry/crypto sweep (docs/strategies/research/
#     20260705_FUTURES_STRATEGY_EXPLORATION_REVIEW.md Sections 2-3, cross-
#     checked against docs/progress/20260704_SHARPE_UPLIFT_PHASE1_SUMMARY.md): 11
#     (7 signal families + 3 combination blends + 1 IDM-cap-1.5-alone variant)
#   Total = 7 + 4 + 4 + 14 + 11 = 40.
CAMPAIGN_CUMULATIVE_TRIALS = 40

_TRADING_DAYS_PER_YEAR = 252


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _add_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def _build_windows(train_months: int, test_months: int, step_months: int,
                    start: date, end: date) -> List[tuple[date, date, date]]:
    """Return (train_start, test_start, test_end) triples, non-overlapping in OOS."""
    windows: List[tuple[date, date, date]] = []
    train_start = start
    while True:
        test_start = _add_months(train_start, train_months)
        if test_start >= end:
            break
        test_end = min(_add_months(test_start, test_months), end)
        windows.append((train_start, test_start, test_end))
        if test_end >= end:
            break
        train_start = _add_months(train_start, step_months)
    return windows


def _oos_returns_dated(equity_curve: List[float], dates: List[date], test_start: date) -> pd.Series:
    """Slice the OOS-dated segment of a window's equity curve and diff to returns.

    Same slice logic as `_oos_returns`, but returns the dated `pd.Series`
    (index = OOS dates) instead of a bare numpy array.
    """
    if len(equity_curve) != len(dates):
        raise ValueError(
            f"equity_curve length {len(equity_curve)} != trading-day count {len(dates)} "
            "-- window date range mismatch between the backtest engine and the panel loader"
        )
    eq = pd.Series(equity_curve, index=pd.Index(dates))
    oos_idx = eq.index[eq.index >= test_start]
    if len(oos_idx) == 0:
        return pd.Series([], dtype=float)
    start_pos = eq.index.get_loc(oos_idx[0])
    # Include one day before the OOS start (if available) so the first OOS
    # return is a real day-over-day change, not a NaN from pct_change's edge.
    segment = eq.iloc[max(start_pos - 1, 0):]
    return segment.pct_change().dropna()


def _oos_returns(equity_curve: List[float], dates: List[date], test_start: date) -> np.ndarray:
    """Slice the OOS-dated segment of a window's equity curve and diff to returns."""
    return _oos_returns_dated(equity_curve, dates, test_start).to_numpy(dtype=float)


def _annualized_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return float("nan")
    std = float(np.nanstd(returns, ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    mean = float(np.nanmean(returns))
    return mean / std * np.sqrt(_TRADING_DAYS_PER_YEAR)


def _compute_pbo(per_window_returns: List[np.ndarray]) -> float:
    """PBO across windows-as-columns (CSCV on the OOS return series per window).

    Each window's stitched-eligible OOS return series is treated as one
    "config" column; PBO here answers whether the OOS ranking of windows is
    stable under CSCV resampling, not a parameter-selection PBO (there is no
    parameter selection for a parameter-free strategy).

    Windows shorter than the CSCV split count `s` are dropped BEFORE
    truncation, so a single short data-end-truncated window does not force
    NaN across all windows. Returns NaN honestly only if fewer than 2
    windows of length >= s survive (genuinely insufficient).
    """
    s = 16
    usable = [r for r in per_window_returns if r.size >= s]
    if len(usable) < 2:
        return float("nan")
    min_len = min(r.size for r in usable)
    matrix = np.column_stack([r[:min_len] for r in usable])
    return pbo(matrix, s=s)


def _verdict(result: Dict[str, Any]) -> str:
    psr_val = result["psr"]
    dsr_val = result["dsr"]
    pbo_val = result["pbo"]
    sharpe = result["oos_sharpe"]
    sharpe_1_5x = result["oos_sharpe_1_5x_cost"]

    if any(np.isnan(x) for x in (psr_val, dsr_val, sharpe)):
        return "INCONCLUSIVE -- insufficient data to compute the statistical gate."

    passes_stat_gate = psr_val >= 0.95 and dsr_val >= 0.95
    passes_cost_gate = sharpe_1_5x > 0.0 and (sharpe <= 0 or sharpe_1_5x >= 0.5 * sharpe)
    passes_pbo = (not np.isnan(pbo_val)) and pbo_val < 0.25

    if sharpe <= 0:
        return "REJECT -- OOS Sharpe is non-positive; no edge to deflate or gate."
    if passes_stat_gate and passes_cost_gate and passes_pbo:
        return "PASS -- clears the combined statistical gate (Section 2.5) and the 1.5x cost gate (Section 4)."
    reasons = []
    if not passes_stat_gate:
        reasons.append(f"PSR/DSR below 0.95 (psr={psr_val:.3f}, dsr={dsr_val:.3f})")
    if not passes_cost_gate:
        reasons.append(f"fails 1.5x cost sensitivity (1x={sharpe:.3f}, 1.5x={sharpe_1_5x:.3f})")
    if not passes_pbo:
        reasons.append(f"PBO not comfortably acceptable (pbo={pbo_val})")
    return "WEAK -- does not clear the combined gate: " + "; ".join(reasons)


def _oos_windows(returns: pd.Series, train_months: int, test_months: int, step_months: int) -> List[pd.Series]:
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


def gate_return_stream(returns: pd.Series, train_months: int = 36,
                        test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    """Walk-forward OOS Sharpe/PSR/DSR/PBO gate for a pre-built return stream."""
    oos = _oos_windows(returns, train_months, test_months, step_months)
    per_window = [w.to_numpy(dtype=float) for w in oos]
    stitched = np.concatenate(per_window) if per_window else np.array([])
    n = int(stitched.size)
    sharpe = _annualized_sharpe(stitched) if n else float("nan")
    s = pd.Series(stitched)
    skew = float(s.skew()) if n > 2 else 0.0
    kurt = float(s.kurtosis()) + 3.0 if n > 3 else 3.0
    return {
        "oos_sharpe": sharpe, "n_oos": n, "n_windows": len(oos),
        "psr": psr(sharpe, 0.0, n, skew, kurt) if n else float("nan"),
        "dsr": dsr(sharpe, [sharpe], n, skew, kurt, n_trials_project=CAMPAIGN_CUMULATIVE_TRIALS) if n else float("nan"),
        "pbo": _compute_pbo(per_window) if len(per_window) > 1 else float("nan"),
        "skew": skew, "kurtosis": kurt,
    }
