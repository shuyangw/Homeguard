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

# DSR deflation distribution (methodology Section 2.3 / 8): the OOS Sharpe of
# every distinct, gradeable, pre-registered campaign trial, extracted verbatim
# from the OOS-Sharpe column of the four SP-A/E/B/C ledgers (never PBO, skew,
# kurtosis, or gate thresholds -- those are different columns). Where a ledger
# reports a pre-fix (contaminated) and a post-fix value, only the post-fix
# value is included (e.g. SP-C #31's roll-masked Sharpes, not the roll-jump-
# contaminated pre-fix numbers). Ungradeable trials (NaN / n_windows=0, e.g.
# SP-A #9 deferred, SP-B #39 pre-FOMC, SP-C #35 all three segments, SP-E #49
# no-data) are excluded -- they contribute no evaluated Sharpe to deflate
# against. Both the 1x and 1.5x cost-sensitivity Sharpes are included where a
# ledger reports both, since methodology Section 4's cost-sensitivity check is
# itself an evaluated trial outcome, not a duplicate of the 1x number. This is
# a documented, reproducible constant (recomputed only when a ledger gains new
# graded rows), not a live query.
CAMPAIGN_TRIAL_SHARPES: List[float] = [
    # SP-A (docs/strategies/research/20260707_FUTURES_SP_A_TRIALS.md), OOS 1x / 1.5x
    0.209, 0.181,    # #3 XS commodity momentum
    0.846, 0.833,    # #10 curve-slope XS
    0.180, 0.166,    # #15 same-month seasonality
    -0.274, -0.279,  # #16 turn-of-month (mis-sampled per ledger, still a real graded Sharpe)
    0.297, 0.288,    # #23 short-horizon reversal
    0.357, 0.336,    # #13 carry-trend gate
    # SP-E (docs/strategies/research/20260707_FUTURES_SP_E_TRIALS.md), OOS 1x
    -0.124,          # #37 CoT tilt
    0.564,           # #26/27 VIX roll-down, POST roll-jump fix (was -0.854 contaminated pre-fix)
    # SP-B1 (docs/strategies/research/20260710_FUTURES_SP_B_TRIALS.md), OOS 1x / 1.5x
    0.792, 0.671,    # #21/25 overnight drift
    -0.023, -0.277,  # #21 NY-Fed hour-slice
    # SP-C continuous engine (docs/strategies/research/20260710_FUTURES_SP_C_TRIALS.md)
    0.329,           # #36 NQ/ES RV
    -0.280,          # #36 RTY/ES RV
    # SP-C convergence engine, roll-masked where applicable
    0.394,           # #31 CL calendar (POST roll-jump mask; pre-fix was 1.183, excluded)
    -0.150,          # #31 NG calendar (POST roll-jump mask; pre-fix was 1.017, excluded)
    0.174,           # #31 ZC calendar
    0.358,           # #31 ZS calendar
    0.263,           # #31 ZW calendar (POST roll-jump mask; pre-fix was 1.019, excluded)
    -0.116,          # #32 crack RB-CL
    -0.215,          # #32 crack HO-CL
    0.136,           # #33 crush ZM+ZL-ZS
    0.269,           # #34 GC/SI ratio
]

def get_campaign_trial_distribution(db_path: Any = None) -> tuple[int, List[float]]:
    """Growing project-wide DSR trial-Sharpe distribution (Gate 0.2, methodology
    Section 9.4).

    Starts from the static, documented `CAMPAIGN_TRIAL_SHARPES` baseline (the
    40 pre-registered trials of the SP-A/B/C/E campaign, fixed before
    `output/experiments.duckdb` existed) and appends one Sharpe per run
    subsequently logged to the registry with a numeric `oos_sharpe` metric.
    Every registry-logged run postdates the static baseline by construction,
    so this is strictly additive -- it never re-counts a trial already baked
    into the 40. As this retest (and any future work) appends runs via
    `src.experiments.registry.append_run`, N grows and SR_zero rises with it
    -- the intended honest, growing-search behavior (never shrinking N to
    make a gate easier).

    Falls back to the static baseline alone -- never raises -- if the
    registry file is missing, empty, or briefly lock-contended (e.g. a fresh
    worktree checkout where `output/` is gitignored and not yet created).
    """
    try:
        from src.experiments.registry import DEFAULT_DB_PATH, _connect_with_retry, init_db
        import json as _json

        path = db_path or DEFAULT_DB_PATH
        init_db(path)
        con = _connect_with_retry(path, read_only=True)
        try:
            rows = con.execute("SELECT metrics FROM runs WHERE metrics IS NOT NULL").fetchall()
        finally:
            con.close()
        extra: List[float] = []
        for (metrics_json,) in rows:
            try:
                m = _json.loads(metrics_json)
            except (TypeError, ValueError):
                continue
            val = m.get("oos_sharpe")
            if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                extra.append(float(val))
        return CAMPAIGN_CUMULATIVE_TRIALS + len(extra), list(CAMPAIGN_TRIAL_SHARPES) + extra
    except Exception:
        return CAMPAIGN_CUMULATIVE_TRIALS, list(CAMPAIGN_TRIAL_SHARPES)


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


def _stitch_oos_dedup(per_window: List[pd.Series]) -> np.ndarray:
    """Concatenate per-window dated OOS return series, drop the calendar day
    shared by adjacent windows (keep-first), and return a bare float array.
    Mirrors the dedup in the seatbelt walk-forward runner so the generic FX
    runner does not double-count a boundary trading day."""
    if not per_window:
        return np.array([], dtype=float)
    s = pd.concat(per_window).sort_index(kind="stable")
    s = s[~s.index.duplicated(keep="first")]
    return s.to_numpy(dtype=float)


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

    Windows shorter than 2*s are dropped BEFORE truncation -- `pbo()` splits
    each column into `s` folds, so a surviving window must guarantee
    `min_len // s >= 2` (min_len >= 2*s) or the CSCV split itself degenerates.
    A window with `s <= size < 2*s` (e.g. 30 rows against s=16) would pass the
    old `>= s` filter yet still NaN the whole PBO once folded. Returns NaN
    honestly only if fewer than 2 windows of length >= 2*s survive (genuinely
    insufficient).
    """
    s = 16
    usable = [r for r in per_window_returns if r.size >= 2 * s]
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
    n_trials, trial_sharpes = get_campaign_trial_distribution()
    return {
        "oos_sharpe": sharpe, "n_oos": n, "n_windows": len(oos),
        "psr": psr(sharpe, 0.0, n, skew, kurt) if n else float("nan"),
        "dsr": dsr(sharpe, trial_sharpes, n, skew, kurt, n_trials_project=n_trials) if n else float("nan"),
        "pbo": _compute_pbo(per_window) if len(per_window) > 1 else float("nan"),
        "skew": skew, "kurtosis": kurt,
        "trial_count": n_trials,
    }
