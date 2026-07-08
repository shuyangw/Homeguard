"""#26/#27 VIX roll-down as a return stream + statistical gate.

Short the front VX future to harvest roll-down when the curve is in CONTANGO
(vx2 > vx1); go FLAT whenever the curve inverts (backwardation, vx1 >= vx2) --
a structural kill-switch for short convexity, part of the strategy definition.
Position sign is determined by the PRIOR day's curve (causal). Gated as a return
stream via the same walk-forward PSR/DSR/PBO helpers used for the carry book."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.backtest_scripts.run_carver_walkforward import (
    TRIAL_COUNT_PARAMETER_FREE, _annualized_sharpe, _compute_pbo,
)
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr


def rolldown_returns(curve: pd.DataFrame) -> pd.Series:
    """Daily return of a short-VX1 roll-down sleeve with a backwardation kill-switch.

    Position (prior-day, causal): short (-1) when vx2 > vx1 (contango), else flat.
    Daily P&L of a short VX1 = -(vx1_t / vx1_{t-1} - 1).

    Roll days are excluded: vx1_settle is a continuous nearest-unexpired front, so at
    each monthly expiry the series switches to a further-out contract and JUMPS. A real
    rolled position never realizes that jump (it rolls at market, no P&L gap), so the
    pct_change on a roll day is a spurious return -- zeroed here. Roll days are detected
    by the front-contract switch (vx1_dte snaps from ~1 back up to ~30, i.e. diff > 0)."""
    c = curve.copy()
    c["date"] = pd.to_datetime(c["date"])
    c = c.sort_values("date").set_index("date")
    contango = (c["vx2_settle"] > c["vx1_settle"]).astype(float)
    position = (-1.0 * contango).shift(1)  # prior-day signal -> today's position (causal)
    vx1_ret = c["vx1_settle"].pct_change(fill_method=None)
    if "vx1_dte" in c.columns:
        roll_day = c["vx1_dte"].diff() > 0
        vx1_ret = vx1_ret.mask(roll_day, 0.0)
    return (position * vx1_ret).rename("rolldown_return")


def _windows(returns: pd.Series, train_months: int, test_months: int, step_months: int) -> List[pd.Series]:
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
    oos = _windows(returns, train_months, test_months, step_months)
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
        "dsr": dsr(sharpe, [sharpe], n, skew, kurt, n_trials_project=TRIAL_COUNT_PARAMETER_FREE) if n else float("nan"),
        "pbo": _compute_pbo(per_window) if len(per_window) > 1 else float("nan"),
        "skew": skew, "kurtosis": kurt,
    }


def run_vix_rolldown(curve: pd.DataFrame, output_dir, train_months: int = 36,
                      test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    """Compute the return stream, gate it, and persist the trade log (returns.csv)
    + gate.json to output_dir. Returns the gate result dict."""
    import json
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    returns = rolldown_returns(curve)
    result = gate_return_stream(returns, train_months=train_months,
                                 test_months=test_months, step_months=step_months)

    returns.to_frame("return").to_csv(out / "returns.csv", index_label="date")
    (out / "gate.json").write_text(json.dumps(result, default=float, indent=2))
    return result
