"""#26/#27 VIX roll-down as a return stream + statistical gate.

Short the front VX future to harvest roll-down when the curve is in CONTANGO
(vx2 > vx1); go FLAT whenever the curve inverts (backwardation, vx1 >= vx2) --
a structural kill-switch for short convexity, part of the strategy definition.
Position sign is determined by the PRIOR day's curve (causal). Gated as a return
stream via the same walk-forward PSR/DSR/PBO helpers used for the carry book."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtesting.walkforward_common import gate_return_stream


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


def subperiod_audit(returns: pd.Series) -> dict:
    """Tail/skew audit for a short-vol return stream (reporting, not a gate).

    Per-calendar-year annualized Sharpe, realized skew/kurtosis, worst single-day
    return, max drawdown, and the realized P&L in the 2018-02 (Volmageddon) and
    2020-03 (COVID) crisis months. Computed on the WHOLE stream -- crash days are
    never excised."""
    r = returns.dropna()
    r.index = pd.to_datetime(r.index)
    by_year = {}
    for yr, grp in r.groupby(r.index.year):
        sd = grp.std()
        by_year[int(yr)] = float(grp.mean() / sd * np.sqrt(252)) if sd and not np.isnan(sd) else float("nan")
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    def _month_sum(y, m):
        mask = (r.index.year == y) & (r.index.month == m)
        return float(r[mask].sum()) if mask.any() else float("nan")

    return {
        "by_year": by_year,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "worst_day": float(r.min()),
        "max_drawdown": float(drawdown.min()),
        "vol_2018_02": _month_sum(2018, 2),
        "vol_2020_03": _month_sum(2020, 3),
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
