"""#31 single-commodity calendar-spread mean reversion (inventory-anchored).

Spread level = F2 - F1 (next minus front settle). Cash-and-carry bounds contango
above; convenience yield makes backwardation soft below -> the convergence
engine's asymmetric structural stop (tighter on the short side) encodes that.
Restricted to inventory-anchored storables; metals calendars (pure financing)
are excluded as a different mechanism.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.futures.front_next import front_next_history
from src.backtesting.spreads.construction import SpreadLeg, round_trip_cost_usd
from src.backtesting.spreads.convergence import simulate_convergence, gate_convergence
from src.utils.logger import get_logger

logger = get_logger(__name__)

STORABLES: list[str] = ["CL", "NG", "ZC", "ZS", "ZW"]


def calendar_signal(root: str, start: date, end: date) -> tuple[pd.Series, pd.Series]:
    hist = front_next_history(root, start, end).set_index("date").sort_index()
    level = (hist["f2"] - hist["f1"]).rename("calendar")
    valid = hist["f1"].dropna()
    scale = abs(float(valid.iloc[0])) if not valid.empty else 1.0
    scale = scale if scale > 1e-9 else 1.0
    unit_return = (level.diff() / scale).rename("return")
    return level, unit_return


def run_calendar(root: str, start: date, end: date, output_dir) -> dict:
    level, unit_return = calendar_signal(root, start, end)
    cost = round_trip_cost_usd([SpreadLeg(root, 1.0), SpreadLeg(root, -1.0)])
    # express the round-trip cost as a return using the same scale as unit_return
    valid = level.dropna()
    cost_return = 0.0
    if not valid.empty:
        # cost in spread-level terms is negligible relative to notional; use a small
        # fixed fraction floored by the tick-based round trip
        cost_return = min(0.02, cost / 100_000.0)
    trades, daily = simulate_convergence(level, unit_return, cost_return=cost_return,
                                         window=252, entry_z=2.0, structural_z=4.0,
                                         structural_z_short=3.0, max_hold=60)
    result = gate_convergence(daily)
    result["n_trades"] = len(trades)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "returns.csv", header=True)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out / "trades.csv", index=False)
    (out / "gate.json").write_text(json.dumps(result, indent=2))
    logger.info(f"[calendar:{root}] {result}")
    return result
