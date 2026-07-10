"""#34 gold/silver ratio RV -- the weak-anchor member.

No physical arbitrage bounds the ratio, so this is a regime-bounded MR trade:
short (2y) z-window, a hard symmetric |z|>3.5 structural stop as the load-bearing
exit (in a monetary-regime break the ratio trends for years). A likely no-edge
verdict is itself the reportable finding.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtesting.spreads.construction import SpreadLeg, build_spread, round_trip_cost_usd
from src.backtesting.spreads.convergence import simulate_convergence, gate_convergence
from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ratio_spread(start: date, end: date):
    panel = load_daily_panel(["GC", "SI"], start, end)
    closes = panel.xs("close", axis=1, level=1)
    legs = [SpreadLeg("GC", 1.0), SpreadLeg("SI", -1.0)]
    return build_spread(legs, closes, mode="multiplicative")


def run_ratio(start: date, end: date, output_dir) -> dict:
    spread = ratio_spread(start, end)
    cost = round_trip_cost_usd([SpreadLeg("GC", 1.0), SpreadLeg("SI", -1.0)])
    cost_return = min(0.02, cost / 100_000.0)
    trades, daily = simulate_convergence(spread.signal, spread.unit_return,
                                         cost_return=cost_return, window=504,
                                         entry_z=2.0, structural_z=3.5,
                                         structural_z_short=3.5, max_hold=120)
    # gate_convergence -> gate_return_stream's walk-forward window math adds
    # pd.DateOffset to the index, which requires a DatetimeIndex; the panel
    # loader's index is python datetime.date objects.
    daily.index = pd.to_datetime(daily.index)
    result = gate_convergence(daily)
    result["n_trades"] = len(trades)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "returns.csv", header=True)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out / "trades.csv", index=False)
    (out / "gate.json").write_text(json.dumps(result, indent=2))
    logger.info(f"[ratio:GC_SI] {result}")
    return result
