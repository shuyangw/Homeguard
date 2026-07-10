"""#32 crack (refining margin) and #33 crush (soy processing margin).

Per-unit-normalized additive spreads. Signals build from daily_raw continuous
fronts. Feasibility: a balanced position is un-tradeable at small size (no micro
product legs); these are gate-evaluated as research-grade -- the verdict is
reported honestly, sizing feasibility is a separate concern.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtesting.spreads.construction import SpreadLeg, SpreadSeries, build_spread, round_trip_cost_usd
from src.backtesting.spreads.convergence import simulate_convergence, gate_convergence
from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GAL_PER_BBL = 42.0


def crack_spread(product_root: str, start: date, end: date) -> SpreadSeries:
    panel = load_daily_panel([product_root, "CL"], start, end)
    closes = panel.xs("close", axis=1, level=1).copy()
    # RB/HO quote $/gal -> x42 to $/bbl so the crack is per-barrel dollars
    closes[product_root] = closes[product_root] * _GAL_PER_BBL
    legs = [SpreadLeg(product_root, 1.0), SpreadLeg("CL", -1.0)]
    return build_spread(legs, closes, mode="additive")


def crush_spread(start: date, end: date) -> SpreadSeries:
    panel = load_daily_panel(["ZM", "ZL", "ZS"], start, end)
    closes = panel.xs("close", axis=1, level=1).copy()
    # board crush per bushel: ZM ($/short ton) x 0.022, ZL (cents/lb) x 0.11 -> $, minus ZS ($/bu)
    closes["ZM"] = closes["ZM"] * 0.022
    closes["ZL"] = closes["ZL"] * 0.11
    legs = [SpreadLeg("ZM", 1.0), SpreadLeg("ZL", 1.0), SpreadLeg("ZS", -1.0)]
    return build_spread(legs, closes, mode="additive")


_SPECS = {
    "crack_RB": (lambda s, e: crack_spread("RB", s, e), ["RB", "CL"]),
    "crack_HO": (lambda s, e: crack_spread("HO", s, e), ["HO", "CL"]),
    "crush": (lambda s, e: crush_spread(s, e), ["ZM", "ZL", "ZS"]),
}


def run_processing(name: str, start: date, end: date, output_dir) -> dict:
    builder, roots = _SPECS[name]
    spread = builder(start, end)
    cost = round_trip_cost_usd([SpreadLeg(r, 1.0) for r in roots])
    cost_return = min(0.02, cost / 100_000.0)
    trades, daily = simulate_convergence(spread.signal, spread.unit_return,
                                         cost_return=cost_return, window=252,
                                         entry_z=2.0, structural_z=4.0,
                                         structural_z_short=3.0, max_hold=60)
    # gate_return_stream's walk-forward window math adds pd.DateOffset to the
    # index, which requires a DatetimeIndex; the panel loader's index is
    # python datetime.date objects (see spread_intermarket_strategy.run_intermarket).
    daily.index = pd.to_datetime(daily.index)
    result = gate_convergence(daily)
    result["n_trades"] = len(trades)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "returns.csv", header=True)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out / "trades.csv", index=False)
    (out / "gate.json").write_text(json.dumps(result, indent=2))
    logger.info(f"[processing:{name}] {result}")
    return result
