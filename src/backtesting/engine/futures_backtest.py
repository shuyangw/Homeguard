"""Config-driven Carver TSMOM futures backtest orchestration (Task 9).

Assembles: daily panel loader -> Carver forecast -> per-day vol-targeted
contract sizing -> margin cap -> daily multi-instrument simulator ->
standard report -> experiment registry. This is the futures-asset-class
counterpart to the equity/crypto config-driven paths in
`src.backtest_runner`; kept in its own module because the futures sizing
math (contracts, margin, MTM cash) does not fit the equity Portfolio
abstractions.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict

import pandas as pd

from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.backtesting.costs.futures import futures_round_trip_usd
from src.backtesting.engine.futures_portfolio_simulator import FuturesPortfolioSimulator
from src.backtesting.margin.futures_margin import MarginModel
from src.backtesting.reporting.standard_report import StandardReportGenerator
from src.features.volatility import close_to_close_rv
from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
from src.utils import logger

_DEFAULT_INITIAL_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20
_DEFAULT_REBALANCE = "weekly"


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def run_futures_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a config-driven Carver TSMOM futures backtest end-to-end.

    Returns a dict with `n_days`, `metrics` (StandardReportGenerator's
    `overall_metrics`), `equity_curve` (list of floats), and `run_id`
    (None if the registry append failed -- logged, not raised).
    """
    strategy_cfg = config.get("strategy", {})
    dates_cfg = config.get("dates", {})
    backtest_cfg = config.get("backtest", {})

    universe = list(strategy_cfg["universe"])
    start = _as_date(dates_cfg["start"])
    end = _as_date(dates_cfg["end"])
    capital = float(backtest_cfg.get("initial_capital", _DEFAULT_INITIAL_CAPITAL))
    vol_target = float(backtest_cfg.get("vol_target_per_instrument", _DEFAULT_VOL_TARGET))
    rebalance = backtest_cfg.get("rebalance", _DEFAULT_REBALANCE)
    cost_mult = float(backtest_cfg.get("cost_mult", 1.0))

    panel = load_daily_panel(universe, start, end)
    close = panel.xs("close", axis=1, level=1)

    forecasts = CarverMomentumStrategy(universe).forecast_panel(close)

    returns = close.pct_change()
    daily_vol = returns.apply(lambda col: close_to_close_rv(col, 25, annualization_factor=1), axis=0)

    margin_model = MarginModel()
    sim = FuturesPortfolioSimulator(
        initial_capital=capital,
        cost_fn=futures_round_trip_usd,
        margin_model=margin_model,
        rebalance=rebalance,
        cost_mult=cost_mult,
    )
    res = sim.run_sized(close, forecasts, daily_vol, vol_target)

    report = StandardReportGenerator().generate_report(
        res.equity_curve, "CarverMomentum", universe,
        str(start), str(end), capital,
    )

    run_id = None
    try:
        from src.experiments import append_run

        run_id = append_run(
            strategy_name="CarverMomentum",
            agent_name="futures-harness",
            metrics=report["overall_metrics"],
            asset_class="futures",
            data_frequency="daily",
            params=config,
            window_start=start,
            window_end=end,
        )
    except Exception as e:
        logger.error(f"[futures_backtest] registry append_run failed (non-fatal): {e}")

    return {
        "n_days": len(res.equity_curve),
        "metrics": report["overall_metrics"],
        "equity_curve": res.equity_curve.tolist(),
        "run_id": run_id,
    }
