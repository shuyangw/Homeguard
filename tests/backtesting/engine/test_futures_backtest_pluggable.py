import pandas as pd
import pytest

from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.strategies.registry import register_strategy

_SLICE = {
    "strategy": {"universe": ["6E", "GC"]},
    "dates": {"start": "2022-01-03", "end": "2022-06-30"},
    "backtest": {"initial_capital": 100000, "vol_target_per_instrument": 0.20,
                 "rebalance": "weekly", "cost_mult": 1.0},
}


class _ZeroForecast:
    """Stub futures strategy: forecast 0 everywhere -> no positions, flat equity."""
    def __init__(self, universe, **params):
        self.universe = list(universe)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=close_panel.index, columns=close_panel.columns)


class _NoForecast:
    """Stub lacking forecast_panel -> must be rejected."""
    def __init__(self, universe, **params):
        self.universe = list(universe)


def test_unknown_strategy_name_raises_fast():
    cfg = {**_SLICE, "strategy": {"name": "NoSuchStrategy", "universe": ["6E", "GC"]}}
    with pytest.raises(ValueError):
        run_futures_backtest(cfg)


def test_strategy_missing_forecast_panel_raises():
    register_strategy("NoForecastStub", _NoForecast)
    cfg = {**_SLICE, "strategy": {"name": "NoForecastStub", "universe": ["6E", "GC"]}}
    with pytest.raises(ValueError):
        run_futures_backtest(cfg)


def test_stub_strategy_is_actually_used():
    # Zero-forecast stub -> no trades -> equity stays flat at initial capital.
    register_strategy("ZeroForecastStub", _ZeroForecast)
    cfg = {**_SLICE, "strategy": {"name": "ZeroForecastStub", "universe": ["6E", "GC"]}}
    res = run_futures_backtest(cfg)
    eq = res["equity_curve"]
    assert eq, "empty equity curve"
    assert all(abs(v - 100000) < 1e-6 for v in eq), "stub not used (equity moved)"


def test_default_name_runs_carver_backward_compat():
    # No strategy.name -> Carver; must produce a non-flat, finite equity curve.
    res = run_futures_backtest(_SLICE)
    eq = res["equity_curve"]
    assert eq and all(isinstance(v, float) for v in eq)
    assert any(abs(v - 100000) > 1e-6 for v in eq), "Carver produced no trades (unexpected)"
