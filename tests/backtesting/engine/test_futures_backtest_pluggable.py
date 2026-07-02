import pandas as pd
import pytest

from src.data.futures.paths import continuous_1min_dir
from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.strategies.registry import register_strategy


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists()


# Mirror the e2e guard: skip (not error) on a machine/CI without the futures store.
pytestmark = pytest.mark.skipif(not _data_present(), reason="futures store not present")

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


class _ParamForecast:
    """Stub whose constant forecast equals params['level'] -> equity is flat iff level == 0."""
    def __init__(self, universe, **params):
        self.universe = list(universe)
        self.level = float(params.get("level", 0.0))

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.level, index=close_panel.index, columns=close_panel.columns)


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


def test_strategy_params_flow_to_constructor():
    # params.level=10 -> non-zero constant forecast -> positions -> equity moves.
    # At 1M capital the sizing is comfortably above the integer-contract floor, so a
    # non-flat curve proves params reached the strategy constructor (level=0 would be flat).
    register_strategy("ParamForecastStub", _ParamForecast)
    cfg = {
        "strategy": {"name": "ParamForecastStub", "universe": ["6E", "GC"],
                     "params": {"level": 10.0}},
        "dates": {"start": "2022-01-03", "end": "2022-06-30"},
        "backtest": {"initial_capital": 1_000_000, "vol_target_per_instrument": 0.20,
                     "rebalance": "weekly", "cost_mult": 1.0},
    }
    res = run_futures_backtest(cfg)
    eq = res["equity_curve"]
    assert eq and any(abs(v - 1_000_000) > 1e-6 for v in eq), \
        "params.level did not reach the strategy constructor (equity flat)"
