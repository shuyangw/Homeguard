import pytest

from src.data.futures.paths import continuous_1min_dir
from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.backtesting.engine.futures_portfolio_simulator import FuturesPortfolioSimulator


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="futures store not present")


def _base_config():
    return {
        "strategy": {"universe": ["MES", "MGC", "6E"]},
        "dates": {"start": "2022-01-01", "end": "2023-12-31"},
        "backtest": {"initial_capital": 25000, "vol_target_per_instrument": 0.20,
                     "rebalance": "weekly"},
    }


def _spy_run_sized(monkeypatch, captured):
    original = FuturesPortfolioSimulator.run_sized

    def wrapper(self, close_panel, forecast_panel, daily_vol_panel, vol_target,
                div_mult=1.0):
        captured["div_mult"] = div_mult
        return original(self, close_panel, forecast_panel, daily_vol_panel, vol_target,
                         div_mult=div_mult)

    monkeypatch.setattr(FuturesPortfolioSimulator, "run_sized", wrapper)


def test_idm_true_passes_per_root_dict_div_mult(monkeypatch):
    captured = {}
    _spy_run_sized(monkeypatch, captured)

    cfg = _base_config()
    cfg["backtest"]["idm"] = True
    result = run_futures_backtest(cfg, register=False)

    assert result["n_days"] > 0
    dm = captured["div_mult"]
    assert isinstance(dm, dict)
    universe = set(cfg["strategy"]["universe"])
    assert set(dm.keys()) <= universe
    assert len(dm) > 0
    for value in dm.values():
        assert value > 0
        assert value == value  # not NaN
        assert value not in (float("inf"), float("-inf"))


def test_idm_absent_passes_scalar_div_mult(monkeypatch):
    captured = {}
    _spy_run_sized(monkeypatch, captured)

    cfg = _base_config()   # no "idm" key -> defaults to False
    result = run_futures_backtest(cfg, register=False)

    assert result["n_days"] > 0
    dm = captured["div_mult"]
    assert isinstance(dm, float)
    assert not isinstance(dm, dict)
    assert dm == 1.0


def test_idm_false_passes_scalar_div_mult(monkeypatch):
    captured = {}
    _spy_run_sized(monkeypatch, captured)

    cfg = _base_config()
    cfg["backtest"]["idm"] = False
    result = run_futures_backtest(cfg, register=False)

    assert result["n_days"] > 0
    dm = captured["div_mult"]
    assert isinstance(dm, float)
    assert not isinstance(dm, dict)
    assert dm == 1.0
