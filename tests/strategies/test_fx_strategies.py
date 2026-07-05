import numpy as np
import pandas as pd

from src.strategies.registry import get_strategy_class


def _price_panel(pairs, n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    data = {p: 1.0 + np.cumsum(rng.normal(0, 0.001, n)) for p in pairs}
    return pd.DataFrame(data, index=idx)


def test_fx_trend_registered_and_forecasts():
    cls = get_strategy_class("FxTrend")
    strat = cls(["EURUSD", "USDJPY"])
    fc = strat.forecast_panel(_price_panel(["EURUSD", "USDJPY"]))
    assert list(fc.columns) == ["EURUSD", "USDJPY"]
    assert fc.abs().max().max() <= 20.0  # forecast cap


def test_fx_value_registered_and_forecasts():
    cls = get_strategy_class("FxValue")
    strat = cls(["EURUSD", "USDJPY"])
    fc = strat.forecast_panel(_price_panel(["EURUSD", "USDJPY"], n=1400))
    assert list(fc.columns) == ["EURUSD", "USDJPY"]
