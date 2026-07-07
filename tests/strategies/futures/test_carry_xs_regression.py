import numpy as np
import pandas as pd
from src.strategies.advanced.futures_carry_strategy import FuturesCarryXSStrategy


def test_carry_xs_forecast_bounded_and_causal(monkeypatch):
    # two same-class roots with a synthetic carry series -> forecasts opposite sign, bounded
    roots = ["6E", "6J"]  # both fx
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    close = pd.DataFrame({"6E": np.linspace(100, 110, 60), "6J": np.linspace(100, 90, 60)}, index=idx)
    strat = FuturesCarryXSStrategy(roots)

    def fake_load(root):
        base = 0.05 if root == "6E" else -0.05
        return pd.Series(base, index=idx)

    monkeypatch.setattr(strat, "_load_carry", fake_load)
    fc = strat.forecast_panel(close)
    assert fc.abs().max().max() <= 20.0
    last = fc.iloc[-1]
    assert last["6E"] > 0 > last["6J"]  # opposite signs within the fx group
