import numpy as np
import pandas as pd
from src.strategies.advanced.futures_conditioning_strategy import FuturesCarryTrendStrategy


def test_gate_zeros_when_signs_disagree(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    close = pd.DataFrame({"ES": np.linspace(100, 110, 10)}, index=idx)
    strat = FuturesCarryTrendStrategy(["ES"])
    base_fc = pd.DataFrame({"ES": np.full(10, 12.0)}, index=idx)   # trend long
    cond_fc = pd.DataFrame({"ES": np.full(10, -5.0)}, index=idx)   # carry short -> disagree
    monkeypatch.setattr(strat, "_base_forecast", lambda c: base_fc)
    monkeypatch.setattr(strat, "_cond_forecast", lambda c: cond_fc)
    fc = strat.forecast_panel(close)
    assert (fc["ES"] == 0.0).all()


def test_gate_passes_base_when_signs_agree(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    close = pd.DataFrame({"ES": np.linspace(100, 110, 10)}, index=idx)
    strat = FuturesCarryTrendStrategy(["ES"])
    base_fc = pd.DataFrame({"ES": np.full(10, 12.0)}, index=idx)
    cond_fc = pd.DataFrame({"ES": np.full(10, 4.0)}, index=idx)  # both long -> agree
    monkeypatch.setattr(strat, "_base_forecast", lambda c: base_fc)
    monkeypatch.setattr(strat, "_cond_forecast", lambda c: cond_fc)
    fc = strat.forecast_panel(close)
    assert (fc["ES"] == 12.0).all()
