import numpy as np
import pandas as pd
from src.strategies.advanced.futures_funding_strategy import FuturesFundingCarryStrategy


def test_positive_funding_gives_positive_forecast(monkeypatch):
    idx = pd.date_range("2021-01-01", periods=120, freq="B")
    close = pd.DataFrame({"BTC": np.linspace(30000, 40000, 120)}, index=idx)
    strat = FuturesFundingCarryStrategy(["BTC"])
    funding = pd.Series(0.20, index=idx)  # rich positive annualized funding
    monkeypatch.setattr(strat, "_load_funding", lambda root: funding)
    fc = strat.forecast_panel(close)
    assert fc.iloc[-1]["BTC"] > 0
    assert fc.abs().max().max() <= 20.0


def test_missing_funding_gives_zero(monkeypatch):
    idx = pd.date_range("2021-01-01", periods=60, freq="B")
    close = pd.DataFrame({"BTC": np.linspace(30000, 40000, 60)}, index=idx)
    strat = FuturesFundingCarryStrategy(["BTC"])
    monkeypatch.setattr(strat, "_load_funding", lambda root: None)
    fc = strat.forecast_panel(close)
    assert (fc["BTC"] == 0.0).all()
