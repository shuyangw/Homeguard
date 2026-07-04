import numpy as np, pandas as pd
from src.strategies.advanced.futures_carry_strategy import FuturesCarryXSStrategy
from src.strategies.registry import get_strategy_class

def _close(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"CL": np.linspace(60,70,n), "NG": np.linspace(3,4,n),
                         "ES": np.linspace(3000,4000,n)}, index=idx)

def test_registered():
    assert get_strategy_class("FuturesCarryXS") is FuturesCarryXSStrategy

def test_within_class_demean_and_cap(monkeypatch):
    close = _close()
    # both energy roots carry +0.05 (a pure common bet); ES alone in equity
    monkeypatch.setattr(FuturesCarryXSStrategy, "_load_carry",
                        lambda self, root: pd.Series(0.05, index=close.index))
    fc = FuturesCarryXSStrategy(["CL","NG","ES"]).forecast_panel(close)
    v = fc.dropna()
    assert ((v >= -20.0) & (v <= 20.0)).all().all()
    # CL and NG share a common energy carry -> after within-energy demean their
    # forecasts are ~equal-and-opposite around 0 (common component removed).
    assert abs(v["CL"].mean() + v["NG"].mean()) < 1e-6
