import numpy as np
import pandas as pd
import pytest

from src.strategies.advanced.futures_carry_strategy import FuturesCarryStrategy
from src.strategies.registry import get_strategy_class


def _close(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # gently trending prices so vol is finite and non-zero
    return pd.DataFrame({"GC": np.linspace(1800, 1900, n), "CL": np.linspace(60, 70, n)}, index=idx)


def test_registered():
    assert get_strategy_class("FuturesCarry") is FuturesCarryStrategy
    assert get_strategy_class("Carry") is FuturesCarryStrategy


def test_forecast_shape_and_cap(monkeypatch):
    close = _close()
    # constant carry 0.05 for every root
    monkeypatch.setattr(FuturesCarryStrategy, "_load_carry",
                        lambda self, root: pd.Series(0.05, index=close.index))
    strat = FuturesCarryStrategy(["GC", "CL"])
    fc = strat.forecast_panel(close)
    assert list(fc.columns) == ["GC", "CL"]
    assert fc.index.equals(close.index)
    valid = fc.dropna()
    assert ((valid >= -20.0) & (valid <= 20.0)).all().all()


def test_missing_cache_gives_nan_column(monkeypatch):
    close = _close()
    monkeypatch.setattr(FuturesCarryStrategy, "_load_carry", lambda self, root: None)
    strat = FuturesCarryStrategy(["GC", "CL"])
    fc = strat.forecast_panel(close)
    assert fc["GC"].isna().all() and fc["CL"].isna().all()


def test_forecast_sign_follows_carry(monkeypatch):
    close = _close()
    monkeypatch.setattr(FuturesCarryStrategy, "_load_carry",
                        lambda self, root: pd.Series(0.05, index=close.index))
    fc = FuturesCarryStrategy(["GC"]).forecast_panel(close).dropna()
    assert (fc["GC"] > 0).all()  # positive carry -> positive (long) forecast
