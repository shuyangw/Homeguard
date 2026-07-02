import numpy as np
import pandas as pd
from src.strategies.advanced.carver_indicators import ewmac_forecast, combined_forecast, FORECAST_SCALARS


def test_scalars_present():
    assert set(FORECAST_SCALARS) == {(4, 16), (16, 64), (64, 256)}


def test_uptrend_positive_forecast():
    prices = pd.Series(np.linspace(100, 200, 400))
    vol = prices * 0.01
    f = ewmac_forecast(prices, 16, 64, vol)
    assert f.iloc[-1] > 0  # sustained uptrend -> positive forecast


def test_forecast_capped():
    prices = pd.Series(np.linspace(100, 100000, 400))  # violent trend
    vol = prices * 0.001
    f = ewmac_forecast(prices, 4, 16, vol)
    assert f.dropna().abs().max() <= 20.0 + 1e-9


def test_combined_averages_and_caps():
    prices = pd.Series(np.linspace(100, 200, 400))
    vol = prices * 0.01
    c = combined_forecast(prices, vol, [(4, 16), (16, 64), (64, 256)])
    assert c.dropna().abs().max() <= 20.0 + 1e-9
