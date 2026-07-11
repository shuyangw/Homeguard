import numpy as np
import pandas as pd
from src.backtesting.vol.har_rv import har_forecast


def test_har_forecast_causal_and_shaped():
    idx = pd.date_range("2015-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    rv = pd.Series(np.abs(rng.normal(1e-4, 3e-5, 500)), index=idx)  # daily variance-ish
    fc = har_forecast(rv, min_train=252)
    # first min_train entries are NaN (no forecast yet)
    assert fc.iloc[:252].isna().all()
    assert fc.iloc[252:].notna().any()
    assert fc.index.equals(rv.index)


def test_har_forecast_does_not_leak_future_spike():
    # flat series with a single, unforecastable spike planted at index k. A causal
    # model fit only on strictly-prior data cannot know the spike is coming, so the
    # forecast made for index k (i.e. fc.iloc[k], produced using data up to k-1)
    # must stay close to the flat baseline, not jump up to anticipate it.
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    rv = pd.Series(1e-4, index=idx)
    k = 350
    rv.iloc[k] = 1.0  # huge, isolated spike unrelated to prior history
    fc = har_forecast(rv, min_train=252)
    assert fc.iloc[k] < 1e-2, "forecast for the spike day must not anticipate the future spike"


def test_har_forecast_positive():
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    rv = pd.Series(1e-4, index=idx)
    fc = har_forecast(rv, min_train=252)
    # constant RV -> forecast approx equals the constant, positive
    assert (fc.dropna() > 0).all()
    assert abs(fc.dropna().iloc[-1] - 1e-4) < 5e-5
