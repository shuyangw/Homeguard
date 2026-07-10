import numpy as np
import pandas as pd
from src.backtesting.spreads.construction import SpreadSeries
from src.backtesting.spreads.continuous import (
    zscore_mr_forecast, momentum_forecast, continuous_return_stream)


def _sig(n=400):
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    return pd.Series(np.cumsum(rng.normal(0, 0.05, n)) + 2.0, index=idx)


def test_zscore_forecast_is_causal_and_capped():
    s = _sig()
    fc = zscore_mr_forecast(s, window=60, cap=2.0)
    assert fc.abs().max() <= 2.0 + 1e-9
    # causal: forecast at t must not use signal at t's own mean beyond shift
    # first `window` entries are NaN (no prior window)
    assert fc.iloc[:60].isna().all()


def test_mr_forecast_sign_opposes_deviation():
    idx = pd.date_range("2019-01-01", periods=120, freq="B")
    s = pd.Series([1.0] * 100 + [5.0] * 20, index=idx)  # big positive deviation at end
    fc = zscore_mr_forecast(s, window=60, cap=2.0)
    assert fc.dropna().iloc[-1] < 0  # high signal -> short (negative) forecast


def test_return_stream_targets_vol_and_charges_cost():
    idx = pd.date_range("2019-01-01", periods=300, freq="B")
    rng = np.random.default_rng(2)
    spread = SpreadSeries(
        signal=pd.Series(np.cumsum(rng.normal(0, 0.05, 300)), index=idx),
        unit_return=pd.Series(rng.normal(0, 0.01, 300), index=idx))
    fc = pd.Series(np.sign(np.sin(np.arange(300) / 10.0)), index=idx)  # flips -> turnover
    r = continuous_return_stream(spread, fc, cost_usd=20.0, target_vol=0.15)
    ann_vol = r.std() * np.sqrt(252)
    assert 0.05 < ann_vol < 0.40  # in the neighborhood of target
    # a zero-cost version earns strictly more (cost is subtracted)
    r0 = continuous_return_stream(spread, fc, cost_usd=0.0, target_vol=0.15)
    assert r0.sum() > r.sum()
