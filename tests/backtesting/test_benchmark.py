import numpy as np
import pandas as pd

from src.backtesting.benchmark import (
    sp500_sharpe_over_dates, correlation_over_dates, information_ratio_vs_sp500)


def _sp_returns(n=500):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def test_sharpe_over_dates_uses_only_given_dates():
    sp = _sp_returns()
    subset = sp.index[100:200]
    got = sp500_sharpe_over_dates(subset, sp_returns=sp)
    expected = sp.reindex(pd.to_datetime(subset)).dropna()
    exp_sharpe = expected.mean() / expected.std(ddof=1) * np.sqrt(252)
    assert abs(got - exp_sharpe) < 1e-9


def test_correlation_of_series_with_itself_is_one():
    sp = _sp_returns()
    assert abs(correlation_over_dates(sp, sp_returns=sp) - 1.0) < 1e-9


def test_information_ratio_is_zero_against_itself():
    sp = _sp_returns()
    # active return (sp - sp) is all zeros -> IR is nan (zero std), handled
    ir = information_ratio_vs_sp500(sp, sp_returns=sp)
    assert np.isnan(ir)
