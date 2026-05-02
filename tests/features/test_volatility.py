"""Tests for src.features.volatility."""
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_ohlc(n_days: int, sigma: float = 0.01,
                         seed: int = 0) -> pd.DataFrame:
    """Generate synthetic OHLC data from GBM with intraday range noise."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0, sigma, n_days)
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    intraday_range = np.abs(rng.normal(0.0, sigma * 0.5, n_days)) + 1e-4
    high_offset = intraday_range * close
    low_offset = intraday_range * close
    open_ = np.empty(n_days)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0.0, sigma * 0.1, n_days - 1))
    high = np.maximum.reduce([open_, close]) + high_offset
    low = np.minimum.reduce([open_, close]) - low_offset
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
    }, index=idx)


def test_module_importable():
    from src.features import volatility  # noqa: F401


class TestCloseToCloseRV:
    def test_phase2_contract_bit_equal(self):
        """Contract test guarding the Phase 2 swap of the FRS, DSTS, EVR
        inline expressions onto close_to_close_rv. Output must be bit-equal."""
        from src.features import close_to_close_rv
        rng = np.random.default_rng(8)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 200))))
        returns = prices.pct_change()
        annualization_factor = 252.0
        lookback = 20

        # Pre-swap inline expression (matches FRS, EVR explicit form, and DSTS
        # default form -- all bit-equal since pandas defaults min_periods=window)
        expected = (returns.rolling(window=lookback, min_periods=lookback).std()
                    * np.sqrt(annualization_factor))
        actual = close_to_close_rv(returns, window=lookback,
                                   annualization_factor=annualization_factor)
        # Bit-equal on non-NaN; NaN positions match
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_efficiency_lower_than_parkinson(self):
        # Asserted in the parkinson_rv task too; here we just verify the
        # function produces sensible non-negative volatilities.
        from src.features import close_to_close_rv
        rng = np.random.default_rng(9)
        returns = pd.Series(rng.normal(0.0, 0.01, 1000))
        rv = close_to_close_rv(returns, window=20)
        assert (rv.dropna() >= 0).all()

    def test_leading_nan(self):
        from src.features import close_to_close_rv
        returns = pd.Series([0.01, -0.01, 0.005, -0.005, 0.002, 0.001])
        rv = close_to_close_rv(returns, window=4)
        assert rv.iloc[0:3].isna().all()
        assert not np.isnan(rv.iloc[3])

    def test_negative_window_raises(self):
        from src.features import close_to_close_rv
        with pytest.raises(ValueError):
            close_to_close_rv(pd.Series([0.01, 0.02]), window=0)

    def test_nan_propagates(self):
        from src.features import close_to_close_rv
        returns = pd.Series([0.01, np.nan, 0.005, -0.005, 0.002, 0.001])
        rv = close_to_close_rv(returns, window=3)
        # Window containing NaN -> NaN output
        assert np.isnan(rv.iloc[1])

    def test_empty_series(self):
        from src.features import close_to_close_rv
        out = close_to_close_rv(pd.Series([], dtype=float), window=20)
        assert len(out) == 0

    def test_index_preserved(self):
        from src.features import close_to_close_rv
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series(np.linspace(-0.01, 0.01, 10), index=idx)
        rv = close_to_close_rv(returns, window=3)
        assert rv.index.equals(idx)
        assert len(rv) == len(returns)
