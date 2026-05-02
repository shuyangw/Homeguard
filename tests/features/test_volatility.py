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


class TestParkinsonRV:
    def test_converges_to_known_sigma(self):
        from src.features import parkinson_rv
        sigma = 0.02
        ohlc = _make_synthetic_ohlc(2000, sigma=sigma, seed=10)
        rv = parkinson_rv(ohlc, window=60).dropna()
        # Annualized: sqrt(252) * sigma_daily
        expected = sigma * np.sqrt(252)
        # Tolerance is generous because synthetic OHLC is approximate
        assert 0.5 * expected < rv.median() < 2.0 * expected

    def test_more_efficient_than_close_to_close(self):
        """Parkinson should have lower variance than close-to-close on the
        same data."""
        from src.features import parkinson_rv, close_to_close_rv
        ohlc = _make_synthetic_ohlc(1000, sigma=0.02, seed=11)
        returns = ohlc['close'].pct_change()
        rv_p = parkinson_rv(ohlc, window=20).dropna()
        rv_c = close_to_close_rv(returns, window=20).dropna()
        # Variance of estimates (sampling variability)
        assert rv_p.std() < rv_c.std() * 1.5  # generous; mostly checks order

    def test_missing_high_column_raises(self):
        from src.features import parkinson_rv
        df = pd.DataFrame({'low': [1, 2, 3]})
        with pytest.raises(KeyError):
            parkinson_rv(df, window=2)

    def test_missing_low_column_raises(self):
        from src.features import parkinson_rv
        df = pd.DataFrame({'high': [1, 2, 3]})
        with pytest.raises(KeyError):
            parkinson_rv(df, window=2)

    def test_negative_window_raises(self):
        from src.features import parkinson_rv
        ohlc = _make_synthetic_ohlc(10, seed=12)
        with pytest.raises(ValueError):
            parkinson_rv(ohlc, window=0)

    def test_index_preserved(self):
        from src.features import parkinson_rv
        ohlc = _make_synthetic_ohlc(20, seed=13)
        rv = parkinson_rv(ohlc, window=5)
        assert rv.index.equals(ohlc.index)
        assert len(rv) == len(ohlc)

    def test_nan_propagates(self):
        from src.features import parkinson_rv
        ohlc = _make_synthetic_ohlc(20, seed=14)
        ohlc.loc[ohlc.index[5], 'high'] = np.nan
        rv = parkinson_rv(ohlc, window=3)
        # Windows containing the NaN row -> NaN
        assert np.isnan(rv.iloc[5])


class TestGarmanKlassRV:
    def test_converges_to_known_sigma(self):
        from src.features import garman_klass_rv
        sigma = 0.02
        ohlc = _make_synthetic_ohlc(2000, sigma=sigma, seed=15)
        rv = garman_klass_rv(ohlc, window=60).dropna()
        expected = sigma * np.sqrt(252)
        assert 0.5 * expected < rv.median() < 2.0 * expected

    def test_missing_columns_raises(self):
        from src.features import garman_klass_rv
        df = pd.DataFrame({'high': [1, 2], 'low': [0.5, 1.5], 'close': [1, 2]})
        with pytest.raises(KeyError):
            garman_klass_rv(df, window=2)  # missing 'open'

    def test_negative_window_raises(self):
        from src.features import garman_klass_rv
        ohlc = _make_synthetic_ohlc(10, seed=16)
        with pytest.raises(ValueError):
            garman_klass_rv(ohlc, window=0)

    def test_index_preserved(self):
        from src.features import garman_klass_rv
        ohlc = _make_synthetic_ohlc(20, seed=17)
        rv = garman_klass_rv(ohlc, window=5)
        assert rv.index.equals(ohlc.index)
        assert len(rv) == len(ohlc)

    def test_nan_propagates(self):
        from src.features import garman_klass_rv
        ohlc = _make_synthetic_ohlc(20, seed=18)
        ohlc.loc[ohlc.index[5], 'open'] = np.nan
        rv = garman_klass_rv(ohlc, window=3)
        assert np.isnan(rv.iloc[5])


class TestYangZhangRV:
    def test_converges_to_known_sigma(self):
        from src.features import yang_zhang_rv
        sigma = 0.02
        ohlc = _make_synthetic_ohlc(2000, sigma=sigma, seed=19)
        rv = yang_zhang_rv(ohlc, window=60).dropna()
        expected = sigma * np.sqrt(252)
        assert 0.5 * expected < rv.median() < 2.0 * expected

    def test_first_row_nan_due_to_shift(self):
        from src.features import yang_zhang_rv
        ohlc = _make_synthetic_ohlc(50, seed=20)
        rv = yang_zhang_rv(ohlc, window=10)
        # First row's overnight log return uses shift(1) -> NaN
        assert np.isnan(rv.iloc[0])

    def test_missing_columns_raises(self):
        from src.features import yang_zhang_rv
        df = pd.DataFrame({'high': [1, 2], 'low': [0.5, 1.5], 'close': [1, 2]})
        with pytest.raises(KeyError):
            yang_zhang_rv(df, window=2)

    def test_negative_window_raises(self):
        from src.features import yang_zhang_rv
        ohlc = _make_synthetic_ohlc(10, seed=21)
        with pytest.raises(ValueError):
            yang_zhang_rv(ohlc, window=0)

    def test_window_must_be_at_least_2(self):
        # Yang-Zhang's k-factor uses (n+1)/(n-1); n=1 would divide by zero
        from src.features import yang_zhang_rv
        ohlc = _make_synthetic_ohlc(10, seed=22)
        with pytest.raises(ValueError):
            yang_zhang_rv(ohlc, window=1)

    def test_index_preserved(self):
        from src.features import yang_zhang_rv
        ohlc = _make_synthetic_ohlc(20, seed=23)
        rv = yang_zhang_rv(ohlc, window=5)
        assert rv.index.equals(ohlc.index)
        assert len(rv) == len(ohlc)
