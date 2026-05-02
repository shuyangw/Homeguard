"""Tests for src.features.normalizers."""
import numpy as np
import pandas as pd
import pytest


def test_module_importable():
    from src.features import normalizers  # noqa: F401


class TestLogTransform:
    def test_inverse_property(self):
        from src.features import log_transform
        rng = np.random.default_rng(0)
        s = pd.Series(rng.uniform(0.1, 100.0, 50))
        recovered = np.exp(log_transform(s))
        np.testing.assert_allclose(recovered.to_numpy(), s.to_numpy(), atol=1e-10)

    def test_non_positive_produces_nan(self):
        from src.features import log_transform
        s = pd.Series([1.0, 0.0, -1.0, 2.0])
        out = log_transform(s)
        assert np.isnan(out.iloc[1])
        assert np.isnan(out.iloc[2])
        assert np.isclose(out.iloc[0], 0.0)
        assert np.isclose(out.iloc[3], np.log(2.0))

    def test_nan_propagates(self):
        from src.features import log_transform
        s = pd.Series([1.0, np.nan, 2.0])
        out = log_transform(s)
        assert np.isnan(out.iloc[1])
        assert not np.isnan(out.iloc[0])
        assert not np.isnan(out.iloc[2])

    def test_empty_series(self):
        from src.features import log_transform
        out = log_transform(pd.Series([], dtype=float))
        assert len(out) == 0
        assert isinstance(out, pd.Series)

    def test_index_preserved(self):
        from src.features import log_transform
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        out = log_transform(s)
        assert out.index.equals(idx)
        assert len(out) == len(s)


class TestLogReturns:
    def test_time_additive(self):
        from src.features import log_returns
        rng = np.random.default_rng(1)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))))
        rets = log_returns(prices)
        # Sum of single-period log-returns equals log(p_end / p_start)
        total = rets.dropna().sum()
        expected = np.log(prices.iloc[-1] / prices.iloc[0])
        assert np.isclose(total, expected, atol=1e-10)

    def test_first_periods_are_nan(self):
        from src.features import log_returns
        prices = pd.Series([100.0, 110.0, 121.0, 133.1])
        rets = log_returns(prices, periods=1)
        assert np.isnan(rets.iloc[0])
        assert not np.isnan(rets.iloc[1])

    def test_periods_parameter(self):
        from src.features import log_returns
        prices = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
        rets = log_returns(prices, periods=2)
        assert np.isnan(rets.iloc[0])
        assert np.isnan(rets.iloc[1])
        assert np.isclose(rets.iloc[2], np.log(121.0 / 100.0))

    def test_negative_periods_raises(self):
        from src.features import log_returns
        with pytest.raises(ValueError):
            log_returns(pd.Series([1.0, 2.0]), periods=0)
        with pytest.raises(ValueError):
            log_returns(pd.Series([1.0, 2.0]), periods=-1)

    def test_nan_propagates(self):
        from src.features import log_returns
        prices = pd.Series([100.0, np.nan, 121.0])
        rets = log_returns(prices)
        assert np.isnan(rets.iloc[1])
        assert np.isnan(rets.iloc[2])  # depends on shifted NaN

    def test_empty_series(self):
        from src.features import log_returns
        out = log_returns(pd.Series([], dtype=float))
        assert len(out) == 0

    def test_index_preserved(self):
        from src.features import log_returns
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41], index=idx)
        out = log_returns(s)
        assert out.index.equals(idx)
        assert len(out) == len(s)


class TestZscoreRolling:
    def test_steady_state_distribution(self):
        from src.features import zscore_rolling
        rng = np.random.default_rng(2)
        s = pd.Series(rng.normal(5.0, 2.0, 1000))
        z = zscore_rolling(s, window=50).dropna()
        # Steady-state mean and std should be close to 0 and 1
        assert abs(z.mean()) < 0.1
        assert abs(z.std() - 1.0) < 0.1

    def test_zero_variance_window_produces_nan(self):
        from src.features import zscore_rolling
        # Constant input over the window -> std=0 -> NaN
        s = pd.Series([5.0] * 10)
        z = zscore_rolling(s, window=5)
        # Indices 4-9 have a full window of constants -> NaN
        assert z.iloc[4:].isna().all()

    def test_no_epsilon_leak(self):
        # Verifies the +1e-10 hack from old ml_crypto_mr is not present.
        from src.features import zscore_rolling
        s = pd.Series([5.0] * 10)
        z = zscore_rolling(s, window=5)
        finite = z.dropna()
        assert len(finite) == 0  # no near-zero leak; all NaN

    def test_insufficient_data_leading_nan(self):
        from src.features import zscore_rolling
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore_rolling(s, window=3)
        assert np.isnan(z.iloc[0])
        assert np.isnan(z.iloc[1])
        assert not np.isnan(z.iloc[2])

    def test_nan_propagates(self):
        from src.features import zscore_rolling
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        z = zscore_rolling(s, window=3)
        # Window containing NaN should produce NaN output
        assert np.isnan(z.iloc[2])
        assert np.isnan(z.iloc[3])

    def test_negative_window_raises(self):
        from src.features import zscore_rolling
        with pytest.raises(ValueError):
            zscore_rolling(pd.Series([1.0, 2.0]), window=0)
        with pytest.raises(ValueError):
            zscore_rolling(pd.Series([1.0, 2.0]), window=-1)

    def test_empty_series(self):
        from src.features import zscore_rolling
        out = zscore_rolling(pd.Series([], dtype=float), window=5)
        assert len(out) == 0

    def test_index_preserved(self):
        from src.features import zscore_rolling
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(np.arange(10, dtype=float), index=idx)
        out = zscore_rolling(s, window=3)
        assert out.index.equals(idx)
        assert len(out) == len(s)
