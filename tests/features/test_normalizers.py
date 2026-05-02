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


class TestRobustZscoreRolling:
    def test_steady_state_median(self):
        from src.features import robust_zscore_rolling
        rng = np.random.default_rng(3)
        s = pd.Series(rng.normal(5.0, 2.0, 1000))
        z = robust_zscore_rolling(s, window=50).dropna()
        assert abs(z.median()) < 0.2

    def test_robust_to_single_outlier(self):
        from src.features import robust_zscore_rolling
        rng = np.random.default_rng(4)
        clean = pd.Series(rng.normal(0.0, 1.0, 200))
        z_clean = robust_zscore_rolling(clean, window=20)

        # Inject a 100-sigma outlier far from any test index we check
        contaminated = clean.copy()
        contaminated.iloc[100] = 100.0
        z_contam = robust_zscore_rolling(contaminated, window=20)

        # Indices BEFORE the outlier window should be unchanged
        np.testing.assert_allclose(
            z_clean.iloc[80:99].dropna().to_numpy(),
            z_contam.iloc[80:99].dropna().to_numpy(),
            atol=1e-10,
        )

    def test_zero_mad_window_produces_nan(self):
        from src.features import robust_zscore_rolling
        s = pd.Series([5.0] * 10)
        z = robust_zscore_rolling(s, window=5)
        assert z.iloc[4:].isna().all()

    def test_insufficient_data_leading_nan(self):
        from src.features import robust_zscore_rolling
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = robust_zscore_rolling(s, window=3)
        assert np.isnan(z.iloc[0])
        assert np.isnan(z.iloc[1])

    def test_negative_window_raises(self):
        from src.features import robust_zscore_rolling
        with pytest.raises(ValueError):
            robust_zscore_rolling(pd.Series([1.0, 2.0]), window=0)

    def test_index_preserved(self):
        from src.features import robust_zscore_rolling
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(np.arange(10, dtype=float), index=idx)
        out = robust_zscore_rolling(s, window=3)
        assert out.index.equals(idx)
        assert len(out) == len(s)

    def test_empty_series(self):
        from src.features import robust_zscore_rolling
        out = robust_zscore_rolling(pd.Series([], dtype=float), window=5)
        assert len(out) == 0

    def test_all_nan_window(self):
        """Coverage: _mad_zscore_window with all NaN values (line 48)."""
        from src.features import robust_zscore_rolling
        s = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, 3.0, 4.0])
        z = robust_zscore_rolling(s, window=3)
        # Window containing all NaN values should produce NaN
        assert np.isnan(z.iloc[4])

    def test_window_last_value_nan(self):
        """Coverage: _mad_zscore_window with NaN as last value (line 51)."""
        from src.features import robust_zscore_rolling
        s = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
        z = robust_zscore_rolling(s, window=3)
        # Last value NaN produces NaN output
        assert np.isnan(z.iloc[4])

    def test_mad_zscore_window_direct_all_nan(self):
        """Direct test of _mad_zscore_window with all NaN (line 48)."""
        from src.features.normalizers import _mad_zscore_window
        window = np.array([np.nan, np.nan, np.nan])
        result = _mad_zscore_window(window)
        assert np.isnan(result)

    def test_mad_zscore_window_direct_last_nan(self):
        """Direct test of _mad_zscore_window with NaN last value (line 51)."""
        from src.features.normalizers import _mad_zscore_window
        window = np.array([1.0, 2.0, np.nan])
        result = _mad_zscore_window(window)
        assert np.isnan(result)


class TestRobustZscoreCrossSectional:
    def test_median_zero_property(self):
        from src.features import robust_zscore_cross_sectional
        rng = np.random.default_rng(5)
        s = pd.Series(rng.normal(10.0, 3.0, 100), index=[f"S{i}" for i in range(100)])
        z = robust_zscore_cross_sectional(s)
        assert abs(z.median()) < 1e-10

    def test_zero_mad_produces_all_nan(self):
        from src.features import robust_zscore_cross_sectional
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        z = robust_zscore_cross_sectional(s)
        assert z.isna().all()

    def test_nan_propagates(self):
        from src.features import robust_zscore_cross_sectional
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        z = robust_zscore_cross_sectional(s)
        assert np.isnan(z.iloc[2])
        # Other entries should be finite
        assert z.iloc[0:2].notna().all()
        assert z.iloc[3:].notna().all()

    def test_empty_series(self):
        from src.features import robust_zscore_cross_sectional
        out = robust_zscore_cross_sectional(pd.Series([], dtype=float))
        assert len(out) == 0

    def test_all_nan_input(self):
        from src.features import robust_zscore_cross_sectional
        s = pd.Series([np.nan, np.nan, np.nan])
        z = robust_zscore_cross_sectional(s)
        assert z.isna().all()
        assert len(z) == 3

    def test_index_preserved(self):
        from src.features import robust_zscore_cross_sectional
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                      index=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA'])
        z = robust_zscore_cross_sectional(s)
        assert list(z.index) == list(s.index)
        assert len(z) == len(s)


class TestWinsorize:
    def test_clips_correct_fraction_uniform(self):
        from src.features import winsorize
        rng = np.random.default_rng(6)
        s = pd.Series(rng.uniform(0.0, 100.0, 10000))
        out = winsorize(s, lower_q=0.01, upper_q=0.99)
        # Total fraction at the boundary values should be ~2%
        lower_bound = s.quantile(0.01)
        upper_bound = s.quantile(0.99)
        clipped = ((out == lower_bound) | (out == upper_bound)).sum()
        assert 150 < clipped < 250  # ~2% of 10000 with a wide tolerance

    def test_lower_q_ge_upper_q_raises(self):
        from src.features import winsorize
        with pytest.raises(ValueError):
            winsorize(pd.Series([1.0, 2.0, 3.0]), lower_q=0.5, upper_q=0.5)
        with pytest.raises(ValueError):
            winsorize(pd.Series([1.0, 2.0, 3.0]), lower_q=0.6, upper_q=0.4)

    def test_quantiles_outside_unit_raises(self):
        from src.features import winsorize
        with pytest.raises(ValueError):
            winsorize(pd.Series([1.0, 2.0]), lower_q=-0.1, upper_q=0.99)
        with pytest.raises(ValueError):
            winsorize(pd.Series([1.0, 2.0]), lower_q=0.01, upper_q=1.1)

    def test_nan_preserved(self):
        from src.features import winsorize
        s = pd.Series([1.0, np.nan, 5.0, 100.0])
        out = winsorize(s, lower_q=0.0, upper_q=0.99)
        assert np.isnan(out.iloc[1])

    def test_empty_series(self):
        from src.features import winsorize
        out = winsorize(pd.Series([], dtype=float))
        assert len(out) == 0

    def test_index_preserved(self):
        from src.features import winsorize
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series([1.0, 50.0, 100.0, 50.0, 200.0], index=idx)
        out = winsorize(s)
        assert out.index.equals(idx)
        assert len(out) == len(s)


class TestRankTransform:
    def test_uniform_distribution(self):
        from src.features import rank_transform
        from scipy import stats
        rng = np.random.default_rng(7)
        s = pd.Series(rng.normal(0.0, 1.0, 1000))
        ranked = rank_transform(s)
        # KS test against uniform [0,1]
        ks_stat, p_value = stats.kstest(ranked.dropna(), 'uniform')
        assert p_value > 0.01  # cannot reject uniformity

    def test_output_in_unit_interval(self):
        from src.features import rank_transform
        s = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0])
        ranked = rank_transform(s)
        assert ranked.min() > 0.0
        assert ranked.max() <= 1.0

    def test_nan_preserved(self):
        from src.features import rank_transform
        s = pd.Series([1.0, np.nan, 3.0, 2.0])
        ranked = rank_transform(s)
        assert np.isnan(ranked.iloc[1])

    def test_method_parameter(self):
        from src.features import rank_transform
        s = pd.Series([1.0, 2.0, 2.0, 3.0])
        out_avg = rank_transform(s, method='average')
        out_min = rank_transform(s, method='min')
        # 'min' assigns the same rank to ties; 'average' assigns the average
        assert out_avg.iloc[1] != out_min.iloc[1] or out_avg.iloc[2] != out_min.iloc[2]

    def test_empty_series(self):
        from src.features import rank_transform
        out = rank_transform(pd.Series([], dtype=float))
        assert len(out) == 0

    def test_index_preserved(self):
        from src.features import rank_transform
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0], index=idx)
        out = rank_transform(s)
        assert out.index.equals(idx)
        assert len(out) == len(s)
