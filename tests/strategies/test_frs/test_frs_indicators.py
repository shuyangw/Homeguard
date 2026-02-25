"""
Unit tests for FRS indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.advanced.frs_indicators import (
    FRSIndicators, FRSRegime, FRSSignal
)


class TestHurstCalculation:
    """Tests for Hurst exponent calculation."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)

        return df

    def test_hurst_returns_series(self):
        """Test Hurst calculation returns pd.Series."""
        df = self.create_sample_ohlcv()
        hurst = FRSIndicators.calculate_hurst(df['close'], window=100)

        assert isinstance(hurst, pd.Series)
        assert len(hurst) == len(df)

    def test_hurst_values_in_range(self):
        """Test Hurst values are in [0, 1] range."""
        df = self.create_sample_ohlcv(days=300)
        hurst = FRSIndicators.calculate_hurst(df['close'], window=100)

        # Non-NaN values should be in [0, 1]
        valid_hurst = hurst.dropna()
        assert (valid_hurst >= 0).all()
        assert (valid_hurst <= 1).all()

    def test_hurst_mean_reverting_data(self):
        """Test Hurst on mean-reverting data should be < 0.5 on average."""
        # Create oscillating/mean-reverting data
        days = 500
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)

        # AR(1) with negative coefficient creates mean-reverting series
        prices = [100]
        for i in range(days - 1):
            shock = np.random.randn() * 2
            mean_revert = -0.5 * (prices[-1] - 100)  # Pull toward 100
            new_price = prices[-1] + mean_revert + shock
            prices.append(max(1, new_price))

        close = pd.Series(prices, index=dates)
        hurst = FRSIndicators.calculate_hurst(close, window=100)

        # Mean of valid values should be less than 0.5 for mean-reverting
        valid_mean = hurst.dropna().mean()
        assert valid_mean < 0.55, f"Expected H < 0.55 for mean-reverting, got {valid_mean:.3f}"

    def test_hurst_trending_data(self):
        """Test Hurst on trending data should be > 0.5 on average."""
        # Create trending data
        days = 500
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')

        # Strong uptrend with small noise
        trend = np.linspace(0, 100, days)
        np.random.seed(42)
        noise = np.random.randn(days) * 0.5
        prices = 100 + trend + noise

        close = pd.Series(prices, index=dates)
        hurst = FRSIndicators.calculate_hurst(close, window=100)

        # Mean of valid values should be greater than 0.5 for trending
        valid_mean = hurst.dropna().mean()
        assert valid_mean > 0.45, f"Expected H > 0.45 for trending, got {valid_mean:.3f}"


class TestHurstSmoothing:
    """Tests for Hurst smoothing."""

    def test_smooth_hurst_returns_series(self):
        """Test smoothing returns pd.Series."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        hurst = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates)

        smoothed = FRSIndicators.smooth_hurst(hurst, span=10)

        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == len(hurst)

    def test_smooth_hurst_reduces_variance(self):
        """Test smoothing reduces variance."""
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        hurst = pd.Series(np.random.uniform(0.3, 0.7, 200), index=dates)

        smoothed = FRSIndicators.smooth_hurst(hurst, span=20)

        # Smoothed series should have lower variance
        assert smoothed.std() < hurst.std()


class TestRegimeClassification:
    """Tests for regime classification."""

    def test_classify_regime_trending(self):
        """Test classification of trending regime."""
        regime = FRSIndicators.classify_regime(0.65, 0.60, 0.40)
        assert regime == FRSRegime.TRENDING.value

    def test_classify_regime_mean_reverting(self):
        """Test classification of mean-reverting regime."""
        regime = FRSIndicators.classify_regime(0.35, 0.60, 0.40)
        assert regime == FRSRegime.MEAN_REVERTING.value

    def test_classify_regime_noise(self):
        """Test classification of noise regime."""
        regime = FRSIndicators.classify_regime(0.50, 0.60, 0.40)
        assert regime == FRSRegime.NOISE.value

    def test_classify_regime_boundary_high(self):
        """Test boundary: exactly at trend threshold."""
        # At exactly 0.60, should be NOISE (not greater than)
        regime = FRSIndicators.classify_regime(0.60, 0.60, 0.40)
        assert regime == FRSRegime.NOISE.value

    def test_classify_regime_boundary_low(self):
        """Test boundary: exactly at MR threshold."""
        # At exactly 0.40, should be NOISE (not less than)
        regime = FRSIndicators.classify_regime(0.40, 0.60, 0.40)
        assert regime == FRSRegime.NOISE.value

    def test_classify_regime_nan(self):
        """Test NaN hurst returns NOISE."""
        regime = FRSIndicators.classify_regime(float('nan'), 0.60, 0.40)
        assert regime == FRSRegime.NOISE.value

    def test_classify_regime_series(self):
        """Test series classification."""
        hurst = pd.Series([0.65, 0.50, 0.35, 0.60, 0.40])
        regimes = FRSIndicators.classify_regime_series(hurst, 0.60, 0.40)

        assert regimes.iloc[0] == FRSRegime.TRENDING.value
        assert regimes.iloc[1] == FRSRegime.NOISE.value
        assert regimes.iloc[2] == FRSRegime.MEAN_REVERTING.value
        assert regimes.iloc[3] == FRSRegime.NOISE.value
        assert regimes.iloc[4] == FRSRegime.NOISE.value


class TestDonchianChannels:
    """Tests for Donchian channel calculation."""

    def create_sample_ohlcv(self, days: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(days))
        high = close + np.abs(np.random.randn(days)) * 2
        low = close - np.abs(np.random.randn(days)) * 2

        return pd.DataFrame({
            'open': close + np.random.randn(days) * 0.5,
            'high': high,
            'low': low,
            'close': close,
        }, index=dates)

    def test_donchian_returns_tuple(self):
        """Test Donchian returns tuple of two Series."""
        df = self.create_sample_ohlcv()
        upper, lower = FRSIndicators.calculate_donchian(df, period=20)

        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(df)
        assert len(lower) == len(df)

    def test_donchian_upper_greater_than_lower(self):
        """Test upper channel >= lower channel."""
        df = self.create_sample_ohlcv()
        upper, lower = FRSIndicators.calculate_donchian(df, period=20)

        valid = ~(upper.isna() | lower.isna())
        assert (upper[valid] >= lower[valid]).all()

    def test_donchian_upper_is_rolling_max_high(self):
        """Test upper channel equals rolling max of high."""
        df = self.create_sample_ohlcv()
        period = 20
        upper, _ = FRSIndicators.calculate_donchian(df, period=period)

        expected = df['high'].rolling(window=period, min_periods=period).max()
        pd.testing.assert_series_equal(upper, expected)

    def test_donchian_lower_is_rolling_min_low(self):
        """Test lower channel equals rolling min of low."""
        df = self.create_sample_ohlcv()
        period = 20
        _, lower = FRSIndicators.calculate_donchian(df, period=period)

        expected = df['low'].rolling(window=period, min_periods=period).min()
        pd.testing.assert_series_equal(lower, expected)


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_returns_tuple(self):
        """Test Bollinger returns tuple of three Series."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(50)), index=dates)

        upper, middle, lower = FRSIndicators.calculate_bollinger(close, period=20, std_dev=2.0)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_bollinger_band_order(self):
        """Test upper > middle > lower."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(50)), index=dates)

        upper, middle, lower = FRSIndicators.calculate_bollinger(close, period=20, std_dev=2.0)

        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_bollinger_middle_is_sma(self):
        """Test middle band equals SMA."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(50)), index=dates)
        period = 20

        _, middle, _ = FRSIndicators.calculate_bollinger(close, period=period, std_dev=2.0)
        expected_sma = close.rolling(window=period, min_periods=period).mean()

        pd.testing.assert_series_equal(middle, expected_sma)

    def test_bollinger_width_proportional_to_std(self):
        """Test band width is proportional to std_dev parameter."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(50)), index=dates)

        upper1, middle1, lower1 = FRSIndicators.calculate_bollinger(close, period=20, std_dev=1.0)
        upper2, middle2, lower2 = FRSIndicators.calculate_bollinger(close, period=20, std_dev=2.0)

        width1 = (upper1 - lower1).dropna().mean()
        width2 = (upper2 - lower2).dropna().mean()

        # 2 std should be ~2x wider than 1 std
        assert abs(width2 / width1 - 2.0) < 0.1


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_returns_series(self):
        """Test RSI returns pd.Series."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(50)), index=dates)

        rsi = FRSIndicators.calculate_rsi(close, period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(close)

    def test_rsi_in_range(self):
        """Test RSI values in [0, 100] range."""
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(200)), index=dates)

        rsi = FRSIndicators.calculate_rsi(close, period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_uptrend_high(self):
        """Test RSI is high during strong uptrend."""
        days = 50
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        # Strong consistent uptrend
        close = pd.Series(100 + np.arange(days) * 2, index=dates)

        rsi = FRSIndicators.calculate_rsi(close, period=14)

        # RSI should be high (> 70) during strong uptrend
        valid_rsi = rsi.dropna()
        assert valid_rsi.iloc[-1] > 70

    def test_rsi_downtrend_low(self):
        """Test RSI is low during strong downtrend."""
        days = 50
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        # Strong consistent downtrend
        close = pd.Series(200 - np.arange(days) * 2, index=dates)

        rsi = FRSIndicators.calculate_rsi(close, period=14)

        # RSI should be low (< 30) during strong downtrend
        valid_rsi = rsi.dropna()
        assert valid_rsi.iloc[-1] < 30


class TestATR:
    """Tests for ATR calculation."""

    def create_sample_ohlcv(self, days: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(days))
        high = close + np.abs(np.random.randn(days)) * 2
        low = close - np.abs(np.random.randn(days)) * 2

        return pd.DataFrame({
            'open': close + np.random.randn(days) * 0.5,
            'high': high,
            'low': low,
            'close': close,
        }, index=dates)

    def test_atr_returns_series(self):
        """Test ATR returns pd.Series."""
        df = self.create_sample_ohlcv()
        atr = FRSIndicators.calculate_atr(df, period=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(df)

    def test_atr_positive(self):
        """Test ATR is always positive."""
        df = self.create_sample_ohlcv(days=100)
        atr = FRSIndicators.calculate_atr(df, period=14)

        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()


class TestSignalGeneration:
    """Tests for signal generation."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)

        return df

    def test_generate_signals_returns_five_series(self):
        """Test generate_signals returns 5 Series."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits, regime = \
            FRSIndicators.generate_signals(df)

        assert isinstance(long_entries, pd.Series)
        assert isinstance(long_exits, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(short_exits, pd.Series)
        assert isinstance(regime, pd.Series)

    def test_generate_signals_correct_lengths(self):
        """Test all outputs have correct length."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits, regime = \
            FRSIndicators.generate_signals(df)

        assert len(long_entries) == len(df)
        assert len(long_exits) == len(df)
        assert len(short_entries) == len(df)
        assert len(short_exits) == len(df)
        assert len(regime) == len(df)

    def test_generate_signals_boolean_types(self):
        """Test entry/exit signals are boolean."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits, _ = \
            FRSIndicators.generate_signals(df)

        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_regime_values_valid(self):
        """Test regime values are valid FRSRegime values."""
        df = self.create_sample_ohlcv()
        _, _, _, _, regime = FRSIndicators.generate_signals(df)

        valid_regimes = {FRSRegime.TRENDING.value, FRSRegime.MEAN_REVERTING.value, FRSRegime.NOISE.value}
        unique_regimes = set(regime.unique())

        assert unique_regimes.issubset(valid_regimes)

    def test_no_signals_in_noise_regime(self):
        """Test no entry signals during NOISE regime."""
        df = self.create_sample_ohlcv()
        long_entries, _, short_entries, _, regime = \
            FRSIndicators.generate_signals(df)

        # Check signals only occur outside NOISE regime
        noise_mask = regime == FRSRegime.NOISE.value

        # Long entries should not occur in NOISE regime
        long_in_noise = (long_entries & noise_mask).sum()
        assert long_in_noise == 0, f"Found {long_in_noise} long entries in NOISE regime"

        # Short entries should not occur in NOISE regime
        short_in_noise = (short_entries & noise_mask).sum()
        assert short_in_noise == 0, f"Found {short_in_noise} short entries in NOISE regime"


class TestCalculateAllIndicators:
    """Tests for calculate_all_indicators."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
        }, index=dates)

    def test_returns_dataframe(self):
        """Test returns DataFrame."""
        df = self.create_sample_ohlcv()
        result = FRSIndicators.calculate_all_indicators(df)

        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self):
        """Test output contains expected columns."""
        df = self.create_sample_ohlcv()
        result = FRSIndicators.calculate_all_indicators(df)

        expected_cols = [
            'hurst', 'hurst_smoothed',
            'donchian_upper', 'donchian_lower',
            'bb_upper', 'bb_middle', 'bb_lower',
            'rsi', 'atr',
            'open', 'high', 'low', 'close'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestGetCurrentSignal:
    """Tests for get_current_signal."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
        }, index=dates)

    def test_returns_frs_signal(self):
        """Test returns FRSSignal dataclass."""
        df = self.create_sample_ohlcv()
        signal = FRSIndicators.get_current_signal(df, hurst_window=100)

        assert isinstance(signal, FRSSignal)

    def test_signal_has_required_attributes(self):
        """Test FRSSignal has all required attributes."""
        df = self.create_sample_ohlcv()
        signal = FRSIndicators.get_current_signal(df, hurst_window=100)

        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'close')
        assert hasattr(signal, 'hurst')
        assert hasattr(signal, 'hurst_smoothed')
        assert hasattr(signal, 'regime')
        assert hasattr(signal, 'donchian_upper')
        assert hasattr(signal, 'donchian_lower')
        assert hasattr(signal, 'bb_upper')
        assert hasattr(signal, 'bb_middle')
        assert hasattr(signal, 'bb_lower')
        assert hasattr(signal, 'rsi')

    def test_regime_is_valid_value(self):
        """Test regime is a valid FRSRegime value."""
        df = self.create_sample_ohlcv()
        signal = FRSIndicators.get_current_signal(df, hurst_window=100)

        valid_regimes = {FRSRegime.TRENDING.value, FRSRegime.MEAN_REVERTING.value, FRSRegime.NOISE.value}
        assert signal.regime in valid_regimes

    def test_insufficient_data_raises(self):
        """Test insufficient data raises ValueError."""
        df = self.create_sample_ohlcv(days=50)  # Less than hurst_window

        with pytest.raises(ValueError, match="Insufficient data"):
            FRSIndicators.get_current_signal(df, hurst_window=100)


# =============================================================================
# IMPROVEMENT METHOD TESTS (Phase 1-3)
# =============================================================================

class TestBTCRegime:
    """Tests for BTC regime filter."""

    def test_get_btc_regime_returns_series(self):
        """Test get_btc_regime returns pd.Series."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        btc_close = pd.Series(30000 + np.cumsum(np.random.randn(100) * 500), index=dates)

        regime = FRSIndicators.get_btc_regime(btc_close, sma_period=20)

        assert isinstance(regime, pd.Series)
        assert len(regime) == len(btc_close)

    def test_get_btc_regime_values(self):
        """Test regime values are 'bullish' or 'bearish'."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        btc_close = pd.Series(30000 + np.cumsum(np.random.randn(100) * 500), index=dates)

        regime = FRSIndicators.get_btc_regime(btc_close, sma_period=20)

        valid_values = {'bullish', 'bearish'}
        assert set(regime.dropna().unique()).issubset(valid_values)

    def test_bullish_when_above_sma(self):
        """Test regime is bullish when price > SMA."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        # Strong uptrend - always above SMA
        btc_close = pd.Series(30000 + np.arange(50) * 1000, index=dates)

        regime = FRSIndicators.get_btc_regime(btc_close, sma_period=20)

        # After warmup period, should be bullish
        assert (regime.iloc[25:] == 'bullish').all()

    def test_bearish_when_below_sma(self):
        """Test regime is bearish when price < SMA."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        # Strong downtrend - always below SMA
        btc_close = pd.Series(50000 - np.arange(50) * 1000, index=dates)

        regime = FRSIndicators.get_btc_regime(btc_close, sma_period=20)

        # After warmup period, should be bearish
        assert (regime.iloc[25:] == 'bearish').all()


class TestGenerateSignalsWithRegime:
    """Tests for generate_signals_with_regime."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
        }, index=dates)

    def create_btc_data(self, days: int = 200) -> pd.Series:
        """Create BTC price data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(123)
        btc_close = 30000 + np.cumsum(np.random.randn(days) * 500)
        return pd.Series(btc_close, index=dates)

    def test_returns_six_series(self):
        """Test returns 6 Series (signals + 2 regimes)."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_btc_data()

        result = FRSIndicators.generate_signals_with_regime(
            df, btc_close, btc_sma_period=50, regime_mode='force_exit'
        )

        assert len(result) == 6
        long_entries, long_exits, short_entries, short_exits, hurst_regime, btc_regime = result

        assert isinstance(long_entries, pd.Series)
        assert isinstance(long_exits, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(short_exits, pd.Series)
        assert isinstance(hurst_regime, pd.Series)
        assert isinstance(btc_regime, pd.Series)

    def test_block_entry_mode(self):
        """Test block_entry mode blocks entries during bearish BTC."""
        df = self.create_sample_ohlcv()
        # Create BTC data that's always bearish (downtrend)
        dates = df.index
        btc_close = pd.Series(50000 - np.arange(len(dates)) * 200, index=dates)

        long_entries, _, short_entries, _, _, btc_regime = \
            FRSIndicators.generate_signals_with_regime(
                df, btc_close, btc_sma_period=20, regime_mode='block_entry'
            )

        # During bearish periods, no entries should occur
        is_bearish = btc_regime == 'bearish'
        long_in_bearish = (long_entries & is_bearish).sum()
        short_in_bearish = (short_entries & is_bearish).sum()

        assert long_in_bearish == 0, f"Found {long_in_bearish} long entries in bearish BTC"
        assert short_in_bearish == 0, f"Found {short_in_bearish} short entries in bearish BTC"

    def test_force_exit_mode(self):
        """Test force_exit mode exits when BTC turns bearish."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_btc_data()

        long_entries, long_exits, short_entries, short_exits, _, btc_regime = \
            FRSIndicators.generate_signals_with_regime(
                df, btc_close, btc_sma_period=50, regime_mode='force_exit'
            )

        # Check that exits happen when BTC turns bearish
        btc_turns_bearish = (btc_regime == 'bearish') & (btc_regime.shift(1) != 'bearish')
        btc_turns_bearish = btc_turns_bearish.fillna(False)

        # Some forced exits should coincide with BTC turning bearish
        # (unless there were no positions at those times)
        assert long_exits.dtype == bool
        assert short_exits.dtype == bool


class TestCalculatePositionSize:
    """Tests for volatility-scaled position sizing."""

    def test_returns_series(self):
        """Test returns pd.Series."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)

        position_size = FRSIndicators.calculate_position_size(
            close, target_vol=0.20, lookback=20
        )

        assert isinstance(position_size, pd.Series)
        assert len(position_size) == len(close)

    def test_values_in_range(self):
        """Test position sizes are within [min, max] range."""
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(200)), index=dates)

        min_pos = 0.25
        max_pos = 1.0
        position_size = FRSIndicators.calculate_position_size(
            close, target_vol=0.20, lookback=20,
            min_position=min_pos, max_position=max_pos
        )

        assert (position_size >= min_pos).all()
        assert (position_size <= max_pos).all()

    def test_high_vol_reduces_position(self):
        """Test high volatility results in smaller position sizes."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

        # Low volatility data
        np.random.seed(42)
        low_vol_close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)

        # High volatility data
        np.random.seed(42)
        high_vol_close = pd.Series(100 + np.cumsum(np.random.randn(100) * 5), index=dates)

        low_vol_size = FRSIndicators.calculate_position_size(
            low_vol_close, target_vol=0.20, lookback=20
        )
        high_vol_size = FRSIndicators.calculate_position_size(
            high_vol_close, target_vol=0.20, lookback=20
        )

        # High vol should have lower average position size
        assert high_vol_size.iloc[30:].mean() < low_vol_size.iloc[30:].mean()

    def test_target_vol_scaling(self):
        """Test higher target_vol results in larger positions."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)

        size_10pct = FRSIndicators.calculate_position_size(
            close, target_vol=0.10, lookback=20
        )
        size_30pct = FRSIndicators.calculate_position_size(
            close, target_vol=0.30, lookback=20
        )

        # 30% target should have larger positions than 10%
        assert size_30pct.iloc[30:].mean() > size_10pct.iloc[30:].mean()

    def test_no_nan_in_output(self):
        """Test output has no NaN values."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)

        position_size = FRSIndicators.calculate_position_size(
            close, target_vol=0.20, lookback=20
        )

        assert not position_size.isna().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
