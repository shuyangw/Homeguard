"""
Unit tests for EVR indicators.
"""

import pytest
import pandas as pd
import numpy as np

from src.strategies.advanced.evr_indicators import EVRIndicators, EVRSignal


class TestRelativeVolume:
    """Tests for relative volume (effort) calculation."""

    def create_sample_volume(self, days: int = 50) -> pd.Series:
        """Create sample volume data."""
        np.random.seed(42)
        base_volume = 1000000
        volume = base_volume + np.random.randn(days) * 100000
        volume = np.maximum(volume, 100000)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        return pd.Series(volume, index=dates)

    def test_returns_series(self):
        """Test calculate_relative_volume returns a Series."""
        volume = self.create_sample_volume()
        result = EVRIndicators.calculate_relative_volume(volume, period=20)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        """Test output length matches input."""
        volume = self.create_sample_volume()
        result = EVRIndicators.calculate_relative_volume(volume, period=20)
        assert len(result) == len(volume)

    def test_values_around_one(self):
        """Test relative volume centers around 1.0."""
        volume = self.create_sample_volume(100)
        result = EVRIndicators.calculate_relative_volume(volume, period=20)
        valid = result.dropna()
        assert 0.5 < valid.mean() < 1.5

    def test_high_volume_spike(self):
        """Test high volume produces relative volume > 1."""
        volume = self.create_sample_volume(50)
        volume.iloc[-1] = volume.iloc[-1] * 3  # 3x spike
        result = EVRIndicators.calculate_relative_volume(volume, period=20)
        assert result.iloc[-1] > 2.0

    def test_low_volume(self):
        """Test low volume produces relative volume < 1."""
        volume = self.create_sample_volume(50)
        volume.iloc[-1] = volume.iloc[-1] * 0.3  # 30% of normal
        result = EVRIndicators.calculate_relative_volume(volume, period=20)
        assert result.iloc[-1] < 0.5

    def test_nan_handling(self):
        """Test NaN values in warmup period."""
        volume = self.create_sample_volume(30)
        result = EVRIndicators.calculate_relative_volume(volume, period=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()


class TestRelativeSpread:
    """Tests for relative spread (result) calculation."""

    def create_sample_ohlc(self, days: int = 50) -> pd.DataFrame:
        """Create sample OHLC data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        close = 100 + np.cumsum(np.random.randn(days) * 2)
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        return pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        }, index=dates)

    def test_returns_series(self):
        """Test calculate_relative_spread returns a Series."""
        df = self.create_sample_ohlc()
        result = EVRIndicators.calculate_relative_spread(df['high'], df['low'], period=20)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        """Test output length matches input."""
        df = self.create_sample_ohlc()
        result = EVRIndicators.calculate_relative_spread(df['high'], df['low'], period=20)
        assert len(result) == len(df)

    def test_values_around_one(self):
        """Test relative spread centers around 1.0."""
        df = self.create_sample_ohlc(100)
        result = EVRIndicators.calculate_relative_spread(df['high'], df['low'], period=20)
        valid = result.dropna()
        assert 0.5 < valid.mean() < 1.5

    def test_narrow_spread(self):
        """Test narrow spread produces relative spread < 1."""
        df = self.create_sample_ohlc(50)
        df.iloc[-1, df.columns.get_loc('high')] = df.iloc[-1]['close'] + 0.1
        df.iloc[-1, df.columns.get_loc('low')] = df.iloc[-1]['close'] - 0.1
        result = EVRIndicators.calculate_relative_spread(df['high'], df['low'], period=20)
        assert result.iloc[-1] < 0.5

    def test_wide_spread(self):
        """Test wide spread produces relative spread > 1."""
        df = self.create_sample_ohlc(50)
        avg_spread = (df['high'] - df['low']).mean()
        df.iloc[-1, df.columns.get_loc('high')] = df.iloc[-1]['close'] + avg_spread * 2
        df.iloc[-1, df.columns.get_loc('low')] = df.iloc[-1]['close'] - avg_spread * 2
        result = EVRIndicators.calculate_relative_spread(df['high'], df['low'], period=20)
        assert result.iloc[-1] > 2.0


class TestAbsorptionDetection:
    """Tests for absorption candle detection."""

    def test_detects_absorption(self):
        """Test absorption detected when high vol + low spread."""
        rel_volume = pd.Series([1.0, 1.0, 2.5, 1.0])
        rel_spread = pd.Series([1.0, 1.0, 0.5, 1.0])
        result = EVRIndicators.detect_absorption(rel_volume, rel_spread)
        assert result.iloc[2] == True
        assert result.iloc[0] == False
        assert result.iloc[3] == False

    def test_no_absorption_low_volume(self):
        """Test no absorption when volume is low."""
        rel_volume = pd.Series([0.5, 0.8, 1.2])
        rel_spread = pd.Series([0.5, 0.5, 0.5])
        result = EVRIndicators.detect_absorption(rel_volume, rel_spread)
        assert result.all() == False

    def test_no_absorption_high_spread(self):
        """Test no absorption when spread is high."""
        rel_volume = pd.Series([2.5, 3.0, 2.2])
        rel_spread = pd.Series([1.5, 1.2, 1.0])
        result = EVRIndicators.detect_absorption(rel_volume, rel_spread)
        assert result.all() == False

    def test_custom_thresholds(self):
        """Test custom thresholds work correctly."""
        rel_volume = pd.Series([1.5, 2.0, 1.8])
        rel_spread = pd.Series([0.9, 0.7, 0.85])

        # Default thresholds (2.0, 0.8) - no absorption
        result1 = EVRIndicators.detect_absorption(rel_volume, rel_spread)
        assert result1.iloc[1] == False

        # Lower thresholds (1.5, 0.9) - absorption detected
        result2 = EVRIndicators.detect_absorption(
            rel_volume, rel_spread, vol_threshold=1.5, spread_threshold=0.9
        )
        assert result2.iloc[1] == True

    def test_nan_handling(self):
        """Test NaN values are handled as False."""
        rel_volume = pd.Series([np.nan, 2.5, 2.5])
        rel_spread = pd.Series([0.5, np.nan, 0.5])
        result = EVRIndicators.detect_absorption(rel_volume, rel_spread)
        assert result.iloc[0] == False
        assert result.iloc[1] == False
        assert result.iloc[2] == True


class TestRSI:
    """Tests for RSI calculation."""

    def create_sample_close(self, days: int = 50) -> pd.Series:
        """Create sample close prices."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        return pd.Series(close, index=dates)

    def test_returns_series(self):
        """Test calculate_rsi returns a Series."""
        close = self.create_sample_close()
        result = EVRIndicators.calculate_rsi(close, period=14)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        """Test output length matches input."""
        close = self.create_sample_close()
        result = EVRIndicators.calculate_rsi(close, period=14)
        assert len(result) == len(close)

    def test_values_in_range(self):
        """Test RSI values are in 0-100 range."""
        close = self.create_sample_close(100)
        result = EVRIndicators.calculate_rsi(close, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_overbought_on_uptrend(self):
        """Test RSI > 70 on strong uptrend."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        close = pd.Series(range(100, 150), index=dates, dtype=float)
        result = EVRIndicators.calculate_rsi(close, period=14)
        assert result.iloc[-1] > 70

    def test_oversold_on_downtrend(self):
        """Test RSI < 30 on strong downtrend."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        close = pd.Series(range(150, 100, -1), index=dates, dtype=float)
        result = EVRIndicators.calculate_rsi(close, period=14)
        assert result.iloc[-1] < 30


class TestBollingerBands:
    """Tests for Bollinger Band calculation."""

    def create_sample_close(self, days: int = 50) -> pd.Series:
        """Create sample close prices."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        close = 100 + np.cumsum(np.random.randn(days) * 2)
        return pd.Series(close, index=dates)

    def test_returns_three_series(self):
        """Test calculate_bollinger returns three Series."""
        close = self.create_sample_close()
        upper, middle, lower = EVRIndicators.calculate_bollinger(close, period=20)
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_band_ordering(self):
        """Test upper > middle > lower."""
        close = self.create_sample_close()
        upper, middle, lower = EVRIndicators.calculate_bollinger(close, period=20)
        valid_idx = middle.dropna().index
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()

    def test_middle_equals_sma(self):
        """Test middle band equals SMA."""
        close = self.create_sample_close()
        upper, middle, lower = EVRIndicators.calculate_bollinger(close, period=20)
        sma = close.rolling(window=20, min_periods=20).mean()
        pd.testing.assert_series_equal(middle, sma)


class TestContext:
    """Tests for context (oversold/overbought) classification."""

    def test_oversold_below_lower_band(self):
        """Test oversold when price below lower BB."""
        close = pd.Series([100.0, 100.0, 80.0])
        rsi = pd.Series([50.0, 50.0, 50.0])
        bb_lower = pd.Series([95.0, 95.0, 95.0])
        bb_upper = pd.Series([105.0, 105.0, 105.0])

        is_oversold, is_overbought = EVRIndicators.calculate_context(
            close, rsi, bb_lower, bb_upper
        )
        assert is_oversold.iloc[2] == True
        assert is_overbought.iloc[2] == False

    def test_oversold_low_rsi(self):
        """Test oversold when RSI < 30."""
        close = pd.Series([100.0, 100.0, 100.0])
        rsi = pd.Series([50.0, 50.0, 25.0])
        bb_lower = pd.Series([95.0, 95.0, 95.0])
        bb_upper = pd.Series([105.0, 105.0, 105.0])

        is_oversold, is_overbought = EVRIndicators.calculate_context(
            close, rsi, bb_lower, bb_upper
        )
        assert is_oversold.iloc[2] == True

    def test_overbought_above_upper_band(self):
        """Test overbought when price above upper BB."""
        close = pd.Series([100.0, 100.0, 120.0])
        rsi = pd.Series([50.0, 50.0, 50.0])
        bb_lower = pd.Series([95.0, 95.0, 95.0])
        bb_upper = pd.Series([105.0, 105.0, 105.0])

        is_oversold, is_overbought = EVRIndicators.calculate_context(
            close, rsi, bb_lower, bb_upper
        )
        assert is_overbought.iloc[2] == True
        assert is_oversold.iloc[2] == False

    def test_overbought_high_rsi(self):
        """Test overbought when RSI > 70."""
        close = pd.Series([100.0, 100.0, 100.0])
        rsi = pd.Series([50.0, 50.0, 75.0])
        bb_lower = pd.Series([95.0, 95.0, 95.0])
        bb_upper = pd.Series([105.0, 105.0, 105.0])

        is_oversold, is_overbought = EVRIndicators.calculate_context(
            close, rsi, bb_lower, bb_upper
        )
        assert is_overbought.iloc[2] == True

    def test_neutral_context(self):
        """Test neutral when price inside bands and RSI normal."""
        close = pd.Series([100.0, 100.0, 100.0])
        rsi = pd.Series([50.0, 50.0, 50.0])
        bb_lower = pd.Series([95.0, 95.0, 95.0])
        bb_upper = pd.Series([105.0, 105.0, 105.0])

        is_oversold, is_overbought = EVRIndicators.calculate_context(
            close, rsi, bb_lower, bb_upper
        )
        assert is_oversold.iloc[2] == False
        assert is_overbought.iloc[2] == False


class TestGenerateSignals:
    """Tests for full signal generation."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_returns_four_series(self):
        """Test generate_signals returns four boolean Series."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(df)

        assert isinstance(long_entries, pd.Series)
        assert isinstance(long_exits, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(short_exits, pd.Series)

    def test_boolean_type(self):
        """Test all outputs are boolean."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(df)

        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_length_matches_input(self):
        """Test output length matches input."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(df)

        assert len(long_entries) == len(df)
        assert len(long_exits) == len(df)
        assert len(short_entries) == len(df)
        assert len(short_exits) == len(df)

    def test_index_preserved(self):
        """Test index is preserved."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(df)

        pd.testing.assert_index_equal(long_entries.index, df.index)

    def test_empty_data(self):
        """Test empty DataFrame returns empty Series."""
        df = pd.DataFrame()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(df)

        assert len(long_entries) == 0
        assert len(long_exits) == 0

    def test_no_nan_in_output(self):
        """Test no NaN values in boolean outputs."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(df)

        assert not long_entries.isna().any()
        assert not long_exits.isna().any()
        assert not short_entries.isna().any()
        assert not short_exits.isna().any()


class TestGetCurrentSignal:
    """Tests for get_current_signal."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_returns_evr_signal(self):
        """Test get_current_signal returns EVRSignal."""
        df = self.create_sample_ohlcv()
        signal = EVRIndicators.get_current_signal(df)
        assert isinstance(signal, EVRSignal)

    def test_signal_attributes(self):
        """Test EVRSignal has all required attributes."""
        df = self.create_sample_ohlcv()
        signal = EVRIndicators.get_current_signal(df)

        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'close')
        assert hasattr(signal, 'volume')
        assert hasattr(signal, 'rel_volume')
        assert hasattr(signal, 'rel_spread')
        assert hasattr(signal, 'is_absorption')
        assert hasattr(signal, 'context')
        assert hasattr(signal, 'rsi')
        assert hasattr(signal, 'bb_upper')
        assert hasattr(signal, 'bb_middle')
        assert hasattr(signal, 'bb_lower')

    def test_context_values(self):
        """Test context is one of oversold/overbought/neutral."""
        df = self.create_sample_ohlcv()
        signal = EVRIndicators.get_current_signal(df)
        assert signal.context in ['oversold', 'overbought', 'neutral']

    def test_insufficient_data_raises(self):
        """Test insufficient data raises ValueError."""
        df = self.create_sample_ohlcv(10)
        with pytest.raises(ValueError, match="Insufficient data"):
            EVRIndicators.get_current_signal(df)


# ===========================================================================
# IMPROVEMENT METHOD TESTS
# ===========================================================================

class TestATRCalculation:
    """Tests for ATR (Average True Range) calculation."""

    def create_sample_ohlcv(self, days: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        close = 100 + np.cumsum(np.random.randn(days) * 2)
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        volume = 1000000 + np.random.randn(days) * 100000
        return pd.DataFrame({
            'open': close + np.random.randn(days) * 0.5,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_returns_series(self):
        """Test calculate_atr returns a Series."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.calculate_atr(df, period=14)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        """Test output length matches input."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.calculate_atr(df, period=14)
        assert len(result) == len(df)

    def test_atr_positive(self):
        """Test ATR values are positive."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.calculate_atr(df, period=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_nan_in_warmup(self):
        """Test NaN values in warmup period."""
        df = self.create_sample_ohlcv(30)
        result = EVRIndicators.calculate_atr(df, period=14)
        assert result.iloc[:13].isna().all()
        assert result.iloc[13:].notna().all()

    def test_higher_volatility_higher_atr(self):
        """Test higher volatility produces higher ATR."""
        # Low volatility
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        close_low = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df_low = pd.DataFrame({
            'open': close_low,
            'high': close_low + 0.5,
            'low': close_low - 0.5,
            'close': close_low,
            'volume': np.ones(50) * 1e6
        }, index=dates)

        # High volatility
        close_high = 100 + np.cumsum(np.random.randn(50) * 2)
        df_high = pd.DataFrame({
            'open': close_high,
            'high': close_high + 2,
            'low': close_high - 2,
            'close': close_high,
            'volume': np.ones(50) * 1e6
        }, index=dates)

        atr_low = EVRIndicators.calculate_atr(df_low, period=14)
        atr_high = EVRIndicators.calculate_atr(df_high, period=14)

        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestBTCRegime:
    """Tests for BTC regime filter."""

    def create_sample_btc_close(self, days: int = 100) -> pd.Series:
        """Create sample BTC close prices."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.03
        close = 50000 * np.exp(np.cumsum(returns))
        return pd.Series(close, index=dates)

    def test_returns_series(self):
        """Test get_btc_regime returns a Series."""
        btc_close = self.create_sample_btc_close()
        result = EVRIndicators.get_btc_regime(btc_close, sma_period=50)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        """Test output length matches input."""
        btc_close = self.create_sample_btc_close()
        result = EVRIndicators.get_btc_regime(btc_close, sma_period=50)
        assert len(result) == len(btc_close)

    def test_only_bullish_or_bearish(self):
        """Test regime is only 'bullish' or 'bearish'."""
        btc_close = self.create_sample_btc_close()
        result = EVRIndicators.get_btc_regime(btc_close, sma_period=50)
        unique_values = result.unique()
        assert set(unique_values).issubset({'bullish', 'bearish'})

    def test_bullish_above_sma(self):
        """Test bullish when price > SMA."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        btc_close = pd.Series(range(40000, 40100), index=dates, dtype=float)
        result = EVRIndicators.get_btc_regime(btc_close, sma_period=20)
        # In an uptrend, recent values should be bullish
        assert result.iloc[-1] == 'bullish'

    def test_bearish_below_sma(self):
        """Test bearish when price < SMA."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        btc_close = pd.Series(range(50000, 49900, -1), index=dates, dtype=float)
        result = EVRIndicators.get_btc_regime(btc_close, sma_period=20)
        # In a downtrend, recent values should be bearish
        assert result.iloc[-1] == 'bearish'


class TestPositionSizing:
    """Tests for volatility-scaled position sizing."""

    def create_sample_close(self, days: int = 100) -> pd.Series:
        """Create sample close prices."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        return pd.Series(close, index=dates)

    def test_returns_series(self):
        """Test calculate_position_size returns a Series."""
        close = self.create_sample_close()
        result = EVRIndicators.calculate_position_size(close)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        """Test output length matches input."""
        close = self.create_sample_close()
        result = EVRIndicators.calculate_position_size(close)
        assert len(result) == len(close)

    def test_values_in_range(self):
        """Test position sizes are in [min, max] range."""
        close = self.create_sample_close()
        result = EVRIndicators.calculate_position_size(
            close, min_position=0.25, max_position=1.0
        )
        assert (result >= 0.25).all()
        assert (result <= 1.0).all()

    def test_high_vol_smaller_position(self):
        """Test higher volatility produces smaller position."""
        # Low volatility price series
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close_low = pd.Series(
            100 + np.cumsum(np.random.randn(100) * 0.5), index=dates
        )
        close_high = pd.Series(
            100 + np.cumsum(np.random.randn(100) * 3), index=dates
        )

        pos_low = EVRIndicators.calculate_position_size(close_low)
        pos_high = EVRIndicators.calculate_position_size(close_high)

        # Higher volatility should generally produce smaller positions
        # (may be clipped, so compare averages)
        assert pos_high.mean() <= pos_low.mean()

    def test_custom_target_vol(self):
        """Test custom target volatility."""
        close = self.create_sample_close()
        pos_20 = EVRIndicators.calculate_position_size(close, target_vol=0.20)
        pos_40 = EVRIndicators.calculate_position_size(close, target_vol=0.40)
        # Higher target vol should produce larger positions
        assert pos_40.mean() >= pos_20.mean()


class TestSignalsWithStops:
    """Tests for signals with ATR-based stops."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_returns_six_series(self):
        """Test generate_signals_with_stops returns six Series."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.generate_signals_with_stops(df)
        assert len(result) == 6

    def test_first_four_are_boolean(self):
        """Test first four outputs are boolean."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits, _, _ = \
            EVRIndicators.generate_signals_with_stops(df)
        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_stop_prices_numeric(self):
        """Test stop prices are numeric."""
        df = self.create_sample_ohlcv()
        _, _, _, _, long_stop, short_stop = \
            EVRIndicators.generate_signals_with_stops(df)
        assert pd.api.types.is_numeric_dtype(long_stop)
        assert pd.api.types.is_numeric_dtype(short_stop)

    def test_long_stop_below_close(self):
        """Test long stop is below close price."""
        df = self.create_sample_ohlcv()
        _, _, _, _, long_stop, _ = EVRIndicators.generate_signals_with_stops(df)
        valid_idx = long_stop.dropna().index
        assert (long_stop[valid_idx] < df.loc[valid_idx, 'close']).all()

    def test_short_stop_above_close(self):
        """Test short stop is above close price."""
        df = self.create_sample_ohlcv()
        _, _, _, _, _, short_stop = EVRIndicators.generate_signals_with_stops(df)
        valid_idx = short_stop.dropna().index
        assert (short_stop[valid_idx] > df.loc[valid_idx, 'close']).all()


class TestSignalsWithBTCRegime:
    """Tests for signals with BTC regime filter."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def create_sample_btc_close(self, days: int = 100) -> pd.Series:
        """Create sample BTC close prices."""
        np.random.seed(123)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.03
        close = 50000 * np.exp(np.cumsum(returns))
        return pd.Series(close, index=dates)

    def test_returns_five_series(self):
        """Test generate_signals_with_btc_regime returns five Series."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_sample_btc_close()
        result = EVRIndicators.generate_signals_with_btc_regime(df, btc_close)
        assert len(result) == 5

    def test_first_four_are_boolean(self):
        """Test first four outputs are boolean."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_sample_btc_close()
        long_entries, long_exits, short_entries, short_exits, _ = \
            EVRIndicators.generate_signals_with_btc_regime(df, btc_close)
        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_regime_is_string(self):
        """Test btc_regime is string Series."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_sample_btc_close()
        _, _, _, _, btc_regime = \
            EVRIndicators.generate_signals_with_btc_regime(df, btc_close)
        assert btc_regime.dtype == object

    def test_block_entry_mode(self):
        """Test block_entry mode blocks entries during bearish."""
        # Create data where BTC is always below SMA (bearish)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        btc_close = pd.Series(range(50000, 49900, -1), index=dates, dtype=float)

        df = self.create_sample_ohlcv()

        long_entries, _, _, _, btc_regime = \
            EVRIndicators.generate_signals_with_btc_regime(
                df, btc_close, regime_mode='block_entry', btc_sma_period=20
            )

        # Check that when regime is bearish, entries are blocked
        bearish_mask = btc_regime == 'bearish'
        if bearish_mask.any():
            assert not long_entries[bearish_mask].any()


class TestSignalsImproved:
    """Tests for improved signal generation."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def create_sample_btc_close(self, days: int = 100) -> pd.Series:
        """Create sample BTC close prices."""
        np.random.seed(123)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.03
        close = 50000 * np.exp(np.cumsum(returns))
        return pd.Series(close, index=dates)

    def test_returns_four_series(self):
        """Test generate_signals_improved returns four boolean Series."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.generate_signals_improved(df)
        assert len(result) == 4

    def test_boolean_type(self):
        """Test all outputs are boolean."""
        df = self.create_sample_ohlcv()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals_improved(df)
        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_with_btc_filter(self):
        """Test with BTC filter enabled."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_sample_btc_close()
        result = EVRIndicators.generate_signals_improved(
            df, btc_close=btc_close, use_btc_filter=True
        )
        assert len(result) == 4

    def test_without_btc_filter(self):
        """Test with BTC filter disabled."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_sample_btc_close()
        result = EVRIndicators.generate_signals_improved(
            df, btc_close=btc_close, use_btc_filter=False
        )
        assert len(result) == 4

    def test_with_atr_stops(self):
        """Test with ATR stops enabled."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.generate_signals_improved(
            df, use_atr_stops=True
        )
        assert len(result) == 4

    def test_without_atr_stops(self):
        """Test with ATR stops disabled."""
        df = self.create_sample_ohlcv()
        result = EVRIndicators.generate_signals_improved(
            df, use_atr_stops=False
        )
        assert len(result) == 4

    def test_tighter_thresholds_fewer_signals(self):
        """Test tighter thresholds generally produce fewer signals."""
        df = self.create_sample_ohlcv()

        # Default (looser) thresholds
        entries_loose, _, _, _ = EVRIndicators.generate_signals(
            df, vol_threshold=2.0, spread_threshold=0.8
        )

        # Tighter thresholds
        entries_tight, _, _, _ = EVRIndicators.generate_signals_improved(
            df, vol_threshold=3.0, spread_threshold=0.5,
            use_btc_filter=False, use_atr_stops=False
        )

        # Tighter thresholds should produce same or fewer signals
        assert entries_tight.sum() <= entries_loose.sum()

    def test_no_nan_in_output(self):
        """Test no NaN values in boolean outputs."""
        df = self.create_sample_ohlcv()
        btc_close = self.create_sample_btc_close()
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals_improved(
                df, btc_close=btc_close,
                use_btc_filter=True, use_atr_stops=True
            )
        assert not long_entries.isna().any()
        assert not long_exits.isna().any()
        assert not short_entries.isna().any()
        assert not short_exits.isna().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
