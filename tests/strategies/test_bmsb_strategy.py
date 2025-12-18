"""
Unit tests for Bull Market Support Band (BMSB) Strategy.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.advanced.bmsb_indicators import BMSBIndicators, BMSBSignal
from src.strategies.advanced.bmsb_strategy import BMSBStrategy
from src.strategies.registry import get_strategy_class


class TestBMSBIndicators:
    """Tests for BMSB indicator calculations."""

    def create_sample_data(self, weeks: int = 30) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        # Create daily data that spans multiple weeks
        days = weeks * 7
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')

        # Create price data with trend
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.02  # 2% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, days).astype(float)
        }, index=dates)

        return df

    def test_resample_to_weekly(self):
        """Test weekly resampling of daily data."""
        daily_data = self.create_sample_data(weeks=10)
        weekly_data = BMSBIndicators.resample_to_weekly(daily_data)

        # Should have approximately 10 weeks of data
        assert len(weekly_data) >= 9
        assert len(weekly_data) <= 11

        # Weekly high should be max of daily highs
        # Weekly low should be min of daily lows
        for week_date in weekly_data.index[:3]:
            week_start = week_date - timedelta(days=6)
            week_mask = (daily_data.index >= week_start) & (daily_data.index <= week_date)
            week_daily = daily_data[week_mask]

            if not week_daily.empty:
                assert weekly_data.loc[week_date, 'high'] == pytest.approx(week_daily['high'].max(), rel=1e-6)
                assert weekly_data.loc[week_date, 'low'] == pytest.approx(week_daily['low'].min(), rel=1e-6)

    def test_calculate_sma(self):
        """Test SMA calculation."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma_3 = BMSBIndicators.calculate_sma(series, 3)

        # First 2 values should be NaN
        assert pd.isna(sma_3.iloc[0])
        assert pd.isna(sma_3.iloc[1])

        # SMA(3) of [1,2,3] = 2
        assert sma_3.iloc[2] == pytest.approx(2.0, rel=1e-6)
        # SMA(3) of [2,3,4] = 3
        assert sma_3.iloc[3] == pytest.approx(3.0, rel=1e-6)

    def test_calculate_ema(self):
        """Test EMA calculation."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ema_3 = BMSBIndicators.calculate_ema(series, 3)

        # EMA should exist after min_periods
        assert not pd.isna(ema_3.iloc[2])

        # EMA should be different from SMA
        sma_3 = BMSBIndicators.calculate_sma(series, 3)
        # Not all values will be equal due to EMA weighting
        assert not np.allclose(ema_3.dropna().values, sma_3.dropna().values)

    def test_calculate_band(self):
        """Test band calculation."""
        daily_data = self.create_sample_data(weeks=30)
        band = BMSBIndicators.calculate_band(daily_data, sma_period=20, ema_period=21)

        assert not band.empty
        assert 'sma' in band.columns
        assert 'ema' in band.columns
        assert 'band_upper' in band.columns
        assert 'band_lower' in band.columns
        assert 'position' in band.columns
        assert 'trend' in band.columns

        # Band upper should always be >= band lower
        assert (band['band_upper'] >= band['band_lower']).all()

        # Position should be one of: above, below, inside
        assert band['position'].isin(['above', 'below', 'inside']).all()

        # Trend should be one of: bullish, bearish, neutral
        assert band['trend'].isin(['bullish', 'bearish', 'neutral']).all()

    def test_calculate_band_insufficient_data(self):
        """Test band calculation with insufficient data."""
        # Create only 10 weeks of data (need 21 for EMA)
        daily_data = self.create_sample_data(weeks=10)
        band = BMSBIndicators.calculate_band(daily_data, sma_period=20, ema_period=21)

        # Should return empty DataFrame due to insufficient data
        assert band.empty

    def test_generate_signals_vectorized(self):
        """Test vectorized signal generation."""
        daily_data = self.create_sample_data(weeks=50)

        long_entries, long_exits, short_entries, short_exits = \
            BMSBIndicators.generate_signals_vectorized(daily_data, sma_period=20, ema_period=21)

        assert len(long_entries) == len(daily_data)
        assert len(long_exits) == len(daily_data)
        assert len(short_entries) == len(daily_data)
        assert len(short_exits) == len(daily_data)

        # Signals should be boolean
        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool


class TestBMSBStrategy:
    """Tests for BMSB Strategy class."""

    def create_sample_data(self, weeks: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        days = weeks * 7
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')

        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, days).astype(float)
        }, index=dates)

        return df

    def test_strategy_initialization(self):
        """Test strategy initialization with default parameters."""
        strategy = BMSBStrategy()

        assert strategy.sma_period == 20
        assert strategy.ema_period == 21
        assert strategy.long_only is False
        assert strategy.use_trailing_stop is True
        assert strategy.trailing_stop_pct == 0.15

    def test_strategy_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = BMSBStrategy(
            sma_period=15,
            ema_period=18,
            long_only=True,
            use_trailing_stop=False,
            trailing_stop_pct=0.10
        )

        assert strategy.sma_period == 15
        assert strategy.ema_period == 18
        assert strategy.long_only is True
        assert strategy.use_trailing_stop is False
        assert strategy.trailing_stop_pct == 0.10

    def test_validate_parameters(self):
        """Test parameter validation."""
        # Invalid sma_period
        with pytest.raises(ValueError):
            BMSBStrategy(sma_period=0)

        # Invalid ema_period
        with pytest.raises(ValueError):
            BMSBStrategy(ema_period=-1)

        # Invalid trailing_stop_pct
        with pytest.raises(ValueError):
            BMSBStrategy(trailing_stop_pct=1.5)

        with pytest.raises(ValueError):
            BMSBStrategy(trailing_stop_pct=-0.1)

    def test_generate_signals(self):
        """Test signal generation."""
        strategy = BMSBStrategy()
        data = self.create_sample_data(weeks=50)

        long_entries, long_exits, short_entries, short_exits = \
            strategy.generate_long_short_signals(data)

        # Should return Series with same index as input
        assert isinstance(long_entries, pd.Series)
        assert isinstance(long_exits, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(short_exits, pd.Series)

        assert len(long_entries) == len(data)
        assert (long_entries.index == data.index).all()

    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data."""
        strategy = BMSBStrategy()
        empty_data = pd.DataFrame()

        long_entries, long_exits, short_entries, short_exits = \
            strategy.generate_long_short_signals(empty_data)

        assert len(long_entries) == 0
        assert len(long_exits) == 0
        assert len(short_entries) == 0
        assert len(short_exits) == 0

    def test_long_only_mode(self):
        """Test that long_only mode doesn't generate short signals."""
        strategy = BMSBStrategy(long_only=True)
        data = self.create_sample_data(weeks=50)

        long_entries, long_exits, short_entries, short_exits = \
            strategy.generate_long_short_signals(data)

        # Short signals should all be False
        assert not short_entries.any()

    def test_registry_lookup(self):
        """Test that strategy can be loaded from registry."""
        strategy_cls = get_strategy_class("BMSBStrategy")
        assert strategy_cls == BMSBStrategy

        # Test display names
        strategy_cls = get_strategy_class("BMSB")
        assert strategy_cls == BMSBStrategy

        strategy_cls = get_strategy_class("Bull Market Support Band")
        assert strategy_cls == BMSBStrategy


class TestBMSBSignalGeneration:
    """Tests for correct signal generation logic."""

    def create_trending_data(self, direction: str = 'up', weeks: int = 50) -> pd.DataFrame:
        """Create data with a clear trend direction."""
        days = weeks * 7
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')

        base_price = 10000
        if direction == 'up':
            # Strong uptrend
            trend = np.linspace(0, 1, days)
        elif direction == 'down':
            # Strong downtrend
            trend = np.linspace(1, 0, days)
        else:
            # Sideways
            trend = np.ones(days) * 0.5

        np.random.seed(42)
        noise = np.random.randn(days) * 0.01
        prices = base_price * (0.5 + trend + noise)

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.005)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.005)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, days).astype(float)
        }, index=dates)

        return df

    def test_uptrend_generates_long_entry(self):
        """Test that strong uptrend generates long entry signal."""
        strategy = BMSBStrategy(signal_on_weekly_close=False)
        data = self.create_trending_data(direction='up', weeks=50)

        long_entries, long_exits, short_entries, short_exits = \
            strategy.generate_long_short_signals(data)

        # Should have at least one long entry in uptrend
        assert long_entries.any(), "Expected long entry signal in uptrend"

    def test_downtrend_generates_short_entry(self):
        """Test that strong downtrend generates short entry signal."""
        strategy = BMSBStrategy(signal_on_weekly_close=False, long_only=False)
        data = self.create_trending_data(direction='down', weeks=50)

        long_entries, long_exits, short_entries, short_exits = \
            strategy.generate_long_short_signals(data)

        # Should have at least one short entry in downtrend
        assert short_entries.any(), "Expected short entry signal in downtrend"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
