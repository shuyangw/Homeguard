"""
Unit Tests for ORB Indicators.

Tests the Opening Range Breakout indicator calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.strategies.advanced.orb_indicators import ORBIndicators


class TestOpeningRange:
    """Test opening range calculation."""

    def test_opening_range_basic(self):
        """Test basic opening range calculation."""
        # Create minute data for 9:30-10:00 AM
        times = pd.date_range(
            '2025-01-06 09:30:00',
            '2025-01-06 10:00:00',
            freq='1min',
            tz='America/New_York'
        )

        # Prices rise from 100 to 105 during first 15 min
        n = len(times)
        df = pd.DataFrame({
            'open': np.linspace(100, 105, n),
            'high': np.linspace(101, 106, n),
            'low': np.linspace(99, 104, n),
            'close': np.linspace(100.5, 105.5, n),
            'volume': [1000] * n
        }, index=times)

        result = ORBIndicators.opening_range(
            df,
            start_time=time(9, 30),
            end_time=time(9, 45)
        )

        assert 'or_high' in result
        assert 'or_low' in result
        assert 'or_height' in result
        assert 'or_midpoint' in result

        # OR high should be max of 9:30-9:45 highs
        # OR low should be min of 9:30-9:45 lows
        or_data = df[df.index.time < time(9, 45)]
        assert result['or_high'] == or_data['high'].max()
        assert result['or_low'] == or_data['low'].min()
        assert result['or_height'] == result['or_high'] - result['or_low']

    def test_opening_range_empty_data(self):
        """Test opening range with empty DataFrame."""
        df = pd.DataFrame()
        result = ORBIndicators.opening_range(df)
        assert result == {}

    def test_opening_range_insufficient_data(self):
        """Test opening range when data doesn't cover OR period."""
        # Create data only for 10:00-10:30 (after OR period)
        times = pd.date_range(
            '2025-01-06 10:00:00',
            '2025-01-06 10:30:00',
            freq='1min',
            tz='America/New_York'
        )
        df = pd.DataFrame({
            'open': [100] * len(times),
            'high': [101] * len(times),
            'low': [99] * len(times),
            'close': [100] * len(times),
            'volume': [1000] * len(times)
        }, index=times)

        result = ORBIndicators.opening_range(df)
        assert result == {}

    def test_opening_range_by_day(self):
        """Test opening range calculation across multiple days."""
        # Create 2 days of minute data
        day1_times = pd.date_range(
            '2025-01-06 09:30:00',
            '2025-01-06 16:00:00',
            freq='1min',
            tz='America/New_York'
        )
        day2_times = pd.date_range(
            '2025-01-07 09:30:00',
            '2025-01-07 16:00:00',
            freq='1min',
            tz='America/New_York'
        )
        times = day1_times.append(day2_times)

        df = pd.DataFrame({
            'open': [100] * len(times),
            'high': [101] * len(times),
            'low': [99] * len(times),
            'close': [100] * len(times),
            'volume': [1000] * len(times)
        }, index=times)

        result = ORBIndicators.opening_range_by_day(df)

        assert len(result) == 2
        assert result.index[0].isoformat() == '2025-01-06'
        assert result.index[1].isoformat() == '2025-01-07'


class TestRelativeVolume:
    """Test RVOL calculation."""

    def test_rvol_calculation(self):
        """Test basic RVOL calculation."""
        # Create volume series with known values
        volume = pd.Series([100] * 19 + [200])  # Last bar is 2x average

        rvol = ORBIndicators.relative_volume(volume, lookback=20)

        # Last RVOL should be ~2.0 (200 / ~100)
        assert rvol.iloc[-1] == pytest.approx(2.0, rel=0.1)

    def test_rvol_first_bars(self):
        """Test RVOL for first bars (uses min_periods=1)."""
        volume = pd.Series([100, 200, 150])

        rvol = ORBIndicators.relative_volume(volume, lookback=20)

        # First bar should be 1.0 (volume / volume = 1)
        assert rvol.iloc[0] == 1.0

    def test_rvol_handles_zero_volume(self):
        """Test RVOL handles zero volume gracefully."""
        volume = pd.Series([100, 0, 100])

        rvol = ORBIndicators.relative_volume(volume, lookback=20)

        # Should not contain inf or nan
        assert not rvol.isna().any()
        assert not np.isinf(rvol).any()


class TestGapDirection:
    """Test gap direction detection."""

    def test_gap_up(self):
        """Test gap up detection."""
        result = ORBIndicators.gap_direction(
            today_open=105.0,
            yesterday_close=100.0,
            threshold_pct=0.001
        )
        assert result == 'gap_up'

    def test_gap_down(self):
        """Test gap down detection."""
        result = ORBIndicators.gap_direction(
            today_open=95.0,
            yesterday_close=100.0,
            threshold_pct=0.001
        )
        assert result == 'gap_down'

    def test_flat(self):
        """Test flat (no gap) detection."""
        result = ORBIndicators.gap_direction(
            today_open=100.05,
            yesterday_close=100.0,
            threshold_pct=0.001
        )
        assert result == 'flat'

    def test_zero_yesterday_close(self):
        """Test handling of zero yesterday close."""
        result = ORBIndicators.gap_direction(
            today_open=100.0,
            yesterday_close=0.0
        )
        assert result == 'flat'


class TestBreakoutDetection:
    """Test breakout detection."""

    def test_long_breakout(self):
        """Test long breakout detection."""
        result = ORBIndicators.detect_breakout(
            current_close=105.0,
            or_high=104.0,
            or_low=100.0,
            prev_close=102.0  # Previous was inside range
        )
        assert result == 'long'

    def test_short_breakout(self):
        """Test short breakout detection."""
        result = ORBIndicators.detect_breakout(
            current_close=99.0,
            or_high=104.0,
            or_low=100.0,
            prev_close=102.0  # Previous was inside range
        )
        assert result == 'short'

    def test_no_breakout_inside_range(self):
        """Test no breakout when price is inside range."""
        result = ORBIndicators.detect_breakout(
            current_close=102.0,
            or_high=104.0,
            or_low=100.0,
            prev_close=102.0
        )
        assert result is None

    def test_no_breakout_prev_already_outside(self):
        """Test no breakout when previous was already outside range."""
        result = ORBIndicators.detect_breakout(
            current_close=106.0,
            or_high=104.0,
            or_low=100.0,
            prev_close=105.0  # Previous was already above range
        )
        assert result is None


class TestBreakoutConfirmation:
    """Test full breakout confirmation with all filters."""

    def test_long_breakout_confirmed(self):
        """Test confirmed long breakout with all filters passing."""
        confirmed, filters = ORBIndicators.is_breakout_confirmed(
            current_close=105.0,
            current_volume=2000,
            or_high=104.0,
            or_low=100.0,
            vwap=103.0,  # Price above VWAP
            rvol=2.0,    # High RVOL
            fast_ema=102.0,
            slow_ema=101.0,  # Fast > Slow (bullish)
            rvol_threshold=1.5,
            direction='long'
        )

        assert confirmed is True
        assert filters['price_breakout'] is True
        assert filters['vwap_above'] is True
        assert filters['trend_bullish'] is True
        assert filters['rvol_sufficient'] is True

    def test_long_breakout_blocked_low_rvol(self):
        """Test long breakout blocked by low RVOL."""
        confirmed, filters = ORBIndicators.is_breakout_confirmed(
            current_close=105.0,
            current_volume=500,
            or_high=104.0,
            or_low=100.0,
            vwap=103.0,
            rvol=1.0,    # Low RVOL - should block
            fast_ema=102.0,
            slow_ema=101.0,
            rvol_threshold=1.5,
            direction='long'
        )

        assert confirmed is False
        assert filters['rvol_sufficient'] is False

    def test_long_breakout_blocked_below_vwap(self):
        """Test long breakout blocked by being below VWAP."""
        confirmed, filters = ORBIndicators.is_breakout_confirmed(
            current_close=105.0,
            current_volume=2000,
            or_high=104.0,
            or_low=100.0,
            vwap=106.0,  # Price below VWAP - should block
            rvol=2.0,
            fast_ema=102.0,
            slow_ema=101.0,
            rvol_threshold=1.5,
            direction='long'
        )

        assert confirmed is False
        assert filters['vwap_above'] is False

    def test_short_breakout_confirmed(self):
        """Test confirmed short breakout."""
        confirmed, filters = ORBIndicators.is_breakout_confirmed(
            current_close=99.0,
            current_volume=2000,
            or_high=104.0,
            or_low=100.0,
            vwap=101.0,  # Price below VWAP
            rvol=2.0,
            fast_ema=101.0,
            slow_ema=102.0,  # Fast < Slow (bearish)
            rvol_threshold=1.5,
            direction='short'
        )

        assert confirmed is True


class TestTargetCalculation:
    """Test stop-loss and target calculation."""

    def test_long_targets(self):
        """Test target calculation for long position."""
        result = ORBIndicators.calculate_targets(
            entry_price=105.0,
            or_high=104.0,
            or_low=100.0,
            direction='long',
            target_multiplier=1.0
        )

        assert result['stop_loss'] == 100.0  # OR low
        assert result['target'] == 109.0     # Entry + OR height (4)
        assert result['risk'] == 5.0         # Entry - stop
        assert result['reward'] == 4.0       # Target - entry
        assert result['rr_ratio'] == pytest.approx(0.8, rel=0.01)

    def test_short_targets(self):
        """Test target calculation for short position."""
        result = ORBIndicators.calculate_targets(
            entry_price=99.0,
            or_high=104.0,
            or_low=100.0,
            direction='short',
            target_multiplier=1.0
        )

        assert result['stop_loss'] == 104.0  # OR high
        assert result['target'] == 95.0      # Entry - OR height (4)
        assert result['risk'] == 5.0         # Stop - entry
        assert result['reward'] == 4.0       # Entry - target

    def test_target_multiplier(self):
        """Test target multiplier effect."""
        result = ORBIndicators.calculate_targets(
            entry_price=105.0,
            or_high=104.0,
            or_low=100.0,
            direction='long',
            target_multiplier=2.0  # 2x OR height target
        )

        assert result['target'] == 113.0  # Entry + 2 * OR height (8)
