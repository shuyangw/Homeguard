"""
Unit Tests for ORB High Volatility Indicators.

Tests SIP score, gap detection, ATR filter, and confidence score calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.strategies.advanced.orb_hv_indicators import ORBHVIndicators


class TestStocksInPlayScore:
    """Test SIP score calculation."""

    def test_sip_score_basic(self):
        """Test basic SIP score calculation."""
        # Create 20 days of minute data with consistent opening volume
        dates = pd.date_range('2025-01-01', periods=20, freq='B')
        dfs = []

        for i, date in enumerate(dates):
            # Create minute data for each day
            times = pd.date_range(
                f'{date.date()} 09:30:00',
                f'{date.date()} 10:00:00',
                freq='1min',
                tz='America/New_York'
            )

            # Base volume of 1000 per minute
            # Last day has 3x volume in first 5 minutes
            if i == len(dates) - 1:
                # High volume day
                volumes = [3000] * 5 + [1000] * (len(times) - 5)
            else:
                volumes = [1000] * len(times)

            df = pd.DataFrame({
                'open': [100] * len(times),
                'high': [101] * len(times),
                'low': [99] * len(times),
                'close': [100] * len(times),
                'volume': volumes
            }, index=times)
            dfs.append(df)

        data = pd.concat(dfs)
        sip_scores = ORBHVIndicators.stocks_in_play_score(
            data,
            lookback_days=14,
            opening_minutes=5
        )

        # Last day should have SIP score around 3.0 (3x average)
        last_date = dates[-1].date()
        assert last_date in sip_scores.index
        assert 2.5 <= sip_scores[last_date] <= 3.5

    def test_sip_score_single_day(self):
        """Test SIP score for single day with historical data."""
        today_volume = 15000  # 3x normal
        historical_volumes = [5000, 5000, 5000, 5000, 5000]  # Consistent 5000

        score = ORBHVIndicators.stocks_in_play_score_single_day(
            today_volume,
            historical_volumes
        )

        assert score == 3.0

    def test_sip_score_empty_data(self):
        """Test SIP score with empty data."""
        df = pd.DataFrame()
        result = ORBHVIndicators.stocks_in_play_score(df)
        assert result.empty

    def test_sip_score_no_history(self):
        """Test SIP score with no historical data."""
        score = ORBHVIndicators.stocks_in_play_score_single_day(
            5000, []
        )
        assert score == 1.0


class TestPremarketGap:
    """Test pre-market gap detection."""

    def test_gap_up(self):
        """Test positive gap detection."""
        result = ORBHVIndicators.premarket_gap(
            today_open=105.0,
            yesterday_close=100.0,
            min_gap_pct=0.02,
            max_gap_pct=0.15
        )

        assert result['gap_pct'] == 0.05
        assert result['gap_direction'] == 'up'
        assert result['is_valid'] == True
        assert result['abs_gap_pct'] == 0.05

    def test_gap_down(self):
        """Test negative gap detection."""
        result = ORBHVIndicators.premarket_gap(
            today_open=95.0,
            yesterday_close=100.0,
            min_gap_pct=0.02,
            max_gap_pct=0.15
        )

        assert result['gap_pct'] == -0.05
        assert result['gap_direction'] == 'down'
        assert result['is_valid'] == True

    def test_gap_too_small(self):
        """Test gap below minimum threshold."""
        result = ORBHVIndicators.premarket_gap(
            today_open=100.5,
            yesterday_close=100.0,
            min_gap_pct=0.02,
            max_gap_pct=0.15
        )

        assert result['gap_direction'] == 'up'
        assert result['is_valid'] == False  # 0.5% < 2% minimum

    def test_gap_too_large(self):
        """Test gap above maximum threshold."""
        result = ORBHVIndicators.premarket_gap(
            today_open=120.0,
            yesterday_close=100.0,
            min_gap_pct=0.02,
            max_gap_pct=0.15
        )

        assert result['gap_direction'] == 'up'
        assert result['is_valid'] == False  # 20% > 15% maximum

    def test_gap_invalid_price(self):
        """Test gap with invalid price."""
        result = ORBHVIndicators.premarket_gap(
            today_open=100.0,
            yesterday_close=0.0
        )

        assert result['is_valid'] == False
        assert result['gap_direction'] == 'flat'


class TestATRVolatilityFilter:
    """Test ATR-based volatility filtering."""

    def test_atr_calculation(self):
        """Test ATR calculation."""
        n = 30
        # Create data with consistent true range
        high = pd.Series([102.0] * n)
        low = pd.Series([98.0] * n)
        close = pd.Series([100.0] * n)

        atr = ORBHVIndicators.atr(high, low, close, period=14)

        # ATR should be around 4 (high - low)
        assert 3.5 <= atr.iloc[-1] <= 4.5

    def test_atr_percent(self):
        """Test ATR as percentage of price."""
        n = 30
        high = pd.Series([102.0] * n)
        low = pd.Series([98.0] * n)
        close = pd.Series([100.0] * n)

        atr_pct = ORBHVIndicators.atr_percent(high, low, close, period=14)

        # ATR% should be around 4% (4 / 100)
        assert 3.5 <= atr_pct.iloc[-1] <= 4.5

    def test_volatility_filter_within_bounds(self):
        """Test volatility filter when within bounds."""
        n = 30
        # 4% ATR range
        high = pd.Series([104.0] * n)
        low = pd.Series([96.0] * n)
        close = pd.Series([100.0] * n)

        result = ORBHVIndicators.volatility_filter(
            high, low, close,
            period=14,
            min_atr_pct=2.0,
            max_atr_pct=10.0
        )

        # All should pass (4% is within 2-10%)
        assert result.iloc[-1] == True

    def test_volatility_filter_too_low(self):
        """Test volatility filter when ATR too low."""
        n = 30
        # 1% ATR range
        high = pd.Series([100.5] * n)
        low = pd.Series([99.5] * n)
        close = pd.Series([100.0] * n)

        result = ORBHVIndicators.volatility_filter(
            high, low, close,
            period=14,
            min_atr_pct=2.0,
            max_atr_pct=10.0
        )

        # Should fail (1% < 2% minimum)
        assert result.iloc[-1] == False


class TestConfidenceScore:
    """Test confidence score calculation."""

    def test_confidence_score_high_sip(self):
        """Test confidence with high SIP score."""
        score = ORBHVIndicators.confidence_score(
            sip_score=5.0,      # 5x average
            sentiment_score=0.0,
            gap_pct=0.05,       # 5% gap up
            volume_ratio=1.5,
            direction='long'
        )

        # High SIP should contribute significantly
        assert 0.3 <= score <= 0.8

    def test_confidence_score_sentiment_aligned(self):
        """Test confidence with aligned sentiment."""
        score = ORBHVIndicators.confidence_score(
            sip_score=3.0,
            sentiment_score=0.5,  # Strong positive
            gap_pct=0.05,
            volume_ratio=1.5,
            direction='long'
        )

        # Aligned sentiment should add to score
        assert score >= 0.4

    def test_confidence_score_range(self):
        """Test that confidence score is always between 0 and 1."""
        # Test extremes
        score_low = ORBHVIndicators.confidence_score(
            sip_score=0.5,
            sentiment_score=-0.5,
            gap_pct=-0.05,
            volume_ratio=0.5,
            direction='long'
        )

        score_high = ORBHVIndicators.confidence_score(
            sip_score=15.0,
            sentiment_score=0.9,
            gap_pct=0.10,
            volume_ratio=5.0,
            direction='long'
        )

        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0


class TestOpeningRange:
    """Test opening range calculation."""

    def test_opening_range_5min(self):
        """Test 5-minute opening range."""
        times = pd.date_range(
            '2025-01-06 09:30:00',
            '2025-01-06 10:00:00',
            freq='1min',
            tz='America/New_York'
        )

        n = len(times)
        df = pd.DataFrame({
            'open': np.linspace(100, 105, n),
            'high': np.linspace(101, 106, n),
            'low': np.linspace(99, 104, n),
            'close': np.linspace(100.5, 105.5, n),
            'volume': [1000] * n
        }, index=times)

        result = ORBHVIndicators.opening_range(
            df,
            start_time=time(9, 30),
            end_time=time(9, 35)  # 5-minute OR
        )

        assert 'or_high' in result
        assert 'or_low' in result
        assert 'or_height' in result
        assert 'or_volume' in result

        # OR should use only first 5 bars
        or_data = df[df.index.time < time(9, 35)]
        assert result['or_high'] == or_data['high'].max()
        assert result['or_low'] == or_data['low'].min()

    def test_opening_range_empty(self):
        """Test opening range with empty data."""
        df = pd.DataFrame()
        result = ORBHVIndicators.opening_range(df)
        assert result == {}


class TestTieredTargets:
    """Test tiered target calculation."""

    def test_long_targets(self):
        """Test target calculation for long position."""
        result = ORBHVIndicators.calculate_tiered_targets(
            entry_price=100.0,
            or_high=100.0,
            or_low=98.0,
            direction='long',
            target1_multiplier=1.0,
            target2_multiplier=2.0
        )

        assert result['stop_loss'] == 98.0  # OR low
        assert result['target1'] == 102.0   # Entry + 1x OR height (2)
        assert result['target2'] == 104.0   # Entry + 2x OR height (4)
        assert result['risk'] == 2.0        # Entry - Stop
        assert result['reward1'] == 2.0     # T1 - Entry
        assert result['rr_ratio1'] == 1.0   # 1:1 risk/reward

    def test_short_targets(self):
        """Test target calculation for short position."""
        result = ORBHVIndicators.calculate_tiered_targets(
            entry_price=98.0,
            or_high=100.0,
            or_low=98.0,
            direction='short',
            target1_multiplier=1.0,
            target2_multiplier=2.0
        )

        assert result['stop_loss'] == 100.0  # OR high
        assert result['target1'] == 96.0     # Entry - 1x OR height (2)
        assert result['target2'] == 94.0     # Entry - 2x OR height (4)
        assert result['risk'] == 2.0         # Stop - Entry


class TestVolatilityRegime:
    """Test volatility regime classification."""

    def test_normal_volatility(self):
        """Test normal volatility regime."""
        # Create SPY data with normal vol
        times = pd.date_range(
            '2025-01-06 09:30:00',
            '2025-01-06 10:00:00',
            freq='1min',
            tz='America/New_York'
        )

        # Small, consistent moves (normal vol)
        close = 500 + np.cumsum(np.random.randn(len(times)) * 0.05)
        df = pd.DataFrame({'close': close}, index=times)

        regime = ORBHVIndicators.volatility_regime(df, lookback_bars=12)
        # Should be normal or low in calm conditions
        assert regime in ['low', 'normal']

    def test_empty_data(self):
        """Test volatility regime with empty data."""
        df = pd.DataFrame()
        regime = ORBHVIndicators.volatility_regime(df)
        assert regime == 'normal'  # Default


class TestPositionSizeAdjustment:
    """Test position size adjustment calculation."""

    def test_normal_adjustment(self):
        """Test adjustment in normal conditions."""
        mult = ORBHVIndicators.position_size_adjustment(
            confidence=0.5,
            volatility_regime='normal'
        )

        # Should be around 0.75 (0.5 + 0.5*0.5 = 0.75)
        assert 0.7 <= mult <= 0.8

    def test_high_vol_reduction(self):
        """Test reduced size in high volatility."""
        mult = ORBHVIndicators.position_size_adjustment(
            confidence=0.5,
            volatility_regime='high'
        )

        # Should be lower due to high vol multiplier (0.7)
        assert mult < 0.7

    def test_extreme_vol_reduction(self):
        """Test significant reduction in extreme volatility."""
        mult = ORBHVIndicators.position_size_adjustment(
            confidence=1.0,
            volatility_regime='extreme'
        )

        # Should be heavily reduced (1.0 * 0.4 = 0.4)
        assert mult <= 0.5
