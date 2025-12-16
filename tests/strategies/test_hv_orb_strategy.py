"""
Unit Tests for ORB High Volatility Strategy.

Tests initialization, parameter validation, and signal generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.strategies.advanced.orb_hv_strategy import (
    ORBHighVolatilityStrategy,
    ORBHVPosition
)


class TestStrategyInitialization:
    """Test strategy initialization."""

    def test_default_parameters(self):
        """Test initialization with default parameters."""
        strategy = ORBHighVolatilityStrategy()

        assert strategy.opening_range_minutes == 5
        assert strategy.sip_min_score == 2.0
        assert strategy.min_gap_pct == 0.02
        assert strategy.max_gap_pct == 0.15
        assert strategy.target1_multiplier == 1.0
        assert strategy.target2_multiplier == 2.0
        assert strategy.use_trailing_stop == True
        assert strategy.long_only == False

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        strategy = ORBHighVolatilityStrategy(
            opening_range_minutes=10,
            sip_min_score=3.0,
            min_gap_pct=0.03,
            max_gap_pct=0.12,
            target1_multiplier=1.5,
            target2_multiplier=2.5,
            use_trailing_stop=False,
            long_only=True
        )

        assert strategy.opening_range_minutes == 10
        assert strategy.sip_min_score == 3.0
        assert strategy.min_gap_pct == 0.03
        assert strategy.max_gap_pct == 0.12
        assert strategy.target1_multiplier == 1.5
        assert strategy.target2_multiplier == 2.5
        assert strategy.use_trailing_stop == False
        assert strategy.long_only == True


class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_opening_range(self):
        """Test invalid opening range minutes."""
        with pytest.raises(ValueError, match="opening_range_minutes"):
            ORBHighVolatilityStrategy(opening_range_minutes=7)

    def test_invalid_sip_min_score(self):
        """Test invalid SIP minimum score."""
        with pytest.raises(ValueError, match="sip_min_score"):
            ORBHighVolatilityStrategy(sip_min_score=0.5)

    def test_invalid_gap_range(self):
        """Test invalid gap percentage range."""
        with pytest.raises(ValueError, match="min_gap_pct"):
            ORBHighVolatilityStrategy(min_gap_pct=0.10, max_gap_pct=0.05)

    def test_invalid_target_multipliers(self):
        """Test invalid target multipliers."""
        with pytest.raises(ValueError, match="target1_multiplier"):
            ORBHighVolatilityStrategy(
                target1_multiplier=2.0,
                target2_multiplier=1.5
            )

    def test_invalid_daily_max_loss(self):
        """Test invalid daily max loss."""
        with pytest.raises(ValueError, match="daily_max_loss_pct"):
            ORBHighVolatilityStrategy(daily_max_loss_pct=1.5)


class TestSignalGeneration:
    """Test signal generation."""

    def _create_test_data(self, days: int = 20) -> pd.DataFrame:
        """Create test minute data."""
        dfs = []
        dates = pd.date_range('2025-01-01', periods=days, freq='B')

        for i, date in enumerate(dates):
            times = pd.date_range(
                f'{date.date()} 09:30:00',
                f'{date.date()} 16:00:00',
                freq='1min',
                tz='America/New_York'
            )

            n = len(times)

            # Base price around 100
            base = 100 + i * 0.5

            # Create OHLCV data
            opens = np.full(n, base)
            closes = np.full(n, base)
            highs = np.full(n, base + 0.5)
            lows = np.full(n, base - 0.5)

            # Add gap on day 15
            if i == 14:
                gap_pct = 0.05  # 5% gap up
                opens = np.full(n, base * (1 + gap_pct))
                closes = np.full(n, base * (1 + gap_pct))
                highs = np.full(n, base * (1 + gap_pct) + 0.5)
                lows = np.full(n, base * (1 + gap_pct) - 0.5)

                # Add breakout above OR after first 5 minutes
                # First 5 minutes form OR high around base * 1.05 + 0.5
                # After 5 min, price breaks above
                or_high = base * (1 + gap_pct) + 0.5
                for j in range(5, min(20, n)):
                    closes[j] = or_high + 0.5
                    highs[j] = or_high + 1.0

            # Volume - high on day 15
            if i == 14:
                volumes = [5000] * 5 + [3000] * (n - 5)  # High opening volume
            else:
                volumes = [1000] * n

            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=times)

            dfs.append(df)

        return pd.concat(dfs)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        strategy = ORBHighVolatilityStrategy()
        df = pd.DataFrame()

        le, lx, se, sx = strategy.generate_long_short_signals(df)

        assert le.empty
        assert lx.empty
        assert se.empty
        assert sx.empty

    def test_missing_columns(self):
        """Test with missing required columns."""
        strategy = ORBHighVolatilityStrategy()

        times = pd.date_range('2025-01-06 09:30:00', periods=30, freq='1min')
        df = pd.DataFrame({
            'open': [100] * 30,
            'close': [100] * 30,
            # Missing high, low, volume
        }, index=times)

        le, lx, se, sx = strategy.generate_long_short_signals(df)

        # Should return False arrays
        assert not le.any()
        assert not lx.any()

    def test_generates_signals(self):
        """Test that strategy generates signals on valid data."""
        strategy = ORBHighVolatilityStrategy(
            sip_min_score=1.5,  # Lower threshold for test
            min_gap_pct=0.01,   # Lower gap requirement
        )

        data = self._create_test_data(days=20)

        le, lx, se, sx = strategy.generate_long_short_signals(data)

        # Should return Series of correct length
        assert len(le) == len(data)
        assert len(lx) == len(data)
        assert len(se) == len(data)
        assert len(sx) == len(data)

    def test_long_only_mode(self):
        """Test long-only mode disables shorts."""
        strategy = ORBHighVolatilityStrategy(long_only=True)

        data = self._create_test_data(days=20)

        le, lx, se, sx = strategy.generate_long_short_signals(data)

        # Short entries should always be False
        assert not se.any()


class TestRiskLimits:
    """Test daily risk limit enforcement."""

    def test_risk_limits_trade_count(self):
        """Test trade count limit."""
        strategy = ORBHighVolatilityStrategy(max_daily_trades=5)

        # Internal method check
        assert strategy._check_risk_limits(daily_trades=4, daily_pnl=0) == True
        assert strategy._check_risk_limits(daily_trades=5, daily_pnl=0) == False
        assert strategy._check_risk_limits(daily_trades=10, daily_pnl=0) == False

    def test_risk_limits_daily_loss(self):
        """Test daily loss limit."""
        strategy = ORBHighVolatilityStrategy(daily_max_loss_pct=0.03)
        capital = 100000

        # -2% loss should be OK
        assert strategy._check_risk_limits(0, -2000, capital) == True

        # -3% loss should stop trading
        assert strategy._check_risk_limits(0, -3000, capital) == False

        # -5% loss should definitely stop
        assert strategy._check_risk_limits(0, -5000, capital) == False


class TestORBHVPosition:
    """Test ORBHVPosition dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        position = ORBHVPosition(
            symbol='TQQQ',
            direction='long',
            entry_price=50.0,
            entry_bar_idx=100,
            entry_time=datetime.now(),
            shares=100,
            stop_loss=49.0,
            target1=51.0,
            target2=52.0,
            or_height=1.0,
            sip_score=3.0,
            gap_pct=0.05,
            confidence=0.7
        )

        assert position.symbol == 'TQQQ'
        assert position.direction == 'long'
        assert position.entry_price == 50.0
        assert position.stop_loss == 49.0
        assert position.target1 == 51.0
        assert position.target2 == 52.0
        assert position.sip_score == 3.0
        assert position.target1_hit == False
        assert position.remaining_pct == 1.0


class TestExitConditions:
    """Test exit condition checking."""

    def test_stop_loss_long(self):
        """Test stop loss detection for long position."""
        strategy = ORBHighVolatilityStrategy()

        position = ORBHVPosition(
            symbol='TQQQ',
            direction='long',
            entry_price=50.0,
            entry_bar_idx=0,
            entry_time=datetime.now(),
            shares=100,
            stop_loss=49.0,
            target1=51.0,
            target2=52.0,
            or_height=1.0,
            sip_score=3.0,
            gap_pct=0.05,
            confidence=0.7,
            current_stop=49.0
        )

        # Price above stop - no exit
        should_exit, exit_type = strategy._check_exit_conditions(
            position=position,
            current_high=50.5,
            current_low=49.5,
            current_close=50.0,
            current_time=time(10, 0)
        )
        assert should_exit == False

        # Price hits stop
        should_exit, exit_type = strategy._check_exit_conditions(
            position=position,
            current_high=50.0,
            current_low=48.5,  # Below stop
            current_close=49.0,
            current_time=time(10, 0)
        )
        assert should_exit == True
        assert exit_type == 'stop_loss'

    def test_target1_long(self):
        """Test target 1 detection for long position."""
        strategy = ORBHighVolatilityStrategy()

        position = ORBHVPosition(
            symbol='TQQQ',
            direction='long',
            entry_price=50.0,
            entry_bar_idx=0,
            entry_time=datetime.now(),
            shares=100,
            stop_loss=49.0,
            target1=51.0,
            target2=52.0,
            or_height=1.0,
            sip_score=3.0,
            gap_pct=0.05,
            confidence=0.7,
            current_stop=49.0
        )

        # Price hits target 1
        should_exit, exit_type = strategy._check_exit_conditions(
            position=position,
            current_high=51.5,  # Above target 1
            current_low=50.5,
            current_close=51.0,
            current_time=time(10, 0)
        )
        assert should_exit == True
        assert exit_type == 'target1'

    def test_eod_exit(self):
        """Test EOD exit."""
        strategy = ORBHighVolatilityStrategy(
            eod_exit_hour=15,
            eod_exit_minute=55
        )

        position = ORBHVPosition(
            symbol='TQQQ',
            direction='long',
            entry_price=50.0,
            entry_bar_idx=0,
            entry_time=datetime.now(),
            shares=100,
            stop_loss=49.0,
            target1=51.0,
            target2=52.0,
            or_height=1.0,
            sip_score=3.0,
            gap_pct=0.05,
            confidence=0.7,
            current_stop=49.0,
            target1_hit=True  # Already hit T1
        )

        # Before EOD - no exit
        should_exit, exit_type = strategy._check_exit_conditions(
            position=position,
            current_high=50.5,
            current_low=49.5,
            current_close=50.0,
            current_time=time(15, 54)
        )
        assert should_exit == False

        # At EOD - force exit
        should_exit, exit_type = strategy._check_exit_conditions(
            position=position,
            current_high=50.5,
            current_low=49.5,
            current_close=50.0,
            current_time=time(15, 55)
        )
        assert should_exit == True
        assert exit_type == 'eod'


class TestRegistration:
    """Test strategy registration."""

    def test_strategy_in_registry(self):
        """Test that strategy is registered."""
        from src.strategies.registry import get_strategy_class, list_strategies

        # Should be in list
        strategies = list_strategies()
        assert 'ORBHighVolatilityStrategy' in strategies

    def test_get_by_class_name(self):
        """Test getting strategy by class name."""
        from src.strategies.registry import get_strategy_class

        cls = get_strategy_class('ORBHighVolatilityStrategy')
        assert cls == ORBHighVolatilityStrategy

    def test_get_by_display_name(self):
        """Test getting strategy by display name."""
        from src.strategies.registry import get_strategy_class

        cls = get_strategy_class('ORB HV')
        assert cls == ORBHighVolatilityStrategy

        cls = get_strategy_class('ORB High Volatility')
        assert cls == ORBHighVolatilityStrategy


class TestSentimentIntegration:
    """Test sentiment integration."""

    def test_sentiment_disabled_by_default(self):
        """Test that sentiment is disabled by default."""
        strategy = ORBHighVolatilityStrategy()
        assert strategy.use_sentiment == False

    def test_sentiment_enabled_parameter(self):
        """Test enabling sentiment via parameter."""
        strategy = ORBHighVolatilityStrategy(
            use_sentiment=True,
            min_sentiment_score=0.3
        )
        assert strategy.use_sentiment == True
        assert strategy.min_sentiment_score == 0.3

    def test_check_sentiment_filter_disabled(self):
        """Test sentiment filter when disabled returns True."""
        strategy = ORBHighVolatilityStrategy(use_sentiment=False)

        passes, score = strategy.check_sentiment_filter(
            symbol='AAPL',
            date=datetime.now(),
            direction='long'
        )

        assert passes == True
        assert score == 0.0

    def test_check_sentiment_filter_long_positive(self):
        """Test sentiment filter for long with positive sentiment."""
        strategy = ORBHighVolatilityStrategy(
            use_sentiment=True,
            min_sentiment_score=0.2
        )

        # Mock sentiment data
        strategy._sentiment_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'timestamp': pd.to_datetime(['2024-01-15 10:00:00'], utc=True),
            'sentiment_score': [0.5],
        })

        passes, score = strategy.check_sentiment_filter(
            symbol='AAPL',
            date=datetime(2024, 1, 15),
            direction='long'
        )

        assert passes == True
        assert score == 0.5

    def test_check_sentiment_filter_long_negative(self):
        """Test sentiment filter rejects long with very negative sentiment."""
        strategy = ORBHighVolatilityStrategy(
            use_sentiment=True,
            min_sentiment_score=0.2
        )

        # Mock negative sentiment data
        strategy._sentiment_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'timestamp': pd.to_datetime(['2024-01-15 10:00:00'], utc=True),
            'sentiment_score': [-0.5],
        })

        passes, score = strategy.check_sentiment_filter(
            symbol='AAPL',
            date=datetime(2024, 1, 15),
            direction='long'
        )

        # -0.5 < -0.2, so should fail
        assert passes == False
        assert score == -0.5

    def test_check_sentiment_filter_short_negative(self):
        """Test sentiment filter for short with negative sentiment."""
        strategy = ORBHighVolatilityStrategy(
            use_sentiment=True,
            min_sentiment_score=0.2
        )

        # Mock negative sentiment data
        strategy._sentiment_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'timestamp': pd.to_datetime(['2024-01-15 10:00:00'], utc=True),
            'sentiment_score': [-0.5],
        })

        passes, score = strategy.check_sentiment_filter(
            symbol='AAPL',
            date=datetime(2024, 1, 15),
            direction='short'
        )

        # For shorts, negative sentiment is good
        assert passes == True
        assert score == -0.5

    def test_get_sentiment_for_date_no_data(self):
        """Test getting sentiment when no data available."""
        strategy = ORBHighVolatilityStrategy(use_sentiment=True)

        score = strategy.get_sentiment_for_date('AAPL', datetime(2024, 1, 15))

        assert score == 0.0

    def test_get_sentiment_for_date_with_data(self):
        """Test getting sentiment with data."""
        strategy = ORBHighVolatilityStrategy(use_sentiment=True)

        # Mock sentiment data with multiple articles on same day
        strategy._sentiment_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT'],
            'timestamp': pd.to_datetime([
                '2024-01-15 09:00:00',
                '2024-01-15 10:00:00',
                '2024-01-15 10:00:00',
            ], utc=True),
            'sentiment_score': [0.6, 0.4, 0.8],
        })

        score = strategy.get_sentiment_for_date('AAPL', datetime(2024, 1, 15))

        # Should average the two AAPL scores: (0.6 + 0.4) / 2 = 0.5
        assert abs(score - 0.5) < 0.001

    def test_confidence_score_with_sentiment(self):
        """Test that sentiment affects confidence score."""
        from src.strategies.advanced.orb_hv_indicators import ORBHVIndicators

        # Without sentiment
        score_no_sentiment = ORBHVIndicators.confidence_score(
            sip_score=3.0,
            sentiment_score=0.0,
            gap_pct=0.05,
            volume_ratio=1.5,
            direction='long'
        )

        # With positive sentiment aligned with long
        score_with_sentiment = ORBHVIndicators.confidence_score(
            sip_score=3.0,
            sentiment_score=0.5,
            gap_pct=0.05,
            volume_ratio=1.5,
            direction='long'
        )

        # Sentiment should increase confidence
        assert score_with_sentiment > score_no_sentiment
