"""
Unit Tests for ICT/SMC Strategy.

Tests cover:
- Strategy initialization and parameter validation
- Signal generation for reversal and continuation setups
- Exit logic (stop loss, target, time exit)
- Regime filtering
- HTF alignment filtering
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.strategies.advanced.ict_strategy import ICTStrategy, ICTPosition
from src.strategies.advanced.ict_indicators import (
    SwingPoint,
    OrderBlock,
    LiquiditySweep,
    SwitchCandle
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def create_day_data():
    """Factory to create minute data for a trading day."""
    def _create(
        date_str: str = '2025-01-06',
        base_price: float = 100.0,
        pattern: str = 'sweep_reversal',
        n_bars: int = 390  # Full trading day
    ):
        times = pd.date_range(
            f'{date_str} 09:30:00',
            periods=n_bars,
            freq='1min',
            tz='America/New_York'
        )

        if pattern == 'sweep_reversal':
            # Create sweep and reversal pattern
            prices = np.concatenate([
                np.linspace(base_price, base_price + 2, 30),   # Up
                np.linspace(base_price + 2, base_price - 1, 30),  # Down to low
                np.linspace(base_price - 1, base_price + 1, 30),  # Rally
                np.linspace(base_price + 1, base_price - 1.5, 30),  # Sweep low
                np.linspace(base_price - 1.5, base_price + 4, 30),  # Reversal
                np.linspace(base_price + 4, base_price + 6, n_bars - 150),  # Continuation
            ])
        elif pattern == 'trending_up':
            prices = np.linspace(base_price, base_price + 10, n_bars)
            # Add some noise
            prices += np.sin(np.linspace(0, 20, n_bars)) * 0.5
        elif pattern == 'trending_down':
            prices = np.linspace(base_price, base_price - 10, n_bars)
            prices += np.sin(np.linspace(0, 20, n_bars)) * 0.5
        elif pattern == 'flat':
            prices = np.full(n_bars, base_price)
            prices += np.random.randn(n_bars) * 0.1
        else:
            prices = np.full(n_bars, base_price)

        return pd.DataFrame({
            'open': prices - 0.1,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.full(n_bars, 10000.0),
            'vwap': prices,
        }, index=times)

    return _create


@pytest.fixture
def simple_price_data():
    """Simple price data for basic tests."""
    np.random.seed(42)
    times = pd.date_range('2025-01-06 09:30', periods=100, freq='1min', tz='America/New_York')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)

    return pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.full(100, 10000.0)
    }, index=times)


# =============================================================================
# Test Classes
# =============================================================================

class TestICTStrategyInit:
    """Test ICTStrategy initialization."""

    def test_initialization_defaults(self):
        """Test default parameter initialization."""
        strategy = ICTStrategy()

        assert strategy.trade_type == 'both'
        assert strategy.swing_lookback == 5
        assert strategy.risk_reward_ratio == 2.0
        assert strategy.use_htf_filter is True
        assert strategy.use_regime is True
        assert strategy.long_only is False
        assert strategy.max_positions_per_day == 3

    def test_initialization_reversal_mode(self):
        """Test reversal mode initialization."""
        strategy = ICTStrategy(
            trade_type='reversal',
            risk_reward_ratio=3.0,
            long_only=True
        )

        assert strategy.trade_type == 'reversal'
        assert strategy.risk_reward_ratio == 3.0
        assert strategy.long_only is True

    def test_initialization_continuation_mode(self):
        """Test continuation mode initialization."""
        strategy = ICTStrategy(
            trade_type='continuation',
            risk_reward_ratio=1.5
        )

        assert strategy.trade_type == 'continuation'
        assert strategy.risk_reward_ratio == 1.5

    def test_invalid_trade_type(self):
        """Test invalid trade type raises error."""
        with pytest.raises(ValueError, match="trade_type"):
            ICTStrategy(trade_type='invalid')

    def test_invalid_swing_lookback(self):
        """Test invalid swing lookback raises error."""
        with pytest.raises(ValueError, match="swing_lookback"):
            ICTStrategy(swing_lookback=1)

    def test_invalid_risk_reward_ratio(self):
        """Test invalid risk/reward ratio raises error."""
        with pytest.raises(ValueError, match="risk_reward_ratio"):
            ICTStrategy(risk_reward_ratio=-1.0)

    def test_invalid_min_swing_size(self):
        """Test invalid min swing size raises error."""
        with pytest.raises(ValueError, match="min_swing_size_pct"):
            ICTStrategy(min_swing_size_pct=-0.01)

    def test_get_parameters(self):
        """Test get_parameters returns correct dict."""
        strategy = ICTStrategy(trade_type='continuation', risk_reward_ratio=2.5)
        params = strategy.get_parameters()

        assert params['trade_type'] == 'continuation'
        assert params['risk_reward_ratio'] == 2.5
        assert 'swing_lookback' in params
        assert 'use_htf_filter' in params
        assert 'use_regime' in params


class TestICTSignalGeneration:
    """Test ICT signal generation."""

    def test_empty_data_returns_empty_signals(self):
        """Test empty DataFrame returns empty signals."""
        strategy = ICTStrategy()
        df = pd.DataFrame()

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(df)

        assert len(long_e) == 0
        assert len(long_x) == 0
        assert len(short_e) == 0
        assert len(short_x) == 0

    def test_signals_are_boolean_series(self, simple_price_data):
        """Test that signals are boolean Series."""
        strategy = ICTStrategy(use_regime=False, use_htf_filter=False)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(
            simple_price_data
        )

        assert isinstance(long_e, pd.Series)
        assert isinstance(long_x, pd.Series)
        assert isinstance(short_e, pd.Series)
        assert isinstance(short_x, pd.Series)

        # Check dtype is bool
        assert long_e.dtype == bool
        assert long_x.dtype == bool

    def test_signals_same_length_as_data(self, simple_price_data):
        """Test signals have same length as input data."""
        strategy = ICTStrategy(use_regime=False, use_htf_filter=False)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(
            simple_price_data
        )

        assert len(long_e) == len(simple_price_data)
        assert len(long_x) == len(simple_price_data)
        assert len(short_e) == len(simple_price_data)
        assert len(short_x) == len(simple_price_data)

    def test_long_only_blocks_shorts(self, simple_price_data):
        """Test long_only mode blocks short entries."""
        strategy = ICTStrategy(
            trade_type='both',
            long_only=True,
            use_regime=False,
            use_htf_filter=False
        )

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(
            simple_price_data
        )

        # Short entries should all be False
        assert not short_e.any()

    def test_reversal_only_mode(self, create_day_data):
        """Test reversal only mode."""
        strategy = ICTStrategy(
            trade_type='reversal',
            use_regime=False,
            use_htf_filter=False
        )

        df = create_day_data(pattern='sweep_reversal')
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(df)

        # Strategy should process without error
        assert isinstance(long_e, pd.Series)

    def test_continuation_only_mode(self, create_day_data):
        """Test continuation only mode."""
        strategy = ICTStrategy(
            trade_type='continuation',
            use_regime=False,
            use_htf_filter=False
        )

        df = create_day_data(pattern='trending_up')
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(df)

        # Strategy should process without error
        assert isinstance(long_e, pd.Series)


class TestICTExits:
    """Test ICT exit logic."""

    @pytest.fixture
    def long_position(self):
        """Create a sample long position."""
        return ICTPosition(
            symbol='SPY',
            direction='long',
            entry_price=100.0,
            entry_time=datetime(2025, 1, 6, 10, 0),
            stop_loss=98.0,
            target=104.0,
            order_block=None,
            liquidity_sweep=None,
            switch_candle=SwitchCandle(
                timestamp=datetime(2025, 1, 6, 10, 0),
                entry_price=100.0,
                stop_loss=98.0,
                direction='long',
                candle_type='hammer',
                index=30
            ),
            trade_type='reversal'
        )

    @pytest.fixture
    def short_position(self):
        """Create a sample short position."""
        return ICTPosition(
            symbol='SPY',
            direction='short',
            entry_price=100.0,
            entry_time=datetime(2025, 1, 6, 10, 0),
            stop_loss=102.0,
            target=96.0,
            order_block=None,
            liquidity_sweep=None,
            switch_candle=SwitchCandle(
                timestamp=datetime(2025, 1, 6, 10, 0),
                entry_price=100.0,
                stop_loss=102.0,
                direction='short',
                candle_type='shooting_star',
                index=30
            ),
            trade_type='reversal'
        )

    def test_stop_loss_exit_long(self, long_position):
        """Test stop-loss exit for long position."""
        strategy = ICTStrategy()

        should_exit, reason = strategy._check_exit(
            long_position,
            current_close=97.5,
            current_low=97.0,  # Hit stop at 98
            current_high=99.0,
            current_time=time(11, 0)
        )

        assert should_exit is True
        assert reason == 'stop_loss'

    def test_stop_loss_exit_short(self, short_position):
        """Test stop-loss exit for short position."""
        strategy = ICTStrategy()

        should_exit, reason = strategy._check_exit(
            short_position,
            current_close=102.5,
            current_low=101.0,
            current_high=103.0,  # Hit stop at 102
            current_time=time(11, 0)
        )

        assert should_exit is True
        assert reason == 'stop_loss'

    def test_target_exit_long(self, long_position):
        """Test target exit for long position."""
        strategy = ICTStrategy()

        should_exit, reason = strategy._check_exit(
            long_position,
            current_close=104.5,
            current_low=103.5,
            current_high=105.0,  # Hit target at 104
            current_time=time(11, 0)
        )

        assert should_exit is True
        assert reason == 'target'

    def test_target_exit_short(self, short_position):
        """Test target exit for short position."""
        strategy = ICTStrategy()

        should_exit, reason = strategy._check_exit(
            short_position,
            current_close=95.5,
            current_low=95.0,  # Hit target at 96
            current_high=96.5,
            current_time=time(11, 0)
        )

        assert should_exit is True
        assert reason == 'target'

    def test_time_exit(self, long_position):
        """Test forced time exit."""
        strategy = ICTStrategy(exit_time_hour=15, exit_time_minute=45)

        should_exit, reason = strategy._check_exit(
            long_position,
            current_close=101.0,
            current_low=100.5,
            current_high=101.5,
            current_time=time(15, 45)  # At exit time
        )

        assert should_exit is True
        assert reason == 'time_exit'

    def test_no_exit_when_in_range(self, long_position):
        """Test no exit when price is between stop and target."""
        strategy = ICTStrategy()

        should_exit, reason = strategy._check_exit(
            long_position,
            current_close=101.0,
            current_low=99.0,  # Above stop (98)
            current_high=102.0,  # Below target (104)
            current_time=time(11, 0)
        )

        assert should_exit is False
        assert reason == ''


class TestRegimeFiltering:
    """Test regime-based filtering."""

    def test_bear_regime_blocks_longs(self):
        """Test BEAR regime blocks long trades."""
        strategy = ICTStrategy(use_regime=True)
        strategy.set_regime('BEAR')

        assert strategy._passes_regime_filter('long') is False
        assert strategy._passes_regime_filter('short') is True

    def test_strong_bull_blocks_shorts(self):
        """Test STRONG_BULL regime blocks short trades."""
        strategy = ICTStrategy(use_regime=True)
        strategy.set_regime('STRONG_BULL')

        assert strategy._passes_regime_filter('long') is True
        assert strategy._passes_regime_filter('short') is False

    def test_sideways_allows_both(self):
        """Test SIDEWAYS regime allows both directions."""
        strategy = ICTStrategy(use_regime=True)
        strategy.set_regime('SIDEWAYS')

        assert strategy._passes_regime_filter('long') is True
        assert strategy._passes_regime_filter('short') is True

    def test_regime_filter_disabled(self):
        """Test regime filter can be disabled."""
        strategy = ICTStrategy(use_regime=False)
        strategy.set_regime('BEAR')

        # Should allow both regardless of regime
        assert strategy._passes_regime_filter('long') is True
        assert strategy._passes_regime_filter('short') is True


class TestHTFFiltering:
    """Test higher timeframe alignment filtering."""

    def test_htf_bullish_allows_longs(self):
        """Test bullish HTF bias allows long trades."""
        strategy = ICTStrategy(use_htf_filter=True)

        assert strategy._passes_htf_filter('long', 'bullish') is True
        assert strategy._passes_htf_filter('short', 'bullish') is False

    def test_htf_bearish_allows_shorts(self):
        """Test bearish HTF bias allows short trades."""
        strategy = ICTStrategy(use_htf_filter=True)

        assert strategy._passes_htf_filter('long', 'bearish') is False
        assert strategy._passes_htf_filter('short', 'bearish') is True

    def test_htf_neutral_allows_both(self):
        """Test neutral HTF bias allows both directions."""
        strategy = ICTStrategy(use_htf_filter=True)

        assert strategy._passes_htf_filter('long', 'neutral') is True
        assert strategy._passes_htf_filter('short', 'neutral') is True

    def test_htf_filter_disabled(self):
        """Test HTF filter can be disabled."""
        strategy = ICTStrategy(use_htf_filter=False)

        # Should allow both regardless of HTF bias
        assert strategy._passes_htf_filter('long', 'bearish') is True
        assert strategy._passes_htf_filter('short', 'bullish') is True


class TestDailyTradeLimit:
    """Test daily trade limit functionality."""

    def test_max_positions_per_day_parameter(self):
        """Test max positions per day can be set."""
        strategy = ICTStrategy(max_positions_per_day=5)
        assert strategy.max_positions_per_day == 5

    def test_reset_state_clears_daily_trades(self):
        """Test reset_state clears daily trade counter."""
        strategy = ICTStrategy()
        strategy._daily_trades = {'2025-01-06': 3}

        strategy.reset_state()

        assert strategy._daily_trades == {}


class TestStateManagement:
    """Test internal state management."""

    def test_reset_state(self):
        """Test reset_state clears all internal state."""
        strategy = ICTStrategy()

        # Add some state
        strategy._swing_points = [SwingPoint(datetime.now(), 100.0, 'HH', 0)]
        strategy._order_blocks = [OrderBlock(
            datetime.now(), 101, 100, 'bullish', False, None, 0.02, 0
        )]
        strategy._liquidity_levels = {'buy_stops': [100.0], 'sell_stops': [99.0]}
        strategy._daily_trades = {'2025-01-06': 2}

        # Reset
        strategy.reset_state()

        # Verify all cleared
        assert strategy._swing_points == []
        assert strategy._order_blocks == []
        assert strategy._liquidity_levels == {}
        assert strategy._daily_trades == {}

    def test_set_regime(self):
        """Test set_regime updates current regime."""
        strategy = ICTStrategy()

        strategy.set_regime('BEAR')
        assert strategy.current_regime == 'BEAR'

        strategy.set_regime('STRONG_BULL')
        assert strategy.current_regime == 'STRONG_BULL'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        strategy = ICTStrategy()

        times = pd.date_range('2025-01-06 09:30', periods=10, freq='1min')
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        }, index=times)

        # Should not raise, but may not generate signals
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(df)
        assert len(long_e) == 10

    def test_flat_price_data(self, create_day_data):
        """Test handling of flat price data."""
        strategy = ICTStrategy(use_regime=False, use_htf_filter=False)

        df = create_day_data(pattern='flat', n_bars=100)
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(df)

        # Should complete without error
        assert isinstance(long_e, pd.Series)

    def test_data_without_timezone(self):
        """Test handling of data without timezone."""
        strategy = ICTStrategy(use_regime=False, use_htf_filter=False)

        # Create data without timezone
        times = pd.date_range('2025-01-06 09:30', periods=100, freq='1min')
        df = pd.DataFrame({
            'open': np.linspace(100, 105, 100),
            'high': np.linspace(100.5, 105.5, 100),
            'low': np.linspace(99.5, 104.5, 100),
            'close': np.linspace(100, 105, 100),
            'volume': [1000] * 100
        }, index=times)

        # Should handle gracefully
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(df)
        assert isinstance(long_e, pd.Series)

    def test_strategy_repr(self):
        """Test strategy string representation."""
        strategy = ICTStrategy(trade_type='reversal')
        repr_str = repr(strategy)

        assert 'ICTStrategy' in repr_str
        assert 'trade_type' in repr_str


class TestIntegrationWithBacktest:
    """Integration tests with backtest patterns."""

    def test_strategy_follows_long_short_interface(self, simple_price_data):
        """Test strategy follows LongShortStrategy interface."""
        strategy = ICTStrategy()

        # Should have generate_signals method from parent
        assert hasattr(strategy, 'generate_signals')

        # generate_long_short_signals should work
        result = strategy.generate_long_short_signals(simple_price_data)
        assert len(result) == 4

    def test_parameters_stored_in_params_dict(self):
        """Test parameters are stored in params dict."""
        strategy = ICTStrategy(
            trade_type='reversal',
            swing_lookback=10
        )

        # Check params dict
        assert 'trade_type' in strategy.params
        assert strategy.params['trade_type'] == 'reversal'
        assert strategy.params['swing_lookback'] == 10
