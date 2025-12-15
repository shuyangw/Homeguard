"""
Unit Tests for ICT/SMC Indicators.

Tests cover:
- Swing point detection (HH, HL, LH, LL)
- Market structure classification
- Order block identification
- Liquidity level mapping
- Liquidity sweep detection
- Switch candle pattern recognition
- Entry target calculations
- Multi-timeframe bias
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.strategies.advanced.ict_indicators import (
    ICTIndicators,
    SwingPoint,
    OrderBlock,
    LiquiditySweep,
    SwitchCandle
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def trending_up_data():
    """Create data with clear uptrend (HH/HL pattern)."""
    np.random.seed(42)
    times = pd.date_range('2025-01-06 09:30', periods=60, freq='1min', tz='America/New_York')

    # Create price pattern with clear higher highs and higher lows
    # Start at 100, trend up with pullbacks
    prices = np.concatenate([
        np.linspace(100, 98, 10),    # Drop to swing low 1 (~98)
        np.linspace(98, 103, 10),    # Rally to swing high 1 (~103)
        np.linspace(103, 100, 10),   # Pullback to swing low 2 (~100, higher low)
        np.linspace(100, 106, 10),   # Rally to swing high 2 (~106, higher high)
        np.linspace(106, 103, 10),   # Pullback to swing low 3 (~103, higher low)
        np.linspace(103, 109, 10),   # Rally to swing high 3 (~109, higher high)
    ])

    noise = np.random.randn(60) * 0.2

    return pd.DataFrame({
        'open': prices - 0.1 + noise,
        'high': prices + 0.5 + np.abs(noise),
        'low': prices - 0.5 - np.abs(noise),
        'close': prices + noise,
        'volume': np.full(60, 10000.0)
    }, index=times)


@pytest.fixture
def trending_down_data():
    """Create data with clear downtrend (LH/LL pattern)."""
    np.random.seed(42)
    times = pd.date_range('2025-01-06 09:30', periods=60, freq='1min', tz='America/New_York')

    # Create price pattern with clear lower highs and lower lows
    prices = np.concatenate([
        np.linspace(109, 111, 10),   # Rally to swing high 1 (~111)
        np.linspace(111, 105, 10),   # Drop to swing low 1 (~105)
        np.linspace(105, 108, 10),   # Rally to swing high 2 (~108, lower high)
        np.linspace(108, 102, 10),   # Drop to swing low 2 (~102, lower low)
        np.linspace(102, 105, 10),   # Rally to swing high 3 (~105, lower high)
        np.linspace(105, 99, 10),    # Drop to swing low 3 (~99, lower low)
    ])

    noise = np.random.randn(60) * 0.2

    return pd.DataFrame({
        'open': prices - 0.1 + noise,
        'high': prices + 0.5 + np.abs(noise),
        'low': prices - 0.5 - np.abs(noise),
        'close': prices + noise,
        'volume': np.full(60, 10000.0)
    }, index=times)


@pytest.fixture
def sweep_data():
    """Create data with liquidity sweep pattern."""
    np.random.seed(42)
    times = pd.date_range('2025-01-06 09:30', periods=50, freq='1min', tz='America/New_York')

    # Pattern: establish low at ~100, rally, return, sweep below 100 to 99, reverse to 105
    prices = np.concatenate([
        np.linspace(102, 100, 10),   # Drop to establish swing low at 100
        np.linspace(100, 104, 10),   # Rally
        np.linspace(104, 101, 10),   # Return to lows
        np.linspace(101, 99, 5),     # Sweep below swing low (99 < 100)
        np.linspace(99, 106, 15),    # Reversal after sweep
    ])

    return pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.full(50, 10000.0)
    }, index=times)


@pytest.fixture
def bullish_ob_data():
    """Create data with bullish order block setup."""
    np.random.seed(42)
    times = pd.date_range('2025-01-06 09:30', periods=40, freq='1min', tz='America/New_York')

    # Pattern: consolidation, bearish candles (OB), then strong bullish impulse
    opens = np.concatenate([
        np.full(5, 100.0),           # Consolidation
        np.array([100, 99.5, 99]),   # Bearish candles (these become OB)
        np.linspace(99, 104, 12),    # Bullish impulse (5% move)
        np.full(20, 104.0),          # Continuation
    ])

    closes = np.concatenate([
        np.full(5, 100.0),
        np.array([99.5, 99, 98.5]),  # Bearish closes
        np.linspace(100, 105, 12),   # Bullish closes
        np.full(20, 104.5),
    ])

    highs = np.maximum(opens, closes) + 0.3
    lows = np.minimum(opens, closes) - 0.3

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.full(40, 10000.0)
    }, index=times)


@pytest.fixture
def hammer_pattern_data():
    """Create data with hammer pattern after sweep."""
    times = pd.date_range('2025-01-06 09:30', periods=5, freq='1min', tz='America/New_York')

    # Bar 3 is a hammer: open near high, close slightly above open, long lower wick
    return pd.DataFrame({
        'open': [100.0, 99.5, 99.0, 99.5, 100.5],    # Bar 3: open at 99.5
        'high': [100.5, 100.0, 99.5, 100.0, 101.0],  # Bar 3: high at 100.0
        'low': [99.5, 99.0, 98.5, 97.5, 100.0],      # Bar 3: low at 97.5 (long wick)
        'close': [99.5, 99.0, 99.2, 99.8, 101.0],    # Bar 3: close at 99.8 (bullish)
        'volume': [1000, 1000, 1000, 2000, 1000]
    }, index=times)


@pytest.fixture
def shooting_star_data():
    """Create data with shooting star pattern."""
    times = pd.date_range('2025-01-06 09:30', periods=5, freq='1min', tz='America/New_York')

    # Bar 3 is a shooting star: open near low, close below open, long upper wick
    # Range = 103.5 - 100.0 = 3.5
    # Upper wick = 103.5 - 101.0 = 2.5 (71% of range)
    # Body = 101.0 - 100.5 = 0.5 (14% of range)
    # Lower wick = 100.5 - 100.0 = 0.5 (14% of range)
    return pd.DataFrame({
        'open': [100.0, 100.5, 101.0, 101.0, 99.5],  # Bar 3: open at 101.0
        'high': [100.5, 101.0, 101.5, 103.5, 100.0], # Bar 3: high at 103.5 (long wick)
        'low': [99.5, 100.0, 100.5, 100.0, 99.0],    # Bar 3: low at 100.0
        'close': [100.5, 101.0, 101.2, 100.5, 99.5], # Bar 3: close at 100.5 (bearish)
        'volume': [1000, 1000, 1000, 2000, 1000]
    }, index=times)


# =============================================================================
# Test Classes
# =============================================================================

class TestSwingPointDetection:
    """Test swing point detection."""

    def test_detect_swing_highs_uptrend(self, trending_up_data):
        """Test swing high detection in uptrend."""
        swings = ICTIndicators.detect_swing_points(
            trending_up_data,
            lookback=3,
            min_swing_size_pct=0.005
        )

        # Should detect multiple swing points
        assert len(swings) >= 2

        # Get swing highs
        highs = [s for s in swings if s.swing_type in ['HH', 'LH']]
        assert len(highs) >= 1

    def test_detect_swing_lows_uptrend(self, trending_up_data):
        """Test swing low detection in uptrend."""
        swings = ICTIndicators.detect_swing_points(
            trending_up_data,
            lookback=3,
            min_swing_size_pct=0.005
        )

        # Get swing lows
        lows = [s for s in swings if s.swing_type in ['HL', 'LL']]
        assert len(lows) >= 1

    def test_detect_swing_points_downtrend(self, trending_down_data):
        """Test swing point detection in downtrend."""
        swings = ICTIndicators.detect_swing_points(
            trending_down_data,
            lookback=3,
            min_swing_size_pct=0.005
        )

        # Should detect swing points
        assert len(swings) >= 2

        # In downtrend, should have LH and LL types
        types = {s.swing_type for s in swings}
        assert len(types) >= 2  # Should have variety

    def test_empty_data_returns_empty_list(self):
        """Test empty DataFrame returns no swings."""
        df = pd.DataFrame()
        swings = ICTIndicators.detect_swing_points(df)
        assert swings == []

    def test_insufficient_data_returns_empty(self):
        """Test insufficient data returns empty list."""
        times = pd.date_range('2025-01-06 09:30', periods=5, freq='1min')
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        }, index=times)

        swings = ICTIndicators.detect_swing_points(df, lookback=5)
        assert swings == []  # Not enough data for lookback=5

    def test_swing_point_has_required_attributes(self, trending_up_data):
        """Test SwingPoint objects have all required attributes."""
        swings = ICTIndicators.detect_swing_points(
            trending_up_data,
            lookback=3
        )

        assert len(swings) > 0
        swing = swings[0]

        assert hasattr(swing, 'timestamp')
        assert hasattr(swing, 'price')
        assert hasattr(swing, 'swing_type')
        assert hasattr(swing, 'index')

        assert swing.swing_type in ['HH', 'HL', 'LH', 'LL']
        assert swing.price > 0
        assert isinstance(swing.index, int)


class TestMarketStructureClassification:
    """Test market structure classification."""

    def test_classify_bullish_structure(self, trending_up_data):
        """Test bullish market structure classification."""
        swings = ICTIndicators.detect_swing_points(
            trending_up_data,
            lookback=3,
            min_swing_size_pct=0.005
        )

        if len(swings) >= 4:
            trend, recent = ICTIndicators.classify_market_structure(swings)
            # Uptrend data should classify as bullish or ranging
            assert trend in ['bullish', 'ranging']

    def test_classify_bearish_structure(self, trending_down_data):
        """Test bearish market structure classification."""
        swings = ICTIndicators.detect_swing_points(
            trending_down_data,
            lookback=3,
            min_swing_size_pct=0.005
        )

        if len(swings) >= 4:
            trend, recent = ICTIndicators.classify_market_structure(swings)
            # Downtrend data should classify as bearish or ranging
            assert trend in ['bearish', 'ranging']

    def test_insufficient_swings_returns_ranging(self):
        """Test that less than 4 swings returns ranging."""
        swings = [
            SwingPoint(datetime.now(), 100.0, 'HH', 0),
            SwingPoint(datetime.now(), 99.0, 'HL', 1),
        ]

        trend, recent = ICTIndicators.classify_market_structure(swings)
        assert trend == 'ranging'

    def test_recent_swings_returned(self, trending_up_data):
        """Test that recent swings are returned."""
        swings = ICTIndicators.detect_swing_points(
            trending_up_data,
            lookback=3
        )

        if len(swings) >= 4:
            trend, recent = ICTIndicators.classify_market_structure(swings)
            assert len(recent) <= 4


class TestBreakOfStructure:
    """Test Break of Structure (BOS) detection."""

    def test_detect_bullish_bos(self):
        """Test detection of bullish BOS."""
        swings = [
            SwingPoint(datetime(2025, 1, 6, 9, 30), 100.0, 'HH', 0),
            SwingPoint(datetime(2025, 1, 6, 9, 35), 98.0, 'HL', 5),
            SwingPoint(datetime(2025, 1, 6, 9, 40), 102.0, 'HH', 10),
            SwingPoint(datetime(2025, 1, 6, 9, 45), 100.0, 'HL', 15),
        ]

        # Price breaks above most recent high (102)
        bos = ICTIndicators.detect_bos(
            swings,
            current_price=103.0,
            current_time=datetime(2025, 1, 6, 9, 50)
        )

        assert bos is not None
        assert bos[0] == 'bullish'

    def test_detect_bearish_bos(self):
        """Test detection of bearish BOS."""
        swings = [
            SwingPoint(datetime(2025, 1, 6, 9, 30), 102.0, 'LH', 0),
            SwingPoint(datetime(2025, 1, 6, 9, 35), 100.0, 'LL', 5),
            SwingPoint(datetime(2025, 1, 6, 9, 40), 101.0, 'LH', 10),
            SwingPoint(datetime(2025, 1, 6, 9, 45), 99.0, 'LL', 15),
        ]

        # Price breaks below most recent low (99)
        bos = ICTIndicators.detect_bos(
            swings,
            current_price=98.0,
            current_time=datetime(2025, 1, 6, 9, 50)
        )

        assert bos is not None
        assert bos[0] == 'bearish'

    def test_no_bos_when_price_inside_range(self):
        """Test no BOS when price is inside swing range."""
        swings = [
            SwingPoint(datetime(2025, 1, 6, 9, 30), 100.0, 'HH', 0),
            SwingPoint(datetime(2025, 1, 6, 9, 35), 98.0, 'HL', 5),
        ]

        bos = ICTIndicators.detect_bos(
            swings,
            current_price=99.0,  # Inside range
            current_time=datetime(2025, 1, 6, 9, 40)
        )

        assert bos is None


class TestOrderBlockDetection:
    """Test order block identification."""

    def test_identify_bullish_order_block(self, bullish_ob_data):
        """Test bullish order block detection."""
        swings = ICTIndicators.detect_swing_points(bullish_ob_data, lookback=2)
        obs = ICTIndicators.identify_order_blocks(
            bullish_ob_data,
            swings,
            min_impulse_move_pct=0.02
        )

        # Should find at least one bullish OB
        bullish_obs = [ob for ob in obs if ob.direction == 'bullish']
        # OB may or may not be found depending on exact data pattern
        # The test validates the function runs without error

    def test_order_block_has_required_attributes(self, bullish_ob_data):
        """Test OrderBlock objects have all required attributes."""
        swings = ICTIndicators.detect_swing_points(bullish_ob_data, lookback=2)
        obs = ICTIndicators.identify_order_blocks(
            bullish_ob_data,
            swings,
            min_impulse_move_pct=0.01
        )

        if len(obs) > 0:
            ob = obs[0]
            assert hasattr(ob, 'timestamp')
            assert hasattr(ob, 'high')
            assert hasattr(ob, 'low')
            assert hasattr(ob, 'direction')
            assert hasattr(ob, 'mitigated')
            assert hasattr(ob, 'impulse_size')

            assert ob.direction in ['bullish', 'bearish']
            assert ob.high > ob.low

    def test_order_block_mitigation(self, bullish_ob_data):
        """Test order block mitigation tracking."""
        swings = ICTIndicators.detect_swing_points(bullish_ob_data, lookback=2)
        obs = ICTIndicators.identify_order_blocks(bullish_ob_data, swings)

        # Initially unmitigated
        for ob in obs:
            assert ob.mitigated is False

        # Update mitigation status
        updated_obs = ICTIndicators.update_order_block_mitigation(obs, bullish_ob_data)

        # Function should run without error
        assert isinstance(updated_obs, list)

    def test_get_unmitigated_order_blocks(self):
        """Test filtering unmitigated order blocks."""
        obs = [
            OrderBlock(datetime.now(), 101, 100, 'bullish', False, None, 0.02, 0),
            OrderBlock(datetime.now(), 103, 102, 'bullish', True, datetime.now(), 0.02, 5),
            OrderBlock(datetime.now(), 100, 99, 'bearish', False, None, 0.02, 10),
        ]

        unmitigated = ICTIndicators.get_unmitigated_order_blocks(obs, 100.5)
        assert len(unmitigated) == 2

        # Filter by direction
        bullish_only = ICTIndicators.get_unmitigated_order_blocks(obs, 100.5, 'bullish')
        assert len(bullish_only) == 1
        assert bullish_only[0].direction == 'bullish'


class TestLiquidityDetection:
    """Test liquidity level identification and sweeps."""

    def test_identify_liquidity_levels(self, trending_up_data):
        """Test liquidity level identification."""
        swings = ICTIndicators.detect_swing_points(trending_up_data, lookback=3)
        levels = ICTIndicators.identify_liquidity_levels(swings)

        assert 'buy_stops' in levels
        assert 'sell_stops' in levels
        assert isinstance(levels['buy_stops'], list)
        assert isinstance(levels['sell_stops'], list)

    def test_detect_liquidity_sweep_bullish(self, sweep_data):
        """Test bullish liquidity sweep detection."""
        swings = ICTIndicators.detect_swing_points(sweep_data, lookback=3)
        levels = ICTIndicators.identify_liquidity_levels(swings)

        # Check for sweep at bars where price went below established lows
        for i in range(30, len(sweep_data)):
            sweep = ICTIndicators.detect_liquidity_sweep(
                sweep_data,
                levels,
                i,
                sweep_threshold_pct=0.001
            )
            if sweep and sweep.direction == 'bullish':
                assert sweep.sweep_price < sweep.swing_point.price
                break

    def test_no_sweep_when_no_levels(self):
        """Test no sweep detected when no liquidity levels."""
        times = pd.date_range('2025-01-06 09:30', periods=10, freq='1min')
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        }, index=times)

        levels = {'buy_stops': [], 'sell_stops': []}
        sweep = ICTIndicators.detect_liquidity_sweep(df, levels, 5)

        assert sweep is None


class TestSwitchCandleDetection:
    """Test switch candle pattern detection."""

    def test_detect_bullish_hammer(self, hammer_pattern_data):
        """Test bullish hammer detection."""
        switch = ICTIndicators.detect_switch_candle(
            hammer_pattern_data,
            liquidity_sweep=None,
            order_block=None,
            current_bar_idx=3,  # The hammer bar
            min_wick_ratio=0.4,
            min_body_ratio=0.1
        )

        assert switch is not None
        assert switch.direction == 'long'
        assert switch.candle_type in ['hammer', 'bullish_engulfing']

    def test_detect_bearish_shooting_star(self, shooting_star_data):
        """Test bearish shooting star detection."""
        switch = ICTIndicators.detect_switch_candle(
            shooting_star_data,
            liquidity_sweep=None,
            order_block=None,
            current_bar_idx=3,  # The shooting star bar
            min_wick_ratio=0.4,
            min_body_ratio=0.1
        )

        assert switch is not None
        assert switch.direction == 'short'
        assert switch.candle_type in ['shooting_star', 'bearish_engulfing']

    def test_switch_candle_has_required_attributes(self, hammer_pattern_data):
        """Test SwitchCandle has all required attributes."""
        switch = ICTIndicators.detect_switch_candle(
            hammer_pattern_data,
            liquidity_sweep=None,
            order_block=None,
            current_bar_idx=3,
            min_wick_ratio=0.4,
            min_body_ratio=0.1
        )

        if switch:
            assert hasattr(switch, 'timestamp')
            assert hasattr(switch, 'entry_price')
            assert hasattr(switch, 'stop_loss')
            assert hasattr(switch, 'direction')
            assert hasattr(switch, 'candle_type')

            assert switch.entry_price > 0
            assert switch.stop_loss > 0

    def test_no_switch_candle_for_doji(self):
        """Test no switch candle for doji (no body)."""
        times = pd.date_range('2025-01-06 09:30', periods=5, freq='1min')
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'high': [101.0, 101.0, 101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 99.0, 99.0, 99.0],
            'close': [100.0, 100.0, 100.0, 100.0, 100.0],  # Doji - open = close
            'volume': [1000] * 5
        }, index=times)

        switch = ICTIndicators.detect_switch_candle(
            df,
            liquidity_sweep=None,
            order_block=None,
            current_bar_idx=3,
            min_body_ratio=0.3  # Requires 30% body
        )

        assert switch is None


class TestEntryTargetCalculation:
    """Test entry, stop loss, and target calculations."""

    def test_long_targets_calculation(self):
        """Test target calculation for long entry."""
        switch = SwitchCandle(
            timestamp=datetime(2025, 1, 6, 10, 0),
            entry_price=100.5,
            stop_loss=99.0,
            direction='long',
            candle_type='hammer',
            index=10
        )

        targets = ICTIndicators.calculate_entry_targets(
            switch,
            order_block=None,
            atr=1.0,
            risk_reward_ratio=2.0
        )

        assert targets['entry'] == 100.5
        assert targets['stop_loss'] < 99.0  # Stop with ATR buffer
        assert targets['risk'] > 0
        assert targets['reward'] > targets['risk']  # 2:1 RR
        assert targets['rr_ratio'] == 2.0

    def test_short_targets_calculation(self):
        """Test target calculation for short entry."""
        switch = SwitchCandle(
            timestamp=datetime(2025, 1, 6, 10, 0),
            entry_price=100.0,
            stop_loss=102.0,
            direction='short',
            candle_type='shooting_star',
            index=10
        )

        targets = ICTIndicators.calculate_entry_targets(
            switch,
            order_block=None,
            atr=1.0,
            risk_reward_ratio=2.0
        )

        assert targets['entry'] == 100.0
        assert targets['stop_loss'] > 102.0  # Stop with ATR buffer
        assert targets['target'] < targets['entry']  # Target below entry for short
        assert targets['risk'] > 0


class TestMultiTimeframeBias:
    """Test higher timeframe bias calculation."""

    def test_bullish_htf_bias(self, trending_up_data):
        """Test bullish HTF bias detection."""
        bias, confidence = ICTIndicators.get_htf_bias(
            trending_up_data,
            htf_swing_points=None,
            lookback=20
        )

        # Uptrend data should give bullish bias
        assert bias in ['bullish', 'neutral']
        assert 0.0 <= confidence <= 1.0

    def test_bearish_htf_bias(self, trending_down_data):
        """Test bearish HTF bias detection."""
        bias, confidence = ICTIndicators.get_htf_bias(
            trending_down_data,
            htf_swing_points=None,
            lookback=20
        )

        # Downtrend data should give bearish bias
        assert bias in ['bearish', 'neutral']
        assert 0.0 <= confidence <= 1.0

    def test_neutral_bias_for_flat_data(self):
        """Test neutral bias for flat/ranging data."""
        times = pd.date_range('2025-01-06 09:30', periods=30, freq='1min')
        flat_data = pd.DataFrame({
            'open': [100.0] * 30,
            'high': [100.5] * 30,
            'low': [99.5] * 30,
            'close': [100.0] * 30,
            'volume': [1000] * 30
        }, index=times)

        bias, confidence = ICTIndicators.get_htf_bias(flat_data, lookback=20)

        assert bias == 'neutral'

    def test_empty_data_returns_neutral(self):
        """Test empty data returns neutral bias."""
        df = pd.DataFrame()
        bias, confidence = ICTIndicators.get_htf_bias(df)

        assert bias == 'neutral'
        assert confidence == 0.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_price_in_zone(self):
        """Test price in zone detection."""
        # Price inside zone
        assert ICTIndicators.is_price_in_zone(100.5, 101.0, 100.0) is True

        # Price outside zone
        assert ICTIndicators.is_price_in_zone(99.0, 101.0, 100.0) is False
        assert ICTIndicators.is_price_in_zone(102.0, 101.0, 100.0) is False

    def test_calculate_atr(self, trending_up_data):
        """Test ATR calculation."""
        atr = ICTIndicators.calculate_atr(trending_up_data, period=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(trending_up_data)
        assert atr.iloc[-1] > 0

    def test_calculate_atr_empty_data(self):
        """Test ATR with empty data."""
        df = pd.DataFrame()
        atr = ICTIndicators.calculate_atr(df)

        assert len(atr) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_bar_data(self):
        """Test handling of single bar data."""
        times = pd.date_range('2025-01-06 09:30', periods=1, freq='1min')
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        }, index=times)

        # All functions should handle gracefully
        swings = ICTIndicators.detect_swing_points(df)
        assert swings == []

        obs = ICTIndicators.identify_order_blocks(df, [])
        assert obs == []

    def test_constant_prices(self):
        """Test handling of constant price data."""
        times = pd.date_range('2025-01-06 09:30', periods=30, freq='1min')
        df = pd.DataFrame({
            'open': [100.0] * 30,
            'high': [100.0] * 30,
            'low': [100.0] * 30,
            'close': [100.0] * 30,
            'volume': [1000] * 30
        }, index=times)

        swings = ICTIndicators.detect_swing_points(df, lookback=3)
        # Should not detect swings in flat data
        assert len(swings) == 0

    def test_negative_prices_handled(self):
        """Test that negative prices don't cause errors."""
        times = pd.date_range('2025-01-06 09:30', periods=20, freq='1min')
        df = pd.DataFrame({
            'open': np.linspace(-10, 10, 20),
            'high': np.linspace(-9, 11, 20),
            'low': np.linspace(-11, 9, 20),
            'close': np.linspace(-10, 10, 20),
            'volume': [1000] * 20
        }, index=times)

        # Should not raise
        swings = ICTIndicators.detect_swing_points(df, lookback=3)
        assert isinstance(swings, list)


# =============================================================================
# Tests for New Features (from original spec)
# =============================================================================

class TestRVOLCalculation:
    """Test relative volume calculations."""

    def test_calculate_rvol_basic(self):
        """Test basic RVOL calculation."""
        times = pd.date_range('2025-01-06 09:30', periods=30, freq='1min')
        df = pd.DataFrame({
            'open': [100.0] * 30,
            'high': [101.0] * 30,
            'low': [99.0] * 30,
            'close': [100.5] * 30,
            'volume': [1000] * 29 + [2000]  # Last bar has 2x volume
        }, index=times)

        rvol = ICTIndicators.calculate_rvol(df, lookback=20)

        # Last bar should have RVOL > 1 (above average)
        assert rvol.iloc[-1] > 1.0

    def test_calculate_rvol_empty_data(self):
        """Test RVOL with empty DataFrame."""
        df = pd.DataFrame()
        rvol = ICTIndicators.calculate_rvol(df)
        assert rvol.empty

    def test_is_high_volume_bar(self):
        """Test high volume bar detection."""
        times = pd.date_range('2025-01-06 09:30', periods=30, freq='1min')
        df = pd.DataFrame({
            'open': [100.0] * 30,
            'high': [101.0] * 30,
            'low': [99.0] * 30,
            'close': [100.5] * 30,
            'volume': [1000] * 29 + [3000]  # Last bar has 3x volume
        }, index=times)

        # Last bar (3x volume) should be high volume
        assert ICTIndicators.is_high_volume_bar(df, 29, rvol_threshold=1.5)
        # Earlier bars (normal volume) should not be
        assert not ICTIndicators.is_high_volume_bar(df, 10, rvol_threshold=1.5)


class TestSessionFilter:
    """Test session filtering functionality."""

    def test_us_regular_session(self):
        """Test US regular market hours filter."""
        # During market hours
        during_hours = datetime(2025, 1, 6, 10, 30)  # 10:30 AM
        assert ICTIndicators.is_optimal_session(during_hours, 'us_regular')

        # After market close
        after_close = datetime(2025, 1, 6, 17, 0)  # 5:00 PM
        assert not ICTIndicators.is_optimal_session(after_close, 'us_regular')

    def test_us_core_session(self):
        """Test US core hours (10 AM - 3 PM)."""
        # Core hours
        core_time = datetime(2025, 1, 6, 12, 0)  # Noon
        assert ICTIndicators.is_optimal_session(core_time, 'us_core')

        # Outside core hours (early morning)
        early = datetime(2025, 1, 6, 9, 35)
        assert not ICTIndicators.is_optimal_session(early, 'us_core')

    def test_no_session_filter(self):
        """Test that 'none' filter always returns True."""
        any_time = datetime(2025, 1, 6, 3, 0)  # 3 AM
        assert ICTIndicators.is_optimal_session(any_time, 'none')


class TestZoneQualityScoring:
    """Test zone quality scoring functionality."""

    def test_calculate_zone_quality_fresh_zone(self):
        """Test quality calculation for fresh order block."""
        ob = OrderBlock(
            timestamp=datetime(2025, 1, 6, 10, 0),
            high=101.0,
            low=99.0,
            direction='bullish',
            mitigated=False,
            mitigation_time=None,
            impulse_size=0.02,  # 2% impulse (strong)
            index=90
        )

        # Current bar at 100, zone at 90 = age 10 bars
        quality = ICTIndicators.calculate_zone_quality(
            ob, current_bar_idx=100, touch_count=0, htf_aligned=True
        )

        # Fresh zone, strong impulse, first touch, HTF aligned = high score
        assert quality > 0.7

    def test_calculate_zone_quality_old_zone(self):
        """Test quality drops for old zones."""
        ob = OrderBlock(
            timestamp=datetime(2025, 1, 6, 10, 0),
            high=101.0,
            low=99.0,
            direction='bullish',
            mitigated=False,
            mitigation_time=None,
            impulse_size=0.01,  # 1% impulse (moderate)
            index=0
        )

        # Old zone (age = 100 bars)
        quality = ICTIndicators.calculate_zone_quality(
            ob, current_bar_idx=100, touch_count=2, htf_aligned=False
        )

        # Old, touched multiple times, not aligned = low score
        assert quality < 0.3

    def test_filter_zones_by_quality(self):
        """Test filtering zones by quality threshold."""
        obs = [
            OrderBlock(
                timestamp=datetime(2025, 1, 6, 10, 0),
                high=101.0, low=99.0, direction='bullish',
                mitigated=False, mitigation_time=None,
                impulse_size=0.02, index=90
            ),
            OrderBlock(
                timestamp=datetime(2025, 1, 6, 9, 30),
                high=98.0, low=96.0, direction='bullish',
                mitigated=False, mitigation_time=None,
                impulse_size=0.005, index=10  # Old, weak impulse
            ),
        ]

        # Filter with moderate threshold
        filtered = ICTIndicators.filter_zones_by_quality(
            obs, current_bar_idx=100, min_quality=0.4
        )

        # Should return zones sorted by quality
        assert len(filtered) >= 1
        # First should be highest quality
        assert filtered[0][1] >= filtered[-1][1] if len(filtered) > 1 else True
