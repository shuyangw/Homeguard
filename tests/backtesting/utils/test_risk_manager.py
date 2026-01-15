"""
Tests for risk_manager.py - Position, StopLoss, and RiskManager classes.

This test module covers:
- Position dataclass and P&L calculations
- PercentageStopLoss for fixed percentage stops
- ATRStopLoss for volatility-based trailing stops
- TimeStopLoss for time-based exits
- ProfitTargetStopLoss for combined take-profit/stop-loss
- RiskManager for portfolio-level risk constraints
"""

import pytest
import pandas as pd
import numpy as np

from src.backtesting.utils.risk_manager import (
    Position,
    PercentageStopLoss,
    ATRStopLoss,
    TimeStopLoss,
    ProfitTargetStopLoss,
    RiskManager,
)
from src.backtesting.utils.risk_config import RiskConfig


# =============================================================================
# Position Tests
# =============================================================================

class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation_long(self):
        """Test creating a long position."""
        pos = Position(
            symbol='AAPL',
            entry_price=150.0,
            shares=100,
            entry_bar=0
        )
        assert pos.symbol == 'AAPL'
        assert pos.entry_price == 150.0
        assert pos.shares == 100
        assert pos.entry_bar == 0
        assert pos.is_short is False

    def test_position_creation_short(self):
        """Test creating a short position."""
        pos = Position(
            symbol='AAPL',
            entry_price=150.0,
            shares=-100,
            entry_bar=0
        )
        assert pos.shares == -100
        assert pos.is_short is True

    def test_highest_price_initialized(self):
        """Test highest_price is auto-initialized to entry_price."""
        pos = Position('AAPL', 150.0, 100, 0)
        assert pos.highest_price == 150.0

    def test_highest_price_custom(self):
        """Test highest_price can be set explicitly."""
        pos = Position('AAPL', 150.0, 100, 0, highest_price=160.0)
        assert pos.highest_price == 160.0

    def test_position_value_long(self):
        """Test position value calculation for long position."""
        pos = Position('AAPL', 150.0, 100, 0)
        assert pos.position_value == 15000.0

    def test_position_value_short(self):
        """Test position value calculation for short position."""
        pos = Position('AAPL', 150.0, -100, 0)
        # position_value uses abs(shares)
        assert pos.position_value == 15000.0

    def test_unrealized_pnl_long_profit(self):
        """Test unrealized P&L for long position with profit."""
        pos = Position('AAPL', 150.0, 100, 0)
        pnl = pos.unrealized_pnl(160.0)  # Price up $10
        assert pnl == 1000.0  # 100 shares × $10

    def test_unrealized_pnl_long_loss(self):
        """Test unrealized P&L for long position with loss."""
        pos = Position('AAPL', 150.0, 100, 0)
        pnl = pos.unrealized_pnl(140.0)  # Price down $10
        assert pnl == -1000.0

    def test_unrealized_pnl_short_profit(self):
        """Test unrealized P&L for short position with profit."""
        pos = Position('AAPL', 150.0, -100, 0)
        pnl = pos.unrealized_pnl(140.0)  # Price down $10 = profit for short
        assert pnl == 1000.0

    def test_unrealized_pnl_short_loss(self):
        """Test unrealized P&L for short position with loss."""
        pos = Position('AAPL', 150.0, -100, 0)
        pnl = pos.unrealized_pnl(160.0)  # Price up $10 = loss for short
        assert pnl == -1000.0

    def test_unrealized_pnl_pct_long(self):
        """Test unrealized P&L percentage for long position."""
        pos = Position('AAPL', 100.0, 100, 0)
        pnl_pct = pos.unrealized_pnl_pct(110.0)  # 10% gain
        assert abs(pnl_pct - 0.10) < 1e-6

    def test_unrealized_pnl_pct_short(self):
        """Test unrealized P&L percentage for short position."""
        pos = Position('AAPL', 100.0, -100, 0)
        pnl_pct = pos.unrealized_pnl_pct(90.0)  # 10% drop = profit for short
        assert abs(pnl_pct - 0.10) < 1e-6

    def test_unrealized_pnl_pct_zero_entry(self):
        """Test P&L percentage returns 0 when entry price is 0."""
        pos = Position('AAPL', 0.0, 100, 0)
        pnl_pct = pos.unrealized_pnl_pct(10.0)
        assert pnl_pct == 0.0


# =============================================================================
# PercentageStopLoss Tests
# =============================================================================

class TestPercentageStopLoss:
    """Tests for PercentageStopLoss class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        stop = PercentageStopLoss(stop_loss_pct=0.02)
        assert stop.stop_loss_pct == 0.02

    def test_initialization_edge_case_max(self):
        """Test initialization at max value (100%)."""
        stop = PercentageStopLoss(stop_loss_pct=1.0)
        assert stop.stop_loss_pct == 1.0

    def test_initialization_invalid_zero(self):
        """Test that zero is rejected."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PercentageStopLoss(stop_loss_pct=0.0)

    def test_initialization_invalid_negative(self):
        """Test that negative value is rejected."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PercentageStopLoss(stop_loss_pct=-0.02)

    def test_initialization_invalid_too_high(self):
        """Test that value > 1 is rejected."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PercentageStopLoss(stop_loss_pct=1.5)

    def test_update_sets_stop_price_long(self):
        """Test update sets correct stop price for long position."""
        stop = PercentageStopLoss(0.02)  # 2% stop
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)
        # Long: stop is below entry (100 × 0.98 = 98)
        assert pos.stop_price == 98.0

    def test_update_sets_stop_price_short(self):
        """Test update sets correct stop price for short position."""
        stop = PercentageStopLoss(0.02)  # 2% stop
        pos = Position('AAPL', 100.0, -100, 0)
        stop.update(pos, 100.0)
        # Short: stop is above entry (100 × 1.02 = 102)
        assert pos.stop_price == 102.0

    def test_should_exit_long_triggered(self):
        """Test long stop triggered when price drops below stop."""
        stop = PercentageStopLoss(0.02)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)  # Stop at 98
        assert stop.should_exit(pos, 97.0, 1) is True

    def test_should_exit_long_at_stop(self):
        """Test long stop triggered when price equals stop."""
        stop = PercentageStopLoss(0.02)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)  # Stop at 98
        assert stop.should_exit(pos, 98.0, 1) is True

    def test_should_exit_long_not_triggered(self):
        """Test long stop not triggered when price above stop."""
        stop = PercentageStopLoss(0.02)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)  # Stop at 98
        assert stop.should_exit(pos, 99.0, 1) is False

    def test_should_exit_short_triggered(self):
        """Test short stop triggered when price rises above stop."""
        stop = PercentageStopLoss(0.02)
        pos = Position('AAPL', 100.0, -100, 0)
        stop.update(pos, 100.0)  # Stop at 102
        assert stop.should_exit(pos, 103.0, 1) is True

    def test_should_exit_short_not_triggered(self):
        """Test short stop not triggered when price below stop."""
        stop = PercentageStopLoss(0.02)
        pos = Position('AAPL', 100.0, -100, 0)
        stop.update(pos, 100.0)  # Stop at 102
        assert stop.should_exit(pos, 101.0, 1) is False

    def test_should_exit_no_stop_price(self):
        """Test should_exit returns False when stop_price is None."""
        stop = PercentageStopLoss(0.02)
        pos = Position('AAPL', 100.0, 100, 0)
        # Don't call update, so stop_price remains None
        assert stop.should_exit(pos, 90.0, 1) is False


# =============================================================================
# ATRStopLoss Tests
# =============================================================================

class TestATRStopLoss:
    """Tests for ATRStopLoss class."""

    @pytest.fixture
    def price_data(self):
        """Create sample price data for ATR calculation."""
        dates = pd.date_range('2024-01-01', periods=20)
        # Simple data with $2 daily range and predictable ATR
        return pd.DataFrame({
            'high': [102.0 + i*0.5 for i in range(20)],
            'low': [98.0 + i*0.5 for i in range(20)],
            'close': [100.0 + i*0.5 for i in range(20)]
        }, index=dates)

    def test_initialization_valid(self):
        """Test valid initialization."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        assert stop.atr_multiplier == 2.0
        assert stop.atr_lookback == 14

    def test_initialization_invalid_multiplier(self):
        """Test that zero/negative multiplier is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            ATRStopLoss(atr_multiplier=0)

    def test_initialization_invalid_lookback(self):
        """Test that zero lookback is rejected."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ATRStopLoss(atr_lookback=0)

    def test_calculate_atr(self, price_data):
        """Test ATR calculation returns positive value."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        atr = stop.calculate_atr(price_data)
        assert atr > 0
        # With our test data (high-low = 4), ATR should be close to 4
        assert 3.0 < atr < 5.0

    def test_calculate_atr_insufficient_data(self):
        """Test ATR raises error with insufficient data."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        short_data = pd.DataFrame({
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'close': [100, 101, 102]
        })
        with pytest.raises(ValueError, match="Need at least"):
            stop.calculate_atr(short_data)

    def test_update_trailing_stop_moves_up(self, price_data):
        """Test trailing stop moves up when price rises."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        pos = Position('AAPL', 100.0, 100, 0)

        # First update at entry
        stop.update(pos, 100.0, price_data)
        initial_stop = pos.stop_price

        # Price rises - stop should trail up
        stop.update(pos, 110.0, price_data)
        assert pos.stop_price > initial_stop

    def test_update_trailing_stop_never_moves_down(self, price_data):
        """Test trailing stop never moves down."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        pos = Position('AAPL', 100.0, 100, 0)

        # First update at high price
        stop.update(pos, 110.0, price_data)
        high_stop = pos.stop_price

        # Price drops - stop should NOT move down
        stop.update(pos, 100.0, price_data)
        assert pos.stop_price == high_stop

    def test_update_fallback_without_price_data(self):
        """Test update uses fallback when no price data."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0, None)
        # Fallback: 98% of highest price
        assert pos.stop_price == 98.0

    def test_should_exit_triggered(self, price_data):
        """Test stop exit triggered."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0, price_data)

        # Price drops well below stop
        assert stop.should_exit(pos, 80.0, 1) == True

    def test_should_exit_not_triggered(self, price_data):
        """Test stop exit not triggered."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0, price_data)

        # Price above stop
        assert stop.should_exit(pos, 100.0, 1) == False


# =============================================================================
# TimeStopLoss Tests
# =============================================================================

class TestTimeStopLoss:
    """Tests for TimeStopLoss class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        stop = TimeStopLoss(max_holding_bars=20)
        assert stop.max_holding_bars == 20

    def test_initialization_invalid_zero(self):
        """Test that zero is rejected."""
        with pytest.raises(ValueError, match="must be >= 1"):
            TimeStopLoss(max_holding_bars=0)

    def test_initialization_invalid_negative(self):
        """Test that negative value is rejected."""
        with pytest.raises(ValueError, match="must be >= 1"):
            TimeStopLoss(max_holding_bars=-1)

    def test_should_exit_after_max_bars(self):
        """Test exit triggered after max holding period."""
        stop = TimeStopLoss(max_holding_bars=20)
        pos = Position('AAPL', 100.0, 100, entry_bar=0)
        assert stop.should_exit(pos, 110.0, current_bar=20) is True

    def test_should_exit_beyond_max_bars(self):
        """Test exit triggered well beyond max bars."""
        stop = TimeStopLoss(max_holding_bars=20)
        pos = Position('AAPL', 100.0, 100, entry_bar=0)
        assert stop.should_exit(pos, 110.0, current_bar=25) is True

    def test_should_not_exit_before_max_bars(self):
        """Test no exit before max holding period."""
        stop = TimeStopLoss(max_holding_bars=20)
        pos = Position('AAPL', 100.0, 100, entry_bar=0)
        assert stop.should_exit(pos, 110.0, current_bar=19) is False

    def test_should_exit_with_offset_entry_bar(self):
        """Test exit calculation with non-zero entry bar."""
        stop = TimeStopLoss(max_holding_bars=10)
        pos = Position('AAPL', 100.0, 100, entry_bar=50)
        # Held for exactly 10 bars (50 to 60)
        assert stop.should_exit(pos, 110.0, current_bar=60) is True
        assert stop.should_exit(pos, 110.0, current_bar=59) is False

    def test_update_is_noop(self):
        """Test update does nothing (time stop doesn't need updating)."""
        stop = TimeStopLoss(max_holding_bars=20)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)
        # Stop price should remain None
        assert pos.stop_price is None


# =============================================================================
# ProfitTargetStopLoss Tests
# =============================================================================

class TestProfitTargetStopLoss:
    """Tests for ProfitTargetStopLoss class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        assert stop.stop_loss_pct == 0.02
        assert stop.take_profit_pct == 0.05

    def test_initialization_invalid_stop_loss(self):
        """Test invalid stop_loss_pct is rejected."""
        with pytest.raises(ValueError, match="stop_loss_pct must be between"):
            ProfitTargetStopLoss(stop_loss_pct=0.0, take_profit_pct=0.05)

    def test_initialization_invalid_take_profit(self):
        """Test invalid take_profit_pct is rejected."""
        with pytest.raises(ValueError, match="take_profit_pct must be between"):
            ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.0)

    def test_update_sets_prices_long(self):
        """Test update sets correct stop and target for long position."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)
        # Long: stop below (98), target above (105)
        assert stop.stop_price == 98.0
        assert stop.target_price == 105.0

    def test_update_sets_prices_short(self):
        """Test update sets correct stop and target for short position."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, -100, 0)
        stop.update(pos, 100.0)
        # Short: stop above (102), target below (95)
        assert stop.stop_price == 102.0
        assert stop.target_price == 95.0

    def test_should_exit_long_stop_loss(self):
        """Test long position exit on stop loss."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)
        # Price drops to stop (98)
        assert stop.should_exit(pos, 97.0, 1) is True

    def test_should_exit_long_profit_target(self):
        """Test long position exit on profit target."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)
        # Price rises to target (105)
        assert stop.should_exit(pos, 106.0, 1) is True

    def test_should_exit_short_stop_loss(self):
        """Test short position exit on stop loss."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, -100, 0)
        stop.update(pos, 100.0)
        # Price rises to stop (102)
        assert stop.should_exit(pos, 103.0, 1) is True

    def test_should_exit_short_profit_target(self):
        """Test short position exit on profit target."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, -100, 0)
        stop.update(pos, 100.0)
        # Price drops to target (95)
        assert stop.should_exit(pos, 94.0, 1) is True

    def test_should_not_exit_between_targets(self):
        """Test no exit when price between stop and target."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, 100, 0)
        stop.update(pos, 100.0)
        # Price in between (99)
        assert stop.should_exit(pos, 99.0, 1) is False

    def test_should_exit_no_prices_set(self):
        """Test should_exit returns False when prices not set."""
        stop = ProfitTargetStopLoss(stop_loss_pct=0.02, take_profit_pct=0.05)
        pos = Position('AAPL', 100.0, 100, 0)
        # Don't call update
        assert stop.should_exit(pos, 90.0, 1) is False


# =============================================================================
# RiskManager Tests
# =============================================================================

class TestRiskManager:
    """Tests for RiskManager class."""

    @pytest.fixture
    def moderate_config(self):
        """Create moderate risk configuration."""
        return RiskConfig(
            enabled=True,
            position_size_pct=0.10,
            use_stop_loss=True,
            stop_loss_pct=0.02,
            stop_loss_type='percentage',
            max_positions=10,
            max_single_position_pct=0.25,
            max_portfolio_heat=0.20
        )

    @pytest.fixture
    def risk_manager(self, moderate_config):
        """Create risk manager with moderate config."""
        return RiskManager(moderate_config)

    def test_initialization_valid(self, moderate_config):
        """Test valid initialization."""
        rm = RiskManager(moderate_config)
        assert rm.config == moderate_config
        assert rm.stop_loss is not None
        assert isinstance(rm.stop_loss, PercentageStopLoss)

    def test_initialization_invalid_config(self):
        """Test initialization with invalid config type."""
        with pytest.raises(TypeError, match="must be RiskConfig"):
            RiskManager("not a config")

    def test_can_open_position_allowed(self, risk_manager):
        """Test position within all limits is allowed."""
        can_open, reason = risk_manager.can_open_position(
            portfolio_value=100000,
            position_value=10000,
            current_positions=5
        )
        assert can_open is True
        assert reason == "OK"

    def test_can_open_position_max_positions(self, risk_manager):
        """Test rejection when max positions reached."""
        can_open, reason = risk_manager.can_open_position(
            portfolio_value=100000,
            position_value=10000,
            current_positions=10  # At max
        )
        assert can_open is False
        assert "Max positions" in reason

    def test_can_open_position_max_single_position(self, risk_manager):
        """Test rejection when position too large."""
        can_open, reason = risk_manager.can_open_position(
            portfolio_value=100000,
            position_value=30000,  # 30% > 25% max
            current_positions=0
        )
        assert can_open is False
        assert "Position size" in reason

    def test_can_open_position_max_heat(self, risk_manager):
        """Test rejection when portfolio heat exceeded."""
        can_open, reason = risk_manager.can_open_position(
            portfolio_value=100000,
            position_value=25000,  # 25% > 20% max heat
            current_positions=0
        )
        assert can_open is False
        assert "portfolio heat" in reason

    def test_can_open_position_disabled(self):
        """Test position allowed when risk management disabled."""
        config = RiskConfig(enabled=False)
        rm = RiskManager(config)
        can_open, reason = rm.can_open_position(
            portfolio_value=100000,
            position_value=100000,  # Would normally be rejected
            current_positions=100
        )
        assert can_open is True
        assert "disabled" in reason

    def test_add_remove_position(self, risk_manager):
        """Test adding and removing positions."""
        risk_manager.add_position('AAPL', 150.0, 100, 0)
        assert 'AAPL' in risk_manager.positions
        assert risk_manager.positions['AAPL'].entry_price == 150.0

        risk_manager.remove_position('AAPL')
        assert 'AAPL' not in risk_manager.positions

    def test_add_position_initializes_stop(self, risk_manager):
        """Test adding position initializes stop loss."""
        risk_manager.add_position('AAPL', 100.0, 100, 0)
        pos = risk_manager.positions['AAPL']
        # 2% stop loss -> stop at 98
        assert pos.stop_price == 98.0

    def test_check_exits_stop_triggered(self, risk_manager):
        """Test check_exits returns exits when stop triggered."""
        risk_manager.add_position('AAPL', 100.0, 100, 0)
        # Price dropped below 2% stop
        exits = risk_manager.check_exits(
            current_prices={'AAPL': 95.0},
            current_bar=1
        )
        assert 'AAPL' in exits
        assert 'Stop loss' in exits['AAPL']

    def test_check_exits_no_exit(self, risk_manager):
        """Test check_exits returns empty when no exits."""
        risk_manager.add_position('AAPL', 100.0, 100, 0)
        # Price above stop
        exits = risk_manager.check_exits(
            current_prices={'AAPL': 105.0},
            current_bar=1
        )
        assert len(exits) == 0

    def test_check_exits_missing_symbol(self, risk_manager):
        """Test check_exits ignores missing price symbols."""
        risk_manager.add_position('AAPL', 100.0, 100, 0)
        # AAPL not in prices dict
        exits = risk_manager.check_exits(
            current_prices={'MSFT': 350.0},
            current_bar=1
        )
        assert len(exits) == 0

    def test_check_exits_no_stop_loss(self):
        """Test check_exits returns empty when no stop loss configured."""
        config = RiskConfig(use_stop_loss=False)
        rm = RiskManager(config)
        rm.add_position('AAPL', 100.0, 100, 0)
        exits = rm.check_exits(
            current_prices={'AAPL': 50.0},  # Huge loss
            current_bar=1
        )
        assert len(exits) == 0

    def test_get_position_info(self, risk_manager):
        """Test get_position_info returns DataFrame with details."""
        risk_manager.add_position('AAPL', 100.0, 100, 0)
        risk_manager.add_position('MSFT', 350.0, 50, 0)

        info = risk_manager.get_position_info({
            'AAPL': 110.0,
            'MSFT': 340.0
        })

        assert len(info) == 2
        assert 'symbol' in info.columns
        assert 'unrealized_pnl' in info.columns

        aapl_row = info[info['symbol'] == 'AAPL'].iloc[0]
        assert aapl_row['entry_price'] == 100.0
        assert aapl_row['unrealized_pnl'] == 1000.0  # 100 shares × $10

    def test_create_stop_loss_atr(self):
        """Test creating ATR stop loss from config."""
        config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='atr',
            atr_multiplier=2.5,
            atr_lookback=20
        )
        rm = RiskManager(config)
        assert isinstance(rm.stop_loss, ATRStopLoss)
        assert rm.stop_loss.atr_multiplier == 2.5
        assert rm.stop_loss.atr_lookback == 20

    def test_create_stop_loss_time(self):
        """Test creating time stop loss from config."""
        config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='time',
            max_holding_bars=30
        )
        rm = RiskManager(config)
        assert isinstance(rm.stop_loss, TimeStopLoss)
        assert rm.stop_loss.max_holding_bars == 30

    def test_create_stop_loss_profit_target(self):
        """Test creating profit target stop loss from config."""
        config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='profit_target',
            stop_loss_pct=0.02,
            take_profit_pct=0.06
        )
        rm = RiskManager(config)
        assert isinstance(rm.stop_loss, ProfitTargetStopLoss)
        assert rm.stop_loss.stop_loss_pct == 0.02
        assert rm.stop_loss.take_profit_pct == 0.06

    def test_create_stop_loss_invalid_type(self):
        """Test error on invalid stop loss type."""
        config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='percentage'  # Valid type
        )
        # Manually set invalid type after creation
        config.stop_loss_type = 'invalid_type'
        with pytest.raises(ValueError, match="Unknown stop_loss_type"):
            RiskManager(config)

    def test_multiple_positions_tracking(self, risk_manager):
        """Test tracking multiple positions."""
        risk_manager.add_position('AAPL', 150.0, 100, 0)
        risk_manager.add_position('MSFT', 350.0, 50, 5)
        risk_manager.add_position('GOOGL', 140.0, 30, 10)

        assert len(risk_manager.positions) == 3
        assert risk_manager.positions['AAPL'].entry_bar == 0
        assert risk_manager.positions['MSFT'].entry_bar == 5
        assert risk_manager.positions['GOOGL'].entry_bar == 10
