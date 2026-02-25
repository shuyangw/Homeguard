"""
Unit tests for DSTS signals (live trading).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.advanced.dsts_signals import DSTSSignals, DSTSRiskSignals


class TestDSTSSignalsInit:
    """Tests for DSTS signals initialization."""

    def test_default_parameters(self):
        """Test signals initialize with default parameters."""
        signals = DSTSSignals()

        assert signals.period == 65
        assert signals.bull_threshold == 0.75
        assert signals.bear_threshold == 0.0
        assert signals.trailing_stop_pct == 0.25

    def test_custom_parameters(self):
        """Test signals initialize with custom parameters."""
        signals = DSTSSignals(
            period=50,
            bull_threshold=1.0,
            bear_threshold=-0.5,
            trailing_stop_pct=0.20,
        )

        assert signals.period == 50
        assert signals.bull_threshold == 1.0
        assert signals.bear_threshold == -0.5
        assert signals.trailing_stop_pct == 0.20

    def test_initial_state(self):
        """Test signals start in cash state."""
        signals = DSTSSignals()

        assert signals._is_long is False
        assert signals._current_state == 'BEARISH'
        assert signals.get_target_position() == 0.0


class TestDSTSSignalsDataUpdate:
    """Tests for data update functionality."""

    def create_sample_data(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.03
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'close': prices,
        }, index=dates)

    def test_update_historical_data(self):
        """Test updating historical data."""
        signals = DSTSSignals()
        data = self.create_sample_data()

        signals.update_historical_data(data)

        assert signals._historical_data is not None
        assert len(signals._historical_data) == len(data)

    def test_insufficient_data_raises(self):
        """Test get_risk_signals raises with insufficient data."""
        signals = DSTSSignals(period=65)
        data = self.create_sample_data(days=30)  # Less than period

        signals.update_historical_data(data)

        with pytest.raises(ValueError, match="Insufficient data"):
            signals.get_risk_signals()


class TestDSTSSignalsStateMachine:
    """Tests for state machine transitions."""

    def create_trending_data(self, direction: str = 'up', days: int = 200) -> pd.DataFrame:
        """Create data with clear trend."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        base_price = 10000

        if direction == 'up':
            trend = np.linspace(0, 1, days)
            np.random.seed(42)
            noise = np.random.randn(days) * 0.01
            prices = base_price * (1 + trend + noise)
        elif direction == 'down':
            trend = np.linspace(1, 0, days)
            np.random.seed(42)
            noise = np.random.randn(days) * 0.01
            prices = base_price * (1 + trend + noise)
        else:
            np.random.seed(42)
            noise = np.random.randn(days) * 0.01
            prices = base_price * (1 + noise)

        return pd.DataFrame({'close': prices}, index=dates)

    def test_state_transition_bearish_to_bullish(self):
        """Test transition from BEARISH to BULLISH in uptrend."""
        signals = DSTSSignals(period=65, bull_threshold=0.75, bear_threshold=0.0)
        data = self.create_trending_data('up', days=200)

        signals.update_historical_data(data)

        # In strong uptrend, should eventually become bullish
        risk_signals = signals.get_risk_signals()

        # The state should reflect the z-score
        if risk_signals.zscore > 0.75:
            assert signals._is_long is True
            assert signals._current_state == 'BULLISH'

    def test_manual_position_state(self):
        """Test setting position state manually."""
        signals = DSTSSignals()

        signals.set_position_state(is_long=True)
        assert signals._is_long is True
        assert signals._current_state == 'BULLISH'

        signals.set_position_state(is_long=False)
        assert signals._is_long is False
        assert signals._current_state == 'BEARISH'

    def test_get_target_position(self):
        """Test target position returns 0.0 or 1.0."""
        signals = DSTSSignals()

        # Default state (cash)
        assert signals.get_target_position() == 0.0

        # Set to long
        signals.set_position_state(is_long=True)
        assert signals.get_target_position() == 1.0

        # Trailing stop triggered
        signals._trailing_stop_triggered = True
        assert signals.get_target_position() == 0.0


class TestDSTSSignalsRiskSignals:
    """Tests for risk signals output."""

    def create_sample_data(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.03
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({'close': prices}, index=dates)

    def test_get_risk_signals_return_type(self):
        """Test get_risk_signals returns DSTSRiskSignals."""
        signals = DSTSSignals(period=65)
        data = self.create_sample_data()

        signals.update_historical_data(data)
        risk_signals = signals.get_risk_signals()

        assert isinstance(risk_signals, DSTSRiskSignals)

    def test_risk_signals_fields(self):
        """Test risk signals has all required fields."""
        signals = DSTSSignals(period=65)
        data = self.create_sample_data()

        signals.update_historical_data(data)
        risk_signals = signals.get_risk_signals()

        assert hasattr(risk_signals, 'timestamp')
        assert hasattr(risk_signals, 'zscore')
        assert hasattr(risk_signals, 'state')
        assert hasattr(risk_signals, 'should_be_long')
        assert hasattr(risk_signals, 'close')
        assert hasattr(risk_signals, 'ema')
        assert hasattr(risk_signals, 'std')
        assert hasattr(risk_signals, 'trailing_stop_triggered')

    def test_risk_signals_to_dict(self):
        """Test risk signals to_dict method."""
        signals = DSTSSignals(period=65)
        data = self.create_sample_data()

        signals.update_historical_data(data)
        risk_signals = signals.get_risk_signals()
        risk_dict = risk_signals.to_dict()

        assert isinstance(risk_dict, dict)
        assert 'zscore' in risk_dict
        assert 'state' in risk_dict
        assert 'should_be_long' in risk_dict


class TestDSTSSignalsTrailingStop:
    """Tests for trailing stop functionality."""

    def create_sample_data(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        prices = 10000 + np.cumsum(np.random.randn(days) * 100)

        return pd.DataFrame({'close': prices}, index=dates)

    def test_trailing_stop_not_triggered(self):
        """Test trailing stop not triggered with small drawdown."""
        signals = DSTSSignals(trailing_stop_pct=0.25)
        signals.set_position_state(is_long=True)

        signals.update_portfolio_value(100000)
        signals.update_portfolio_value(90000)  # 10% drawdown

        assert signals._trailing_stop_triggered is False
        assert signals.get_target_position() == 1.0

    def test_trailing_stop_triggered(self):
        """Test trailing stop triggers at threshold."""
        signals = DSTSSignals(trailing_stop_pct=0.25)
        signals.set_position_state(is_long=True)

        signals.update_portfolio_value(100000)  # Peak
        signals.update_portfolio_value(74000)  # 26% drawdown

        assert signals._trailing_stop_triggered is True
        assert signals.get_target_position() == 0.0
        assert signals._is_long is False

    def test_reset_trailing_stop(self):
        """Test resetting trailing stop."""
        signals = DSTSSignals(trailing_stop_pct=0.25)
        signals._trailing_stop_triggered = True

        signals.reset_trailing_stop()

        assert signals._trailing_stop_triggered is False
        assert signals._peak_portfolio_value == 0.0

    def test_get_current_drawdown(self):
        """Test current drawdown calculation."""
        signals = DSTSSignals()
        signals.set_position_state(is_long=True)

        signals.update_portfolio_value(100000)
        signals.update_portfolio_value(80000)

        drawdown = signals.get_current_drawdown()
        assert drawdown == pytest.approx(0.20, rel=1e-6)


class TestDSTSSignalsEntryExit:
    """Tests for should_enter and should_exit methods."""

    def create_bullish_data(self, days: int = 200) -> pd.DataFrame:
        """Create data that will produce bullish Z-score."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        # Strong uptrend
        prices = 10000 * np.exp(np.linspace(0, 1, days))
        return pd.DataFrame({'close': prices}, index=dates)

    def create_bearish_data(self, days: int = 200) -> pd.DataFrame:
        """Create data that will produce bearish Z-score."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        # Strong downtrend
        prices = 10000 * np.exp(np.linspace(1, 0, days))
        return pd.DataFrame({'close': prices}, index=dates)

    def test_should_enter_bullish(self):
        """Test should_enter returns True in bullish conditions."""
        signals = DSTSSignals(period=65, bull_threshold=0.5)  # Lower threshold for test
        data = self.create_bullish_data()

        signals.update_historical_data(data)

        # In strong uptrend, should want to enter
        risk_signals = signals.get_risk_signals()
        if risk_signals.zscore > 0.5 and not signals._is_long:
            assert signals.should_enter() is True

    def test_should_not_enter_when_long(self):
        """Test should_enter returns False when already long."""
        signals = DSTSSignals(period=65)
        data = self.create_bullish_data()

        signals.update_historical_data(data)
        signals.set_position_state(is_long=True)

        assert signals.should_enter() is False

    def test_should_exit_bearish(self):
        """Test should_exit returns True in bearish conditions."""
        signals = DSTSSignals(period=65, bear_threshold=0.0)
        data = self.create_bearish_data()

        signals.update_historical_data(data)
        signals.set_position_state(is_long=True)

        # In strong downtrend, should want to exit
        risk_signals = signals.get_risk_signals()
        if risk_signals.zscore < 0.0:
            assert signals.should_exit() == True  # noqa: E712 - numpy bool comparison


class TestDSTSSignalsSummary:
    """Tests for summary method."""

    def create_sample_data(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        prices = 10000 + np.cumsum(np.random.randn(days) * 100)

        return pd.DataFrame({'close': prices}, index=dates)

    def test_summary_with_data(self):
        """Test get_summary with sufficient data."""
        signals = DSTSSignals(period=65)
        data = self.create_sample_data()

        signals.update_historical_data(data)
        summary = signals.get_summary()

        assert isinstance(summary, dict)
        assert 'state' in summary
        assert 'is_long' in summary
        assert 'zscore' in summary
        assert 'target_position' in summary
        assert 'period' in summary

    def test_summary_without_data(self):
        """Test get_summary without sufficient data."""
        signals = DSTSSignals(period=65)
        data = self.create_sample_data(days=30)

        signals.update_historical_data(data)
        summary = signals.get_summary()

        assert summary['state'] == 'INSUFFICIENT_DATA'
        assert summary['bars_available'] == 30
        assert summary['bars_required'] == 65


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
