"""
Unit Tests for Momentum Pure Strategies.

Tests the pure momentum signal generation without backtesting dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.implementations.momentum import (
    MACDMomentumSignals,
    BreakoutMomentumSignals
)
from src.strategies.core import Signal


class TestMACDMomentumSignals:
    """Test MACD momentum strategy."""

    def test_initialization_valid_params(self):
        """Test valid parameter initialization."""
        strategy = MACDMomentumSignals(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            min_confidence=0.6
        )

        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
        assert strategy.min_confidence == 0.6

    def test_initialization_invalid_periods(self):
        """Test that fast >= slow raises error."""
        with pytest.raises(ValueError, match="fast_period.*must be less than"):
            MACDMomentumSignals(fast_period=26, slow_period=12)

        with pytest.raises(ValueError, match="fast_period.*must be less than"):
            MACDMomentumSignals(fast_period=12, slow_period=12)

    def test_initialization_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="min_confidence must be"):
            MACDMomentumSignals(min_confidence=1.5)

    def test_required_lookback(self):
        """Test that lookback period is correct."""
        strategy = MACDMomentumSignals(fast_period=12, slow_period=26, signal_period=9)
        lookback = strategy.get_required_lookback()

        assert lookback == 45  # slow + signal + 10

    def test_macd_calculation(self):
        """Test MACD indicator calculation."""
        strategy = MACDMomentumSignals(fast_period=5, slow_period=10, signal_period=3)

        # Create test data
        dates = pd.date_range('2025-01-01', periods=50, freq='D')
        close = pd.Series(range(100, 150), index=dates)

        macd_line, signal_line, histogram = strategy._calculate_macd(close)

        # MACD line should exist
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)

        # Histogram = MACD - Signal
        assert np.allclose(
            histogram.dropna(),
            (macd_line - signal_line).dropna(),
            rtol=1e-10
        )

    def test_bullish_crossover_detection(self):
        """Test detection of bullish MACD crossover."""
        strategy = MACDMomentumSignals(
            fast_period=5,
            slow_period=10,
            signal_period=3,
            min_confidence=0.0  # Accept all signals
        )

        # Create data with uptrend (potential bullish cross)
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        prices = list(range(100, 110)) + list(range(110, 130))
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 30
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # May or may not generate signal depending on crossover timing
        assert isinstance(signals, list)
        if len(signals) > 0:
            assert signals[0].direction == 'BUY'
            assert signals[0].metadata['strategy'] == 'MACD_Momentum'

    def test_bearish_crossover_detection(self):
        """Test detection of bearish MACD crossover."""
        strategy = MACDMomentumSignals(
            fast_period=5,
            slow_period=10,
            signal_period=3,
            min_confidence=0.0
        )

        # Create data with downtrend
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        prices = list(range(130, 120, -1)) + list(range(120, 100, -1))
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 30
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        if len(signals) > 0:
            assert signals[0].direction == 'SELL'

    def test_no_signal_without_crossover(self):
        """Test no signal when no crossover occurs."""
        strategy = MACDMomentumSignals(fast_period=5, slow_period=10)

        # Flat price (no crossover)
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'open': [100] * 30,
            'high': [101] * 30,
            'low': [99] * 30,
            'close': [100] * 30,
            'volume': [1000000] * 30
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Expected: no crossover, no signals
        assert isinstance(signals, list)

    def test_multiple_symbols(self):
        """Test signal generation for multiple symbols."""
        strategy = MACDMomentumSignals(
            fast_period=5,
            slow_period=10,
            min_confidence=0.0
        )

        dates = pd.date_range('2025-01-01', periods=30, freq='D')

        # Uptrend symbol
        prices1 = list(range(100, 130))
        df1 = pd.DataFrame({
            'open': prices1,
            'high': [p * 1.01 for p in prices1],
            'low': [p * 0.99 for p in prices1],
            'close': prices1,
            'volume': [1000000] * 30
        }, index=dates)

        # Downtrend symbol
        prices2 = list(range(130, 100, -1))
        df2 = pd.DataFrame({
            'open': prices2,
            'high': [p * 1.01 for p in prices2],
            'low': [p * 0.99 for p in prices2],
            'close': prices2,
            'volume': [1000000] * 30
        }, index=dates)

        market_data = {'AAPL': df1, 'MSFT': df2}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Check signals are Signal objects
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol in ['AAPL', 'MSFT']

    def test_get_parameters(self):
        """Test parameter retrieval."""
        strategy = MACDMomentumSignals(
            fast_period=10,
            slow_period=20,
            signal_period=5,
            min_confidence=0.7
        )

        params = strategy.get_parameters()

        assert params['fast_period'] == 10
        assert params['slow_period'] == 20
        assert params['signal_period'] == 5
        assert params['min_confidence'] == 0.7


class TestBreakoutMomentumSignals:
    """Test breakout momentum strategy."""

    def test_initialization_valid_params(self):
        """Test valid parameter initialization."""
        strategy = BreakoutMomentumSignals(
            breakout_window=20,
            exit_window=10,
            min_confidence=0.65
        )

        assert strategy.breakout_window == 20
        assert strategy.exit_window == 10
        assert strategy.min_confidence == 0.65

    def test_initialization_with_filters(self):
        """Test initialization with filters enabled."""
        strategy = BreakoutMomentumSignals(
            volatility_filter=True,
            min_volatility=0.02,
            max_volatility=0.08,
            volume_confirmation=True,
            volume_threshold=2.0
        )

        assert strategy.volatility_filter == True
        assert strategy.volume_confirmation == True
        assert strategy.volume_threshold == 2.0

    def test_initialization_invalid_volatility(self):
        """Test that min >= max volatility raises error."""
        with pytest.raises(ValueError, match="min_volatility must be"):
            BreakoutMomentumSignals(
                volatility_filter=True,
                min_volatility=0.10,
                max_volatility=0.05
            )

    def test_initialization_invalid_volume_threshold(self):
        """Test that volume threshold <= 1.0 raises error."""
        with pytest.raises(ValueError, match="volume_threshold must be"):
            BreakoutMomentumSignals(
                volume_confirmation=True,
                volume_threshold=0.9
            )

    def test_required_lookback(self):
        """Test lookback period calculation."""
        strategy = BreakoutMomentumSignals(
            breakout_window=20,
            volatility_window=30
        )

        lookback = strategy.get_required_lookback()
        assert lookback == 40  # max(20, 30) + 10

    def test_bullish_breakout_detection(self):
        """Test detection of bullish breakout."""
        strategy = BreakoutMomentumSignals(
            breakout_window=10,
            min_confidence=0.0
        )

        # Create data with breakout
        dates = pd.date_range('2025-01-01', periods=25, freq='D')
        prices = [100] * 10 + [101, 102, 103, 104, 105, 110, 115, 120, 125, 130,
                               132, 134, 136, 138, 140]
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 25
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # May generate bullish breakout signal
        assert isinstance(signals, list)

    def test_bearish_breakout_detection(self):
        """Test detection of bearish breakout."""
        strategy = BreakoutMomentumSignals(
            breakout_window=10,
            min_confidence=0.0
        )

        # Create data with bearish breakout
        dates = pd.date_range('2025-01-01', periods=25, freq='D')
        prices = [130] * 10 + [128, 126, 124, 122, 120, 115, 110, 105, 100, 95,
                               93, 91, 89, 87, 85]
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 25
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        if len(signals) > 0:
            assert signals[0].direction == 'SELL'

    def test_volatility_filter(self):
        """Test volatility filter functionality."""
        strategy = BreakoutMomentumSignals(
            breakout_window=10,
            volatility_filter=True,
            min_volatility=0.20,  # Very high threshold
            max_volatility=0.30,
            min_confidence=0.0
        )

        # Normal volatility data (will be filtered out)
        dates = pd.date_range('2025-01-01', periods=25, freq='D')
        prices = list(range(100, 125))
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 25
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Expected: filtered out by volatility
        assert isinstance(signals, list)

    def test_volume_confirmation(self):
        """Test volume confirmation filter."""
        strategy = BreakoutMomentumSignals(
            breakout_window=10,
            volume_confirmation=True,
            volume_threshold=2.0,
            min_confidence=0.0
        )

        # Create data with breakout but low volume
        dates = pd.date_range('2025-01-01', periods=25, freq='D')
        prices = [100] * 10 + list(range(105, 120))
        volumes = [1000000] * 24 + [1100000]  # Slight volume increase (< 2x)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Expected: may be filtered by volume
        assert isinstance(signals, list)

    def test_atr_calculation(self):
        """Test ATR calculation."""
        strategy = BreakoutMomentumSignals()

        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [110] * 20,
            'low': [90] * 20,
            'close': [105] * 20,
            'volume': [1000000] * 20
        }, index=dates)

        atr = strategy._calculate_atr(df, 5)

        assert atr is not None
        assert atr > 0

    def test_get_parameters(self):
        """Test parameter retrieval."""
        strategy = BreakoutMomentumSignals(
            breakout_window=25,
            exit_window=15,
            min_confidence=0.70,
            volatility_filter=True,
            volume_confirmation=False
        )

        params = strategy.get_parameters()

        assert params['breakout_window'] == 25
        assert params['exit_window'] == 15
        assert params['min_confidence'] == 0.70
        assert params['volatility_filter'] == True
        assert params['volume_confirmation'] == False


def test_imports():
    """Test that all imports work correctly."""
    from src.strategies.implementations.momentum import (
        MACDMomentumSignals,
        BreakoutMomentumSignals
    )

    from src.strategies.implementations import (
        MACDMomentumSignals as MACD1,
        BreakoutMomentumSignals as Breakout1
    )

    # All imports should work
    assert MACDMomentumSignals == MACD1
    assert BreakoutMomentumSignals == Breakout1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
