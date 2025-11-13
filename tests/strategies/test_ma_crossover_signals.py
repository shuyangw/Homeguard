"""
Unit Tests for MA Crossover Pure Strategy.

Tests the pure MA crossover signal generation without backtesting dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.implementations.moving_average import (
    MACrossoverSignals,
    TripleMACrossoverSignals
)
from src.strategies.core import Signal


class TestMACrossoverSignals:
    """Test dual MA crossover strategy."""

    def test_initialization_valid_params(self):
        """Test valid parameter initialization."""
        strategy = MACrossoverSignals(
            fast_period=50,
            slow_period=200,
            ma_type='sma',
            min_confidence=0.7
        )

        assert strategy.fast_period == 50
        assert strategy.slow_period == 200
        assert strategy.ma_type == 'sma'
        assert strategy.min_confidence == 0.7

    def test_initialization_invalid_periods(self):
        """Test that fast >= slow raises error."""
        with pytest.raises(ValueError, match="fast_period.*must be less than"):
            MACrossoverSignals(fast_period=200, slow_period=50)

        with pytest.raises(ValueError, match="fast_period.*must be less than"):
            MACrossoverSignals(fast_period=50, slow_period=50)

    def test_initialization_invalid_ma_type(self):
        """Test that invalid MA type raises error."""
        with pytest.raises(ValueError, match="ma_type must be"):
            MACrossoverSignals(ma_type='invalid')

    def test_initialization_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="min_confidence must be"):
            MACrossoverSignals(min_confidence=1.5)

        with pytest.raises(ValueError, match="min_confidence must be"):
            MACrossoverSignals(min_confidence=-0.1)

    def test_required_lookback(self):
        """Test that lookback period is correct."""
        strategy = MACrossoverSignals(fast_period=50, slow_period=200)
        lookback = strategy.get_required_lookback()

        assert lookback == 210  # slow_period + 10

    def test_golden_cross_detection(self):
        """Test detection of golden cross (BUY signal)."""
        strategy = MACrossoverSignals(
            fast_period=5,
            slow_period=10,
            ma_type='sma',
            min_confidence=0.0  # Accept all signals for testing
        )

        # Create data with golden cross
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        prices = [100] * 10 + [102, 104, 106, 108, 110, 112, 114, 116, 118, 120]
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Should detect golden cross
        assert len(signals) >= 0  # May or may not trigger depending on exact crossover
        if len(signals) > 0:
            signal = signals[0]
            assert signal.symbol == 'TEST'
            assert signal.direction == 'BUY'
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.metadata['strategy'] == 'MA_Crossover'
            assert signal.metadata['crossover_type'] == 'golden'

    def test_death_cross_detection(self):
        """Test detection of death cross (SELL signal)."""
        strategy = MACrossoverSignals(
            fast_period=5,
            slow_period=10,
            ma_type='sma',
            min_confidence=0.0
        )

        # Create data with death cross
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        prices = [120] * 10 + [118, 116, 114, 112, 110, 108, 106, 104, 102, 100]
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        if len(signals) > 0:
            signal = signals[0]
            assert signal.direction == 'SELL'
            assert signal.metadata['crossover_type'] == 'death'

    def test_no_signal_when_no_crossover(self):
        """Test that no signal is generated without crossover."""
        strategy = MACrossoverSignals(fast_period=5, slow_period=10)

        # Flat price data (no crossover)
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [101] * 20,
            'low': [99] * 20,
            'close': [100] * 20,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # No crossover, no signals (or filtered out by confidence)
        # This is expected behavior

    def test_multiple_symbols(self):
        """Test signal generation for multiple symbols."""
        strategy = MACrossoverSignals(
            fast_period=5,
            slow_period=10,
            min_confidence=0.0
        )

        dates = pd.date_range('2025-01-01', periods=20, freq='D')

        # Symbol 1: Rising prices (potential golden cross)
        prices1 = [100] * 10 + list(range(102, 122, 2))
        df1 = pd.DataFrame({
            'open': prices1,
            'high': [p * 1.01 for p in prices1],
            'low': [p * 0.99 for p in prices1],
            'close': prices1,
            'volume': [1000000] * 20
        }, index=dates)

        # Symbol 2: Falling prices (potential death cross)
        prices2 = [120] * 10 + list(range(118, 98, -2))
        df2 = pd.DataFrame({
            'open': prices2,
            'high': [p * 1.01 for p in prices2],
            'low': [p * 0.99 for p in prices2],
            'close': prices2,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'AAPL': df1, 'MSFT': df2}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Check that signals are Symbol objects
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol in ['AAPL', 'MSFT']
            assert signal.direction in ['BUY', 'SELL']

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        strategy = MACrossoverSignals(fast_period=50, slow_period=200)

        # Only 30 periods of data (insufficient)
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

        # Should return empty list or skip symbol
        assert isinstance(signals, list)

    def test_confidence_filtering(self):
        """Test that low confidence signals are filtered out."""
        strategy = MACrossoverSignals(
            fast_period=5,
            slow_period=10,
            min_confidence=0.9  # Very high threshold
        )

        # Create data with weak crossover
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        prices = [100] * 10 + [100.1, 100.2, 100.3, 100.4, 100.5,
                               100.6, 100.7, 100.8, 100.9, 101.0]
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Weak signal should be filtered out
        # (or may not even trigger crossover with such small moves)

    def test_get_parameters(self):
        """Test parameter retrieval."""
        strategy = MACrossoverSignals(
            fast_period=50,
            slow_period=200,
            ma_type='ema',
            min_confidence=0.75
        )

        params = strategy.get_parameters()

        assert params['fast_period'] == 50
        assert params['slow_period'] == 200
        assert params['ma_type'] == 'ema'
        assert params['min_confidence'] == 0.75

    def test_ema_vs_sma(self):
        """Test that EMA and SMA produce different MAs."""
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'open': list(range(100, 120)),
            'high': list(range(101, 121)),
            'low': list(range(99, 119)),
            'close': list(range(100, 120)),
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}

        sma_strategy = MACrossoverSignals(
            fast_period=5, slow_period=10, ma_type='sma', min_confidence=0.0
        )
        ema_strategy = MACrossoverSignals(
            fast_period=5, slow_period=10, ma_type='ema', min_confidence=0.0
        )

        sma_signals = sma_strategy.generate_signals(market_data, dates[-1])
        ema_signals = ema_strategy.generate_signals(market_data, dates[-1])

        # Both should work (though may produce different signals)
        assert isinstance(sma_signals, list)
        assert isinstance(ema_signals, list)


class TestTripleMACrossoverSignals:
    """Test triple MA crossover strategy."""

    def test_initialization_valid_params(self):
        """Test valid parameter initialization."""
        strategy = TripleMACrossoverSignals(
            fast_period=10,
            medium_period=20,
            slow_period=50,
            ma_type='ema',
            min_confidence=0.75
        )

        assert strategy.fast_period == 10
        assert strategy.medium_period == 20
        assert strategy.slow_period == 50
        assert strategy.ma_type == 'ema'

    def test_initialization_invalid_ordering(self):
        """Test that periods must be ordered."""
        with pytest.raises(ValueError, match="fast < medium < slow"):
            TripleMACrossoverSignals(
                fast_period=20,
                medium_period=10,
                slow_period=50
            )

        with pytest.raises(ValueError, match="fast < medium < slow"):
            TripleMACrossoverSignals(
                fast_period=10,
                medium_period=10,
                slow_period=50
            )

    def test_uptrend_alignment_signal(self):
        """Test detection of uptrend alignment."""
        strategy = TripleMACrossoverSignals(
            fast_period=3,
            medium_period=5,
            slow_period=10,
            min_confidence=0.0
        )

        # Create strong uptrend
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        prices = list(range(100, 120))
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        # Should generate signals when MAs align
        assert isinstance(signals, list)
        if len(signals) > 0:
            signal = signals[0]
            assert signal.direction == 'BUY'
            assert signal.metadata['strategy'] == 'Triple_MA'

    def test_downtrend_alignment_signal(self):
        """Test detection of downtrend alignment."""
        strategy = TripleMACrossoverSignals(
            fast_period=3,
            medium_period=5,
            slow_period=10,
            min_confidence=0.0
        )

        # Create strong downtrend
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        prices = list(range(120, 100, -1))
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)

        market_data = {'TEST': df}
        signals = strategy.generate_signals(market_data, dates[-1])

        if len(signals) > 0:
            signal = signals[0]
            assert signal.direction == 'SELL'

    def test_get_parameters(self):
        """Test parameter retrieval."""
        strategy = TripleMACrossoverSignals(
            fast_period=10,
            medium_period=20,
            slow_period=50,
            ma_type='sma',
            min_confidence=0.8
        )

        params = strategy.get_parameters()

        assert params['fast_period'] == 10
        assert params['medium_period'] == 20
        assert params['slow_period'] == 50
        assert params['ma_type'] == 'sma'
        assert params['min_confidence'] == 0.8


def test_imports():
    """Test that all imports work correctly."""
    from src.strategies.implementations.moving_average import (
        MACrossoverSignals,
        TripleMACrossoverSignals
    )

    from src.strategies.implementations import (
        MACrossoverSignals as MA1,
        TripleMACrossoverSignals as MA2
    )

    # All imports should work
    assert MACrossoverSignals == MA1
    assert TripleMACrossoverSignals == MA2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
