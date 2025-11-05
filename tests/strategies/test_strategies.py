"""
Unit tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from strategies.base_strategies.moving_average import MovingAverageCrossover, TripleMovingAverage
from strategies.base_strategies.mean_reversion import MeanReversion, RSIMeanReversion
from strategies.base_strategies.momentum import MomentumStrategy, BreakoutStrategy


class TestMovingAverageCrossover:
    """Test MovingAverageCrossover strategy."""

    def test_strategy_initialization(self):
        """Test strategy initializes with valid parameters."""
        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)

        assert strategy.params['fast_window'] == 10
        assert strategy.params['slow_window'] == 20
        assert strategy.params['ma_type'] == 'sma'

    def test_invalid_window_sizes(self):
        """Test validation rejects invalid window sizes."""
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=20, slow_window=10)

        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=0, slow_window=20)

    def test_invalid_ma_type(self):
        """Test validation rejects invalid MA type."""
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=10, slow_window=20, ma_type='invalid')

    def test_signal_generation(self, simple_price_data):
        """Test strategy generates valid signals."""
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
        entries, exits = strategy.generate_signals(simple_price_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert entries.dtype == bool
        assert exits.dtype == bool
        assert len(entries) == len(simple_price_data)

    def test_signals_on_uptrend(self, simple_price_data):
        """Test strategy generates entry signals on uptrend."""
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
        entries, exits = strategy.generate_signals(simple_price_data)

        assert entries.sum() > 0, "Should generate entry signals on uptrend"

    def test_ema_vs_sma(self, simple_price_data):
        """Test EMA and SMA generate different signals."""
        sma_strategy = MovingAverageCrossover(fast_window=5, slow_window=15, ma_type='sma')
        ema_strategy = MovingAverageCrossover(fast_window=5, slow_window=15, ma_type='ema')

        sma_entries, _ = sma_strategy.generate_signals(simple_price_data)
        ema_entries, _ = ema_strategy.generate_signals(simple_price_data)

        assert not sma_entries.equals(ema_entries), "SMA and EMA should generate different signals"


class TestMeanReversion:
    """Test MeanReversion (Bollinger Bands) strategy."""

    def test_strategy_initialization(self):
        """Test strategy initializes with valid parameters."""
        strategy = MeanReversion(window=20, num_std=2.0)

        assert strategy.params['window'] == 20
        assert strategy.params['num_std'] == 2.0

    def test_signal_generation(self, oscillating_price_data):
        """Test strategy generates signals on oscillating data."""
        strategy = MeanReversion(window=15, num_std=2.0)
        entries, exits = strategy.generate_signals(oscillating_price_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert entries.sum() > 0, "Should generate entry signals on oscillating data"


class TestRSIMeanReversion:
    """Test RSI Mean Reversion strategy."""

    def test_strategy_initialization(self):
        """Test strategy initializes with valid parameters."""
        strategy = RSIMeanReversion(rsi_window=14, oversold=30, overbought=70)

        assert strategy.params['rsi_window'] == 14
        assert strategy.params['oversold'] == 30
        assert strategy.params['overbought'] == 70

    def test_invalid_rsi_levels(self):
        """Test validation rejects invalid RSI levels."""
        with pytest.raises(ValueError):
            RSIMeanReversion(rsi_window=14, oversold=50, overbought=40)

        with pytest.raises(ValueError):
            RSIMeanReversion(rsi_window=14, oversold=-10, overbought=70)

    def test_signal_generation(self, oscillating_price_data):
        """Test RSI strategy generates signals."""
        strategy = RSIMeanReversion(rsi_window=14, oversold=30, overbought=70)
        entries, exits = strategy.generate_signals(oscillating_price_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)


class TestMomentumStrategy:
    """Test MACD Momentum strategy."""

    def test_strategy_initialization(self):
        """Test strategy initializes with valid parameters."""
        strategy = MomentumStrategy(fast=12, slow=26, signal=9)

        assert strategy.params['fast'] == 12
        assert strategy.params['slow'] == 26
        assert strategy.params['signal'] == 9

    def test_signal_generation(self, simple_price_data):
        """Test MACD strategy generates signals."""
        strategy = MomentumStrategy(fast=8, slow=17, signal=9)
        entries, exits = strategy.generate_signals(simple_price_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(simple_price_data)


class TestBreakoutStrategy:
    """Test Breakout strategy."""

    def test_strategy_initialization(self):
        """Test strategy initializes with valid parameters."""
        strategy = BreakoutStrategy(breakout_window=20)

        assert strategy.params['breakout_window'] == 20

    def test_signal_generation(self, simple_price_data):
        """Test breakout strategy generates signals."""
        strategy = BreakoutStrategy(breakout_window=15)
        entries, exits = strategy.generate_signals(simple_price_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(simple_price_data)

    def test_breakout_on_trending_data(self, simple_price_data):
        """Test breakout generates entries on strong uptrend."""
        strategy = BreakoutStrategy(breakout_window=10)
        entries, exits = strategy.generate_signals(simple_price_data)

        assert entries.sum() > 0, "Should detect breakouts on trending data"


class TestTripleMovingAverage:
    """Test Triple Moving Average strategy."""

    def test_strategy_initialization(self):
        """Test strategy initializes with valid parameters."""
        strategy = TripleMovingAverage(fast_window=10, medium_window=20, slow_window=50)

        assert strategy.params['fast_window'] == 10
        assert strategy.params['medium_window'] == 20
        assert strategy.params['slow_window'] == 50

    def test_invalid_window_order(self):
        """Test validation rejects invalid window ordering."""
        with pytest.raises(ValueError):
            TripleMovingAverage(fast_window=20, medium_window=10, slow_window=50)

        with pytest.raises(ValueError):
            TripleMovingAverage(fast_window=10, medium_window=50, slow_window=30)

    def test_signal_generation(self, simple_price_data):
        """Test triple MA strategy generates signals."""
        strategy = TripleMovingAverage(fast_window=5, medium_window=10, slow_window=20)
        entries, exits = strategy.generate_signals(simple_price_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(simple_price_data)
