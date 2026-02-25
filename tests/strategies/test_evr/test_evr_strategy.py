"""
Unit tests for EVR strategy.
"""

import pytest
import pandas as pd
import numpy as np

from src.strategies.advanced.evr_strategy import EVRStrategy
from src.backtesting.base.strategy import LongShortStrategy


class TestEVRStrategyInit:
    """Tests for EVR strategy initialization."""

    def test_default_parameters(self):
        """Test strategy initializes with default parameters."""
        strategy = EVRStrategy()

        assert strategy.vol_lookback == 20
        assert strategy.spread_lookback == 20
        assert strategy.vol_threshold == 2.0
        assert strategy.spread_threshold == 0.8
        assert strategy.rsi_period == 14
        assert strategy.rsi_oversold == 30
        assert strategy.rsi_overbought == 70
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0
        assert strategy.require_confirmation == True

    def test_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = EVRStrategy(
            vol_lookback=30,
            vol_threshold=2.5,
            spread_threshold=0.7,
            rsi_period=10,
            rsi_oversold=25,
            require_confirmation=False,
        )

        assert strategy.vol_lookback == 30
        assert strategy.vol_threshold == 2.5
        assert strategy.spread_threshold == 0.7
        assert strategy.rsi_period == 10
        assert strategy.rsi_oversold == 25
        assert strategy.require_confirmation == False

    def test_inherits_long_short_strategy(self):
        """Test strategy inherits from LongShortStrategy."""
        strategy = EVRStrategy()
        assert isinstance(strategy, LongShortStrategy)

    def test_params_dict(self):
        """Test params dict is populated correctly."""
        strategy = EVRStrategy(vol_lookback=30, vol_threshold=2.5)

        params = strategy.get_parameters()
        assert params['vol_lookback'] == 30
        assert params['vol_threshold'] == 2.5


class TestEVRStrategyValidation:
    """Tests for parameter validation."""

    def test_vol_lookback_too_small(self):
        """Test vol_lookback < 5 raises ValueError."""
        with pytest.raises(ValueError, match="vol_lookback must be >= 5"):
            EVRStrategy(vol_lookback=3)

    def test_spread_lookback_too_small(self):
        """Test spread_lookback < 5 raises ValueError."""
        with pytest.raises(ValueError, match="spread_lookback must be >= 5"):
            EVRStrategy(spread_lookback=3)

    def test_vol_threshold_too_low(self):
        """Test vol_threshold <= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="vol_threshold must be > 1.0"):
            EVRStrategy(vol_threshold=0.8)

        with pytest.raises(ValueError, match="vol_threshold must be > 1.0"):
            EVRStrategy(vol_threshold=1.0)

    def test_spread_threshold_out_of_range(self):
        """Test spread_threshold must be in (0, 1.0)."""
        with pytest.raises(ValueError, match="spread_threshold must be in"):
            EVRStrategy(spread_threshold=0.0)

        with pytest.raises(ValueError, match="spread_threshold must be in"):
            EVRStrategy(spread_threshold=1.0)

        with pytest.raises(ValueError, match="spread_threshold must be in"):
            EVRStrategy(spread_threshold=1.5)

    def test_rsi_period_too_small(self):
        """Test rsi_period < 2 raises ValueError."""
        with pytest.raises(ValueError, match="rsi_period must be >= 2"):
            EVRStrategy(rsi_period=1)

    def test_rsi_threshold_order(self):
        """Test RSI thresholds must satisfy oversold < overbought."""
        with pytest.raises(ValueError, match="RSI thresholds"):
            EVRStrategy(rsi_oversold=80, rsi_overbought=20)

        with pytest.raises(ValueError, match="RSI thresholds"):
            EVRStrategy(rsi_oversold=50, rsi_overbought=50)

    def test_bb_period_too_small(self):
        """Test bb_period < 5 raises ValueError."""
        with pytest.raises(ValueError, match="bb_period must be >= 5"):
            EVRStrategy(bb_period=3)

    def test_bb_std_non_positive(self):
        """Test bb_std <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="bb_std must be > 0"):
            EVRStrategy(bb_std=0)

        with pytest.raises(ValueError, match="bb_std must be > 0"):
            EVRStrategy(bb_std=-1)

    def test_valid_parameters(self):
        """Test valid parameter combinations."""
        strategy = EVRStrategy(
            vol_lookback=20,
            spread_lookback=20,
            vol_threshold=2.0,
            spread_threshold=0.8,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            bb_period=20,
            bb_std=2.0,
        )

        assert strategy.vol_threshold == 2.0
        assert strategy.spread_threshold == 0.8


class TestEVRStrategySignals:
    """Tests for signal generation."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_generate_signals_return_type(self):
        """Test generate_signals returns correct types."""
        strategy = EVRStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        assert isinstance(long_entries, pd.Series)
        assert isinstance(long_exits, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(short_exits, pd.Series)

    def test_generate_signals_boolean_type(self):
        """Test generate_signals returns boolean Series."""
        strategy = EVRStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_generate_signals_length(self):
        """Test generate_signals returns same length as input."""
        strategy = EVRStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        assert len(long_entries) == len(data)
        assert len(long_exits) == len(data)
        assert len(short_entries) == len(data)
        assert len(short_exits) == len(data)

    def test_generate_signals_index(self):
        """Test generate_signals preserves index."""
        strategy = EVRStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        pd.testing.assert_index_equal(long_entries.index, data.index)
        pd.testing.assert_index_equal(long_exits.index, data.index)
        pd.testing.assert_index_equal(short_entries.index, data.index)
        pd.testing.assert_index_equal(short_exits.index, data.index)

    def test_generate_signals_empty_data(self):
        """Test generate_signals handles empty data."""
        strategy = EVRStrategy()
        empty_data = pd.DataFrame()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(empty_data)

        assert len(long_entries) == 0
        assert len(long_exits) == 0
        assert len(short_entries) == 0
        assert len(short_exits) == 0

    def test_generate_signals_missing_columns(self):
        """Test generate_signals raises on missing required columns."""
        strategy = EVRStrategy()
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            # Missing 'low', 'close', 'volume'
        })

        with pytest.raises(ValueError, match="missing required columns"):
            strategy.generate_signals(data)


class TestEVRStrategyHelpers:
    """Tests for helper methods."""

    def create_sample_ohlcv(self, days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(np.random.randn(days)) * 2 + 1
        high = close + spread / 2
        low = close - spread / 2
        open_ = close + np.random.randn(days) * 0.5
        volume = 1000000 + np.random.randn(days) * 100000

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_get_parameters(self):
        """Test get_parameters returns correct dict."""
        strategy = EVRStrategy(
            vol_lookback=30,
            vol_threshold=2.5,
            spread_threshold=0.7,
        )

        params = strategy.get_parameters()

        assert params['vol_lookback'] == 30
        assert params['vol_threshold'] == 2.5
        assert params['spread_threshold'] == 0.7

    def test_get_indicators(self):
        """Test get_indicators returns indicator DataFrame."""
        strategy = EVRStrategy()
        data = self.create_sample_ohlcv()

        indicators = strategy.get_indicators(data)

        assert isinstance(indicators, pd.DataFrame)
        assert len(indicators) == len(data)

        # Check expected columns
        expected_cols = ['rel_volume', 'rel_spread', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower']
        for col in expected_cols:
            assert col in indicators.columns

    def test_get_current_signal(self):
        """Test get_current_signal returns EVRSignal."""
        strategy = EVRStrategy()
        data = self.create_sample_ohlcv()

        signal = strategy.get_current_signal(data)

        # Check it has required attributes
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'close')
        assert hasattr(signal, 'volume')
        assert hasattr(signal, 'rel_volume')
        assert hasattr(signal, 'rel_spread')
        assert hasattr(signal, 'is_absorption')
        assert hasattr(signal, 'context')

    def test_str_repr(self):
        """Test string representation."""
        strategy = EVRStrategy(vol_lookback=30, vol_threshold=2.5)

        repr_str = repr(strategy)

        assert 'EVRStrategy' in repr_str
        assert 'vol_lookback=30' in repr_str
        assert 'vol_threshold=2.5' in repr_str


class TestEVRStrategyColumnHandling:
    """Tests for column name handling (uppercase/lowercase)."""

    def test_lowercase_columns(self):
        """Test strategy handles lowercase column names."""
        strategy = EVRStrategy()
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100))

        data = pd.DataFrame({
            'open': close * 1.001,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': np.random.randint(1000000, 2000000, 100),
        }, index=dates)

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
        assert len(long_entries) == 100

    def test_uppercase_columns(self):
        """Test strategy handles uppercase column names."""
        strategy = EVRStrategy()
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100))

        data = pd.DataFrame({
            'Open': close * 1.001,
            'High': close * 1.01,
            'Low': close * 0.99,
            'Close': close,
            'Volume': np.random.randint(1000000, 2000000, 100),
        }, index=dates)

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
        assert len(long_entries) == 100

    def test_mixed_case_columns(self):
        """Test strategy handles mixed case column names."""
        strategy = EVRStrategy()
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100))

        data = pd.DataFrame({
            'OPEN': close * 1.001,
            'High': close * 1.01,
            'low': close * 0.99,
            'Close': close,
            'VOLUME': np.random.randint(1000000, 2000000, 100),
        }, index=dates)

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
        assert len(long_entries) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
