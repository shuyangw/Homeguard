"""
Unit tests for DSTS strategy.
"""

import pytest
import pandas as pd
import numpy as np

from src.strategies.advanced.dsts_strategy import DSTSStrategy
from src.backtesting.base.strategy import LongOnlyStrategy


class TestDSTSStrategyInit:
    """Tests for DSTS strategy initialization."""

    def test_default_parameters(self):
        """Test strategy initializes with default parameters."""
        strategy = DSTSStrategy()

        assert strategy.period == 65
        assert strategy.bull_threshold == 0.75
        assert strategy.bear_threshold == 0.0

    def test_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = DSTSStrategy(
            period=50,
            bull_threshold=1.0,
            bear_threshold=-0.5,
        )

        assert strategy.period == 50
        assert strategy.bull_threshold == 1.0
        assert strategy.bear_threshold == -0.5

    def test_inherits_long_only_strategy(self):
        """Test strategy inherits from LongOnlyStrategy."""
        strategy = DSTSStrategy()
        assert isinstance(strategy, LongOnlyStrategy)

    def test_params_dict(self):
        """Test params dict is populated correctly."""
        strategy = DSTSStrategy(period=50, bull_threshold=1.0)

        assert strategy.params['period'] == 50
        assert strategy.params['bull_threshold'] == 1.0


class TestDSTSStrategyValidation:
    """Tests for parameter validation."""

    def test_period_too_small(self):
        """Test period < 10 raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 10"):
            DSTSStrategy(period=5)

    def test_period_boundary(self):
        """Test period = 10 is accepted."""
        strategy = DSTSStrategy(period=10)
        assert strategy.period == 10

    def test_bull_must_exceed_bear(self):
        """Test bull_threshold <= bear_threshold raises ValueError."""
        with pytest.raises(ValueError, match="bull_threshold .* must be > bear_threshold"):
            DSTSStrategy(bull_threshold=0.5, bear_threshold=0.5)

        with pytest.raises(ValueError, match="bull_threshold .* must be > bear_threshold"):
            DSTSStrategy(bull_threshold=0.5, bear_threshold=0.75)

    def test_valid_thresholds(self):
        """Test valid threshold combinations."""
        # Positive thresholds
        strategy = DSTSStrategy(bull_threshold=1.0, bear_threshold=0.5)
        assert strategy.bull_threshold == 1.0

        # Negative bear threshold
        strategy = DSTSStrategy(bull_threshold=0.5, bear_threshold=-0.5)
        assert strategy.bear_threshold == -0.5


class TestDSTSStrategySignals:
    """Tests for signal generation."""

    def create_sample_data(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.03
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, days).astype(float),
        }, index=dates)

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

        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.005)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.005)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, days).astype(float),
        }, index=dates)

    def test_generate_signals_return_type(self):
        """Test generate_signals returns correct types."""
        strategy = DSTSStrategy()
        data = self.create_sample_data()

        entries, exits = strategy.generate_signals(data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_generate_signals_length(self):
        """Test generate_signals returns same length as input."""
        strategy = DSTSStrategy()
        data = self.create_sample_data()

        entries, exits = strategy.generate_signals(data)

        assert len(entries) == len(data)
        assert len(exits) == len(data)

    def test_generate_signals_index(self):
        """Test generate_signals preserves index."""
        strategy = DSTSStrategy()
        data = self.create_sample_data()

        entries, exits = strategy.generate_signals(data)

        pd.testing.assert_index_equal(entries.index, data.index)
        pd.testing.assert_index_equal(exits.index, data.index)

    def test_generate_signals_empty_data(self):
        """Test generate_signals handles empty data."""
        strategy = DSTSStrategy()
        empty_data = pd.DataFrame()

        entries, exits = strategy.generate_signals(empty_data)

        assert len(entries) == 0
        assert len(exits) == 0

    def test_generate_signals_missing_close(self):
        """Test generate_signals raises on missing close column."""
        strategy = DSTSStrategy()
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            'low': [1, 2, 3],
        })

        with pytest.raises(ValueError, match="close"):
            strategy.generate_signals(data)

    def test_uptrend_generates_entry(self):
        """Test uptrend generates at least one entry signal."""
        strategy = DSTSStrategy()
        data = self.create_trending_data('up', days=200)

        entries, exits = strategy.generate_signals(data)

        assert entries.any(), "Expected entry signal in uptrend"

    def test_long_only_signals(self):
        """Test generate_long_signals is called by generate_signals."""
        strategy = DSTSStrategy()
        data = self.create_sample_data()

        # Both methods should return same results
        entries1, exits1 = strategy.generate_signals(data)
        entries2, exits2 = strategy.generate_long_signals(data)

        pd.testing.assert_series_equal(entries1, entries2)
        pd.testing.assert_series_equal(exits1, exits2)


class TestDSTSStrategyHelpers:
    """Tests for helper methods."""

    def create_sample_data(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.03
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'close': prices,
        }, index=dates)

    def test_get_parameters(self):
        """Test get_parameters returns correct dict."""
        strategy = DSTSStrategy(period=50, bull_threshold=1.0, bear_threshold=-0.25)

        params = strategy.get_parameters()

        assert params == {
            'period': 50,
            'bull_threshold': 1.0,
            'bear_threshold': -0.25,
        }

    def test_get_zscore(self):
        """Test get_zscore returns Z-score series."""
        strategy = DSTSStrategy(period=65)
        data = self.create_sample_data()

        zscore = strategy.get_zscore(data)

        assert isinstance(zscore, pd.Series)
        assert len(zscore) == len(data)

    def test_get_indicators(self):
        """Test get_indicators returns indicator DataFrame."""
        strategy = DSTSStrategy(period=65)
        data = self.create_sample_data()

        indicators = strategy.get_indicators(data)

        assert isinstance(indicators, pd.DataFrame)
        assert list(indicators.columns) == ['close', 'ema', 'std', 'zscore']
        assert len(indicators) == len(data)

    def test_str_repr(self):
        """Test string representation."""
        strategy = DSTSStrategy(period=50, bull_threshold=1.0, bear_threshold=0.0)

        repr_str = repr(strategy)

        assert 'DSTSStrategy' in repr_str
        assert 'period=50' in repr_str


class TestDSTSStrategyColumnHandling:
    """Tests for column name handling (uppercase/lowercase)."""

    def test_lowercase_close(self):
        """Test strategy handles lowercase 'close' column."""
        strategy = DSTSStrategy()
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.linspace(100, 200, 100),
        }, index=dates)

        entries, exits = strategy.generate_signals(data)
        assert len(entries) == 100

    def test_uppercase_close(self):
        """Test strategy handles uppercase 'Close' column."""
        strategy = DSTSStrategy()
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.linspace(100, 200, 100),
        }, index=dates)

        entries, exits = strategy.generate_signals(data)
        assert len(entries) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
