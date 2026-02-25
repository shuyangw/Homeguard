"""
Unit tests for FRS strategy.
"""

import pytest
import pandas as pd
import numpy as np

from src.strategies.advanced.frs_strategy import FRSStrategy
from src.strategies.advanced.frs_indicators import FRSRegime
from src.backtesting.base.strategy import LongShortStrategy


class TestFRSStrategyInit:
    """Tests for FRS strategy initialization."""

    def test_default_parameters(self):
        """Test strategy initializes with default parameters."""
        strategy = FRSStrategy()

        assert strategy.hurst_window == 100
        assert strategy.hurst_smooth_span == 10
        assert strategy.trend_threshold == 0.60
        assert strategy.mr_threshold == 0.40
        assert strategy.donchian_period == 20
        assert strategy.bollinger_period == 20
        assert strategy.bollinger_std == 2.0
        assert strategy.rsi_period == 14
        assert strategy.rsi_overbought == 70
        assert strategy.rsi_oversold == 30

    def test_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = FRSStrategy(
            hurst_window=50,
            trend_threshold=0.65,
            mr_threshold=0.35,
            donchian_period=25,
            rsi_overbought=80,
        )

        assert strategy.hurst_window == 50
        assert strategy.trend_threshold == 0.65
        assert strategy.mr_threshold == 0.35
        assert strategy.donchian_period == 25
        assert strategy.rsi_overbought == 80

    def test_inherits_long_short_strategy(self):
        """Test strategy inherits from LongShortStrategy."""
        strategy = FRSStrategy()
        assert isinstance(strategy, LongShortStrategy)

    def test_params_dict(self):
        """Test params dict is populated correctly."""
        strategy = FRSStrategy(hurst_window=50, trend_threshold=0.65)

        assert strategy.params['hurst_window'] == 50
        assert strategy.params['trend_threshold'] == 0.65


class TestFRSStrategyValidation:
    """Tests for parameter validation."""

    def test_hurst_window_too_small(self):
        """Test hurst_window < 20 raises ValueError."""
        with pytest.raises(ValueError, match="hurst_window must be >= 20"):
            FRSStrategy(hurst_window=10)

    def test_hurst_window_boundary(self):
        """Test hurst_window = 20 is accepted."""
        strategy = FRSStrategy(hurst_window=20)
        assert strategy.hurst_window == 20

    def test_trend_must_exceed_mr(self):
        """Test trend_threshold <= mr_threshold raises ValueError."""
        with pytest.raises(ValueError, match="trend_threshold .* must be > mr_threshold"):
            FRSStrategy(trend_threshold=0.40, mr_threshold=0.40)

        with pytest.raises(ValueError, match="trend_threshold .* must be > mr_threshold"):
            FRSStrategy(trend_threshold=0.40, mr_threshold=0.50)

    def test_mr_threshold_range(self):
        """Test mr_threshold must be in (0, 0.5)."""
        with pytest.raises(ValueError, match="mr_threshold should be in"):
            FRSStrategy(mr_threshold=0.55)

        with pytest.raises(ValueError, match="mr_threshold should be in"):
            FRSStrategy(mr_threshold=0.0)

    def test_trend_threshold_range(self):
        """Test trend_threshold must be in (0.5, 1.0)."""
        with pytest.raises(ValueError, match="trend_threshold should be in"):
            FRSStrategy(trend_threshold=0.45)

        with pytest.raises(ValueError, match="trend_threshold should be in"):
            FRSStrategy(trend_threshold=1.0)

    def test_donchian_period_too_small(self):
        """Test donchian_period < 5 raises ValueError."""
        with pytest.raises(ValueError, match="donchian_period must be >= 5"):
            FRSStrategy(donchian_period=3)

    def test_bollinger_period_too_small(self):
        """Test bollinger_period < 5 raises ValueError."""
        with pytest.raises(ValueError, match="bollinger_period must be >= 5"):
            FRSStrategy(bollinger_period=3)

    def test_rsi_period_too_small(self):
        """Test rsi_period < 2 raises ValueError."""
        with pytest.raises(ValueError, match="rsi_period must be >= 2"):
            FRSStrategy(rsi_period=1)

    def test_rsi_threshold_order(self):
        """Test RSI thresholds must satisfy oversold < overbought."""
        with pytest.raises(ValueError, match="RSI thresholds"):
            FRSStrategy(rsi_oversold=80, rsi_overbought=20)

        with pytest.raises(ValueError, match="RSI thresholds"):
            FRSStrategy(rsi_oversold=50, rsi_overbought=50)

    def test_valid_parameters(self):
        """Test valid parameter combinations."""
        strategy = FRSStrategy(
            hurst_window=100,
            trend_threshold=0.65,
            mr_threshold=0.35,
            donchian_period=20,
            bollinger_period=20,
            rsi_period=14,
            rsi_overbought=80,
            rsi_oversold=20,
        )

        assert strategy.trend_threshold == 0.65
        assert strategy.mr_threshold == 0.35


class TestFRSStrategySignals:
    """Tests for signal generation."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)

        return df

    def test_generate_signals_return_type(self):
        """Test generate_signals returns correct types."""
        strategy = FRSStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        assert isinstance(long_entries, pd.Series)
        assert isinstance(long_exits, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(short_exits, pd.Series)

    def test_generate_signals_boolean_type(self):
        """Test generate_signals returns boolean Series."""
        strategy = FRSStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        assert long_entries.dtype == bool
        assert long_exits.dtype == bool
        assert short_entries.dtype == bool
        assert short_exits.dtype == bool

    def test_generate_signals_length(self):
        """Test generate_signals returns same length as input."""
        strategy = FRSStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        assert len(long_entries) == len(data)
        assert len(long_exits) == len(data)
        assert len(short_entries) == len(data)
        assert len(short_exits) == len(data)

    def test_generate_signals_index(self):
        """Test generate_signals preserves index."""
        strategy = FRSStrategy()
        data = self.create_sample_ohlcv()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)

        pd.testing.assert_index_equal(long_entries.index, data.index)
        pd.testing.assert_index_equal(long_exits.index, data.index)
        pd.testing.assert_index_equal(short_entries.index, data.index)
        pd.testing.assert_index_equal(short_exits.index, data.index)

    def test_generate_signals_empty_data(self):
        """Test generate_signals handles empty data."""
        strategy = FRSStrategy()
        empty_data = pd.DataFrame()

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(empty_data)

        assert len(long_entries) == 0
        assert len(long_exits) == 0
        assert len(short_entries) == 0
        assert len(short_exits) == 0

    def test_generate_signals_missing_columns(self):
        """Test generate_signals raises on missing required columns."""
        strategy = FRSStrategy()
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            # Missing 'low' and 'close'
        })

        with pytest.raises(ValueError, match="missing required columns"):
            strategy.generate_signals(data)


class TestFRSStrategyHelpers:
    """Tests for helper methods."""

    def create_sample_ohlcv(self, days: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(days) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': close * (1 + np.random.randn(days) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(days)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(days)) * 0.01),
            'close': close,
        }, index=dates)

    def test_get_parameters(self):
        """Test get_parameters returns correct dict."""
        strategy = FRSStrategy(
            hurst_window=50,
            trend_threshold=0.65,
            mr_threshold=0.35,
        )

        params = strategy.get_parameters()

        assert params['hurst_window'] == 50
        assert params['trend_threshold'] == 0.65
        assert params['mr_threshold'] == 0.35

    def test_get_regime(self):
        """Test get_regime returns regime series."""
        strategy = FRSStrategy(hurst_window=100)
        data = self.create_sample_ohlcv()

        regime = strategy.get_regime(data)

        assert isinstance(regime, pd.Series)
        assert len(regime) == len(data)

        # Check all values are valid regimes
        valid_regimes = {FRSRegime.TRENDING.value, FRSRegime.MEAN_REVERTING.value, FRSRegime.NOISE.value}
        assert set(regime.unique()).issubset(valid_regimes)

    def test_get_indicators(self):
        """Test get_indicators returns indicator DataFrame."""
        strategy = FRSStrategy(hurst_window=100)
        data = self.create_sample_ohlcv()

        indicators = strategy.get_indicators(data)

        assert isinstance(indicators, pd.DataFrame)
        assert len(indicators) == len(data)

        # Check expected columns
        expected_cols = ['hurst', 'hurst_smoothed', 'donchian_upper', 'donchian_lower',
                        'bb_upper', 'bb_middle', 'bb_lower', 'rsi', 'atr']
        for col in expected_cols:
            assert col in indicators.columns

    def test_get_current_signal(self):
        """Test get_current_signal returns FRSSignal."""
        strategy = FRSStrategy(hurst_window=100)
        data = self.create_sample_ohlcv()

        signal = strategy.get_current_signal(data)

        # Check it has required attributes
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'close')
        assert hasattr(signal, 'hurst')
        assert hasattr(signal, 'regime')

    def test_get_regime_distribution(self):
        """Test get_regime_distribution returns percentages."""
        strategy = FRSStrategy(hurst_window=100)
        data = self.create_sample_ohlcv()

        dist = strategy.get_regime_distribution(data)

        assert 'TRENDING' in dist
        assert 'MEAN_REVERTING' in dist
        assert 'NOISE' in dist

        # Percentages should sum to ~1.0
        total = dist['TRENDING'] + dist['MEAN_REVERTING'] + dist['NOISE']
        assert abs(total - 1.0) < 0.01

    def test_str_repr(self):
        """Test string representation."""
        strategy = FRSStrategy(hurst_window=50, trend_threshold=0.65)

        repr_str = repr(strategy)

        assert 'FRSStrategy' in repr_str
        assert 'hurst_window=50' in repr_str


class TestFRSStrategyColumnHandling:
    """Tests for column name handling (uppercase/lowercase)."""

    def test_lowercase_columns(self):
        """Test strategy handles lowercase column names."""
        strategy = FRSStrategy()
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200))

        data = pd.DataFrame({
            'open': close * 1.001,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
        }, index=dates)

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
        assert len(long_entries) == 200

    def test_uppercase_columns(self):
        """Test strategy handles uppercase column names."""
        strategy = FRSStrategy()
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200))

        data = pd.DataFrame({
            'Open': close * 1.001,
            'High': close * 1.01,
            'Low': close * 0.99,
            'Close': close,
        }, index=dates)

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
        assert len(long_entries) == 200

    def test_mixed_case_columns(self):
        """Test strategy handles mixed case column names."""
        strategy = FRSStrategy()
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200))

        data = pd.DataFrame({
            'OPEN': close * 1.001,
            'High': close * 1.01,
            'low': close * 0.99,
            'Close': close,
        }, index=dates)

        long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
        assert len(long_entries) == 200


class TestFRSStrategyRegistry:
    """Tests for strategy registry integration."""

    def test_strategy_in_registry(self):
        """Test FRSStrategy is in the strategy registry."""
        from src.strategies.registry import get_strategy_class, list_strategies

        strategies = list_strategies()
        assert 'FRSStrategy' in strategies

    def test_get_strategy_class(self):
        """Test getting FRSStrategy from registry."""
        from src.strategies.registry import get_strategy_class

        strategy_cls = get_strategy_class('FRSStrategy')
        assert strategy_cls == FRSStrategy

    def test_get_by_display_name(self):
        """Test getting strategy by display name."""
        from src.strategies.registry import get_strategy_class

        # Test various display names
        assert get_strategy_class('FRS') == FRSStrategy
        assert get_strategy_class('Fractal Regime Switching') == FRSStrategy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
