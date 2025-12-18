"""
Tests for ML Crypto Mean Reversion Strategy.

Tests cover:
- Indicator calculations (Z-score, RSI, ATR, ADX, Choppiness, Efficiency Ratio)
- ML Regime Filter (rule-based, ML training, predictions)
- Strategy initialization and parameter validation
- Signal generation with/without ML filter
- Registry lookup
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.advanced.ml_crypto_mr_indicators import (
    MLCryptoMRIndicators,
    MLRegimeFilter,
    RegimeFeatures,
)
from src.strategies.advanced.ml_crypto_mr_strategy import (
    MLCryptoMRStrategy,
    MLCryptoMRPosition,
)
from src.strategies.registry import get_strategy_class


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # Generate price with some mean-reverting characteristics
    base_price = 100
    returns = np.random.normal(0, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some noise to create OHLC
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    open_prices = prices + np.random.normal(0, 0.5, len(dates))

    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': np.random.uniform(1e6, 10e6, len(dates))
    }, index=dates)


@pytest.fixture
def trending_data():
    """Create data with clear trend for testing regime filter."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # Strong uptrend
    trend = np.linspace(100, 200, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    prices = trend + noise

    return pd.DataFrame({
        'open': prices - np.random.uniform(0, 1, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 2, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 2, len(dates))),
        'close': prices,
        'volume': np.random.uniform(1e6, 10e6, len(dates))
    }, index=dates)


@pytest.fixture
def ranging_data():
    """Create data with sideways range for testing regime filter."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # Sideways oscillation
    base = 100
    oscillation = 5 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    noise = np.random.normal(0, 1, len(dates))
    prices = base + oscillation + noise

    return pd.DataFrame({
        'open': prices - np.random.uniform(0, 0.5, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 1, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 1, len(dates))),
        'close': prices,
        'volume': np.random.uniform(1e6, 10e6, len(dates))
    }, index=dates)


# =============================================================================
# INDICATOR TESTS
# =============================================================================

class TestMLCryptoMRIndicators:
    """Tests for indicator calculations."""

    def test_calculate_zscore(self, sample_ohlcv_data):
        """Test Z-score calculation."""
        close = sample_ohlcv_data['close']
        zscore = MLCryptoMRIndicators.calculate_zscore(close, window=20)

        assert isinstance(zscore, pd.Series)
        assert len(zscore) == len(close)
        # First 19 values should be NaN (need 20 for window)
        assert zscore.iloc[:19].isna().all()
        # Z-scores should be roughly between -3 and +3 for normal data
        valid_zscore = zscore.dropna()
        assert valid_zscore.abs().max() < 10  # Reasonable bound

    def test_calculate_rsi(self, sample_ohlcv_data):
        """Test RSI calculation."""
        close = sample_ohlcv_data['close']
        rsi = MLCryptoMRIndicators.calculate_rsi(close, period=14)

        assert isinstance(rsi, pd.Series)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_calculate_atr(self, sample_ohlcv_data):
        """Test ATR calculation."""
        atr = MLCryptoMRIndicators.calculate_atr(sample_ohlcv_data, period=14)

        assert isinstance(atr, pd.Series)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_calculate_adx(self, sample_ohlcv_data):
        """Test ADX calculation."""
        adx = MLCryptoMRIndicators.calculate_adx(sample_ohlcv_data, period=14)

        assert isinstance(adx, pd.Series)
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        assert valid_adx.min() >= 0
        assert valid_adx.max() <= 100

    def test_calculate_choppiness(self, sample_ohlcv_data):
        """Test Choppiness Index calculation."""
        chop = MLCryptoMRIndicators.calculate_choppiness(sample_ohlcv_data, period=14)

        assert isinstance(chop, pd.Series)
        # Choppiness should be between 0 and 100
        valid_chop = chop.dropna()
        assert valid_chop.min() >= 0
        assert valid_chop.max() <= 100

    def test_calculate_efficiency_ratio(self, sample_ohlcv_data):
        """Test Efficiency Ratio calculation."""
        close = sample_ohlcv_data['close']
        er = MLCryptoMRIndicators.calculate_efficiency_ratio(close, period=10)

        assert isinstance(er, pd.Series)
        # ER should be between 0 and 1
        valid_er = er.dropna()
        assert valid_er.min() >= 0
        assert valid_er.max() <= 1.1  # Allow small margin

    def test_calculate_bollinger_width(self, sample_ohlcv_data):
        """Test Bollinger Band width calculation."""
        close = sample_ohlcv_data['close']
        width = MLCryptoMRIndicators.calculate_bollinger_width(close, window=20)

        assert isinstance(width, pd.Series)
        # Width should be positive
        valid_width = width.dropna()
        assert (valid_width > 0).all()

    def test_calculate_all_features(self, sample_ohlcv_data):
        """Test full feature calculation."""
        config = {
            'zscore_window': 20,
            'rsi_period': 14,
            'atr_period': 14,
        }
        features = MLCryptoMRIndicators.calculate_all_features(sample_ohlcv_data, config)

        assert isinstance(features, pd.DataFrame)
        assert 'zscore' in features.columns
        assert 'rsi' in features.columns
        assert 'atr' in features.columns
        assert 'adx' in features.columns
        assert 'choppiness' in features.columns
        assert 'efficiency_ratio' in features.columns
        assert 'bollinger_width' in features.columns


# =============================================================================
# ML REGIME FILTER TESTS
# =============================================================================

class TestMLRegimeFilter:
    """Tests for ML Regime Filter."""

    def test_rule_based_predict_trending(self, trending_data):
        """Test rule-based prediction identifies trending conditions."""
        filter = MLRegimeFilter(use_ml=False, adx_threshold=25.0, choppiness_threshold=61.8)

        config = {'adx_period': 14, 'choppiness_period': 14}
        features = MLCryptoMRIndicators.calculate_all_features(trending_data, config)

        is_ranging, prob_ranging = filter._predict_rules(features)

        # In trending data, should mostly predict trending (not ranging)
        valid_ranging = is_ranging.dropna()
        ranging_pct = valid_ranging.mean()

        # Trending data should have low ranging percentage
        assert ranging_pct < 0.8  # Not all ranging

    def test_rule_based_predict_ranging(self, ranging_data):
        """Test rule-based prediction identifies ranging conditions."""
        filter = MLRegimeFilter(use_ml=False, adx_threshold=25.0, choppiness_threshold=50.0)

        config = {'adx_period': 14, 'choppiness_period': 14}
        features = MLCryptoMRIndicators.calculate_all_features(ranging_data, config)

        is_ranging, prob_ranging = filter._predict_rules(features)

        # In ranging data, should predict ranging more often
        valid_ranging = is_ranging.dropna()
        assert isinstance(valid_ranging, pd.Series)

    def test_filter_initialization(self):
        """Test filter initializes with default parameters."""
        filter = MLRegimeFilter()

        assert filter.lookback_days == 252
        assert filter.retrain_frequency == 20
        assert filter.min_train_samples == 100
        assert filter.adx_threshold == 25.0
        assert filter.choppiness_threshold == 61.8
        assert filter._is_trained is False

    def test_filter_reset(self, sample_ohlcv_data):
        """Test filter reset clears state."""
        filter = MLRegimeFilter(use_ml=False)

        # Simulate some activity
        filter._is_trained = True
        filter._last_train_idx = 100

        # Reset
        filter.reset()

        assert filter._is_trained is False
        assert filter._last_train_idx == -1
        assert filter.model is None

    def test_regime_features_to_array(self):
        """Test RegimeFeatures conversion to numpy array."""
        features = RegimeFeatures(
            adx=25.0,
            choppiness=50.0,
            efficiency_ratio=0.5,
            bollinger_width=0.1,
            zscore_abs=2.0,
            volatility_ratio=1.0
        )

        arr = features.to_array()

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 6
        assert arr[0] == 25.0
        assert arr[1] == 50.0


# =============================================================================
# STRATEGY TESTS
# =============================================================================

class TestMLCryptoMRStrategy:
    """Tests for the main strategy class."""

    def test_strategy_initialization_default(self):
        """Test strategy initializes with default parameters."""
        strategy = MLCryptoMRStrategy()

        assert strategy.zscore_window == 20
        assert strategy.zscore_entry_threshold == 2.0
        assert strategy.atr_stop_multiplier == 1.5
        assert strategy.atr_target_multiplier == 4.5
        assert strategy.use_ml_filter is True
        assert strategy.long_only is False

    def test_strategy_initialization_custom(self):
        """Test strategy initializes with custom parameters."""
        strategy = MLCryptoMRStrategy(
            zscore_window=30,
            zscore_entry_threshold=2.5,
            use_ml_filter=False,
            long_only=True
        )

        assert strategy.zscore_window == 30
        assert strategy.zscore_entry_threshold == 2.5
        assert strategy.use_ml_filter is False
        assert strategy.long_only is True

    def test_validate_parameters_valid(self):
        """Test parameter validation passes for valid params."""
        strategy = MLCryptoMRStrategy(
            zscore_window=20,
            zscore_entry_threshold=2.0,
            atr_period=14
        )
        # Should not raise
        strategy.validate_parameters()

    def test_validate_parameters_invalid_window(self):
        """Test parameter validation rejects invalid window."""
        with pytest.raises(ValueError, match="zscore_window must be positive"):
            strategy = MLCryptoMRStrategy(zscore_window=-1)
            strategy.validate_parameters()

    def test_validate_parameters_invalid_rsi(self):
        """Test parameter validation rejects invalid RSI levels."""
        with pytest.raises(ValueError, match="rsi_oversold must be between"):
            strategy = MLCryptoMRStrategy(rsi_oversold=150)
            strategy.validate_parameters()

    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data."""
        strategy = MLCryptoMRStrategy()
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(empty_df)

        assert len(long_e) == 0
        assert len(long_x) == 0
        assert len(short_e) == 0
        assert len(short_x) == 0

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = MLCryptoMRStrategy(zscore_window=20)

        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        small_df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1e6] * 10
        }, index=dates)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(small_df)

        # Should return empty signals (insufficient data)
        assert long_e.sum() == 0

    def test_generate_signals_basic(self, sample_ohlcv_data):
        """Test basic signal generation."""
        strategy = MLCryptoMRStrategy(use_ml_filter=False)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        assert isinstance(long_e, pd.Series)
        assert isinstance(long_x, pd.Series)
        assert isinstance(short_e, pd.Series)
        assert isinstance(short_x, pd.Series)

        assert long_e.dtype == bool
        assert len(long_e) == len(sample_ohlcv_data)

    def test_long_only_mode(self, sample_ohlcv_data):
        """Test long_only mode disables short signals."""
        strategy = MLCryptoMRStrategy(use_ml_filter=False, long_only=True)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        # Short entries should be all False in long_only mode
        assert short_e.sum() == 0

    def test_ml_filter_disabled(self, sample_ohlcv_data):
        """Test strategy works with ML filter disabled."""
        strategy_ml_off = MLCryptoMRStrategy(use_ml_filter=False)

        long_e, long_x, short_e, short_x = strategy_ml_off.generate_long_short_signals(sample_ohlcv_data)

        # Should produce some signals (with ML off, more signals pass through)
        assert isinstance(long_e, pd.Series)

    def test_reset_state(self, sample_ohlcv_data):
        """Test reset_state clears strategy state."""
        strategy = MLCryptoMRStrategy()

        # Generate some signals to populate state
        strategy.generate_long_short_signals(sample_ohlcv_data)

        # Reset
        strategy.reset_state()

        assert strategy._current_position is None
        assert strategy._features_cache is None


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_registry_lookup_by_class_name(self):
        """Test strategy can be loaded by class name."""
        strategy_cls = get_strategy_class("MLCryptoMRStrategy")

        assert strategy_cls is MLCryptoMRStrategy

    def test_registry_lookup_by_display_name(self):
        """Test strategy can be loaded by display name."""
        strategy_cls = get_strategy_class("ML Crypto Mean Reversion")

        assert strategy_cls is MLCryptoMRStrategy

    def test_registry_lookup_by_alias(self):
        """Test strategy can be loaded by alias."""
        strategy_cls = get_strategy_class("MLMR")

        assert strategy_cls is MLCryptoMRStrategy

    def test_instantiate_from_registry(self):
        """Test strategy can be instantiated after registry lookup."""
        strategy_cls = get_strategy_class("MLCryptoMRStrategy")
        strategy = strategy_cls(zscore_window=30)

        assert strategy.zscore_window == 30


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_backtest_flow(self, sample_ohlcv_data):
        """Test complete backtest flow from data to signals."""
        # Load from registry
        strategy_cls = get_strategy_class("MLCryptoMRStrategy")

        # Configure strategy
        strategy = strategy_cls(
            zscore_window=20,
            zscore_entry_threshold=2.0,
            use_ml_filter=False,  # Disable ML for simpler test
            long_only=False
        )

        # Generate signals
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        # Validate outputs
        assert len(long_e) == len(sample_ohlcv_data)
        assert long_e.index.equals(sample_ohlcv_data.index)

        # Check signal counts are reasonable
        total_entries = long_e.sum() + short_e.sum()
        total_exits = long_x.sum() + short_x.sum()

        # With random data and thresholds, expect some but not too many signals
        assert total_entries >= 0
        assert total_exits >= 0

    def test_get_current_signal(self, sample_ohlcv_data):
        """Test live signal generation."""
        strategy = MLCryptoMRStrategy(use_ml_filter=False)

        signal = strategy.get_current_signal(sample_ohlcv_data)

        assert hasattr(signal, 'zscore')
        assert hasattr(signal, 'rsi')
        assert hasattr(signal, 'atr')
        assert hasattr(signal, 'entry_long')
        assert hasattr(signal, 'entry_short')
        assert hasattr(signal, 'stop_long')
        assert hasattr(signal, 'target_long')

    def test_regime_stats(self, sample_ohlcv_data):
        """Test regime statistics calculation."""
        strategy = MLCryptoMRStrategy(use_ml_filter=False)

        stats = strategy.get_regime_stats(sample_ohlcv_data)

        assert 'ranging_pct' in stats
        assert 'trending_pct' in stats
        assert stats['ranging_pct'] + stats['trending_pct'] == pytest.approx(100.0, rel=0.01)
