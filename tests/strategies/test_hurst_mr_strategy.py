"""
Tests for Hurst Mean Reversion Strategy.

Tests cover:
- Strategy initialization and parameter validation
- Signal generation with Hurst filter
- Registry lookup
- Integration with backtest engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.advanced.hurst_mr_strategy import (
    HurstMRStrategy,
    HurstMRPosition,
    HurstMRSignal,
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
    """Create data with clear trend (should have high Hurst)."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

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
def mean_reverting_data():
    """Create data with mean-reverting characteristics (should have low Hurst)."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # Mean-reverting oscillation around 100
    base = 100
    # Create oscillation that reverts to mean
    oscillation = 10 * np.sin(np.linspace(0, 16*np.pi, len(dates)))
    # Add some noise
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
# STRATEGY INITIALIZATION TESTS
# =============================================================================

class TestHurstMRStrategyInit:
    """Tests for strategy initialization."""

    def test_default_initialization(self):
        """Test strategy initializes with default parameters."""
        strategy = HurstMRStrategy()

        assert strategy.hurst_window == 100
        assert strategy.hurst_threshold == 0.45
        assert strategy.zscore_window == 20
        assert strategy.zscore_entry_threshold == 2.0
        assert strategy.zscore_exit_threshold == 0.5
        assert strategy.atr_period == 14
        assert strategy.atr_stop_multiplier == 1.5
        assert strategy.atr_target_multiplier == 4.5
        assert strategy.long_only is False
        assert strategy.max_hold_bars == 10

    def test_custom_initialization(self):
        """Test strategy initializes with custom parameters."""
        strategy = HurstMRStrategy(
            hurst_window=50,
            hurst_threshold=0.40,
            zscore_entry_threshold=1.5,
            long_only=True
        )

        assert strategy.hurst_window == 50
        assert strategy.hurst_threshold == 0.40
        assert strategy.zscore_entry_threshold == 1.5
        assert strategy.long_only is True

    def test_fixed_pct_exits(self):
        """Test fixed percentage exits configuration."""
        strategy = HurstMRStrategy(
            use_fixed_pct_exits=True,
            fixed_stop_pct=0.05,
            fixed_target_pct=0.15
        )

        assert strategy.use_fixed_pct_exits is True
        assert strategy.fixed_stop_pct == 0.05
        assert strategy.fixed_target_pct == 0.15


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

class TestHurstMRStrategyValidation:
    """Tests for parameter validation."""

    def test_validate_valid_params(self):
        """Test validation passes for valid parameters."""
        strategy = HurstMRStrategy()
        strategy.validate_parameters()  # Should not raise

    def test_validate_invalid_hurst_window(self):
        """Test validation rejects too small Hurst window."""
        with pytest.raises(ValueError, match="hurst_window must be > 20"):
            strategy = HurstMRStrategy(hurst_window=10)
            strategy.validate_parameters()

    def test_validate_invalid_hurst_threshold(self):
        """Test validation rejects invalid Hurst threshold."""
        with pytest.raises(ValueError, match="hurst_threshold must be between"):
            strategy = HurstMRStrategy(hurst_threshold=1.5)
            strategy.validate_parameters()

    def test_validate_invalid_zscore_window(self):
        """Test validation rejects invalid Z-score window."""
        with pytest.raises(ValueError, match="zscore_window must be positive"):
            strategy = HurstMRStrategy(zscore_window=-1)
            strategy.validate_parameters()

    def test_validate_invalid_max_hold_bars(self):
        """Test validation rejects invalid max_hold_bars."""
        with pytest.raises(ValueError, match="max_hold_bars must be positive"):
            strategy = HurstMRStrategy(max_hold_bars=0)
            strategy.validate_parameters()


# =============================================================================
# SIGNAL GENERATION TESTS
# =============================================================================

class TestHurstMRStrategySignals:
    """Tests for signal generation."""

    def test_generate_signals_returns_correct_types(self, sample_ohlcv_data):
        """Test signal generation returns correct types."""
        strategy = HurstMRStrategy()

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        assert isinstance(long_e, pd.Series)
        assert isinstance(long_x, pd.Series)
        assert isinstance(short_e, pd.Series)
        assert isinstance(short_x, pd.Series)
        assert long_e.dtype == bool
        assert len(long_e) == len(sample_ohlcv_data)

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = HurstMRStrategy(hurst_window=100)

        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        small_df = pd.DataFrame({
            'open': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'close': [100] * 50,
            'volume': [1e6] * 50
        }, index=dates)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(small_df)

        # Should return empty signals (insufficient data for Hurst)
        assert long_e.sum() == 0

    def test_long_only_mode(self, sample_ohlcv_data):
        """Test long_only mode disables short signals."""
        strategy = HurstMRStrategy(long_only=True)

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        # Short entries should be all False in long_only mode
        assert short_e.sum() == 0

    def test_reset_state(self, sample_ohlcv_data):
        """Test reset_state clears strategy state."""
        strategy = HurstMRStrategy()

        # Generate some signals
        strategy.generate_long_short_signals(sample_ohlcv_data)

        # Reset
        strategy.reset_state()

        assert strategy._current_position is None


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestHurstMRStrategyRegistry:
    """Tests for registry integration."""

    def test_registry_lookup_by_class_name(self):
        """Test strategy can be loaded by class name."""
        strategy_cls = get_strategy_class("HurstMRStrategy")

        assert strategy_cls is HurstMRStrategy

    def test_registry_lookup_by_display_name(self):
        """Test strategy can be loaded by display name."""
        strategy_cls = get_strategy_class("Hurst Mean Reversion")

        assert strategy_cls is HurstMRStrategy

    def test_registry_lookup_by_alias(self):
        """Test strategy can be loaded by alias."""
        strategy_cls = get_strategy_class("Hurst MR")

        assert strategy_cls is HurstMRStrategy

    def test_instantiate_from_registry(self):
        """Test strategy can be instantiated after registry lookup."""
        strategy_cls = get_strategy_class("HurstMRStrategy")
        strategy = strategy_cls(hurst_window=50)

        assert strategy.hurst_window == 50


# =============================================================================
# SIGNAL STATE TESTS
# =============================================================================

class TestHurstMRSignal:
    """Tests for live signal state."""

    def test_get_current_signal(self, sample_ohlcv_data):
        """Test current signal generation."""
        strategy = HurstMRStrategy()

        signal = strategy.get_current_signal(sample_ohlcv_data)

        assert isinstance(signal, HurstMRSignal)
        assert hasattr(signal, 'zscore')
        assert hasattr(signal, 'hurst')
        assert hasattr(signal, 'atr')
        assert hasattr(signal, 'is_mean_reverting')
        assert hasattr(signal, 'entry_long')
        assert hasattr(signal, 'entry_short')
        assert hasattr(signal, 'stop_long')
        assert hasattr(signal, 'target_long')

    def test_hurst_stats(self, sample_ohlcv_data):
        """Test Hurst statistics calculation."""
        strategy = HurstMRStrategy()

        stats = strategy.get_hurst_stats(sample_ohlcv_data)

        assert 'mean_hurst' in stats
        assert 'std_hurst' in stats
        assert 'mean_reverting_pct' in stats
        assert 'trending_pct' in stats
        assert 'random_walk_pct' in stats

        # Percentages should sum to 100
        total_pct = stats['mean_reverting_pct'] + stats['random_walk_pct'] + stats['trending_pct']
        assert total_pct == pytest.approx(100.0, rel=0.01)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHurstMRStrategyIntegration:
    """Integration tests."""

    def test_full_backtest_flow(self, sample_ohlcv_data):
        """Test complete backtest flow from data to signals."""
        # Load from registry
        strategy_cls = get_strategy_class("HurstMRStrategy")

        # Configure strategy
        strategy = strategy_cls(
            hurst_window=50,
            hurst_threshold=0.50,
            zscore_entry_threshold=1.5,
            long_only=False
        )

        # Generate signals
        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        # Validate outputs
        assert len(long_e) == len(sample_ohlcv_data)
        assert long_e.index.equals(sample_ohlcv_data.index)

    def test_entries_have_exits(self, sample_ohlcv_data):
        """Test that all entries have corresponding exits."""
        strategy = HurstMRStrategy(
            hurst_window=50,
            hurst_threshold=0.60,  # Relaxed threshold for more signals
            zscore_entry_threshold=1.0,  # Relaxed for more signals
            max_hold_bars=10
        )

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        # Count entries and exits
        long_entries = long_e.sum()
        long_exits = long_x.sum()

        # Due to max_hold_bars enforcement, exits should be >= entries
        # (some may have multiple exit conditions on same bar)
        if long_entries > 0:
            assert long_exits >= long_entries

    def test_max_hold_bars_enforced(self, sample_ohlcv_data):
        """Test that positions don't exceed max_hold_bars."""
        max_bars = 5
        strategy = HurstMRStrategy(
            hurst_window=50,
            hurst_threshold=0.60,
            zscore_entry_threshold=1.0,
            max_hold_bars=max_bars
        )

        long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(sample_ohlcv_data)

        # Find entry-exit pairs and check duration
        entry_indices = np.where(long_e)[0]
        exit_indices = np.where(long_x)[0]

        for entry_idx in entry_indices:
            # Find next exit after this entry
            exits_after = exit_indices[exit_indices > entry_idx]
            if len(exits_after) > 0:
                exit_idx = exits_after[0]
                hold_duration = exit_idx - entry_idx
                assert hold_duration <= max_bars, f"Hold duration {hold_duration} exceeds max {max_bars}"
