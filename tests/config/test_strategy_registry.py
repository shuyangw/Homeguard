"""
Tests for strategy registry.
"""

import pytest

from src.strategies.registry import (
    get_strategy_class,
    list_strategies,
    list_strategy_display_names,
    get_strategy_info,
    register_strategy,
    clear_cache,
)
from src.backtesting.base.strategy import BaseStrategy


class TestListStrategies:
    """Tests for list_strategies function."""

    def test_returns_list(self):
        """Test that list_strategies returns a list."""
        strategies = list_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_includes_known_strategies(self):
        """Test that known strategies are in the list."""
        strategies = list_strategies()
        assert "MovingAverageCrossover" in strategies
        assert "MeanReversion" in strategies
        assert "MomentumStrategy" in strategies

    def test_list_is_sorted(self):
        """Test that the list is sorted alphabetically."""
        strategies = list_strategies()
        assert strategies == sorted(strategies)


class TestListStrategyDisplayNames:
    """Tests for list_strategy_display_names function."""

    def test_returns_dict(self):
        """Test that display names returns a dict."""
        names = list_strategy_display_names()
        assert isinstance(names, dict)
        assert len(names) > 0

    def test_includes_known_display_names(self):
        """Test that known display names are present."""
        names = list_strategy_display_names()
        assert "Moving Average Crossover" in names
        assert "Mean Reversion" in names
        assert "Momentum Strategy" in names

    def test_display_names_map_to_class_names(self):
        """Test that display names map to valid class names."""
        names = list_strategy_display_names()
        strategies = list_strategies()

        for display_name, class_name in names.items():
            assert class_name in strategies, f"{display_name} maps to unknown class {class_name}"


class TestGetStrategyClass:
    """Tests for get_strategy_class function."""

    def test_get_by_class_name(self):
        """Test getting strategy by class name."""
        cls = get_strategy_class("MovingAverageCrossover")
        assert cls is not None
        assert cls.__name__ == "MovingAverageCrossover"

    def test_get_by_display_name(self):
        """Test getting strategy by display name."""
        cls = get_strategy_class("Moving Average Crossover")
        assert cls is not None
        assert cls.__name__ == "MovingAverageCrossover"

    def test_case_insensitive_lookup(self):
        """Test case-insensitive name lookup."""
        cls1 = get_strategy_class("movingaveragecrossover")
        cls2 = get_strategy_class("MOVINGAVERAGECROSSOVER")
        assert cls1 == cls2

    def test_returns_subclass_of_base_strategy(self):
        """Test that returned class is a strategy subclass."""
        cls = get_strategy_class("MeanReversion")
        assert issubclass(cls, BaseStrategy)

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_strategy_class("NonExistentStrategy")
        assert "Unknown strategy" in str(exc_info.value)

    def test_strategy_can_be_instantiated(self):
        """Test that returned strategy can be instantiated."""
        cls = get_strategy_class("MovingAverageCrossover")
        instance = cls()
        assert instance is not None

    def test_strategy_with_parameters(self):
        """Test instantiating strategy with parameters."""
        cls = get_strategy_class("MovingAverageCrossover")
        instance = cls(fast_window=5, slow_window=20)
        assert instance.params["fast_window"] == 5
        assert instance.params["slow_window"] == 20


class TestGetStrategyInfo:
    """Tests for get_strategy_info function."""

    def test_returns_dict(self):
        """Test that strategy info returns a dict."""
        info = get_strategy_info("MovingAverageCrossover")
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        """Test that info contains required keys."""
        info = get_strategy_info("MovingAverageCrossover")
        assert "class_name" in info
        assert "module" in info
        assert "description" in info
        assert "parameters" in info

    def test_parameters_is_dict(self):
        """Test that parameters is a dict."""
        info = get_strategy_info("MovingAverageCrossover")
        assert isinstance(info["parameters"], dict)


class TestRegisterStrategy:
    """Tests for register_strategy function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        # Create a minimal test strategy
        class TestStrategy(BaseStrategy):
            def __init__(self, param1=10):
                super().__init__()
                self.params = {"param1": param1}

            def generate_signals(self, data):
                import pandas as pd
                return pd.Series(False, index=data.index), pd.Series(False, index=data.index)

        register_strategy("TestStrategy", TestStrategy, "Test Strategy")

        # Should now be retrievable
        cls = get_strategy_class("TestStrategy")
        assert cls == TestStrategy

        # Should also work by display name
        cls = get_strategy_class("Test Strategy")
        assert cls == TestStrategy


class TestCaching:
    """Tests for strategy class caching."""

    def test_cache_works(self):
        """Test that strategy classes are cached."""
        clear_cache()

        # First load
        cls1 = get_strategy_class("MeanReversion")
        # Second load (should be cached)
        cls2 = get_strategy_class("MeanReversion")

        assert cls1 is cls2

    def test_clear_cache(self):
        """Test that clear_cache works."""
        # Load a strategy
        get_strategy_class("MeanReversion")

        # Clear cache
        clear_cache()

        # Should still work (will reload)
        cls = get_strategy_class("MeanReversion")
        assert cls is not None
