"""Tests for HomeguardOrderFactory."""

from datetime import datetime

import pandas as pd
import pytest

from src.backtesting.adapters.nautilus.order_factory import (
    HomeguardOrderFactory,
    MockOrder,
)
from src.strategies.core.signal import Signal


class TestOrderFactoryInit:
    """Tests for HomeguardOrderFactory initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        factory = HomeguardOrderFactory()
        assert factory.default_quantity == 100
        assert factory.venue == "XNAS"
        assert factory.strategy is None

    def test_init_custom(self):
        """Test initialization with custom values."""
        factory = HomeguardOrderFactory(
            default_quantity=50,
            venue="NYSE"
        )
        assert factory.default_quantity == 50
        assert factory.venue == "NYSE"


class TestSignalToOrder:
    """Tests for signal_to_order conversion."""

    def test_buy_signal_to_order(self):
        """Test BUY signal creates correct order."""
        factory = HomeguardOrderFactory()
        signal = Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="AAPL",
            direction="BUY",
            confidence=0.85,
            price=185.50,
        )

        order = factory.signal_to_order(signal)

        assert order is not None
        assert isinstance(order, MockOrder)
        assert order.side == "BUY"
        assert "AAPL" in order.instrument_id
        assert order.quantity == 100  # default

    def test_sell_signal_to_order(self):
        """Test SELL signal creates correct order."""
        factory = HomeguardOrderFactory()
        signal = Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="MSFT",
            direction="SELL",
            confidence=0.75,
            price=376.20,
        )

        order = factory.signal_to_order(signal)

        assert order is not None
        assert order.side == "SELL"
        assert "MSFT" in order.instrument_id

    def test_hold_signal_returns_none(self):
        """Test HOLD signal returns None (no order)."""
        factory = HomeguardOrderFactory()
        signal = Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="GOOGL",
            direction="HOLD",
            confidence=0.50,
            price=140.80,
        )

        order = factory.signal_to_order(signal)

        assert order is None

    def test_custom_quantity(self):
        """Test signal with custom quantity."""
        factory = HomeguardOrderFactory()
        signal = Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="AAPL",
            direction="BUY",
            confidence=0.85,
            price=185.50,
        )

        order = factory.signal_to_order(signal, quantity=250)

        assert order.quantity == 250


class TestBooleanSeriesToOrders:
    """Tests for boolean_series_to_orders conversion."""

    def test_entry_signal_creates_buy(self):
        """Test entry signal creates BUY order."""
        factory = HomeguardOrderFactory()

        # Create boolean Series with entry at index 5
        entries = pd.Series([False] * 10)
        entries.iloc[5] = True
        exits = pd.Series([False] * 10)
        prices = pd.Series([100.0 + i for i in range(10)])

        orders = factory.boolean_series_to_orders(
            symbol="AAPL",
            entries=entries,
            exits=exits,
            prices=prices,
            bar_idx=5,
        )

        assert len(orders) == 1
        assert orders[0].side == "BUY"

    def test_exit_signal_creates_sell(self):
        """Test exit signal creates SELL order."""
        factory = HomeguardOrderFactory()

        entries = pd.Series([False] * 10)
        exits = pd.Series([False] * 10)
        exits.iloc[7] = True
        prices = pd.Series([100.0 + i for i in range(10)])

        orders = factory.boolean_series_to_orders(
            symbol="AAPL",
            entries=entries,
            exits=exits,
            prices=prices,
            bar_idx=7,
        )

        assert len(orders) == 1
        assert orders[0].side == "SELL"

    def test_both_signals_creates_two_orders(self):
        """Test entry and exit at same bar creates two orders."""
        factory = HomeguardOrderFactory()

        entries = pd.Series([False] * 10)
        entries.iloc[5] = True
        exits = pd.Series([False] * 10)
        exits.iloc[5] = True
        prices = pd.Series([100.0 + i for i in range(10)])

        orders = factory.boolean_series_to_orders(
            symbol="AAPL",
            entries=entries,
            exits=exits,
            prices=prices,
            bar_idx=5,
        )

        assert len(orders) == 2
        sides = {o.side for o in orders}
        assert "BUY" in sides
        assert "SELL" in sides

    def test_no_signals_creates_empty_list(self):
        """Test no signals creates empty list."""
        factory = HomeguardOrderFactory()

        entries = pd.Series([False] * 10)
        exits = pd.Series([False] * 10)
        prices = pd.Series([100.0 + i for i in range(10)])

        orders = factory.boolean_series_to_orders(
            symbol="AAPL",
            entries=entries,
            exits=exits,
            prices=prices,
            bar_idx=5,
        )

        assert len(orders) == 0

    def test_out_of_bounds_index(self):
        """Test out of bounds index returns empty list."""
        factory = HomeguardOrderFactory()

        entries = pd.Series([True] * 5)
        exits = pd.Series([False] * 5)
        prices = pd.Series([100.0] * 5)

        # Index beyond series length
        orders = factory.boolean_series_to_orders(
            symbol="AAPL",
            entries=entries,
            exits=exits,
            prices=prices,
            bar_idx=10,
        )

        assert len(orders) == 0


class TestSignalsToOrders:
    """Tests for batch signal conversion."""

    def test_multiple_signals(self, sample_signals):
        """Test converting multiple signals."""
        factory = HomeguardOrderFactory()

        orders = factory.signals_to_orders(sample_signals)

        # Should have 2 orders (BUY and SELL), HOLD filtered out
        assert len(orders) == 2

        symbols = {o.instrument_id.split(".")[0] for o in orders}
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_with_custom_quantities(self, sample_signals):
        """Test with custom quantities per symbol."""
        factory = HomeguardOrderFactory()

        quantities = {
            "AAPL": 200,
            "MSFT": 50,
        }

        orders = factory.signals_to_orders(sample_signals, quantities=quantities)

        for order in orders:
            symbol = order.instrument_id.split(".")[0]
            if symbol == "AAPL":
                assert order.quantity == 200
            elif symbol == "MSFT":
                assert order.quantity == 50

    def test_empty_signals(self):
        """Test with empty signal list."""
        factory = HomeguardOrderFactory()

        orders = factory.signals_to_orders([])

        assert len(orders) == 0


class TestOrderIdGeneration:
    """Tests for unique order ID generation."""

    def test_unique_order_ids(self):
        """Test that each order gets unique ID."""
        factory = HomeguardOrderFactory()

        signal = Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="AAPL",
            direction="BUY",
            confidence=0.85,
            price=185.50,
        )

        order1 = factory.signal_to_order(signal)
        order2 = factory.signal_to_order(signal)
        order3 = factory.signal_to_order(signal)

        ids = {order1.client_order_id, order2.client_order_id, order3.client_order_id}
        assert len(ids) == 3  # All unique
