"""
BrokerInterface Unit Tests

Tests that verify broker implementations conform to the BrokerInterface contract.
"""

import pytest
from datetime import datetime

from src.trading.brokers.broker_interface import (
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)
from tests.trading.mock_broker import MockBroker


class TestBrokerInterface:
    """
    Test suite for BrokerInterface compliance.

    These tests verify that broker implementations correctly implement
    the BrokerInterface contract.
    """

    @pytest.fixture
    def broker(self):
        """Create mock broker for testing."""
        return MockBroker(initial_cash=100000.0)

    # ==================== Account Tests ====================

    def test_get_account(self, broker):
        """Test getting account information."""
        account = broker.get_account()

        assert 'account_id' in account
        assert 'cash' in account
        assert 'buying_power' in account
        assert 'portfolio_value' in account

        assert account['cash'] == 100000.0
        assert account['portfolio_value'] == 100000.0

    def test_get_positions_empty(self, broker):
        """Test getting positions when none exist."""
        positions = broker.get_positions()
        assert positions == []

    def test_get_position_nonexistent(self, broker):
        """Test getting nonexistent position."""
        position = broker.get_position('AAPL')
        assert position is None

    # ==================== Order Tests ====================

    def test_place_market_order_buy(self, broker):
        """Test placing market buy order."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        assert order['symbol'] == 'AAPL'
        assert order['quantity'] == 10
        assert order['side'] == OrderSide.BUY.value
        assert order['status'] == OrderStatus.FILLED.value
        assert order['filled_qty'] == 10
        assert 'order_id' in order
        assert 'filled_avg_price' in order

    def test_place_market_order_sell(self, broker):
        """Test placing market sell order."""
        # First buy to create position
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.place_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        # Then sell
        broker.set_quote('AAPL', bid=155.0, ask=155.05)
        order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET
        )

        assert order['symbol'] == 'AAPL'
        assert order['quantity'] == 10
        assert order['side'] == OrderSide.SELL.value
        assert order['status'] == OrderStatus.FILLED.value

    def test_place_limit_order(self, broker):
        """Test placing limit order."""
        order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )

        assert order['order_type'] == OrderType.LIMIT.value
        assert order['limit_price'] == 150.0

    def test_place_stop_order(self, broker):
        """Test placing stop order."""
        order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=150.0
        )

        assert order['order_type'] == OrderType.STOP.value
        assert order['stop_price'] == 150.0

    def test_get_order(self, broker):
        """Test getting order by ID."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        placed_order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        retrieved_order = broker.get_order(placed_order['order_id'])
        assert retrieved_order['order_id'] == placed_order['order_id']
        assert retrieved_order['symbol'] == 'AAPL'

    def test_get_orders(self, broker):
        """Test getting all orders."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.place_order('AAPL', 10, OrderSide.BUY)
        broker.place_order('AAPL', 5, OrderSide.BUY)

        orders = broker.get_orders()
        assert len(orders) >= 2

    def test_cancel_order(self, broker):
        """Test canceling order."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        order = broker.place_order('AAPL', 10, OrderSide.BUY)

        # Note: In mock broker, order is already filled
        # In real broker, you'd cancel before fill
        cancelled = broker.cancel_order(order['order_id'])
        assert cancelled['order_id'] == order['order_id']

    # ==================== Position Tests ====================

    def test_position_after_buy(self, broker):
        """Test that position is created after buy order."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.place_order('AAPL', 10, OrderSide.BUY)

        position = broker.get_position('AAPL')
        assert position is not None
        assert position['symbol'] == 'AAPL'
        assert position['quantity'] == 10

        positions = broker.get_positions()
        assert len(positions) == 1

    def test_close_position(self, broker):
        """Test closing position."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.place_order('AAPL', 10, OrderSide.BUY)

        close_order = broker.close_position('AAPL')
        assert close_order['symbol'] == 'AAPL'
        assert close_order['side'] == OrderSide.SELL.value

        position = broker.get_position('AAPL')
        assert position is None

    def test_close_all_positions(self, broker):
        """Test closing all positions."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.set_quote('MSFT', bid=300.0, ask=300.05)

        broker.place_order('AAPL', 10, OrderSide.BUY)
        broker.place_order('MSFT', 5, OrderSide.BUY)

        closed_orders = broker.close_all_positions()
        assert len(closed_orders) == 2

        positions = broker.get_positions()
        assert len(positions) == 0

    # ==================== Market Data Tests ====================

    def test_get_latest_quote(self, broker):
        """Test getting latest quote."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        quote = broker.get_latest_quote('AAPL')

        assert quote['symbol'] == 'AAPL'
        assert 'bid' in quote
        assert 'ask' in quote
        assert 'timestamp' in quote

    def test_get_latest_trade(self, broker):
        """Test getting latest trade."""
        trade = broker.get_latest_trade('AAPL')

        assert trade['symbol'] == 'AAPL'
        assert 'price' in trade
        assert 'size' in trade
        assert 'timestamp' in trade

    def test_get_bars(self, broker):
        """Test getting historical bars."""
        bars = broker.get_bars('AAPL', '1Day', limit=10)

        assert len(bars) > 0
        bar = bars[0]
        assert 'timestamp' in bar
        assert 'open' in bar
        assert 'high' in bar
        assert 'low' in bar
        assert 'close' in bar
        assert 'volume' in bar

    # ==================== Utility Tests ====================

    def test_is_market_open(self, broker):
        """Test checking market open status."""
        is_open = broker.is_market_open()
        assert isinstance(is_open, bool)

    def test_get_market_hours(self, broker):
        """Test getting market hours."""
        hours = broker.get_market_hours()

        assert 'is_open' in hours
        assert isinstance(hours['is_open'], bool)

    def test_test_connection(self, broker):
        """Test connection test."""
        result = broker.test_connection()
        assert result is True

    # ==================== Integration Tests ====================

    def test_buy_and_sell_workflow(self, broker):
        """Test complete buy and sell workflow."""
        initial_cash = broker.cash

        # Set quote
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        # Buy
        buy_order = broker.place_order('AAPL', 10, OrderSide.BUY)
        assert buy_order['status'] == OrderStatus.FILLED.value

        # Check position
        position = broker.get_position('AAPL')
        assert position['quantity'] == 10

        # Check cash decreased
        assert broker.cash < initial_cash

        # Sell
        broker.set_quote('AAPL', bid=155.0, ask=155.05)
        sell_order = broker.place_order('AAPL', 10, OrderSide.SELL)
        assert sell_order['status'] == OrderStatus.FILLED.value

        # Check position closed
        position = broker.get_position('AAPL')
        assert position is None

        # Check cash increased (profit)
        assert broker.cash > initial_cash

    def test_multiple_positions(self, broker):
        """Test managing multiple positions."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.set_quote('MSFT', bid=300.0, ask=300.05)
        broker.set_quote('TSLA', bid=200.0, ask=200.05)

        # Open 3 positions
        broker.place_order('AAPL', 10, OrderSide.BUY)
        broker.place_order('MSFT', 5, OrderSide.BUY)
        broker.place_order('TSLA', 8, OrderSide.BUY)

        positions = broker.get_positions()
        assert len(positions) == 3

        symbols = {p['symbol'] for p in positions}
        assert symbols == {'AAPL', 'MSFT', 'TSLA'}

    def test_order_time_in_force(self, broker):
        """Test different time in force options."""
        for tif in [TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK]:
            order = broker.place_order(
                symbol='AAPL',
                quantity=1,
                side=OrderSide.BUY,
                time_in_force=tif
            )
            assert order['time_in_force'] == tif.value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
