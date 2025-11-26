"""
Interface Compliance Tests

Verifies that broker implementations correctly implement all interface contracts.
Tests backward compatibility aliases and type compliance.
"""

import pytest
from abc import ABC
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.trading.brokers import (
    # Interfaces
    AccountInterface,
    MarketHoursInterface,
    MarketDataInterface,
    OrderManagementInterface,
    StockTradingInterface,
    OptionsTradingInterface,
    BrokerInterface,
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    OptionType,
    OptionRight,
    # Implementations
    AlpacaBroker,
)
from tests.trading.mock_broker import MockBroker


class TestInterfaceHierarchy:
    """Test that interface inheritance is correct."""

    def test_stock_trading_inherits_order_management(self):
        """StockTradingInterface should inherit from OrderManagementInterface."""
        assert issubclass(StockTradingInterface, OrderManagementInterface)

    def test_options_trading_inherits_order_management(self):
        """OptionsTradingInterface should inherit from OrderManagementInterface."""
        assert issubclass(OptionsTradingInterface, OrderManagementInterface)

    def test_broker_interface_inherits_all_stock_interfaces(self):
        """BrokerInterface should inherit from all stock-related interfaces."""
        assert issubclass(BrokerInterface, AccountInterface)
        assert issubclass(BrokerInterface, MarketHoursInterface)
        assert issubclass(BrokerInterface, MarketDataInterface)
        assert issubclass(BrokerInterface, StockTradingInterface)

    def test_broker_interface_does_not_inherit_options(self):
        """BrokerInterface should NOT inherit from OptionsTradingInterface."""
        assert not issubclass(BrokerInterface, OptionsTradingInterface)

    def test_all_interfaces_are_abstract(self):
        """All interfaces should be abstract base classes."""
        interfaces = [
            AccountInterface,
            MarketHoursInterface,
            MarketDataInterface,
            OrderManagementInterface,
            StockTradingInterface,
            OptionsTradingInterface,
        ]
        for interface in interfaces:
            assert issubclass(interface, ABC), f"{interface.__name__} should be ABC"


class TestMockBrokerCompliance:
    """Test that MockBroker implements all required interfaces."""

    @pytest.fixture
    def broker(self):
        return MockBroker(initial_cash=100000.0)

    # ==================== AccountInterface ====================

    def test_implements_account_interface(self, broker):
        """MockBroker should implement AccountInterface."""
        assert isinstance(broker, AccountInterface)

    def test_get_account_returns_dict(self, broker):
        """get_account should return Dict with required keys."""
        account = broker.get_account()
        assert isinstance(account, dict)
        assert 'account_id' in account
        assert 'cash' in account
        assert 'buying_power' in account
        assert 'portfolio_value' in account

    def test_test_connection_returns_bool(self, broker):
        """test_connection should return bool."""
        result = broker.test_connection()
        assert isinstance(result, bool)

    # ==================== MarketHoursInterface ====================

    def test_implements_market_hours_interface(self, broker):
        """MockBroker should implement MarketHoursInterface."""
        assert isinstance(broker, MarketHoursInterface)

    def test_is_market_open_returns_bool(self, broker):
        """is_market_open should return bool."""
        result = broker.is_market_open()
        assert isinstance(result, bool)

    def test_get_market_hours_returns_tuple(self, broker):
        """get_market_hours should return Tuple[datetime, datetime]."""
        result = broker.get_market_hours(datetime.now())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], datetime)
        assert isinstance(result[1], datetime)

    # ==================== MarketDataInterface ====================

    def test_implements_market_data_interface(self, broker):
        """MockBroker should implement MarketDataInterface."""
        assert isinstance(broker, MarketDataInterface)

    def test_get_latest_quote_returns_dict(self, broker):
        """get_latest_quote should return Dict with required keys."""
        quote = broker.get_latest_quote('AAPL')
        assert isinstance(quote, dict)
        assert 'symbol' in quote
        assert 'bid' in quote
        assert 'ask' in quote
        assert 'timestamp' in quote

    def test_get_latest_trade_returns_dict(self, broker):
        """get_latest_trade should return Dict with required keys."""
        trade = broker.get_latest_trade('AAPL')
        assert isinstance(trade, dict)
        assert 'symbol' in trade
        assert 'price' in trade
        assert 'size' in trade
        assert 'timestamp' in trade

    def test_get_bars_returns_dataframe(self, broker):
        """get_bars should return pandas DataFrame."""
        end = datetime.now()
        start = end - timedelta(days=5)
        bars = broker.get_bars(['AAPL'], '1Day', start, end)
        assert isinstance(bars, pd.DataFrame)
        assert 'open' in bars.columns
        assert 'high' in bars.columns
        assert 'low' in bars.columns
        assert 'close' in bars.columns
        assert 'volume' in bars.columns

    # ==================== OrderManagementInterface ====================

    def test_implements_order_management_interface(self, broker):
        """MockBroker should implement OrderManagementInterface."""
        assert isinstance(broker, OrderManagementInterface)

    def test_cancel_order_returns_bool(self, broker):
        """cancel_order should return bool."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        order = broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        result = broker.cancel_order(order['order_id'])
        assert isinstance(result, bool)
        assert result is True

    def test_get_order_returns_dict(self, broker):
        """get_order should return Dict."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        placed = broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        order = broker.get_order(placed['order_id'])
        assert isinstance(order, dict)
        assert 'order_id' in order

    def test_get_orders_returns_list(self, broker):
        """get_orders should return List[Dict]."""
        orders = broker.get_orders()
        assert isinstance(orders, list)

    # ==================== StockTradingInterface ====================

    def test_implements_stock_trading_interface(self, broker):
        """MockBroker should implement StockTradingInterface."""
        assert isinstance(broker, StockTradingInterface)

    def test_get_stock_positions_returns_list(self, broker):
        """get_stock_positions should return List[Dict]."""
        positions = broker.get_stock_positions()
        assert isinstance(positions, list)

    def test_get_stock_position_returns_dict_or_none(self, broker):
        """get_stock_position should return Dict or None."""
        position = broker.get_stock_position('NONEXISTENT')
        assert position is None

        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        position = broker.get_stock_position('AAPL')
        assert isinstance(position, dict)

    def test_place_stock_order_returns_dict(self, broker):
        """place_stock_order should return Dict with order details."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        order = broker.place_stock_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        assert isinstance(order, dict)
        assert 'order_id' in order
        assert 'symbol' in order
        assert 'quantity' in order
        assert 'side' in order
        assert 'status' in order

    def test_close_stock_position_returns_dict(self, broker):
        """close_stock_position should return Dict."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        order = broker.close_stock_position('AAPL')
        assert isinstance(order, dict)

    def test_close_all_stock_positions_returns_list(self, broker):
        """close_all_stock_positions should return List[Dict]."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.set_quote('MSFT', bid=200.0, ask=200.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        broker.place_stock_order('MSFT', 5, OrderSide.BUY)
        orders = broker.close_all_stock_positions()
        assert isinstance(orders, list)
        assert len(orders) == 2


class TestBackwardCompatibilityAliases:
    """Test that backward compatibility method aliases work correctly."""

    @pytest.fixture
    def broker(self):
        return MockBroker(initial_cash=100000.0)

    def test_get_positions_alias(self, broker):
        """get_positions should call get_stock_positions."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)

        old_result = broker.get_positions()
        new_result = broker.get_stock_positions()
        assert old_result == new_result

    def test_get_position_alias(self, broker):
        """get_position should call get_stock_position."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)

        old_result = broker.get_position('AAPL')
        new_result = broker.get_stock_position('AAPL')
        assert old_result == new_result

    def test_place_order_alias(self, broker):
        """place_order should call place_stock_order."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        old_result = broker.place_order('AAPL', 10, OrderSide.BUY)
        assert 'order_id' in old_result

    def test_close_position_alias(self, broker):
        """close_position should call close_stock_position."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)

        order = broker.close_position('AAPL')
        assert order['side'] == OrderSide.SELL.value

    def test_close_all_positions_alias(self, broker):
        """close_all_positions should call close_all_stock_positions."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)

        orders = broker.close_all_positions()
        assert len(orders) == 1


class TestEnumsExported:
    """Test that all enums are properly exported."""

    def test_order_side_values(self):
        """OrderSide enum should have BUY and SELL."""
        assert OrderSide.BUY.value == 'buy'
        assert OrderSide.SELL.value == 'sell'

    def test_order_type_values(self):
        """OrderType enum should have all order types."""
        assert OrderType.MARKET.value == 'market'
        assert OrderType.LIMIT.value == 'limit'
        assert OrderType.STOP.value == 'stop'
        assert OrderType.STOP_LIMIT.value == 'stop_limit'

    def test_order_status_values(self):
        """OrderStatus enum should have all statuses."""
        assert OrderStatus.PENDING.value == 'pending'
        assert OrderStatus.FILLED.value == 'filled'
        assert OrderStatus.CANCELLED.value == 'cancelled'
        assert OrderStatus.REJECTED.value == 'rejected'
        assert OrderStatus.PARTIALLY_FILLED.value == 'partially_filled'

    def test_time_in_force_values(self):
        """TimeInForce enum should have all TIF values."""
        assert TimeInForce.DAY.value == 'day'
        assert TimeInForce.GTC.value == 'gtc'
        assert TimeInForce.IOC.value == 'ioc'
        assert TimeInForce.FOK.value == 'fok'

    def test_option_type_values(self):
        """OptionType enum should have CALL and PUT."""
        assert OptionType.CALL.value == 'call'
        assert OptionType.PUT.value == 'put'

    def test_option_right_values(self):
        """OptionRight enum should have AMERICAN and EUROPEAN."""
        assert OptionRight.AMERICAN.value == 'american'
        assert OptionRight.EUROPEAN.value == 'european'


class TestOrderDictFields:
    """Test that order dicts have consistent field names."""

    @pytest.fixture
    def broker(self):
        return MockBroker(initial_cash=100000.0)

    def test_order_has_filled_avg_price(self, broker):
        """Order dict should have filled_avg_price field."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        order = broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        assert 'filled_avg_price' in order

    def test_order_timestamps(self, broker):
        """Order dict should have timestamp fields."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        order = broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        assert 'created_at' in order
        assert 'submitted_at' in order
        assert isinstance(order['created_at'], datetime)


class TestPositionDictFields:
    """Test that position dicts have consistent field names."""

    @pytest.fixture
    def broker(self):
        return MockBroker(initial_cash=100000.0)

    def test_position_has_required_fields(self, broker):
        """Position dict should have all required fields."""
        broker.set_quote('AAPL', bid=100.0, ask=100.05)
        broker.place_stock_order('AAPL', 10, OrderSide.BUY)
        position = broker.get_stock_position('AAPL')

        assert 'symbol' in position
        assert 'quantity' in position
        assert 'avg_entry_price' in position
        assert 'current_price' in position
        assert 'market_value' in position
        assert 'unrealized_pnl' in position
        assert 'side' in position


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
