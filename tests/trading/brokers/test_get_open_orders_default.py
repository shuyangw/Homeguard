"""
Regression test for the 2026-04-23 RAMP rebalance failure.

PortfolioHealthChecker.check_before_entry calls broker.get_open_orders().
AlpacaBroker implements it as an Alpaca-specific helper, but IBKRBroker
did not -- so the health check raised AttributeError, which the checker
caught as a failure and blocked entry. Result: two consecutive days of
RAMP rebalances with 0 orders placed.

Fix: OrderManagementInterface now has a default `get_open_orders`
implementation that delegates to `get_orders(status=PENDING)`. Any
concrete broker implementing the interface inherits it.

These tests verify:
1. IBKRBroker has `get_open_orders` (was missing)
2. The default delegates to get_orders with PENDING status
3. AlpacaBroker's custom override still wins when present
"""

import inspect

from src.trading.brokers.interfaces.base import OrderStatus
from src.trading.brokers.interfaces.order_management import OrderManagementInterface


def test_interface_has_default_get_open_orders():
    """The interface now provides a concrete default implementation."""
    assert hasattr(OrderManagementInterface, 'get_open_orders')
    # Must be concrete, not abstract -- subclasses shouldn't be forced to override.
    method = OrderManagementInterface.get_open_orders
    assert not getattr(method, '__isabstractmethod__', False)


def test_ibkr_broker_inherits_get_open_orders():
    """IBKRBroker didn't have get_open_orders -- today's bug. Interface default fixes it."""
    from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
    assert hasattr(IBKRBroker, 'get_open_orders')
    # Not defined on IBKRBroker itself; inherited from the interface default.
    assert 'get_open_orders' not in IBKRBroker.__dict__
    # Interface resolution order should find it on OrderManagementInterface
    source_cls = None
    for cls in IBKRBroker.__mro__:
        if 'get_open_orders' in cls.__dict__:
            source_cls = cls
            break
    assert source_cls is OrderManagementInterface


def test_alpaca_broker_has_custom_override():
    """AlpacaBroker already had a custom get_open_orders -- it must still win.
    Otherwise we regress Alpaca's broader semantics (includes PARTIALLY_FILLED)."""
    from src.trading.brokers.alpaca_broker import AlpacaBroker
    assert 'get_open_orders' in AlpacaBroker.__dict__


def test_default_delegates_to_get_orders_with_pending_status():
    """The default must call get_orders(status=PENDING). Using a stub class
    that implements the abstracts."""
    from typing import List, Dict, Optional
    from datetime import datetime

    calls = []

    class StubBroker(OrderManagementInterface):
        def cancel_order(self, order_id: str) -> bool:
            return True

        def get_order(self, order_id: str) -> Dict:
            return {}

        def get_orders(
            self,
            status: Optional[OrderStatus] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
        ) -> List[Dict]:
            calls.append({'status': status, 'start_date': start_date, 'end_date': end_date})
            return [{'order_id': '1', 'status': 'pending'}]

    b = StubBroker()
    result = b.get_open_orders()

    assert calls == [{'status': OrderStatus.PENDING, 'start_date': None, 'end_date': None}]
    assert result == [{'order_id': '1', 'status': 'pending'}]
