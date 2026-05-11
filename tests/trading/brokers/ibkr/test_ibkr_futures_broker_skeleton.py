"""Tests for IBKRFuturesBroker skeleton (no IBKR connection required).

These tests verify class shape only -- signatures, ABC compliance, and
that methods raise NotImplementedError (since the actual IBKR API calls
are wired in sub-chunks 6c-6j).
"""
import inspect

import pytest

from src.trading.brokers.interfaces.futures_trading import FuturesTradingInterface
from src.trading.brokers.ibkr.ibkr_futures_broker import IBKRFuturesBroker


def test_ibkr_futures_broker_implements_interface():
    """IBKRFuturesBroker is concrete (instantiable) implementing FuturesTradingInterface."""
    assert issubclass(IBKRFuturesBroker, FuturesTradingInterface)
    # Note: instantiation requires an IBKRConfig; we don't actually start a
    # connection here -- subsequent tests rely on skeleton behavior only.


def test_all_abstract_methods_implemented():
    """Concrete class must implement every abstract method from the interface."""
    abstract_methods = FuturesTradingInterface.__abstractmethods__
    for m in abstract_methods:
        assert hasattr(IBKRFuturesBroker, m), f"missing implementation of {m}"
    # The concrete class itself should have no remaining abstract methods
    remaining = getattr(IBKRFuturesBroker, "__abstractmethods__", frozenset())
    assert remaining == frozenset(), (
        f"IBKRFuturesBroker still has abstract methods: {remaining}"
    )


def test_method_signatures_match_interface():
    """Concrete method signatures match the interface (parameter names)."""
    required = [
        ("place_futures_order",
         ["symbol_root", "contract_month", "side", "quantity", "order_type",
          "limit_price", "stop_price", "time_in_force"]),
        ("place_futures_combo_order",
         ["legs", "order_type", "limit_price", "time_in_force"]),
        ("get_futures_positions", []),
        ("get_futures_position",
         ["symbol_root", "contract_month"]),
        ("close_futures_position",
         ["symbol_root", "contract_month"]),
        ("close_all_futures_positions", []),
        ("what_if_order",
         ["symbol_root", "contract_month", "side", "quantity",
          "order_type", "limit_price"]),
        ("get_margin_status", []),
    ]
    for method_name, expected_params in required:
        method = getattr(IBKRFuturesBroker, method_name)
        sig = inspect.signature(method)
        actual_params = [p for p in sig.parameters.keys() if p != "self"]
        for ep in expected_params:
            assert ep in actual_params, (
                f"{method_name}: expected '{ep}' in {actual_params}"
            )


def test_skeleton_methods_raise_not_implemented():
    """Skeleton methods raise NotImplementedError until sub-chunks 6c-6j wire them."""
    # We construct an instance without starting a connection -- the skeleton
    # __init__ stashes config but doesn't connect.
    from src.trading.brokers.ibkr.config import IBKRConfig
    broker = IBKRFuturesBroker(IBKRConfig(port=4002))

    from src.trading.brokers.interfaces.base import OrderSide, OrderType

    with pytest.raises(NotImplementedError):
        broker.place_futures_order(
            symbol_root="MES", contract_month="202603",
            side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET,
        )
    with pytest.raises(NotImplementedError):
        broker.get_futures_positions()
    with pytest.raises(NotImplementedError):
        broker.get_margin_status()
    with pytest.raises(NotImplementedError):
        broker.what_if_order(
            symbol_root="MES", contract_month="202603",
            side=OrderSide.BUY, quantity=1,
        )
