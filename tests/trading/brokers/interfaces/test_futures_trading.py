"""Tests for FuturesTradingInterface ABC."""
import inspect

import pytest

from src.trading.brokers.interfaces.futures_trading import FuturesTradingInterface
from src.trading.brokers.interfaces.order_management import OrderManagementInterface


def test_futures_trading_extends_order_management():
    """FuturesTradingInterface inherits OrderManagementInterface."""
    assert issubclass(FuturesTradingInterface, OrderManagementInterface)


def test_required_futures_methods_are_abstract():
    """Every futures-specific method is marked abstract on the interface."""
    required = [
        "place_futures_order",
        "place_futures_combo_order",
        "get_futures_positions",
        "get_futures_position",
        "close_futures_position",
        "close_all_futures_positions",
        "what_if_order",
        "get_margin_status",
    ]
    abstract_methods = FuturesTradingInterface.__abstractmethods__
    for m in required:
        assert m in abstract_methods, f"{m} should be abstract on FuturesTradingInterface"


def test_place_futures_order_signature():
    """Signature includes the parameters callers (ExecutionEngine/strategy adapters) need."""
    sig = inspect.signature(FuturesTradingInterface.place_futures_order)
    params = list(sig.parameters.keys())
    # self plus the documented params
    for p in ["symbol_root", "contract_month", "side", "quantity",
              "order_type", "limit_price", "stop_price", "time_in_force"]:
        assert p in params, f"place_futures_order missing parameter '{p}'"


def test_place_futures_combo_order_signature():
    """submit_combo_order takes a ComboOrder or sequence of legs."""
    sig = inspect.signature(FuturesTradingInterface.place_futures_combo_order)
    params = list(sig.parameters.keys())
    # at minimum, a 'legs' or 'combo' parameter
    assert "legs" in params or "combo" in params, (
        f"place_futures_combo_order signature {params} missing 'legs' or 'combo'"
    )


def test_what_if_order_signature():
    """what_if_order returns projected margin without placing the order."""
    sig = inspect.signature(FuturesTradingInterface.what_if_order)
    params = list(sig.parameters.keys())
    for p in ["symbol_root", "contract_month", "side", "quantity"]:
        assert p in params


def test_cannot_instantiate_abstract():
    """ABC enforces -- you cannot instantiate FuturesTradingInterface directly."""
    with pytest.raises(TypeError):
        FuturesTradingInterface()
