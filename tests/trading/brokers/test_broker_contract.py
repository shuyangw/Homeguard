"""
Broker contract tests.

Static assertion that every concrete broker class implements every method
ExecutionEngine (and other core trading code) calls on `self.broker`, with
signatures compatible with what the caller passes.

Catches the class of regressions where a broker silently falls off the
interface (e.g. IBKRBroker missing `place_order` because it inherits from
fragmented interface mixins instead of the deprecated-but-shimmed
BrokerInterface). Runs in the default test suite -- no credentials, no
broker instantiation.

If ExecutionEngine adds a new broker call, update REQUIRED_STOCK_METHODS.
"""

import inspect

import pytest

from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker


# (method_name, required_parameter_names)
# These are the methods ExecutionEngine (src/trading/core/execution_engine.py)
# calls on self.broker, with the parameters it passes by name. Kept in sync
# with that file -- if ExecutionEngine starts calling a new broker method,
# add it here.
REQUIRED_STOCK_METHODS = [
    (
        "place_stock_order",
        ["symbol", "quantity", "side", "order_type",
         "limit_price", "stop_price", "time_in_force"],
    ),
    ("cancel_order",         ["order_id"]),
    ("get_order",            ["order_id"]),
    ("get_stock_position",   ["symbol"]),
    ("get_stock_positions",  []),
    ("close_stock_position", ["symbol"]),
]


STOCK_BROKERS = [AlpacaBroker, IBKRBroker]


@pytest.mark.parametrize("broker_cls", STOCK_BROKERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "method_name,required_params",
    REQUIRED_STOCK_METHODS,
    ids=lambda p: p if isinstance(p, str) else ",".join(p) or "<none>",
)
def test_stock_broker_has_method(broker_cls, method_name, required_params):
    """Every stock broker must expose each method ExecutionEngine calls."""
    assert hasattr(broker_cls, method_name), (
        f"{broker_cls.__name__} is missing {method_name!r} -- "
        f"ExecutionEngine calls broker.{method_name}(...) and will crash "
        f"at runtime with 'AttributeError: has no attribute {method_name!r}'."
    )


@pytest.mark.parametrize("broker_cls", STOCK_BROKERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "method_name,required_params",
    REQUIRED_STOCK_METHODS,
    ids=lambda p: p if isinstance(p, str) else ",".join(p) or "<none>",
)
def test_stock_broker_method_accepts_required_params(
    broker_cls, method_name, required_params
):
    """Each method must accept the parameters ExecutionEngine passes by name.

    Uses inspect.signature rather than instantiation so this test doesn't
    need credentials or a live gateway.
    """
    method = getattr(broker_cls, method_name, None)
    if method is None:
        pytest.skip(f"{broker_cls.__name__} lacks {method_name} -- see other test")

    sig = inspect.signature(method)
    actual_params = [p for p in sig.parameters if p != "self"]

    # Accept either: the exact param by name, or a **kwargs catch-all that
    # could swallow it.
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    for required in required_params:
        if has_var_keyword:
            continue  # **kwargs catches anything
        assert required in actual_params, (
            f"{broker_cls.__name__}.{method_name} does not accept "
            f"required parameter {required!r}. Signature: {sig}. "
            f"ExecutionEngine passes this by name."
        )


# --- Futures broker contract test (added 2026-05-11, sub-chunk 6b) ---

# Methods that strategies and ExecutionEngine call on a FuturesTradingInterface
# concrete broker. Kept in sync with src/trading/brokers/interfaces/futures_trading.py
# -- if FuturesTradingInterface gains a method that strategies call, add it here.
REQUIRED_FUTURES_METHODS = [
    (
        "place_futures_order",
        ["symbol_root", "contract_month", "side", "quantity", "order_type",
         "limit_price", "stop_price", "time_in_force"],
    ),
    (
        "place_futures_combo_order",
        ["legs", "order_type", "limit_price", "time_in_force"],
    ),
    ("get_futures_positions",       []),
    ("get_futures_position",        ["symbol_root", "contract_month"]),
    ("close_futures_position",      ["symbol_root", "contract_month"]),
    ("close_all_futures_positions", []),
    (
        "what_if_order",
        ["symbol_root", "contract_month", "side", "quantity",
         "order_type", "limit_price"],
    ),
    ("get_margin_status",           []),
    # Inherited from OrderManagementInterface
    ("cancel_order",                ["order_id"]),
    ("get_order",                   ["order_id"]),
]


@pytest.mark.parametrize("method_name,required_params", REQUIRED_FUTURES_METHODS)
def test_ibkr_futures_broker_has_required_method(method_name, required_params):
    """IBKRFuturesBroker exposes every method strategies/ExecutionEngine need.

    Catches regressions where the broker silently falls off the interface.
    """
    from src.trading.brokers.ibkr.ibkr_futures_broker import IBKRFuturesBroker

    method = getattr(IBKRFuturesBroker, method_name, None)
    assert method is not None, (
        f"IBKRFuturesBroker is missing required method '{method_name}'"
    )

    sig = inspect.signature(method)
    actual_params = [p for p in sig.parameters.keys() if p != "self"]
    for p in required_params:
        assert p in actual_params, (
            f"IBKRFuturesBroker.{method_name} signature {actual_params} "
            f"missing required parameter '{p}'"
        )
