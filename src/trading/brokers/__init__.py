"""
Broker Abstraction Layer

Provides broker-agnostic interface for trading operations.

Components:
- BrokerInterface: Abstract protocol defining all broker operations
- BrokerFactory: Factory for creating broker instances from config
- AlpacaBroker: Alpaca implementation of BrokerInterface
- IBBroker: Interactive Brokers implementation (future)
- TDAmeritradeBroker: TD Ameritrade implementation (future)
"""

from .broker_interface import (
    BrokerInterface,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerError,
    BrokerConnectionError,
    BrokerAuthError,
    InvalidOrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    NoPositionError,
    SymbolNotFoundError,
)
from .broker_factory import BrokerFactory
from .alpaca_broker import AlpacaBroker

__all__ = [
    # Interface
    "BrokerInterface",
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    # Exceptions
    "BrokerError",
    "BrokerConnectionError",
    "BrokerAuthError",
    "InvalidOrderError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    "NoPositionError",
    "SymbolNotFoundError",
    # Factory
    "BrokerFactory",
    # Implementations
    "AlpacaBroker",
]
