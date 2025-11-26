"""
Broker Abstraction Layer

Provides broker-agnostic interface for trading operations.

Components:
- interfaces/: Focused, composable interfaces (ISP)
    - AccountInterface: Account info and connection
    - MarketHoursInterface: Market schedule
    - MarketDataInterface: Quotes, trades, bars
    - OrderManagementInterface: Order retrieval/cancellation
    - StockTradingInterface: Stock positions and orders
    - OptionsTradingInterface: Options chains, positions, orders
- BrokerInterface: Composite interface (backward compatibility)
- BrokerFactory: Factory for creating broker instances
- AlpacaBroker: Alpaca implementation
"""

# New focused interfaces (preferred)
from .interfaces import (
    # Interfaces
    AccountInterface,
    MarketHoursInterface,
    MarketDataInterface,
    OrderManagementInterface,
    StockTradingInterface,
    OptionsTradingInterface,
    OptionLeg,
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    OptionType,
    OptionRight,
    # Exceptions
    BrokerError,
    BrokerConnectionError,
    BrokerAuthError,
    InvalidOrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    NoPositionError,
    SymbolNotFoundError,
    OptionsNotSupportedError,
)

# Composite interface (backward compatibility)
from .broker_interface import BrokerInterface

# Factory and implementations
from .broker_factory import BrokerFactory
from .alpaca_broker import AlpacaBroker

__all__ = [
    # New focused interfaces
    "AccountInterface",
    "MarketHoursInterface",
    "MarketDataInterface",
    "OrderManagementInterface",
    "StockTradingInterface",
    "OptionsTradingInterface",
    "OptionLeg",
    # Composite interface (backward compat)
    "BrokerInterface",
    # Order enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    # Options enums
    "OptionType",
    "OptionRight",
    # Exceptions
    "BrokerError",
    "BrokerConnectionError",
    "BrokerAuthError",
    "InvalidOrderError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    "NoPositionError",
    "SymbolNotFoundError",
    "OptionsNotSupportedError",
    # Factory
    "BrokerFactory",
    # Implementations
    "AlpacaBroker",
]
