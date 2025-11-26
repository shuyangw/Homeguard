"""
Broker Interfaces Package.

Provides focused, composable interfaces following the Interface Segregation
Principle (ISP). Brokers implement only the interfaces they support.

Interface Hierarchy:
    AccountInterface - Account info and connection
    MarketHoursInterface - Market schedule
    MarketDataInterface - Quotes, trades, bars
    OrderManagementInterface - Order retrieval and cancellation (shared)
        └── StockTradingInterface - Stock positions and orders
        └── OptionsTradingInterface - Options chains, positions, orders

Example Usage:
    # Stock-only broker (e.g., Alpaca)
    class AlpacaBroker(
        AccountInterface,
        MarketHoursInterface,
        MarketDataInterface,
        StockTradingInterface
    ):
        ...

    # Full-featured broker (e.g., IBKR)
    class IBKRBroker(
        AccountInterface,
        MarketHoursInterface,
        MarketDataInterface,
        StockTradingInterface,
        OptionsTradingInterface
    ):
        ...
"""

# Base types (enums, exceptions)
from .base import (
    # Order enums
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    # Options enums
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

# Interfaces
from .account import AccountInterface
from .market_hours import MarketHoursInterface
from .market_data import MarketDataInterface
from .order_management import OrderManagementInterface
from .stock_trading import StockTradingInterface
from .options_trading import OptionsTradingInterface, OptionLeg


__all__ = [
    # Order enums
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    # Options enums
    'OptionType',
    'OptionRight',
    # Exceptions
    'BrokerError',
    'BrokerConnectionError',
    'BrokerAuthError',
    'InvalidOrderError',
    'InsufficientFundsError',
    'OrderNotFoundError',
    'NoPositionError',
    'SymbolNotFoundError',
    'OptionsNotSupportedError',
    # Interfaces
    'AccountInterface',
    'MarketHoursInterface',
    'MarketDataInterface',
    'OrderManagementInterface',
    'StockTradingInterface',
    'OptionsTradingInterface',
    'OptionLeg',
]
