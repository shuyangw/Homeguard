"""
Base types for broker interfaces.

Contains shared enums, exceptions, and type definitions used across
all broker interfaces (stock, options, futures, etc.).
"""

from enum import Enum
from typing import Dict, List, Optional, TypedDict
from datetime import datetime


# ==================== Order Enums ====================

class OrderSide(Enum):
    """Order side enum (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(Enum):
    """Time in force enum."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


# ==================== Options Enums ====================

class OptionType(Enum):
    """Option type enum (call/put)."""
    CALL = "call"
    PUT = "put"


class OptionRight(Enum):
    """Option exercise right."""
    AMERICAN = "american"
    EUROPEAN = "european"


# ==================== Custom Exceptions ====================

class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class BrokerConnectionError(BrokerError):
    """Broker connection error."""
    pass


class BrokerAuthError(BrokerError):
    """Broker authentication error."""
    pass


class InvalidOrderError(BrokerError):
    """Invalid order parameters."""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order."""
    pass


class OrderNotFoundError(BrokerError):
    """Order not found."""
    pass


class NoPositionError(BrokerError):
    """No position exists for symbol."""
    pass


class SymbolNotFoundError(BrokerError):
    """Symbol not found."""
    pass


class OptionsNotSupportedError(BrokerError):
    """Broker does not support options trading."""
    pass
