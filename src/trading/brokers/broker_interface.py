"""
Broker Interface - Composite Interface for Backward Compatibility.

DEPRECATED: This module is maintained for backward compatibility only.
New code should import from `src.trading.brokers.interfaces` instead.

This composite interface inherits from all stock-related interfaces,
maintaining the original BrokerInterface API for existing code.

Migration Guide:
    Old: from src.trading.brokers.broker_interface import BrokerInterface
    New: from src.trading.brokers.interfaces import (
             AccountInterface, MarketDataInterface, StockTradingInterface
         )
"""

import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# Import all interfaces and types from the new structure
from .interfaces import (
    # Interfaces
    AccountInterface,
    MarketHoursInterface,
    MarketDataInterface,
    StockTradingInterface,
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    # Exceptions
    BrokerError,
    BrokerConnectionError,
    BrokerAuthError,
    InvalidOrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    NoPositionError,
    SymbolNotFoundError,
)

# Re-export for backward compatibility
__all__ = [
    'BrokerInterface',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    'BrokerError',
    'BrokerConnectionError',
    'BrokerAuthError',
    'InvalidOrderError',
    'InsufficientFundsError',
    'OrderNotFoundError',
    'NoPositionError',
    'SymbolNotFoundError',
]


class BrokerInterface(
    AccountInterface,
    MarketHoursInterface,
    MarketDataInterface,
    StockTradingInterface
):
    """
    Composite broker interface for trading operations.

    DEPRECATED: This class is maintained for backward compatibility.
    New implementations should use specific interfaces:
        - AccountInterface for account operations
        - MarketHoursInterface for market schedule
        - MarketDataInterface for quotes and bars
        - StockTradingInterface for stock trading

    All broker implementations must implement this interface (or the
    component interfaces) to ensure core trading logic is broker-agnostic.

    Example Usage:
        >>> broker = BrokerFactory.create_broker('alpaca', config)
        >>> account = broker.get_account()
        >>> order = broker.place_order('SPY', 10, OrderSide.BUY)
    """

    # ==================== Backward Compatibility Aliases ====================
    # These methods delegate to the new interface methods with proper names.
    # They maintain the original API for existing code.

    def get_positions(self) -> List[Dict]:
        """
        Get all current positions.

        DEPRECATED: Use get_stock_positions() instead.

        Returns:
            List[Dict] with position details
        """
        return self.get_stock_positions()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get specific position by symbol.

        DEPRECATED: Use get_stock_position() instead.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with position details or None
        """
        return self.get_stock_position(symbol)

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs
    ) -> Dict:
        """
        Place an order.

        DEPRECATED: Use place_stock_order() instead.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            **kwargs: Additional parameters

        Returns:
            Dict with order details
        """
        return self.place_stock_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs
        )

    def close_position(
        self,
        symbol: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """
        Close a position.

        DEPRECATED: Use close_stock_position() instead.

        Args:
            symbol: Stock symbol
            quantity: Number of shares to close

        Returns:
            Dict with order details
        """
        return self.close_stock_position(symbol, quantity)

    def close_all_positions(self) -> List[Dict]:
        """
        Close all open positions.

        DEPRECATED: Use close_all_stock_positions() instead.

        Returns:
            List[Dict] of order details
        """
        return self.close_all_stock_positions()
