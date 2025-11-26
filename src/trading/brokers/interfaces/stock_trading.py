"""
Stock Trading Interface - Stock-specific trading operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .base import OrderSide, OrderType, TimeInForce
from .order_management import OrderManagementInterface


class StockTradingInterface(OrderManagementInterface):
    """
    Abstract interface for stock trading operations.

    Inherits order management from OrderManagementInterface.
    Provides stock-specific position and order methods.
    """

    # ==================== Position Methods ====================

    @abstractmethod
    def get_stock_positions(self) -> List[Dict]:
        """
        Get all current stock positions.

        Returns:
            List[Dict] where each Dict contains:
                - symbol (str): Stock symbol
                - quantity (int): Number of shares (negative = short)
                - avg_entry_price (float): Average entry price
                - current_price (float): Current market price
                - market_value (float): Current market value
                - unrealized_pnl (float): Unrealized P&L
                - unrealized_pnl_pct (float): Unrealized P&L percentage
                - side (str): 'long' or 'short'

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    @abstractmethod
    def get_stock_position(self, symbol: str) -> Optional[Dict]:
        """
        Get specific stock position by symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Dict with position details (same format as get_stock_positions)
            or None if no position exists

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    # ==================== Order Methods ====================

    @abstractmethod
    def place_stock_order(
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
        Place a stock order.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            quantity: Number of shares (positive integer)
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Order type (market, limit, stop, etc.)
            limit_price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)
            time_in_force: Time in force (day, gtc, ioc, fok)
            **kwargs: Additional broker-specific parameters

        Returns:
            Dict with order details:
                - order_id (str): Unique order identifier
                - symbol (str): Stock symbol
                - quantity (int): Number of shares
                - side (str): 'buy' or 'sell'
                - order_type (str): Order type
                - status (str): Order status
                - limit_price (float): Limit price (if applicable)
                - stop_price (float): Stop price (if applicable)
                - created_at (datetime): Order creation time
                - filled_qty (int): Filled quantity
                - avg_fill_price (float): Average fill price

        Raises:
            BrokerConnectionError: If broker connection fails
            InvalidOrderError: If order parameters are invalid
            InsufficientFundsError: If insufficient buying power
        """
        pass

    @abstractmethod
    def close_stock_position(
        self,
        symbol: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """
        Close a stock position (or partial position).

        Args:
            symbol: Stock symbol
            quantity: Number of shares to close (None = close all)

        Returns:
            Dict with order details (same format as place_stock_order)

        Raises:
            BrokerConnectionError: If broker connection fails
            NoPositionError: If no position exists
        """
        pass

    @abstractmethod
    def close_all_stock_positions(self) -> List[Dict]:
        """
        Close all open stock positions.

        Returns:
            List[Dict] of order details for all closed positions

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass
