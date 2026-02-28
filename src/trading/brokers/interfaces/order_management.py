"""
Order Management Interface - Shared order operations.

This interface is inherited by both StockTradingInterface and
OptionsTradingInterface, providing common order management
functionality across instrument types.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from .base import OrderStatus


class OrderManagementInterface(ABC):
    """
    Abstract interface for order management operations.

    Shared across stocks, options, and other instrument types.
    Provides order retrieval and cancellation functionality.
    """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Returns:
            True if cancellation successful, False otherwise

        Raises:
            BrokerConnectionError: If broker connection fails
            OrderNotFoundError: If order doesn't exist
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Dict:
        """
        Get order details by ID.

        Returns:
            Dict with order details:
                - order_id (str): Unique order identifier
                - symbol (str): Symbol
                - quantity (int): Quantity
                - side (str): 'buy' or 'sell'
                - order_type (str): Order type
                - status (str): Order status
                - limit_price (float): Limit price (if applicable)
                - stop_price (float): Stop price (if applicable)
                - created_at (datetime): Order creation time
                - filled_qty (int): Filled quantity
                - filled_avg_price (float): Average fill price

        Raises:
            BrokerConnectionError: If broker connection fails
            OrderNotFoundError: If order doesn't exist
        """
        pass

    @abstractmethod
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get orders with optional filters.

        Returns:
            List[Dict] of orders (same format as get_order)

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass
