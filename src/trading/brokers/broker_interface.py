"""
Broker Interface - Abstract Protocol for Trading Operations

All broker implementations (Alpaca, Interactive Brokers, TD Ameritrade, etc.)
must implement this interface to ensure core trading logic remains broker-agnostic.

Design Principles:
- Dependency Inversion: Core depends on this abstraction, not concrete implementations
- Adapter Pattern: Translate broker-specific APIs to standardized format
- Single Responsibility: Broker handles only broker operations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd


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
    DAY = "day"  # Good for day
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class BrokerInterface(ABC):
    """
    Abstract broker interface for trading operations.

    All broker implementations must implement this interface.
    This ensures core trading logic is broker-agnostic and can work
    with any broker (Alpaca, Interactive Brokers, TD Ameritrade, etc.).

    Example Usage:
        >>> broker = BrokerFactory.create_broker('alpaca', config)
        >>> account = broker.get_account()
        >>> order = broker.place_order('SPY', 10, OrderSide.BUY)
    """

    # ==================== Account Methods ====================

    @abstractmethod
    def get_account(self) -> Dict:
        """
        Get account information.

        Returns:
            Dict with standardized keys:
                - account_id (str): Account identifier
                - buying_power (float): Available buying power
                - cash (float): Cash balance
                - portfolio_value (float): Total portfolio value
                - equity (float): Total equity
                - currency (str): Account currency (USD, etc.)

        Raises:
            BrokerConnectionError: If broker connection fails
            BrokerAuthError: If authentication fails
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """
        Get all current positions.

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
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get specific position by symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Dict with position details (same format as get_positions)
            or None if no position exists

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    # ==================== Order Methods ====================

    @abstractmethod
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
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

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

        Args:
            order_id: Order ID

        Returns:
            Dict with order details (same format as place_order)

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

        Args:
            status: Filter by order status (None = all)
            start_date: Filter orders after this date
            end_date: Filter orders before this date

        Returns:
            List[Dict] of orders (same format as place_order)

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """
        Close a position (or partial position).

        Args:
            symbol: Stock symbol
            quantity: Number of shares to close (None = close all)

        Returns:
            Dict with order details (same format as place_order)

        Raises:
            BrokerConnectionError: If broker connection fails
            NoPositionError: If no position exists
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> List[Dict]:
        """
        Close all open positions.

        Returns:
            List[Dict] of order details for all closed positions

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    # ==================== Market Data Methods ====================

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Dict:
        """
        Get latest bid/ask quote.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quote details:
                - symbol (str): Stock symbol
                - bid (float): Bid price
                - ask (float): Ask price
                - bid_size (int): Bid size
                - ask_size (int): Ask size
                - timestamp (datetime): Quote timestamp

        Raises:
            BrokerConnectionError: If broker connection fails
            SymbolNotFoundError: If symbol doesn't exist
        """
        pass

    @abstractmethod
    def get_latest_trade(self, symbol: str) -> Dict:
        """
        Get latest trade.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with trade details:
                - symbol (str): Stock symbol
                - price (float): Trade price
                - size (int): Trade size
                - timestamp (datetime): Trade timestamp

        Raises:
            BrokerConnectionError: If broker connection fails
            SymbolNotFoundError: If symbol doesn't exist
        """
        pass

    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars.

        Args:
            symbols: List of stock symbols
            timeframe: Timeframe ('1Min', '5Min', '1Hour', '1Day')
            start: Start datetime
            end: End datetime

        Returns:
            pd.DataFrame with MultiIndex (symbol, timestamp) and columns:
                - open (float)
                - high (float)
                - low (float)
                - close (float)
                - volume (int)

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    # ==================== Utility Methods ====================

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open, False otherwise

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    @abstractmethod
    def get_market_hours(self, date: datetime) -> Tuple[datetime, datetime]:
        """
        Get market hours for a specific date.

        Args:
            date: Date to check

        Returns:
            Tuple of (open_time, close_time)

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test broker connection.

        Returns:
            True if connection successful, False otherwise
        """
        pass


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
