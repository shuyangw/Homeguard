"""
Crypto Trading Interface - Cryptocurrency-specific trading operations.

Provides abstract interface for crypto trading across different exchanges
(Coinbase, Alpaca Crypto, Binance, etc.).

Key differences from stock trading:
- 24/7 market (no market hours)
- Fractional quantities (e.g., 0.001 BTC)
- Different symbol format (BTC/USD vs BTC-USD)
- No short selling on most exchanges (long only)
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional

from .base import OrderSide, OrderType, TimeInForce
from .order_management import OrderManagementInterface


class CryptoTradingInterface(OrderManagementInterface):
    """
    Abstract interface for cryptocurrency trading operations.

    Inherits order management from OrderManagementInterface.
    Provides crypto-specific position and order methods.

    Key Features:
    - Fractional quantity support (Decimal)
    - 24/7 trading (no market hours)
    - Symbol normalization (BTC/USD, BTC-USD, BTCUSD)
    """

    @abstractmethod
    def get_crypto_positions(self) -> List[Dict]:
        """
        Get all current crypto positions.

        Returns:
            List[Dict] where each Dict contains:
                - symbol (str): Crypto symbol (e.g., 'BTC/USD')
                - quantity (Decimal): Amount held (can be fractional)
                - avg_entry_price (float): Average entry price
                - current_price (float): Current market price
                - market_value (float): Current market value (quantity x price)
                - unrealized_pnl (float): Unrealized P&L
                - unrealized_pnl_pct (float): Unrealized P&L percentage

        Raises:
            BrokerConnectionError: If exchange connection fails
        """
        pass

    @abstractmethod
    def get_crypto_position(self, symbol: str) -> Optional[Dict]:
        """
        Get specific crypto position by symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')

        Returns:
            Dict with position details (same format as get_crypto_positions)
            or None if no position exists

        Raises:
            BrokerConnectionError: If exchange connection fails
        """
        pass

    @abstractmethod
    def place_crypto_order(
        self,
        symbol: str,
        quantity: Decimal,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        **kwargs
    ) -> Dict:
        """
        Place a crypto order.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            quantity: Amount to buy/sell (Decimal for precision)
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Order type (MARKET, LIMIT)
            limit_price: Limit price for limit orders
            time_in_force: GTC for crypto (default)
            **kwargs: Exchange-specific parameters

        Returns:
            Dict with order details:
                - order_id (str): Unique order identifier
                - symbol (str): Crypto symbol
                - quantity (Decimal): Order quantity
                - side (str): 'buy' or 'sell'
                - order_type (str): Order type
                - status (str): Order status
                - limit_price (float): Limit price (if applicable)
                - created_at (datetime): Order creation time
                - filled_qty (Decimal): Filled quantity
                - filled_avg_price (float): Average fill price
                - fees (float): Trading fees

        Raises:
            BrokerConnectionError: If exchange connection fails
            InvalidOrderError: If order parameters are invalid
            InsufficientFundsError: If insufficient balance
        """
        pass

    @abstractmethod
    def close_crypto_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """
        Close a crypto position (or partial position).

        Args:
            symbol: Crypto symbol
            quantity: Amount to close (None = close all)

        Returns:
            Dict with order details (same format as place_crypto_order)

        Raises:
            BrokerConnectionError: If exchange connection fails
            NoPositionError: If no position exists
        """
        pass

    @abstractmethod
    def close_all_crypto_positions(self) -> List[Dict]:
        """
        Close all open crypto positions.

        Returns:
            List[Dict] of order details for all closed positions

        Raises:
            BrokerConnectionError: If exchange connection fails
        """
        pass

    @abstractmethod
    def get_crypto_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a crypto symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Dict with quote data:
                - symbol (str): Crypto symbol
                - bid (float): Best bid price
                - ask (float): Best ask price
                - last (float): Last trade price
                - volume_24h (float): 24-hour volume
                - timestamp (datetime): Quote timestamp

        Raises:
            BrokerConnectionError: If exchange connection fails
            SymbolNotFoundError: If symbol not found
        """
        pass

    @abstractmethod
    def get_crypto_balance(self, currency: str = 'USD') -> Dict:
        """
        Get crypto account balance.

        Args:
            currency: Base currency (default 'USD')

        Returns:
            Dict with balance data:
                - total (float): Total balance
                - available (float): Available for trading
                - held (float): Held in open orders

        Raises:
            BrokerConnectionError: If exchange connection fails
        """
        pass

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize crypto symbol to exchange format.

        Override in subclass if exchange uses different format.
        Default: 'BTC/USD' format.

        Args:
            symbol: Input symbol (various formats)

        Returns:
            Normalized symbol string
        """
        # Handle common formats
        symbol = symbol.upper()
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')

        # Add /USD if not present
        if '/' not in symbol and not symbol.endswith('USD'):
            symbol = f"{symbol}/USD"

        return symbol
