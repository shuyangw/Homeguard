"""
Options Trading Interface - Options-specific trading operations.

This is an abstract interface only. No broker implementation exists yet.
Future implementations: TastyTrade, Interactive Brokers, etc.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional

from .base import OptionType, OrderSide, OrderType, TimeInForce
from .order_management import OrderManagementInterface


class OptionLeg:
    """Represents a single leg in a multi-leg options order."""

    def __init__(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        option_type: OptionType,
        quantity: int,
        side: OrderSide
    ):
        self.underlying = underlying
        self.expiration = expiration
        self.strike = strike
        self.option_type = option_type
        self.quantity = quantity
        self.side = side

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'underlying': self.underlying,
            'expiration': self.expiration.isoformat(),
            'strike': self.strike,
            'option_type': self.option_type.value,
            'quantity': self.quantity,
            'side': self.side.value
        }


class OptionsTradingInterface(OrderManagementInterface):
    """
    Abstract interface for options trading operations.

    Inherits order management from OrderManagementInterface.
    Provides options-specific chain, position, and order methods.

    Note: This is an abstract interface only. Implement with a broker
    adapter (e.g., TastyTradeBroker, IBKRBroker) when needed.
    """

    # ==================== Options Chain ====================

    @abstractmethod
    def get_options_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None
    ) -> List[Dict]:
        """
        Get options chain for an underlying symbol.

        Args:
            underlying: Underlying stock symbol (e.g., "AAPL")
            expiration: Specific expiration date (None = all expirations)

        Returns:
            List[Dict] where each Dict contains:
                - contract_id (str): Unique contract identifier
                - underlying (str): Underlying symbol
                - expiration (date): Expiration date
                - strike (float): Strike price
                - option_type (str): 'call' or 'put'
                - bid (float): Bid price
                - ask (float): Ask price
                - last (float): Last trade price
                - volume (int): Volume
                - open_interest (int): Open interest
                - implied_volatility (float): IV

        Raises:
            BrokerConnectionError: If broker connection fails
            SymbolNotFoundError: If underlying doesn't exist
        """
        pass

    # ==================== Position Methods ====================

    @abstractmethod
    def get_options_positions(self) -> List[Dict]:
        """
        Get all current options positions.

        Returns:
            List[Dict] where each Dict contains:
                - contract_id (str): Unique contract identifier
                - underlying (str): Underlying symbol
                - expiration (date): Expiration date
                - strike (float): Strike price
                - option_type (str): 'call' or 'put'
                - quantity (int): Number of contracts (negative = short)
                - avg_entry_price (float): Average entry price per contract
                - current_price (float): Current contract price
                - market_value (float): Current market value
                - unrealized_pnl (float): Unrealized P&L
                - delta (float): Position delta
                - gamma (float): Position gamma
                - theta (float): Position theta
                - vega (float): Position vega

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    @abstractmethod
    def get_options_position(self, contract_id: str) -> Optional[Dict]:
        """
        Get specific options position by contract ID.

        Args:
            contract_id: Options contract identifier

        Returns:
            Dict with position details (same format as get_options_positions)
            or None if no position exists

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    # ==================== Order Methods ====================

    @abstractmethod
    def place_options_order(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        option_type: OptionType,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs
    ) -> Dict:
        """
        Place a single-leg options order.

        Args:
            underlying: Underlying stock symbol (e.g., "AAPL")
            expiration: Option expiration date
            strike: Strike price
            option_type: OptionType.CALL or OptionType.PUT
            quantity: Number of contracts
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Order type (market, limit)
            limit_price: Limit price (required for LIMIT orders)
            time_in_force: Time in force
            **kwargs: Additional broker-specific parameters

        Returns:
            Dict with order details:
                - order_id (str): Unique order identifier
                - contract_id (str): Options contract identifier
                - underlying (str): Underlying symbol
                - expiration (date): Expiration date
                - strike (float): Strike price
                - option_type (str): 'call' or 'put'
                - quantity (int): Number of contracts
                - side (str): 'buy' or 'sell'
                - order_type (str): Order type
                - status (str): Order status
                - limit_price (float): Limit price (if applicable)
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
    def place_multi_leg_order(
        self,
        legs: List[OptionLeg],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs
    ) -> Dict:
        """
        Place a multi-leg options order (spreads, straddles, etc.).

        Args:
            legs: List of OptionLeg objects defining the strategy
            order_type: Order type (usually LIMIT for spreads)
            limit_price: Net debit/credit for the spread
            time_in_force: Time in force
            **kwargs: Additional broker-specific parameters

        Returns:
            Dict with order details:
                - order_id (str): Unique order identifier
                - legs (List[Dict]): Details of each leg
                - order_type (str): Order type
                - status (str): Order status
                - limit_price (float): Net limit price
                - created_at (datetime): Order creation time
                - filled_qty (int): Filled quantity
                - avg_fill_price (float): Average fill price

        Raises:
            BrokerConnectionError: If broker connection fails
            InvalidOrderError: If order parameters are invalid
            InsufficientFundsError: If insufficient buying power
        """
        pass

    # ==================== Greeks ====================

    @abstractmethod
    def get_greeks(self, contract_id: str) -> Dict:
        """
        Get option Greeks for a specific contract.

        Args:
            contract_id: Options contract identifier

        Returns:
            Dict with Greeks:
                - delta (float): Delta
                - gamma (float): Gamma
                - theta (float): Theta (per day)
                - vega (float): Vega
                - rho (float): Rho
                - implied_volatility (float): IV

        Raises:
            BrokerConnectionError: If broker connection fails
            SymbolNotFoundError: If contract doesn't exist
        """
        pass

    # ==================== Position Closing ====================

    @abstractmethod
    def close_options_position(
        self,
        contract_id: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """
        Close an options position (or partial position).

        Args:
            contract_id: Options contract identifier
            quantity: Number of contracts to close (None = close all)

        Returns:
            Dict with order details (same format as place_options_order)

        Raises:
            BrokerConnectionError: If broker connection fails
            NoPositionError: If no position exists
        """
        pass

    @abstractmethod
    def close_all_options_positions(self) -> List[Dict]:
        """
        Close all open options positions.

        Returns:
            List[Dict] of order details for all closed positions

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass
