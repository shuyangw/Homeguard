"""
Mock Broker for Testing

Implements BrokerInterface for testing purposes without requiring real API access.
"""

from typing import Dict, List, Optional
from datetime import datetime, time
import uuid

from src.trading.brokers.broker_interface import (
    BrokerInterface,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)


class MockBroker(BrokerInterface):
    """
    Mock broker implementation for testing.

    Simulates all broker operations without making real API calls.
    Useful for testing components that depend on BrokerInterface.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        account_id: str = "MOCK_ACCOUNT_123"
    ):
        """
        Initialize mock broker.

        Args:
            initial_cash: Starting cash balance
            account_id: Mock account ID
        """
        self.account_id = account_id
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.market_open = True
        self.quotes: Dict[str, Dict] = {}

    # ==================== Account Operations ====================

    def get_account(self) -> Dict:
        """Get mock account information."""
        portfolio_value = self.cash + sum(
            p['quantity'] * p['current_price']
            for p in self.positions.values()
        )

        return {
            'account_id': self.account_id,
            'cash': self.cash,
            'buying_power': self.cash * 2,  # Assume 2x leverage
            'portfolio_value': portfolio_value,
            'equity': portfolio_value,
        }

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol."""
        return self.positions.get(symbol)

    # ==================== Order Operations ====================

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
        """Place mock order."""
        order_id = str(uuid.uuid4())

        # Simulate order price
        if order_type == OrderType.MARKET:
            fill_price = self.quotes.get(symbol, {}).get('ask', 100.0)
        elif order_type == OrderType.LIMIT:
            fill_price = limit_price
        elif order_type == OrderType.STOP:
            fill_price = stop_price
        elif order_type == OrderType.STOP_LIMIT:
            fill_price = limit_price
        else:
            fill_price = 100.0

        order = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'side': side.value,
            'order_type': order_type.value,
            'status': OrderStatus.FILLED.value,
            'filled_qty': quantity,
            'filled_avg_price': fill_price,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'time_in_force': time_in_force.value,
            'submitted_at': datetime.now(),
            'filled_at': datetime.now(),
        }

        self.orders[order_id] = order

        # Update positions
        if side == OrderSide.BUY:
            self._add_position(symbol, quantity, fill_price)
        else:
            self._remove_position(symbol, quantity, fill_price)

        return order

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel mock order."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.orders[order_id]
        order['status'] = OrderStatus.CANCELLED.value
        return order

    def get_order(self, order_id: str) -> Dict:
        """Get mock order by ID."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]

    def get_orders(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get all mock orders."""
        orders = list(self.orders.values())

        if status:
            orders = [o for o in orders if o['status'] == status]

        if limit:
            orders = orders[:limit]

        return orders

    def close_position(self, symbol: str) -> Dict:
        """Close mock position."""
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        position = self.positions[symbol]
        quantity = position['quantity']
        side = OrderSide.SELL if quantity > 0 else OrderSide.BUY

        return self.place_order(
            symbol=symbol,
            quantity=abs(quantity),
            side=side,
            order_type=OrderType.MARKET
        )

    def close_all_positions(self) -> List[Dict]:
        """Close all mock positions."""
        closed_orders = []
        for symbol in list(self.positions.keys()):
            order = self.close_position(symbol)
            closed_orders.append(order)
        return closed_orders

    # ==================== Market Data ====================

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get mock quote."""
        if symbol in self.quotes:
            return self.quotes[symbol]

        # Generate mock quote
        return {
            'symbol': symbol,
            'bid': 100.0,
            'ask': 100.05,
            'bid_size': 100,
            'ask_size': 100,
            'timestamp': datetime.now(),
        }

    def get_latest_trade(self, symbol: str) -> Dict:
        """Get mock trade."""
        return {
            'symbol': symbol,
            'price': 100.0,
            'size': 100,
            'timestamp': datetime.now(),
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get mock bars."""
        return [
            {
                'timestamp': datetime.now(),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000000,
            }
        ]

    # ==================== Utility ====================

    def is_market_open(self) -> bool:
        """Check if mock market is open."""
        return self.market_open

    def get_market_hours(self, date: Optional[datetime] = None) -> Dict:
        """Get mock market hours."""
        return {
            'is_open': self.market_open,
            'market_open': time(9, 30),
            'market_close': time(16, 0),
        }

    def test_connection(self) -> bool:
        """Test mock connection (always succeeds)."""
        return True

    # ==================== Helper Methods ====================

    def _add_position(self, symbol: str, quantity: int, price: float):
        """Add or update position."""
        cost = quantity * price
        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            old_qty = pos['quantity']
            old_avg_price = pos['avg_entry_price']
            new_qty = old_qty + quantity
            new_avg_price = ((old_qty * old_avg_price) + (quantity * price)) / new_qty

            pos['quantity'] = new_qty
            pos['avg_entry_price'] = new_avg_price
            pos['current_price'] = price
        else:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'avg_entry_price': price,
                'current_price': price,
                'market_value': quantity * price,
                'unrealized_pnl': 0.0,
            }

    def _remove_position(self, symbol: str, quantity: int, price: float):
        """Remove or update position."""
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        pos = self.positions[symbol]
        proceeds = quantity * price
        self.cash += proceeds

        pos['quantity'] -= quantity
        if pos['quantity'] == 0:
            del self.positions[symbol]
        else:
            pos['current_price'] = price

    def set_quote(self, symbol: str, bid: float, ask: float):
        """Set mock quote for testing."""
        self.quotes[symbol] = {
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'bid_size': 100,
            'ask_size': 100,
            'timestamp': datetime.now(),
        }

    def set_market_open(self, is_open: bool):
        """Set mock market open status."""
        self.market_open = is_open
