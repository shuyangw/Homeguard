"""
Mock Broker for Testing

Implements BrokerInterface for testing purposes without requiring real API access.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
import uuid
import pandas as pd

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
            'currency': 'USD',
        }

    def test_connection(self) -> bool:
        """Test mock connection (always succeeds)."""
        return True

    # ==================== Market Hours ====================

    def is_market_open(self) -> bool:
        """Check if mock market is open."""
        return self.market_open

    def get_market_hours(self, date: datetime) -> Tuple[datetime, datetime]:
        """Get mock market hours."""
        target_date = date.date() if hasattr(date, 'date') else date
        open_time = datetime.combine(target_date, time(9, 30))
        close_time = datetime.combine(target_date, time(16, 0))
        return (open_time, close_time)

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
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get mock bars as DataFrame with MultiIndex."""
        records = []
        for symbol in symbols:
            # Generate mock data
            current = start
            while current <= end:
                records.append({
                    'symbol': symbol,
                    'timestamp': current,
                    'open': 100.0,
                    'high': 101.0,
                    'low': 99.0,
                    'close': 100.5,
                    'volume': 1000000,
                })
                current += timedelta(days=1)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.set_index(['symbol', 'timestamp'])
        return df

    # ==================== Order Management ====================

    def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.orders[order_id]
        order['status'] = OrderStatus.CANCELLED.value
        return True

    def get_order(self, order_id: str) -> Dict:
        """Get mock order by ID."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get all mock orders with optional filters."""
        orders = list(self.orders.values())

        if status:
            status_value = status.value if isinstance(status, OrderStatus) else status
            orders = [o for o in orders if o['status'] == status_value]

        if start_date:
            orders = [o for o in orders if o.get('submitted_at', datetime.min) >= start_date]

        if end_date:
            orders = [o for o in orders if o.get('submitted_at', datetime.max) <= end_date]

        return orders

    # ==================== Stock Trading ====================

    def get_stock_positions(self) -> List[Dict]:
        """Get all open stock positions."""
        return list(self.positions.values())

    def get_stock_position(self, symbol: str) -> Optional[Dict]:
        """Get stock position for specific symbol."""
        return self.positions.get(symbol)

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
        """Place mock stock order."""
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
            'filled_avg_price': fill_price,  # Matches interface spec
            'avg_fill_price': fill_price,    # Alias for compatibility
            'limit_price': limit_price,
            'stop_price': stop_price,
            'time_in_force': time_in_force.value,
            'created_at': datetime.now(),
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

    def close_stock_position(
        self,
        symbol: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """Close mock stock position."""
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        position = self.positions[symbol]
        qty = quantity if quantity is not None else abs(position['quantity'])
        side = OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY

        return self.place_stock_order(
            symbol=symbol,
            quantity=qty,
            side=side,
            order_type=OrderType.MARKET
        )

    def close_all_stock_positions(self) -> List[Dict]:
        """Close all mock stock positions."""
        closed_orders = []
        for symbol in list(self.positions.keys()):
            order = self.close_stock_position(symbol)
            closed_orders.append(order)
        return closed_orders

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
            pos['market_value'] = new_qty * price
            pos['unrealized_pnl'] = new_qty * (price - new_avg_price)
            pos['unrealized_pnl_pct'] = (price - new_avg_price) / new_avg_price if new_avg_price else 0
            pos['side'] = 'long' if new_qty > 0 else 'short'
        else:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'avg_entry_price': price,
                'current_price': price,
                'market_value': quantity * price,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'side': 'long' if quantity > 0 else 'short',
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
            pos['market_value'] = pos['quantity'] * price
            pos['unrealized_pnl'] = pos['quantity'] * (price - pos['avg_entry_price'])

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
