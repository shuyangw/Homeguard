"""
Coinbase Advanced Trade Broker Adapter.

Implements CryptoTradingInterface using Coinbase's official Python SDK.
Used as primary crypto broker for CSCM strategy.

Requirements:
    pip install coinbase-advanced-py

Environment Variables:
    COINBASE_API_KEY: CDP API key
    COINBASE_API_SECRET: CDP API secret

Documentation:
    https://docs.cdp.coinbase.com/advanced-trade/docs/welcome
    https://github.com/coinbase/coinbase-advanced-py
"""

import os
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from src.trading.brokers.interfaces import (
    CryptoTradingInterface,
    AccountInterface,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerConnectionError,
    BrokerAuthError,
    InvalidOrderError,
    InsufficientFundsError,
    NoPositionError,
    SymbolNotFoundError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Lazy import to avoid import errors if SDK not installed
def _get_coinbase_client():
    """Get Coinbase REST client with lazy import."""
    try:
        from coinbase.rest import RESTClient
        return RESTClient
    except ImportError:
        raise ImportError(
            "coinbase-advanced-py not installed. "
            "Install with: pip install coinbase-advanced-py"
        )


class CoinbaseBroker(CryptoTradingInterface, AccountInterface):
    """
    Coinbase Advanced Trade broker adapter.

    Implements crypto trading interface for Coinbase Advanced Trade API.
    Uses the official coinbase-advanced-py SDK.

    Features:
    - 24/7 crypto trading
    - Market and limit orders
    - Fractional quantities
    - Real-time quotes

    Usage:
        broker = CoinbaseBroker()  # Uses env vars for auth
        broker = CoinbaseBroker(api_key='...', api_secret='...')

        # Get positions
        positions = broker.get_crypto_positions()

        # Place order
        order = broker.place_crypto_order(
            symbol='BTC/USD',
            quantity=Decimal('0.001'),
            side=OrderSide.BUY
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = False
    ):
        """
        Initialize Coinbase broker.

        Args:
            api_key: CDP API key (or env COINBASE_API_KEY)
            api_secret: CDP API secret (or env COINBASE_API_SECRET)
            paper: If True, use sandbox (not currently supported by Coinbase)
        """
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise BrokerAuthError(
                "Coinbase API credentials required. "
                "Set COINBASE_API_KEY and COINBASE_API_SECRET environment variables."
            )

        self.paper = paper  # Note: Coinbase doesn't have sandbox for Advanced Trade

        # Initialize client
        try:
            RESTClient = _get_coinbase_client()
            self._client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            logger.info("[Coinbase] Initialized broker connection")
        except Exception as e:
            raise BrokerConnectionError(f"Failed to initialize Coinbase client: {e}")

    # ==================== Account Interface ====================

    def get_account(self) -> Dict:
        """Get account information."""
        try:
            accounts = self._client.get_accounts()
            # Calculate totals from all currency accounts
            total_balance = 0.0
            available_balance = 0.0
            holdings = []

            for account in accounts.get('accounts', []):
                currency = account.get('currency', '')
                balance = float(account.get('available_balance', {}).get('value', 0))

                if balance > 0 or currency == 'USD':
                    holdings.append({
                        'currency': currency,
                        'balance': balance,
                    })

                # Sum USD value if available
                if currency == 'USD':
                    available_balance += balance
                    total_balance += balance

            return {
                'account_id': accounts.get('accounts', [{}])[0].get('uuid', ''),
                'status': 'active',
                'currency': 'USD',
                'cash': available_balance,
                'buying_power': available_balance,
                'portfolio_value': total_balance,
                'holdings': holdings,
            }

        except Exception as e:
            logger.error(f"[Coinbase] Failed to get account: {e}")
            raise BrokerConnectionError(f"Failed to get account: {e}")

    def test_connection(self) -> bool:
        """Test broker connection."""
        try:
            self._client.get_accounts()
            return True
        except Exception as e:
            logger.error(f"[Coinbase] Connection test failed: {e}")
            return False

    # ==================== Crypto Trading Interface ====================

    def get_crypto_positions(self) -> List[Dict]:
        """Get all crypto positions."""
        try:
            accounts = self._client.get_accounts()
            positions = []

            for account in accounts.get('accounts', []):
                currency = account.get('currency', '')
                if currency == 'USD':
                    continue  # Skip USD, it's not a position

                balance = float(account.get('available_balance', {}).get('value', 0))
                if balance <= 0:
                    continue

                # Get current price
                symbol = f"{currency}-USD"
                try:
                    product = self._client.get_product(symbol)
                    current_price = float(product.get('price', 0))
                except (KeyError, ValueError, TypeError):
                    current_price = 0

                market_value = balance * current_price

                positions.append({
                    'symbol': f"{currency}/USD",
                    'quantity': Decimal(str(balance)),
                    'avg_entry_price': 0.0,  # Not available from account
                    'current_price': current_price,
                    'market_value': market_value,
                    'unrealized_pnl': 0.0,  # Would need entry price
                    'unrealized_pnl_pct': 0.0,
                })

            return positions

        except Exception as e:
            logger.error(f"[Coinbase] Failed to get positions: {e}")
            raise BrokerConnectionError(f"Failed to get positions: {e}")

    def get_crypto_position(self, symbol: str) -> Optional[Dict]:
        """Get specific crypto position."""
        symbol = self.normalize_symbol(symbol)
        positions = self.get_crypto_positions()

        for pos in positions:
            if pos['symbol'] == symbol:
                return pos

        return None

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
        """Place a crypto order."""
        try:
            # Normalize symbol to Coinbase format (BTC-USD)
            product_id = symbol.replace('/', '-').upper()
            client_order_id = str(uuid.uuid4())

            # Determine order configuration
            if order_type == OrderType.MARKET:
                if side == OrderSide.BUY:
                    # Market buy requires quote_size (USD amount) or base_size
                    order = self._client.market_order_buy(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(quantity)
                    )
                else:
                    order = self._client.market_order_sell(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(quantity)
                    )

            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise InvalidOrderError("Limit price required for limit orders")

                if side == OrderSide.BUY:
                    order = self._client.limit_order_gtc_buy(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(quantity),
                        limit_price=str(limit_price)
                    )
                else:
                    order = self._client.limit_order_gtc_sell(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(quantity),
                        limit_price=str(limit_price)
                    )
            else:
                raise InvalidOrderError(f"Unsupported order type: {order_type}")

            # Parse response
            order_data = order.get('order', order)
            order_id = order_data.get('order_id', client_order_id)
            status = self._parse_order_status(order_data.get('status', 'PENDING'))

            logger.info(
                f"[Coinbase] Order placed: {side.value} {quantity} {symbol} "
                f"@ {order_type.value} - ID: {order_id}"
            )

            return {
                'order_id': order_id,
                'client_order_id': client_order_id,
                'symbol': symbol,
                'quantity': quantity,
                'side': side.value,
                'order_type': order_type.value,
                'status': status.value,
                'limit_price': limit_price,
                'created_at': datetime.now(),
                'filled_qty': Decimal('0'),
                'avg_fill_price': 0.0,
                'fees': 0.0,
            }

        except InvalidOrderError:
            raise
        except Exception as e:
            error_msg = str(e)
            if 'insufficient' in error_msg.lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            logger.error(f"[Coinbase] Order failed: {e}")
            raise BrokerConnectionError(f"Failed to place order: {e}")

    def close_crypto_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """Close a crypto position."""
        position = self.get_crypto_position(symbol)
        if position is None:
            raise NoPositionError(f"No position for {symbol}")

        close_qty = quantity if quantity else position['quantity']

        return self.place_crypto_order(
            symbol=symbol,
            quantity=close_qty,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET
        )

    def close_all_crypto_positions(self) -> List[Dict]:
        """Close all crypto positions."""
        positions = self.get_crypto_positions()
        orders = []

        for pos in positions:
            try:
                order = self.close_crypto_position(pos['symbol'])
                orders.append(order)
            except Exception as e:
                logger.error(f"[Coinbase] Failed to close {pos['symbol']}: {e}")

        return orders

    def get_crypto_quote(self, symbol: str) -> Dict:
        """Get current quote for a crypto symbol."""
        try:
            product_id = symbol.replace('/', '-').upper()
            product = self._client.get_product(product_id)

            return {
                'symbol': symbol,
                'bid': float(product.get('bid', 0)),
                'ask': float(product.get('ask', 0)),
                'last': float(product.get('price', 0)),
                'volume_24h': float(product.get('volume_24h', 0)),
                'timestamp': datetime.now(),
            }

        except Exception as e:
            if 'not found' in str(e).lower():
                raise SymbolNotFoundError(f"Symbol not found: {symbol}")
            raise BrokerConnectionError(f"Failed to get quote: {e}")

    def get_crypto_balance(self, currency: str = 'USD') -> Dict:
        """Get crypto account balance."""
        try:
            accounts = self._client.get_accounts()

            for account in accounts.get('accounts', []):
                if account.get('currency', '').upper() == currency.upper():
                    available = float(account.get('available_balance', {}).get('value', 0))
                    hold = float(account.get('hold', {}).get('value', 0))
                    return {
                        'total': available + hold,
                        'available': available,
                        'held': hold,
                    }

            return {'total': 0.0, 'available': 0.0, 'held': 0.0}

        except Exception as e:
            raise BrokerConnectionError(f"Failed to get balance: {e}")

    # ==================== Order Management (from interface) ====================

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        try:
            order = self._client.get_order(order_id)
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"[Coinbase] Failed to get order {order_id}: {e}")
            return None

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Get orders with optional filters."""
        try:
            params = {}
            if status:
                params['order_status'] = status.value.upper()
            if symbol:
                params['product_id'] = symbol.replace('/', '-').upper()

            orders = self._client.list_orders(**params)
            return [self._parse_order(o) for o in orders.get('orders', [])]

        except Exception as e:
            logger.error(f"[Coinbase] Failed to get orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self._client.cancel_orders([order_id])
            logger.info(f"[Coinbase] Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[Coinbase] Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        try:
            result = self._client.cancel_orders([])  # Empty = cancel all
            cancelled = len(result.get('results', []))
            logger.info(f"[Coinbase] Cancelled {cancelled} orders")
            return cancelled
        except Exception as e:
            logger.error(f"[Coinbase] Failed to cancel all orders: {e}")
            return 0

    # ==================== Helper Methods ====================

    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Coinbase order status to enum."""
        status_map = {
            'PENDING': OrderStatus.PENDING,
            'OPEN': OrderStatus.PENDING,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'CANCELED': OrderStatus.CANCELLED,
            'FAILED': OrderStatus.REJECTED,
        }
        return status_map.get(status.upper(), OrderStatus.PENDING)

    def _parse_order(self, order: Dict) -> Dict:
        """Parse Coinbase order to standard format."""
        return {
            'order_id': order.get('order_id', ''),
            'client_order_id': order.get('client_order_id', ''),
            'symbol': order.get('product_id', '').replace('-', '/'),
            'quantity': Decimal(order.get('size', '0')),
            'side': order.get('side', '').lower(),
            'order_type': order.get('order_type', 'market').lower(),
            'status': self._parse_order_status(order.get('status', '')).value,
            'limit_price': float(order.get('limit_price', 0) or 0),
            'created_at': order.get('created_time'),
            'filled_qty': Decimal(order.get('filled_size', '0')),
            'avg_fill_price': float(order.get('average_filled_price', 0) or 0),
            'fees': float(order.get('total_fees', 0) or 0),
        }

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format (BTC/USD)."""
        symbol = symbol.upper()
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        if '/' not in symbol and not symbol.endswith('USD'):
            symbol = f"{symbol}/USD"
        return symbol
