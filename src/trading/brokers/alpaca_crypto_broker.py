"""
Alpaca Crypto Broker Adapter.

Implements CryptoTradingInterface using Alpaca's Crypto Trading API.
Used as secondary/failover broker for CSCM strategy.

Alpaca Crypto Trading:
- 24/7 trading
- Fractional quantities supported
- Market and limit orders
- Symbol format: BTC/USD

Documentation:
    https://docs.alpaca.markets/docs/crypto-trading
"""

import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
    TimeInForce as AlpacaTimeInForce,
    AssetClass,
)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

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


class AlpacaCryptoBroker(CryptoTradingInterface, AccountInterface):
    """
    Alpaca Crypto broker adapter.

    Implements crypto trading interface for Alpaca's Crypto API.
    Uses the same API keys as stock trading but different asset class.

    Features:
    - 24/7 crypto trading
    - Market and limit orders
    - Fractional quantities
    - Same credentials as Alpaca stock

    Usage:
        broker = AlpacaCryptoBroker()  # Uses env vars for auth
        broker = AlpacaCryptoBroker(api_key='...', secret_key='...', paper=True)

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
        secret_key: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca crypto broker.

        Args:
            api_key: Alpaca API key (or env ALPACA_PAPER_KEY_ID / APCA_API_KEY_ID)
            secret_key: Alpaca secret key (or env ALPACA_PAPER_SECRET_KEY / APCA_API_SECRET_KEY)
            paper: If True, use paper trading endpoint
        """
        # Check paper-specific env vars first, then fall back to standard Alpaca vars
        if paper:
            self.api_key = api_key or os.getenv('ALPACA_PAPER_KEY_ID') or os.getenv('APCA_API_KEY_ID')
            self.secret_key = secret_key or os.getenv('ALPACA_PAPER_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
        else:
            self.api_key = api_key or os.getenv('APCA_API_KEY_ID')
            self.secret_key = secret_key or os.getenv('APCA_API_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise BrokerAuthError(
                "Alpaca API credentials required. "
                "Set ALPACA_PAPER_KEY_ID/ALPACA_PAPER_SECRET_KEY (paper) or "
                "APCA_API_KEY_ID/APCA_API_SECRET_KEY environment variables."
            )

        self.paper = paper

        try:
            # Trading client (same for stocks and crypto)
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=paper
            )
            # Crypto data client
            self._data_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            logger.info(f"[Alpaca Crypto] Initialized broker (paper={paper})")

        except Exception as e:
            raise BrokerConnectionError(f"Failed to initialize Alpaca client: {e}")

    # ==================== Account Interface ====================

    def get_account(self) -> Dict:
        """Get account information."""
        try:
            account = self._trading_client.get_account()

            return {
                'account_id': account.id,
                'status': account.status,
                'currency': account.currency,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
            }

        except Exception as e:
            logger.error(f"[Alpaca Crypto] Failed to get account: {e}")
            raise BrokerConnectionError(f"Failed to get account: {e}")

    def test_connection(self) -> bool:
        """Test broker connection."""
        try:
            self._trading_client.get_account()
            return True
        except Exception as e:
            logger.error(f"[Alpaca Crypto] Connection test failed: {e}")
            return False

    # ==================== Crypto Trading Interface ====================

    def get_crypto_positions(self) -> List[Dict]:
        """Get all crypto positions."""
        try:
            all_positions = self._trading_client.get_all_positions()
            crypto_positions = []

            for pos in all_positions:
                # Filter for crypto assets only
                if hasattr(pos, 'asset_class') and pos.asset_class == AssetClass.CRYPTO:
                    crypto_positions.append({
                        'symbol': pos.symbol.replace('-', '/'),  # BTC-USD -> BTC/USD
                        'quantity': Decimal(str(pos.qty)),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pl),
                        'unrealized_pnl_pct': float(pos.unrealized_plpc),
                    })
                elif '/' in pos.symbol or pos.symbol.endswith('USD'):
                    # Fallback for crypto detection by symbol
                    crypto_positions.append({
                        'symbol': pos.symbol.replace('-', '/'),
                        'quantity': Decimal(str(pos.qty)),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pl),
                        'unrealized_pnl_pct': float(pos.unrealized_plpc),
                    })

            return crypto_positions

        except Exception as e:
            logger.error(f"[Alpaca Crypto] Failed to get positions: {e}")
            raise BrokerConnectionError(f"Failed to get positions: {e}")

    def get_crypto_position(self, symbol: str) -> Optional[Dict]:
        """Get specific crypto position."""
        # Normalize to Alpaca format (BTC/USD or BTCUSD)
        alpaca_symbol = symbol.replace('/', '')

        try:
            pos = self._trading_client.get_open_position(alpaca_symbol)

            return {
                'symbol': symbol,
                'quantity': Decimal(str(pos.qty)),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pnl_pct': float(pos.unrealized_plpc),
            }

        except Exception as e:
            logger.debug(f"[Alpaca Crypto] No position for {symbol}: {e}")
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
            # Convert to Alpaca format
            alpaca_symbol = symbol.replace('/', '')  # BTC/USD -> BTCUSD
            alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
            alpaca_tif = self._convert_time_in_force(time_in_force)

            if order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=float(quantity),
                    side=alpaca_side,
                    time_in_force=alpaca_tif
                )
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise InvalidOrderError("Limit price required for limit orders")
                order_request = LimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=float(quantity),
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price
                )
            else:
                raise InvalidOrderError(f"Unsupported order type: {order_type}")

            order = self._trading_client.submit_order(order_request)

            logger.info(
                f"[Alpaca Crypto] Order placed: {side.value} {quantity} {symbol} "
                f"@ {order_type.value} - ID: {order.id}"
            )

            return {
                'order_id': str(order.id),
                'client_order_id': order.client_order_id,
                'symbol': symbol,
                'quantity': Decimal(str(order.qty)),
                'side': side.value,
                'order_type': order_type.value,
                'status': self._convert_order_status(order.status).value,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'created_at': order.created_at,
                'filled_qty': Decimal(str(order.filled_qty or 0)),
                'avg_fill_price': float(order.filled_avg_price or 0),
                'fees': 0.0,  # Alpaca includes fees in price
            }

        except InvalidOrderError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if 'insufficient' in error_msg:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            logger.error(f"[Alpaca Crypto] Order failed: {e}")
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
                logger.error(f"[Alpaca Crypto] Failed to close {pos['symbol']}: {e}")

        return orders

    def get_crypto_quote(self, symbol: str) -> Dict:
        """Get current quote for a crypto symbol."""
        try:
            # Data client expects "BTC/USD" format (with slash)
            # Normalize to ensure slash format
            alpaca_symbol = self.normalize_symbol(symbol)
            request = CryptoLatestQuoteRequest(symbol_or_symbols=[alpaca_symbol])
            quotes = self._data_client.get_crypto_latest_quote(request)

            quote = quotes.get(alpaca_symbol)
            if not quote:
                raise SymbolNotFoundError(f"No quote for {symbol}")

            return {
                'symbol': symbol,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'last': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                'volume_24h': 0.0,  # Not in quote
                'timestamp': quote.timestamp,
            }

        except SymbolNotFoundError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower():
                raise SymbolNotFoundError(f"Symbol not found: {symbol}")
            raise BrokerConnectionError(f"Failed to get quote: {e}")

    def get_crypto_balance(self, currency: str = 'USD') -> Dict:
        """Get crypto account balance."""
        try:
            account = self._trading_client.get_account()

            return {
                'total': float(account.portfolio_value),
                'available': float(account.buying_power),
                'held': float(account.portfolio_value) - float(account.buying_power),
            }

        except Exception as e:
            raise BrokerConnectionError(f"Failed to get balance: {e}")

    # ==================== Order Management ====================

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        try:
            order = self._trading_client.get_order_by_id(order_id)
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"[Alpaca Crypto] Failed to get order {order_id}: {e}")
            return None

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Get orders with optional filters."""
        try:
            orders = self._trading_client.get_orders()
            result = []

            for order in orders:
                # Filter by symbol if specified
                if symbol:
                    alpaca_symbol = symbol.replace('/', '')
                    if order.symbol != alpaca_symbol:
                        continue

                # Filter by status if specified
                if status:
                    order_status = self._convert_order_status(order.status)
                    if order_status != status:
                        continue

                result.append(self._parse_order(order))

            return result

        except Exception as e:
            logger.error(f"[Alpaca Crypto] Failed to get orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info(f"[Alpaca Crypto] Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[Alpaca Crypto] Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        try:
            result = self._trading_client.cancel_orders()
            cancelled = len(result)
            logger.info(f"[Alpaca Crypto] Cancelled {cancelled} orders")
            return cancelled
        except Exception as e:
            logger.error(f"[Alpaca Crypto] Failed to cancel all orders: {e}")
            return 0

    # ==================== Helper Methods ====================

    def _convert_time_in_force(self, tif: TimeInForce) -> AlpacaTimeInForce:
        """Convert TimeInForce enum to Alpaca format."""
        mapping = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
        }
        return mapping.get(tif, AlpacaTimeInForce.GTC)

    def _convert_order_status(self, status) -> OrderStatus:
        """Convert Alpaca order status to enum."""
        status_str = str(status).upper()
        mapping = {
            'NEW': OrderStatus.PENDING,
            'ACCEPTED': OrderStatus.PENDING,
            'PENDING_NEW': OrderStatus.PENDING,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.CANCELLED,
        }
        return mapping.get(status_str, OrderStatus.PENDING)

    def _parse_order(self, order) -> Dict:
        """Parse Alpaca order to standard format."""
        return {
            'order_id': str(order.id),
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'quantity': Decimal(str(order.qty)),
            'side': str(order.side).lower(),
            'order_type': str(order.order_type).lower(),
            'status': self._convert_order_status(order.status).value,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'created_at': order.created_at,
            'filled_qty': Decimal(str(order.filled_qty or 0)),
            'avg_fill_price': float(order.filled_avg_price or 0),
            'fees': 0.0,
        }

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format (BTC/USD)."""
        symbol = symbol.upper()
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        if '/' not in symbol:
            # Assume it's BTCUSD format, split at USD
            if symbol.endswith('USD'):
                symbol = f"{symbol[:-3]}/USD"
        return symbol
