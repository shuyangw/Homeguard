"""
Alpaca Broker - Alpaca Implementation of BrokerInterface

Translates between the broker-agnostic BrokerInterface and Alpaca's
specific API. All Alpaca-specific logic is contained here.

Usage:
    >>> broker = AlpacaBroker(api_key='KEY', secret_key='SECRET', paper=True)
    >>> account = broker.get_account()
    >>> order = broker.place_order('SPY', 10, OrderSide.BUY)

Design Principle:
- Adapter Pattern: Translate Alpaca API to standardized format
- Error Handling: Convert Alpaca exceptions to broker exceptions
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce as AlpacaTimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .broker_interface import (
    BrokerInterface,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerError,
    BrokerConnectionError,
    BrokerAuthError,
    InvalidOrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    NoPositionError,
    SymbolNotFoundError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker implementation.

    Translates between BrokerInterface and Alpaca-specific API.
    All Alpaca-specific logic is contained here.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (default: True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        try:
            # Alpaca-specific clients
            self.trading_client = TradingClient(api_key, secret_key, paper=paper)
            self.data_client = StockHistoricalDataClient(api_key, secret_key)

            logger.info(f"Initialized AlpacaBroker (paper={paper})")

        except Exception as e:
            logger.error(f"Failed to initialize AlpacaBroker: {e}")
            raise BrokerConnectionError(f"Failed to initialize Alpaca client: {e}")

    # ==================== Account Methods ====================

    def get_account(self) -> Dict:
        """Get account information (translated to standard format)."""
        try:
            account = self.trading_client.get_account()

            # Translate Alpaca account to standardized format
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'currency': account.currency,
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_positions(self) -> List[Dict]:
        """Get all current positions (translated to standard format)."""
        try:
            positions = self.trading_client.get_all_positions()

            # Translate Alpaca positions to standardized format
            return [
                {
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc),
                    'side': pos.side,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get specific position by symbol."""
        try:
            pos = self.trading_client.get_open_position(symbol)

            return {
                'symbol': pos.symbol,
                'quantity': int(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pnl_pct': float(pos.unrealized_plpc),
                'side': pos.side,
            }
        except Exception as e:
            # Alpaca raises exception if no position found
            logger.debug(f"No position found for {symbol}: {e}")
            return None

    # ==================== Order Methods ====================

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
        """Place order (broker-agnostic interface)."""
        try:
            # Translate to Alpaca-specific enums
            alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
            alpaca_tif = self._translate_time_in_force(time_in_force)

            # Create order request based on type
            if order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif
                )
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise InvalidOrderError("Limit price required for LIMIT orders")
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price
                )
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise InvalidOrderError("Stop price required for STOP orders")
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price
                )
            else:
                raise InvalidOrderError(f"Order type {order_type} not supported yet")

            # Submit order
            order = self.trading_client.submit_order(order_request)

            logger.info(f"Order placed: {symbol} {side.value} {quantity} @ {order_type.value}")

            # Translate to standardized format
            return self._translate_order(order)

        except InvalidOrderError:
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            if "not found" in str(e).lower():
                raise OrderNotFoundError(f"Order {order_id} not found")
            return False

    def get_order(self, order_id: str) -> Dict:
        """Get order details by ID."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return self._translate_order(order)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            if "not found" in str(e).lower():
                raise OrderNotFoundError(f"Order {order_id} not found")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get orders with optional filters."""
        try:
            # Translate status filter
            status_filter = status.value if status else 'all'

            # Get orders from Alpaca
            orders = self.trading_client.get_orders(
                status=status_filter,
                after=start_date,
                until=end_date
            )

            # Translate to standardized format
            return [self._translate_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def close_position(
        self,
        symbol: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """Close a position (or partial position)."""
        try:
            # Get current position
            position = self.get_position(symbol)
            if position is None:
                raise NoPositionError(f"No position for {symbol}")

            # Determine quantity to close
            qty_to_close = quantity if quantity is not None else abs(position['quantity'])

            # Determine side (opposite of current position)
            current_qty = position['quantity']
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY

            # Place market order to close
            logger.info(f"Closing position: {symbol} {qty_to_close} shares")
            return self.place_order(
                symbol=symbol,
                quantity=qty_to_close,
                side=side,
                order_type=OrderType.MARKET
            )
        except NoPositionError:
            raise
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def close_all_positions(self) -> List[Dict]:
        """Close all open positions."""
        try:
            # Alpaca has a built-in method for this
            result = self.trading_client.close_all_positions(cancel_orders=True)

            logger.info(f"Closed all positions")
            return [{'status': 'closed_all', 'result': str(result)}]
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    # ==================== Market Data Methods ====================

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get latest bid/ask quote."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]

            return {
                'symbol': symbol,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            if "not found" in str(e).lower():
                raise SymbolNotFoundError(f"Symbol {symbol} not found")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_latest_trade(self, symbol: str) -> Dict:
        """Get latest trade."""
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = self.data_client.get_stock_latest_trade(request)
            trade = trades[symbol]

            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': int(trade.size),
                'timestamp': trade.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get trade for {symbol}: {e}")
            if "not found" in str(e).lower():
                raise SymbolNotFoundError(f"Symbol {symbol} not found")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get historical OHLCV bars."""
        try:
            # Translate timeframe string to Alpaca TimeFrame
            tf = self._translate_timeframe(timeframe)

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end
            )

            bars = self.data_client.get_stock_bars(request)
            return bars.df
        except Exception as e:
            logger.error(f"Failed to get bars: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    # ==================== Utility Methods ====================

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market hours: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_market_hours(self, date: datetime) -> Tuple[datetime, datetime]:
        """Get market hours for a specific date."""
        try:
            # Use current date if none specified
            target_date = date.date() if date else datetime.now().date()

            # Create calendar request
            request = GetCalendarRequest(start=target_date, end=target_date)
            calendar = self.trading_client.get_calendar(filters=request)

            if len(calendar) == 0:
                raise BrokerConnectionError(f"Market closed on {target_date}")

            day = calendar[0]
            return (day.open, day.close)
        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def test_connection(self) -> bool:
        """Test broker connection."""
        try:
            account = self.trading_client.get_account()
            logger.success(f"Alpaca connection successful (account: {account.id})")
            return True
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False

    # ==================== Helper Methods ====================

    def _translate_order(self, alpaca_order) -> Dict:
        """Translate Alpaca order to standardized format."""
        return {
            'order_id': alpaca_order.id,
            'symbol': alpaca_order.symbol,
            'quantity': int(alpaca_order.qty),
            'side': alpaca_order.side.value,
            'order_type': alpaca_order.type.value,
            'status': alpaca_order.status.value,
            'limit_price': float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            'stop_price': float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            'created_at': alpaca_order.created_at,
            'filled_qty': int(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0,
            'avg_fill_price': float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
        }

    def _translate_time_in_force(self, tif: TimeInForce) -> AlpacaTimeInForce:
        """Translate TimeInForce to Alpaca enum."""
        mapping = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
        }
        return mapping.get(tif, AlpacaTimeInForce.DAY)

    def _translate_timeframe(self, timeframe: str) -> TimeFrame:
        """Translate timeframe string to Alpaca TimeFrame."""
        # Parse timeframe string (e.g., "1Min", "5Min", "1Hour", "1Day")
        timeframe = timeframe.lower()

        if 'min' in timeframe:
            amount = int(timeframe.replace('min', ''))
            return TimeFrame(amount, TimeFrameUnit.Minute)
        elif 'hour' in timeframe:
            amount = int(timeframe.replace('hour', ''))
            return TimeFrame(amount, TimeFrameUnit.Hour)
        elif 'day' in timeframe:
            amount = int(timeframe.replace('day', ''))
            return TimeFrame(amount, TimeFrameUnit.Day)
        elif 'week' in timeframe:
            amount = int(timeframe.replace('week', ''))
            return TimeFrame(amount, TimeFrameUnit.Week)
        elif 'month' in timeframe:
            amount = int(timeframe.replace('month', ''))
            return TimeFrame(amount, TimeFrameUnit.Month)
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
