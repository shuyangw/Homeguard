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
import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce as AlpacaTimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

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

logger = get_logger()  # Use global logger (no file creation)


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

    def get_stock_positions(self) -> List[Dict]:
        """Get all current stock positions (translated to standard format)."""
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

    def get_stock_position(self, symbol: str) -> Optional[Dict]:
        """Get specific stock position by symbol."""
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
            logger.info(f"No position found for {symbol}: {e}")
            return None

    # ==================== Order Methods ====================

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
        """Place stock order (broker-agnostic interface)."""
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
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            # Translate status filter (handle both enum and string)
            if status is None:
                status_filter = QueryOrderStatus.ALL
            elif isinstance(status, str):
                status_map = {
                    'open': QueryOrderStatus.OPEN,
                    'closed': QueryOrderStatus.CLOSED,
                    'all': QueryOrderStatus.ALL
                }
                status_filter = status_map.get(status.lower(), QueryOrderStatus.ALL)
            else:
                # Assume it's our internal OrderStatus enum
                status_map = {
                    'open': QueryOrderStatus.OPEN,
                    'closed': QueryOrderStatus.CLOSED,
                }
                status_filter = status_map.get(status.value.lower(), QueryOrderStatus.ALL)

            # Build request object
            request = GetOrdersRequest(
                status=status_filter,
                after=start_date,
                until=end_date
            )

            # Get orders from Alpaca
            orders = self.trading_client.get_orders(filter=request)

            # Translate to standardized format
            return [self._translate_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders (convenience method)."""
        try:
            # Import Alpaca request class
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            # Get only open orders from Alpaca using request object
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.trading_client.get_orders(filter=request)

            # Translate to standardized format
            return [self._translate_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def close_stock_position(
        self,
        symbol: str,
        quantity: Optional[int] = None
    ) -> Dict:
        """Close a stock position (or partial position)."""
        try:
            # Get current position
            position = self.get_stock_position(symbol)
            if position is None:
                raise NoPositionError(f"No position for {symbol}")

            # Determine quantity to close
            qty_to_close = quantity if quantity is not None else abs(position['quantity'])

            # Determine side (opposite of current position)
            current_qty = position['quantity']
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY

            # Place market order to close
            logger.info(f"Closing position: {symbol} {qty_to_close} shares")
            return self.place_stock_order(
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

    def close_all_stock_positions(self) -> List[Dict]:
        """Close all open stock positions."""
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
            feed = DataFeed.IEX if self.paper else DataFeed.SIP
            request = StockLatestQuoteRequest(
                symbol_or_symbols=[symbol],
                feed=feed
            )
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
            feed = DataFeed.IEX if self.paper else DataFeed.SIP
            request = StockLatestTradeRequest(
                symbol_or_symbols=[symbol],
                feed=feed
            )
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

            # Use IEX feed for paper trading (free tier)
            # SIP feed requires paid subscription
            feed = DataFeed.IEX if self.paper else DataFeed.SIP

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
                feed=feed
            )

            bars = self.data_client.get_stock_bars(request)
            df = bars.df

            # Convert UTC timestamps to Eastern Time
            # Contract: All AlpacaBroker data is returned in ET
            df = self._ensure_et_timezone(df)

            return df
        except Exception as e:
            logger.error(f"Failed to get bars: {e}")
            raise BrokerConnectionError(f"Alpaca API error: {e}")

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars for a single symbol.

        This method is used by strategy adapters for fetching market data.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string (e.g., '1D', '1Min', '5Min', '1Hour')

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        try:
            # Use get_bars with single symbol
            df = self.get_bars(
                symbols=[symbol],
                timeframe=timeframe,
                start=start,
                end=end
            )

            # Extract data for the symbol and reset index
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # If multi-index (symbol, timestamp), extract just this symbol
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')

            return df

        except Exception as e:
            logger.error(f"Failed to get historical bars for {symbol}: {e}")
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

    def _ensure_et_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame index to Eastern Time.

        Alpaca API returns all timestamps in UTC. This method converts the index
        to US/Eastern so that time-based operations (like between_time) work
        correctly for market hours.

        Contract: All data returned from AlpacaBroker is in Eastern Time.

        Args:
            df: DataFrame with UTC timestamp index

        Returns:
            DataFrame with Eastern Time index
        """
        if df.empty:
            return df

        ET = pytz.timezone('America/New_York')

        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex case: (symbol, timestamp)
            # Get the timestamp level (level 1)
            ts_level = df.index.get_level_values(1)
            if hasattr(ts_level, 'tz') and ts_level.tz is not None:
                # Convert to ET
                new_ts = ts_level.tz_convert(ET)
                # Rebuild the MultiIndex with converted timestamps
                df.index = pd.MultiIndex.from_arrays(
                    [df.index.get_level_values(0), new_ts],
                    names=df.index.names
                )
        else:
            # Simple DatetimeIndex case
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_convert(ET)
            elif hasattr(df.index, 'tz_localize'):
                # Naive datetime - assume UTC
                df.index = df.index.tz_localize('UTC').tz_convert(ET)

        return df

    def _translate_order(self, alpaca_order) -> Dict:
        """Translate Alpaca order to standardized format."""
        return {
            'order_id': str(alpaca_order.id),
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
        import re

        # Extract number and unit using regex before lowercasing
        match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        amount = int(match.group(1))
        unit = match.group(2).lower()

        if 'min' in unit:
            return TimeFrame(amount, TimeFrameUnit.Minute)
        elif 'hour' in unit or unit == 'h':
            return TimeFrame(amount, TimeFrameUnit.Hour)
        elif 'day' in unit or unit == 'd':
            return TimeFrame(amount, TimeFrameUnit.Day)
        elif 'week' in unit or unit == 'w':
            return TimeFrame(amount, TimeFrameUnit.Week)
        elif 'month' in unit or unit == 'm':
            return TimeFrame(amount, TimeFrameUnit.Month)
        else:
            raise ValueError(f"Invalid timeframe unit: {unit}")
