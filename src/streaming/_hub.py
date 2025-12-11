"""
Central coordinator for streaming market data.

Manages the stream manager, buffers, and fallback poller.
Provides a unified interface for data access and subscriptions.
"""

import uuid
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

from src.streaming.types import Bar, Quote, Trade
from src.streaming._buffer import BarBuffer, QuoteBuffer, TradeBuffer
from src.streaming._stream import StreamManager
from src.streaming._fallback import FallbackPoller
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataHub:
    """
    Central coordinator for all market data operations.

    Coordinates:
    - WebSocket streaming via StreamManager
    - In-memory buffers for recent data
    - Fallback polling when streaming unavailable
    - Subscription management for callbacks

    Example:
        hub = MarketDataHub(api_key, secret_key, feed='sip')
        hub.start(['TQQQ', 'SPY'])

        # Get data from buffer
        quote = hub.get_latest_quote('TQQQ')
        bars = hub.get_bars('TQQQ', n=15)

        # Register callbacks
        hub.on_bar(['TQQQ'], my_handler)

        hub.stop()
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = "iex",
        max_bars_per_symbol: int = 500,
        fallback_enabled: bool = True,
    ):
        """
        Initialize market data hub.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            feed: Data feed - 'iex' (free) or 'sip' (paid)
            max_bars_per_symbol: Max bars to keep in buffer per symbol
            fallback_enabled: Enable polling fallback when streaming fails
        """
        self._api_key = api_key
        self._secret_key = secret_key
        self._feed = feed
        self._fallback_enabled = fallback_enabled

        # Core components
        self._stream = StreamManager(api_key, secret_key, feed)
        self._bar_buffer = BarBuffer(max_bars_per_symbol)
        self._quote_buffer = QuoteBuffer()
        self._trade_buffer = TradeBuffer()
        self._fallback = FallbackPoller(api_key, secret_key, feed=feed) if fallback_enabled else None

        # Subscription tracking
        self._bar_handlers: Dict[str, Dict[str, Callable]] = {}  # symbol -> {sub_id -> handler}
        self._quote_handlers: Dict[str, Dict[str, Callable]] = {}
        self._trade_handlers: Dict[str, Dict[str, Callable]] = {}

        # Subscribed symbols
        self._subscribed_symbols: Set[str] = set()

        logger.info(f"MarketDataHub initialized with {feed.upper()} feed")

    def start(self, symbols: Optional[List[str]] = None) -> None:
        """
        Start streaming and subscribe to symbols.

        Args:
            symbols: Symbols to subscribe to (optional)
        """
        if symbols:
            self._subscribe_to_symbols(symbols)

        self._stream.start()
        logger.info("MarketDataHub started")

    def stop(self) -> None:
        """Stop streaming and cleanup."""
        self._stream.stop()
        logger.info("MarketDataHub stopped")

    def _subscribe_to_symbols(self, symbols: List[str]) -> None:
        """Subscribe to streaming data for symbols."""
        # Subscribe to bars and quotes
        self._stream.subscribe_bars(symbols, self._handle_bar)
        self._stream.subscribe_quotes(symbols, self._handle_quote)
        self._stream.subscribe_trades(symbols, self._handle_trade)

        self._subscribed_symbols.update(symbols)
        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def _handle_bar(self, bar: Bar) -> None:
        """Handle incoming bar from stream."""
        # Store in buffer
        self._bar_buffer.add(bar)

        # Notify handlers
        if bar.symbol in self._bar_handlers:
            for handler in self._bar_handlers[bar.symbol].values():
                try:
                    await handler(bar)
                except Exception as e:
                    logger.error(f"Error in bar handler for {bar.symbol}: {e}")

    async def _handle_quote(self, quote: Quote) -> None:
        """Handle incoming quote from stream."""
        # Store in buffer
        self._quote_buffer.update(quote)

        # Notify handlers
        if quote.symbol in self._quote_handlers:
            for handler in self._quote_handlers[quote.symbol].values():
                try:
                    await handler(quote)
                except Exception as e:
                    logger.error(f"Error in quote handler for {quote.symbol}: {e}")

    async def _handle_trade(self, trade: Trade) -> None:
        """Handle incoming trade from stream."""
        # Store in buffer
        self._trade_buffer.update(trade)

        # Notify handlers
        if trade.symbol in self._trade_handlers:
            for handler in self._trade_handlers[trade.symbol].values():
                try:
                    await handler(trade)
                except Exception as e:
                    logger.error(f"Error in trade handler for {trade.symbol}: {e}")

    # === Data Access (from buffer or fallback) ===

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get latest bar for a symbol.

        Tries buffer first, falls back to polling if buffer empty.
        """
        bar = self._bar_buffer.get_latest(symbol)
        if bar is None and self._fallback is not None:
            bar = self._fallback.get_latest_bar(symbol)
        return bar

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get latest quote for a symbol.

        Tries buffer first, falls back to polling if buffer empty.
        """
        quote = self._quote_buffer.get(symbol)
        if quote is None and self._fallback is not None:
            quote = self._fallback.get_latest_quote(symbol)
        return quote

    def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """
        Get latest trade for a symbol.

        Tries buffer first, falls back to polling if buffer empty.
        """
        trade = self._trade_buffer.get(symbol)
        if trade is None and self._fallback is not None:
            trade = self._fallback.get_latest_trade(symbol)
        return trade

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.

        Uses last trade price from buffer or fallback.
        """
        # Try trade buffer first
        price = self._trade_buffer.get_price(symbol)
        if price is not None:
            return price

        # Try quote mid
        price = self._quote_buffer.get_price(symbol)
        if price is not None:
            return price

        # Fallback to polling
        if self._fallback is not None:
            return self._fallback.get_latest_price(symbol)

        return None

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """
        Get recent bars for a symbol.

        Tries buffer first. If buffer has insufficient data (e.g., bot restarted mid-day),
        falls back to REST API to fetch complete data from market open.

        Args:
            symbol: Ticker symbol
            n: Number of bars (None = all available)

        Returns:
            DataFrame with OHLCV data
        """
        bars = self._bar_buffer.get_bars(symbol, n)

        # Check if buffer has sufficient data
        if n is not None and bars:
            # If requesting N bars, buffer should have at least 90% to be considered "sufficient"
            required_bars = int(n * 0.9)
            has_sufficient = len(bars) >= required_bars

            if not has_sufficient and self._fallback is not None:
                logger.warning(
                    f"Buffer for {symbol} has {len(bars)}/{n} bars requested ({len(bars)/n*100:.1f}%). "
                    f"Falling back to REST API for complete data from market open."
                )
                return self._fallback_from_market_open(symbol)

        # Buffer is empty - use fallback
        if not bars:
            if self._fallback is not None:
                logger.info(f"Buffer empty for {symbol}, falling back to REST API")
                return self._fallback_from_market_open(symbol)
            return pd.DataFrame()

        # Convert bars to DataFrame
        data = {
            "timestamp": [b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "trade_count": [b.trade_count for b in bars],
            "vwap": [b.vwap for b in bars],
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _fallback_from_market_open(self, symbol: str) -> pd.DataFrame:
        """
        Fetch bars from market open (9:30 AM ET) to now via REST API.

        This is used when buffer has insufficient data (e.g., bot restarted mid-day).

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with OHLCV data from market open to now
        """
        from src.utils.timezone import tz
        from alpaca.data.timeframe import TimeFrame

        # Get current time in ET
        now = tz.now()

        # Calculate market open time (9:30 AM ET today)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

        # If current time is before market open, fetch from yesterday's open
        if now < market_open:
            from datetime import timedelta
            market_open = market_open - timedelta(days=1)

        logger.info(
            f"Fetching {symbol} bars from market open ({market_open.strftime('%H:%M')}) "
            f"to now ({now.strftime('%H:%M')}) via REST API"
        )

        return self._fallback.get_bars(
            symbol=symbol,
            start=market_open,
            end=now,
            timeframe=TimeFrame.Minute
        )

    def get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP for a symbol."""
        bar = self.get_latest_bar(symbol)
        if bar is None:
            return None
        return bar.vwap

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current bid-ask spread for a symbol."""
        quote = self.get_latest_quote(symbol)
        if quote is None:
            return None
        return quote.spread

    # === Subscription Management ===

    def on_bar(self, symbols: List[str], handler: Callable) -> str:
        """
        Register callback for bar events.

        Args:
            symbols: Symbols to subscribe to
            handler: Async callback - async def handler(bar: Bar) -> None

        Returns:
            Subscription ID for later unsubscription
        """
        sub_id = str(uuid.uuid4())

        for symbol in symbols:
            if symbol not in self._bar_handlers:
                self._bar_handlers[symbol] = {}
            self._bar_handlers[symbol][sub_id] = handler

            # Subscribe to stream if not already
            if symbol not in self._subscribed_symbols:
                self._stream.subscribe_bars([symbol], self._handle_bar)
                self._subscribed_symbols.add(symbol)

        logger.debug(f"Registered bar handler {sub_id} for {symbols}")
        return sub_id

    def on_quote(self, symbols: List[str], handler: Callable) -> str:
        """
        Register callback for quote events.

        Args:
            symbols: Symbols to subscribe to
            handler: Async callback - async def handler(quote: Quote) -> None

        Returns:
            Subscription ID for later unsubscription
        """
        sub_id = str(uuid.uuid4())

        for symbol in symbols:
            if symbol not in self._quote_handlers:
                self._quote_handlers[symbol] = {}
            self._quote_handlers[symbol][sub_id] = handler

            # Subscribe to stream if not already
            if symbol not in self._subscribed_symbols:
                self._stream.subscribe_quotes([symbol], self._handle_quote)
                self._subscribed_symbols.add(symbol)

        logger.debug(f"Registered quote handler {sub_id} for {symbols}")
        return sub_id

    def on_trade(self, symbols: List[str], handler: Callable) -> str:
        """
        Register callback for trade events.

        Args:
            symbols: Symbols to subscribe to
            handler: Async callback - async def handler(trade: Trade) -> None

        Returns:
            Subscription ID for later unsubscription
        """
        sub_id = str(uuid.uuid4())

        for symbol in symbols:
            if symbol not in self._trade_handlers:
                self._trade_handlers[symbol] = {}
            self._trade_handlers[symbol][sub_id] = handler

            # Subscribe to stream if not already
            if symbol not in self._subscribed_symbols:
                self._stream.subscribe_trades([symbol], self._handle_trade)
                self._subscribed_symbols.add(symbol)

        logger.debug(f"Registered trade handler {sub_id} for {symbols}")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """
        Remove a callback subscription.

        Args:
            subscription_id: ID returned from on_bar/on_quote/on_trade
        """
        # Remove from all handler dicts
        for handlers_by_symbol in [self._bar_handlers, self._quote_handlers, self._trade_handlers]:
            for symbol, handlers in handlers_by_symbol.items():
                if subscription_id in handlers:
                    del handlers[subscription_id]

        logger.debug(f"Unsubscribed {subscription_id}")

    # === Status ===

    def is_connected(self) -> bool:
        """Check if WebSocket streaming is active."""
        return self._stream.is_running()

    def get_subscribed_symbols(self) -> Set[str]:
        """Get set of subscribed symbols."""
        return self._subscribed_symbols.copy()

    def clear_buffers(self) -> None:
        """Clear all buffers."""
        self._bar_buffer.clear()
        self._quote_buffer.clear()
        self._trade_buffer.clear()
        logger.info("Buffers cleared")
