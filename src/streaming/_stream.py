"""
WebSocket stream manager for Alpaca real-time market data.

Wraps Alpaca's StockDataStream to provide a simpler interface
and handle threading concerns.

Reference: https://alpaca.markets/sdks/python/api_reference/data/stock/live.html
"""

import threading
from typing import Callable, Optional, Set

from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from src.streaming.types import Bar, Quote, Trade
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamManager:
    """
    Manages Alpaca WebSocket connection for streaming market data.

    IMPORTANT: Alpaca's run() is BLOCKING. We run it in a background thread
    to allow the rest of the application to continue.

    Example:
        manager = StreamManager(api_key, secret_key, feed='sip')
        manager.subscribe_bars(['TQQQ'], handle_bar)
        manager.start()  # Starts background thread
        # ... application continues
        manager.stop()
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = "iex",
    ):
        """
        Initialize stream manager.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            feed: Data feed - 'iex' (free) or 'sip' (paid, full coverage)
        """
        # Map feed string to Alpaca DataFeed enum
        feed_map = {
            "iex": DataFeed.IEX,
            "sip": DataFeed.SIP,
        }
        self._feed_enum = feed_map.get(feed.lower(), DataFeed.IEX)
        self._feed = feed

        self._client = StockDataStream(api_key, secret_key, feed=self._feed_enum)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Track subscribed symbols
        self._bar_symbols: Set[str] = set()
        self._quote_symbols: Set[str] = set()
        self._trade_symbols: Set[str] = set()

        logger.info(f"StreamManager initialized with {feed.upper()} feed")

    def start(self) -> None:
        """
        Start WebSocket connection in background thread.

        The connection runs until stop() is called.
        """
        if self._running:
            logger.warning("StreamManager already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_client, daemon=True)
        self._thread.start()
        logger.info("StreamManager started in background thread")

    def _run_client(self) -> None:
        """Run the blocking WebSocket client."""
        try:
            logger.info("WebSocket event loop starting")
            self._client.run()  # BLOCKING
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self._running = False
            logger.info("WebSocket event loop ended")

    def stop(self) -> None:
        """Stop WebSocket connection."""
        if not self._running:
            logger.warning("StreamManager not running")
            return

        logger.info("Stopping StreamManager")
        self._client.stop()
        self._running = False

        # Wait for thread to finish
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def subscribe_bars(self, symbols: list, handler: Callable) -> None:
        """
        Subscribe to minute bars for symbols.

        Args:
            symbols: List of ticker symbols, or ["*"] for all
            handler: Async callback - async def handler(bar: Bar) -> None
        """
        async def wrapper(alpaca_bar):
            bar = Bar.from_alpaca(alpaca_bar)
            await handler(bar)

        self._client.subscribe_bars(wrapper, *symbols)
        self._bar_symbols.update(symbols)
        logger.debug(f"Subscribed to bars: {symbols}")

    def subscribe_quotes(self, symbols: list, handler: Callable) -> None:
        """
        Subscribe to quotes for symbols.

        Args:
            symbols: List of ticker symbols, or ["*"] for all
            handler: Async callback - async def handler(quote: Quote) -> None
        """
        async def wrapper(alpaca_quote):
            quote = Quote.from_alpaca(alpaca_quote)
            await handler(quote)

        self._client.subscribe_quotes(wrapper, *symbols)
        self._quote_symbols.update(symbols)
        logger.debug(f"Subscribed to quotes: {symbols}")

    def subscribe_trades(self, symbols: list, handler: Callable) -> None:
        """
        Subscribe to trades for symbols.

        Args:
            symbols: List of ticker symbols, or ["*"] for all
            handler: Async callback - async def handler(trade: Trade) -> None
        """
        async def wrapper(alpaca_trade):
            trade = Trade.from_alpaca(alpaca_trade)
            await handler(trade)

        self._client.subscribe_trades(wrapper, *symbols)
        self._trade_symbols.update(symbols)
        logger.debug(f"Subscribed to trades: {symbols}")

    def unsubscribe_bars(self, symbols: list) -> None:
        """Unsubscribe from bars for symbols."""
        self._client.unsubscribe_bars(*symbols)
        self._bar_symbols -= set(symbols)
        logger.debug(f"Unsubscribed from bars: {symbols}")

    def unsubscribe_quotes(self, symbols: list) -> None:
        """Unsubscribe from quotes for symbols."""
        self._client.unsubscribe_quotes(*symbols)
        self._quote_symbols -= set(symbols)
        logger.debug(f"Unsubscribed from quotes: {symbols}")

    def unsubscribe_trades(self, symbols: list) -> None:
        """Unsubscribe from trades for symbols."""
        self._client.unsubscribe_trades(*symbols)
        self._trade_symbols -= set(symbols)
        logger.debug(f"Unsubscribed from trades: {symbols}")

    def is_running(self) -> bool:
        """Check if WebSocket is running."""
        return self._running

    @property
    def subscribed_bar_symbols(self) -> Set[str]:
        """Get set of symbols subscribed for bars."""
        return self._bar_symbols.copy()

    @property
    def subscribed_quote_symbols(self) -> Set[str]:
        """Get set of symbols subscribed for quotes."""
        return self._quote_symbols.copy()

    @property
    def subscribed_trade_symbols(self) -> Set[str]:
        """Get set of symbols subscribed for trades."""
        return self._trade_symbols.copy()
