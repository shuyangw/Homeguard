"""
Public interface for real-time streaming market data.

This is the ONLY class that trading strategies should import from src/streaming.
All implementation details are hidden behind this interface.

Example:
    from src.streaming import LiveDataProvider

    provider = LiveDataProvider(feed='sip')
    provider.start(['TQQQ', 'SPY', 'SOXL'])

    # Get data instantly (from memory buffer)
    price = provider.get_price('TQQQ')
    quote = provider.get_quote('TQQQ')
    bars = provider.get_bars('TQQQ', n=15)

    # Register for real-time callbacks
    async def handle_bar(bar):
        print(f"New bar: {bar.symbol} @ {bar.close}")

    provider.on_bar(['TQQQ'], handle_bar)

    # Cleanup
    provider.stop()
"""

import os
from typing import Callable, List, Optional

import pandas as pd

from src.streaming.types import Bar, Quote, Trade
from src.streaming.interface import StreamingProviderInterface
from src.streaming._hub import MarketDataHub
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_alpaca_credentials() -> tuple:
    """
    Get Alpaca API credentials from environment variables.

    Checks in order:
    1. ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY (paper trading)
    2. API_KEY / API_SECRET (legacy)

    Returns:
        Tuple of (api_key, secret_key)

    Raises:
        ValueError: If no credentials are found
    """
    api_key = os.environ.get("ALPACA_PAPER_KEY_ID") or os.environ.get("API_KEY")
    secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY") or os.environ.get("API_SECRET")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found. Set ALPACA_PAPER_KEY_ID and "
            "ALPACA_PAPER_SECRET_KEY environment variables, or pass api_key/secret_key "
            "to LiveDataProvider constructor."
        )

    return api_key, secret_key


class LiveDataProvider(StreamingProviderInterface):
    """
    Public interface for all live market data.

    Replaces broker.get_latest_quote(), broker.get_latest_trade(),
    and broker.get_historical_bars() with a unified streaming interface.

    Features:
    - Single WebSocket connection shared by all strategies
    - In-memory buffer for instant data access
    - Real-time callbacks for intraday strategies
    - Automatic fallback to polling if WebSocket fails
    - Thread-safe data access
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: str = "iex",
        max_bars_per_symbol: int = 500,
        fallback_enabled: bool = True,
    ):
        """
        Initialize live data provider.

        Args:
            api_key: Alpaca API key (loads from env vars if not provided)
            secret_key: Alpaca secret key (loads from env vars if not provided)
            feed: Data feed - 'iex' (free) or 'sip' (paid, full coverage)
            max_bars_per_symbol: Max bars to keep in buffer per symbol (~8hrs of 1-min bars)
            fallback_enabled: Enable polling fallback when streaming fails
        """
        # Load credentials from environment if not provided
        if api_key is None or secret_key is None:
            env_key, env_secret = _get_alpaca_credentials()
            api_key = api_key or env_key
            secret_key = secret_key or env_secret

        self._hub = MarketDataHub(
            api_key=api_key,
            secret_key=secret_key,
            feed=feed,
            max_bars_per_symbol=max_bars_per_symbol,
            fallback_enabled=fallback_enabled,
        )
        self._feed = feed
        self._name = f"streaming-{feed}"

        logger.info(f"LiveDataProvider initialized with {feed.upper()} feed")

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return self._name

    # === Lifecycle ===

    def start(self, symbols: Optional[List[str]] = None) -> None:
        """
        Start streaming connection.

        Args:
            symbols: Initial symbols to subscribe to (optional)
        """
        self._hub.start(symbols)
        logger.info("LiveDataProvider started")

    def stop(self) -> None:
        """Stop streaming and cleanup."""
        self._hub.stop()
        logger.info("LiveDataProvider stopped")

    def is_connected(self) -> bool:
        """Check if WebSocket streaming is active."""
        return self._hub.is_connected()

    # === On-Demand Data (Replaces Polling) ===

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.

        Uses last trade price, or quote mid if no trade available.
        Replaces: broker.get_latest_trade()

        Args:
            symbol: Ticker symbol

        Returns:
            Latest price or None if no data
        """
        return self._hub.get_price(symbol)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get latest bid/ask quote for a symbol.

        Replaces: broker.get_latest_quote()

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Quote or None if no data
        """
        return self._hub.get_latest_quote(symbol)

    def get_trade(self, symbol: str) -> Optional[Trade]:
        """
        Get latest trade for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Trade or None if no data
        """
        return self._hub.get_latest_trade(symbol)

    def get_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get latest bar for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Bar or None if no data
        """
        return self._hub.get_latest_bar(symbol)

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """
        Get recent bars from buffer.

        Replaces: broker.get_historical_bars() for recent data

        Args:
            symbol: Ticker symbol
            n: Number of bars to return (None = all available, up to max_bars_per_symbol)

        Returns:
            DataFrame with OHLCV columns, indexed by timestamp
        """
        return self._hub.get_bars(symbol, n)

    def get_vwap(self, symbol: str) -> Optional[float]:
        """
        Get current VWAP for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Current VWAP or None if no data
        """
        return self._hub.get_vwap(symbol)

    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get current bid-ask spread for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Spread (ask - bid) or None if no data
        """
        return self._hub.get_spread(symbol)

    # === Real-Time Callbacks ===

    def on_bar(self, symbols: List[str], handler: Callable[[Bar], None]) -> str:
        """
        Register callback for new minute bars.

        The handler is called each time a new bar is received.
        Use this for intraday strategies like ORB.

        Args:
            symbols: Symbols to subscribe to
            handler: Async callback - async def handler(bar: Bar) -> None

        Returns:
            Subscription ID (use with unsubscribe())

        Example:
            async def handle_bar(bar):
                if bar.timestamp.time() >= time(9, 45):
                    check_breakout(bar)

            sub_id = provider.on_bar(['TQQQ'], handle_bar)
        """
        return self._hub.on_bar(symbols, handler)

    def on_quote(self, symbols: List[str], handler: Callable[[Quote], None]) -> str:
        """
        Register callback for new quotes.

        Args:
            symbols: Symbols to subscribe to
            handler: Async callback - async def handler(quote: Quote) -> None

        Returns:
            Subscription ID
        """
        return self._hub.on_quote(symbols, handler)

    def on_trade(self, symbols: List[str], handler: Callable[[Trade], None]) -> str:
        """
        Register callback for new trades.

        Args:
            symbols: Symbols to subscribe to
            handler: Async callback - async def handler(trade: Trade) -> None

        Returns:
            Subscription ID
        """
        return self._hub.on_trade(symbols, handler)

    def unsubscribe(self, subscription_id: str) -> None:
        """
        Remove a callback subscription.

        Args:
            subscription_id: ID returned from on_bar/on_quote/on_trade
        """
        self._hub.unsubscribe(subscription_id)

    # === Utility ===

    def get_subscribed_symbols(self) -> set:
        """Get set of currently subscribed symbols."""
        return self._hub.get_subscribed_symbols()

    def clear_buffers(self) -> None:
        """Clear all in-memory data buffers."""
        self._hub.clear_buffers()

    @property
    def feed(self) -> str:
        """Get current data feed ('iex' or 'sip')."""
        return self._feed
