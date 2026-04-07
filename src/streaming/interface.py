"""
Abstract base class for streaming market data providers.

All streaming providers (Alpaca LiveDataProvider, future IBKR provider, etc.)
must implement this interface. Strategy adapters use isinstance checks against
StreamingProviderInterface rather than duck-typing with hasattr.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import pandas as pd

from src.streaming.types import Bar, Quote, Trade


class StreamingProviderInterface(ABC):
    """
    Abstract interface for real-time streaming market data providers.

    Concrete implementations:
        - LiveDataProvider (Alpaca WebSocket)
        - IBKRStreamingProvider (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier (e.g., 'streaming-iex', 'ibkr-live')."""
        ...

    # --- Lifecycle ---

    @abstractmethod
    def start(self, symbols: Optional[List[str]] = None) -> None:
        """Start streaming connection, optionally subscribing to symbols."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and cleanup resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the streaming connection is active."""
        ...

    # --- On-Demand Data ---

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest bid/ask quote for a symbol."""
        ...

    @abstractmethod
    def get_trade(self, symbol: str) -> Optional[Trade]:
        """Get latest trade for a symbol."""
        ...

    @abstractmethod
    def get_bar(self, symbol: str) -> Optional[Bar]:
        """Get latest bar for a symbol."""
        ...

    @abstractmethod
    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """Get recent bars from buffer as a DataFrame."""
        ...

    @abstractmethod
    def get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP for a symbol."""
        ...

    @abstractmethod
    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current bid-ask spread for a symbol."""
        ...

    # --- Real-Time Callbacks ---

    @abstractmethod
    def on_bar(self, symbols: List[str], handler: Callable[[Bar], None]) -> str:
        """Register callback for new minute bars. Returns subscription ID."""
        ...

    @abstractmethod
    def on_quote(self, symbols: List[str], handler: Callable[[Quote], None]) -> str:
        """Register callback for new quotes. Returns subscription ID."""
        ...

    @abstractmethod
    def on_trade(self, symbols: List[str], handler: Callable[[Trade], None]) -> str:
        """Register callback for new trades. Returns subscription ID."""
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a callback subscription by ID."""
        ...

    # --- Utility ---

    @abstractmethod
    def get_subscribed_symbols(self) -> set:
        """Get set of currently subscribed symbols."""
        ...
