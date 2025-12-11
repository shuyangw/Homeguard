"""
Thread-safe in-memory buffers for streaming market data.

These buffers store recent bars, quotes, and trades for quick access
without requiring API calls.
"""

import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.streaming.types import Bar, Quote, Trade


class BarBuffer:
    """
    Thread-safe in-memory buffer for recent bars.

    Maintains a rolling window of the most recent bars per symbol.
    Designed for concurrent access from WebSocket thread and main thread.
    """

    def __init__(self, max_bars_per_symbol: int = 500):
        """
        Initialize bar buffer.

        Args:
            max_bars_per_symbol: Maximum bars to keep per symbol (~8 hours of minute data)
        """
        self._buffers: Dict[str, deque] = {}
        self._max_bars = max_bars_per_symbol
        self._lock = threading.Lock()

    def add(self, bar: Bar) -> None:
        """
        Add a bar to the buffer.

        Thread-safe. Automatically evicts oldest bar if at capacity.

        Args:
            bar: Bar to add
        """
        from src.utils.logger import get_logger
        logger = get_logger(__name__)

        with self._lock:
            if bar.symbol not in self._buffers:
                self._buffers[bar.symbol] = deque(maxlen=self._max_bars)
                logger.info(f"[Buffer] Created buffer for {bar.symbol}")
            self._buffers[bar.symbol].append(bar)

            # Log every 10th bar to avoid spam, or first 3 bars for new symbol
            bar_count = len(self._buffers[bar.symbol])
            if bar_count <= 3 or bar_count % 10 == 0:
                logger.info(
                    f"[Buffer] {bar.symbol}: {bar_count} bars stored "
                    f"(latest: {bar.timestamp.strftime('%H:%M:%S')} @ ${bar.close:.2f})"
                )

    def get_latest(self, symbol: str) -> Optional[Bar]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Most recent Bar or None if no bars exist
        """
        with self._lock:
            if symbol not in self._buffers or len(self._buffers[symbol]) == 0:
                return None
            return self._buffers[symbol][-1]

    def get_bars(self, symbol: str, n: Optional[int] = None) -> List[Bar]:
        """
        Get recent bars for a symbol.

        Args:
            symbol: Ticker symbol
            n: Number of bars to return (None = all available)

        Returns:
            List of bars, oldest first
        """
        with self._lock:
            if symbol not in self._buffers:
                return []
            bars = list(self._buffers[symbol])
            if n is not None:
                return bars[-n:]
            return bars

    def to_dataframe(self, symbol: str) -> pd.DataFrame:
        """
        Convert buffered bars to DataFrame.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        bars = self.get_bars(symbol)
        if not bars:
            return pd.DataFrame()

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

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear buffer contents.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        with self._lock:
            if symbol is None:
                self._buffers.clear()
            elif symbol in self._buffers:
                self._buffers[symbol].clear()

    def symbols(self) -> List[str]:
        """Get list of symbols in buffer."""
        with self._lock:
            return list(self._buffers.keys())

    def count(self, symbol: str) -> int:
        """Get number of bars for a symbol."""
        with self._lock:
            if symbol not in self._buffers:
                return 0
            return len(self._buffers[symbol])


class QuoteBuffer:
    """
    Thread-safe buffer for latest quotes.

    Stores only the most recent quote per symbol for quick access.
    """

    def __init__(self):
        """Initialize quote buffer."""
        self._quotes: Dict[str, Quote] = {}
        self._lock = threading.Lock()

    def update(self, quote: Quote) -> None:
        """
        Update the latest quote for a symbol.

        Args:
            quote: New quote (replaces existing)
        """
        with self._lock:
            self._quotes[quote.symbol] = quote

    def get(self, symbol: str) -> Optional[Quote]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Quote or None if no quote exists
        """
        with self._lock:
            return self._quotes.get(symbol)

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the mid price for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Mid price or None if no quote exists
        """
        quote = self.get(symbol)
        if quote is None:
            return None
        return quote.mid

    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get the bid-ask spread for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Spread or None if no quote exists
        """
        quote = self.get(symbol)
        if quote is None:
            return None
        return quote.spread

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear buffer contents.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        with self._lock:
            if symbol is None:
                self._quotes.clear()
            elif symbol in self._quotes:
                del self._quotes[symbol]

    def symbols(self) -> List[str]:
        """Get list of symbols in buffer."""
        with self._lock:
            return list(self._quotes.keys())


class TradeBuffer:
    """
    Thread-safe buffer for latest trades.

    Stores only the most recent trade per symbol for quick price access.
    """

    def __init__(self):
        """Initialize trade buffer."""
        self._trades: Dict[str, Trade] = {}
        self._lock = threading.Lock()

    def update(self, trade: Trade) -> None:
        """
        Update the latest trade for a symbol.

        Args:
            trade: New trade (replaces existing)
        """
        with self._lock:
            self._trades[trade.symbol] = trade

    def get(self, symbol: str) -> Optional[Trade]:
        """
        Get the latest trade for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Trade or None if no trade exists
        """
        with self._lock:
            return self._trades.get(symbol)

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the last trade price for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Last trade price or None if no trade exists
        """
        trade = self.get(symbol)
        if trade is None:
            return None
        return trade.price

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear buffer contents.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        with self._lock:
            if symbol is None:
                self._trades.clear()
            elif symbol in self._trades:
                del self._trades[symbol]

    def symbols(self) -> List[str]:
        """Get list of symbols in buffer."""
        with self._lock:
            return list(self._trades.keys())
