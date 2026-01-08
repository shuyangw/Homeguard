"""
Thread-safe 24-hour sliding window buffer for crypto klines.

Stores up to 1440 bars (24 hours of 1-minute data) per symbol,
optimized for crypto trading strategies that need recent price history.
"""

import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.streaming.types import Bar
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoBarBuffer:
    """
    24-hour sliding window buffer for crypto bars.

    Maintains up to 1440 1-minute bars per symbol (24 hours of data).
    Thread-safe for concurrent access from WebSocket and main threads.

    Memory estimate: ~2.9 MB for 20 symbols (100 bytes/bar x 1440 x 20)

    Example:
        buffer = CryptoBarBuffer()
        buffer.add_bar(bar)

        # Get latest price
        price = buffer.get_latest_price('BTC/USD')

        # Get OHLCV DataFrame for analysis
        df = buffer.get_ohlcv_dataframe('BTC/USD')
    """

    BARS_PER_DAY = 1440  # 24 * 60 minutes

    def __init__(self, max_bars: int = BARS_PER_DAY):
        """
        Initialize crypto buffer.

        Args:
            max_bars: Maximum bars per symbol (default: 1440 for 24h)
        """
        self._buffers: Dict[str, deque] = {}
        self._max_bars = max_bars
        self._lock = threading.RLock()  # Reentrant for nested calls

        # Track statistics per symbol
        self._stats: Dict[str, Dict] = {}

    def add_bar(self, bar: Bar) -> None:
        """
        Add a bar to the buffer.

        Thread-safe. Automatically evicts oldest bar when at capacity.

        Args:
            bar: Bar to add
        """
        with self._lock:
            symbol = bar.symbol

            if symbol not in self._buffers:
                self._buffers[symbol] = deque(maxlen=self._max_bars)
                self._stats[symbol] = {
                    "bars_added": 0,
                    "first_bar_time": bar.timestamp,
                    "last_bar_time": None,
                }
                logger.info(f"[CryptoBuffer] Created buffer for {symbol}")

            self._buffers[symbol].append(bar)
            self._stats[symbol]["bars_added"] += 1
            self._stats[symbol]["last_bar_time"] = bar.timestamp

            # Log periodically (every 60 bars = 1 hour)
            bar_count = len(self._buffers[symbol])
            if bar_count == 1 or bar_count % 60 == 0:
                logger.debug(
                    f"[CryptoBuffer] {symbol}: {bar_count}/{self._max_bars} bars "
                    f"(latest: {bar.timestamp.strftime('%H:%M')} @ ${bar.close:.2f})"
                )

    def get_latest(self, symbol: str) -> Optional[Bar]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')

        Returns:
            Most recent Bar or None if no bars exist
        """
        with self._lock:
            if symbol not in self._buffers or len(self._buffers[symbol]) == 0:
                return None
            return self._buffers[symbol][-1]

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the most recent close price for a symbol.

        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')

        Returns:
            Latest close price or None if no bars exist
        """
        bar = self.get_latest(symbol)
        return bar.close if bar else None

    def get_bars(self, symbol: str, n: Optional[int] = None) -> List[Bar]:
        """
        Get recent bars for a symbol.

        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')
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

    def get_ohlcv_dataframe(self, symbol: str) -> pd.DataFrame:
        """
        Convert buffered bars to OHLCV DataFrame.

        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Indexed by timestamp
        """
        bars = self.get_bars(symbol)
        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        data = {
            "timestamp": [b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "trade_count": [b.trade_count for b in bars],
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def get_all_prices(self) -> Dict[str, float]:
        """
        Get latest prices for all symbols in buffer.

        Returns:
            Dict mapping symbol to latest close price
        """
        with self._lock:
            prices = {}
            for symbol in self._buffers:
                price = self.get_latest_price(symbol)
                if price is not None:
                    prices[symbol] = price
            return prices

    def has_symbol(self, symbol: str) -> bool:
        """Check if buffer has data for symbol."""
        with self._lock:
            return symbol in self._buffers and len(self._buffers[symbol]) > 0

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

    def total_bars(self) -> int:
        """Get total number of bars across all symbols."""
        with self._lock:
            return sum(len(buf) for buf in self._buffers.values())

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear buffer contents.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        with self._lock:
            if symbol is None:
                self._buffers.clear()
                self._stats.clear()
                logger.info("[CryptoBuffer] Cleared all buffers")
            elif symbol in self._buffers:
                self._buffers[symbol].clear()
                if symbol in self._stats:
                    del self._stats[symbol]
                logger.info(f"[CryptoBuffer] Cleared buffer for {symbol}")

    def get_stats(self) -> Dict:
        """
        Get buffer statistics.

        Returns:
            Dict with overall and per-symbol statistics
        """
        with self._lock:
            total_bars = self.total_bars()
            memory_bytes = total_bars * 100  # Rough estimate: 100 bytes/bar

            return {
                "symbols": len(self._buffers),
                "total_bars": total_bars,
                "max_bars_per_symbol": self._max_bars,
                "memory_mb": memory_bytes / (1024 * 1024),
                "per_symbol": {
                    symbol: {
                        "bars": len(self._buffers[symbol]),
                        "fill_pct": len(self._buffers[symbol])
                        / self._max_bars
                        * 100,
                        **self._stats.get(symbol, {}),
                    }
                    for symbol in self._buffers
                },
            }

    def is_full(self, symbol: str) -> bool:
        """
        Check if buffer for symbol is at capacity.

        Args:
            symbol: Symbol to check

        Returns:
            True if buffer has max_bars bars
        """
        with self._lock:
            if symbol not in self._buffers:
                return False
            return len(self._buffers[symbol]) >= self._max_bars

    def time_range(self, symbol: str) -> Optional[tuple]:
        """
        Get time range of bars in buffer.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (oldest_timestamp, newest_timestamp) or None
        """
        with self._lock:
            if symbol not in self._buffers or len(self._buffers[symbol]) == 0:
                return None
            bars = self._buffers[symbol]
            return (bars[0].timestamp, bars[-1].timestamp)
