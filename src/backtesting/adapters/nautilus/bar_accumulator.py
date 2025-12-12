"""
Bar Accumulator for NautilusTrader Adapters.

Accumulates incoming bars from NautilusTrader and converts them to
pandas DataFrames with canonical schema for Homeguard strategies.

This module does not depend on NautilusTrader being installed.
"""

from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

import pandas as pd


class BarProtocol(Protocol):
    """Protocol for bar-like objects (works with both real and mock bars)."""
    @property
    def open(self) -> Any: ...
    @property
    def high(self) -> Any: ...
    @property
    def low(self) -> Any: ...
    @property
    def close(self) -> Any: ...
    @property
    def volume(self) -> Any: ...
    @property
    def ts_event(self) -> int: ...


class BarAccumulator:
    """
    Accumulates bars and converts to pandas DataFrames.

    Maintains a rolling window of bars per symbol, converting them to
    the canonical DataFrame format expected by Homeguard strategies.

    Attributes:
        max_periods: Maximum number of bars to keep per symbol
    """

    def __init__(self, max_periods: int) -> None:
        """
        Initialize bar accumulator.

        Args:
            max_periods: Maximum number of periods to accumulate per symbol.
                         Older bars are dropped when this limit is exceeded.

        Raises:
            ValueError: If max_periods is less than 1
        """
        if max_periods < 1:
            raise ValueError("max_periods must be at least 1")

        self.max_periods = max_periods
        self._bars: Dict[str, deque] = {}

    def add_bar(self, symbol: str, bar: BarProtocol) -> None:
        """
        Add a bar to the accumulator.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            bar: Bar object with open, high, low, close, volume, ts_event

        Note:
            If the number of bars exceeds max_periods, the oldest bar is removed.
        """
        if symbol not in self._bars:
            self._bars[symbol] = deque(maxlen=self.max_periods)

        # Extract bar data
        bar_data = {
            'timestamp': pd.Timestamp(bar.ts_event, unit='ns', tz='UTC'),
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume),
        }
        self._bars[symbol].append(bar_data)

    def get_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get accumulated bars as a pandas DataFrame.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with canonical schema (lowercase columns, timestamp index)
            or None if no bars for symbol.

        Schema:
            Index: timestamp (datetime64[ns, UTC])
            Columns: open, high, low, close, volume (all float64)
        """
        if symbol not in self._bars or len(self._bars[symbol]) == 0:
            return None

        df = pd.DataFrame(list(self._bars[symbol]))
        df.set_index('timestamp', inplace=True)
        return df

    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get accumulated bars for all symbols.

        Returns:
            Dict mapping symbol -> DataFrame with canonical schema.
            Only includes symbols that have at least one bar.
        """
        result = {}
        for symbol in self._bars:
            df = self.get_dataframe(symbol)
            if df is not None:
                result[symbol] = df
        return result

    def has_sufficient_data(self, symbol: str, required_periods: int) -> bool:
        """
        Check if symbol has sufficient data for strategy calculation.

        Args:
            symbol: Stock symbol
            required_periods: Minimum number of periods needed

        Returns:
            True if symbol has at least required_periods of data
        """
        if symbol not in self._bars:
            return False
        return len(self._bars[symbol]) >= required_periods

    def get_bar_count(self, symbol: str) -> int:
        """
        Get number of accumulated bars for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Number of bars accumulated (0 if symbol not tracked)
        """
        if symbol not in self._bars:
            return 0
        return len(self._bars[symbol])

    def get_symbols(self) -> list:
        """
        Get list of symbols being tracked.

        Returns:
            List of symbol strings
        """
        return list(self._bars.keys())

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear accumulated bars.

        Args:
            symbol: If provided, clear only this symbol. Otherwise clear all.
        """
        if symbol is not None:
            if symbol in self._bars:
                self._bars[symbol].clear()
        else:
            self._bars.clear()

    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with bar data or None if no bars
        """
        if symbol not in self._bars or len(self._bars[symbol]) == 0:
            return None
        return self._bars[symbol][-1].copy()

    def get_latest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Get timestamp of the most recent bar for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Timestamp or None if no bars
        """
        bar = self.get_latest_bar(symbol)
        if bar is None:
            return None
        return bar['timestamp']
