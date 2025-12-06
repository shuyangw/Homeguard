"""
Data Provider Interface - Abstract base for all market data providers.

All implementations must:
1. Return data in standardized format (lowercase columns, ET timezone)
2. Handle errors gracefully (return None, don't raise)
3. Support both single-symbol and batch operations
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class DataProviderInterface(ABC):
    """
    Abstract interface for market data providers.

    Contract:
    - Index: DatetimeIndex with America/New_York timezone
    - Columns: open, high, low, close, volume (lowercase)
    - Returns: pd.DataFrame or None on failure (enables fallback)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging (e.g., 'Alpaca', 'yfinance')."""
        pass

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV bars for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TQQQ')
            start: Start datetime (timezone-aware recommended)
            end: End datetime (timezone-aware recommended)
            timeframe: '1D', '1Min', '5Min', '1Hour'
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            DataFrame with OHLCV data, or None on failure
        """
        pass

    @abstractmethod
    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple symbols.

        Args:
            symbols: List of symbols
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            Dict mapping symbol -> DataFrame (empty dict on total failure)
        """
        pass

    def is_available(self) -> bool:
        """Check if provider is currently available."""
        return True

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if provider supports the given timeframe."""
        return True


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class SymbolNotFoundError(DataProviderError):
    """Raised when a symbol cannot be found."""
    pass


class DataUnavailableError(DataProviderError):
    """Raised when data is temporarily unavailable."""
    pass
