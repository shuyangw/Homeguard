"""
Market Data Interface - Market data retrieval operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

import pandas as pd


class MarketDataInterface(ABC):
    """
    Abstract interface for market data operations.

    Provides access to quotes, trades, and historical bars.
    Works for any instrument type (stocks, options, futures, etc.).
    """

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Dict:
        """
        Get latest bid/ask quote.

        Args:
            symbol: Symbol identifier

        Returns:
            Dict with quote details:
                - symbol (str): Symbol
                - bid (float): Bid price
                - ask (float): Ask price
                - bid_size (int): Bid size
                - ask_size (int): Ask size
                - timestamp (datetime): Quote timestamp

        Raises:
            BrokerConnectionError: If broker connection fails
            SymbolNotFoundError: If symbol doesn't exist
        """
        pass

    @abstractmethod
    def get_latest_trade(self, symbol: str) -> Dict:
        """
        Get latest trade.

        Args:
            symbol: Symbol identifier

        Returns:
            Dict with trade details:
                - symbol (str): Symbol
                - price (float): Trade price
                - size (int): Trade size
                - timestamp (datetime): Trade timestamp

        Raises:
            BrokerConnectionError: If broker connection fails
            SymbolNotFoundError: If symbol doesn't exist
        """
        pass

    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars.

        Args:
            symbols: List of symbols
            timeframe: Timeframe ('1Min', '5Min', '1Hour', '1Day')
            start: Start datetime
            end: End datetime

        Returns:
            pd.DataFrame with MultiIndex (symbol, timestamp) and columns:
                - open (float)
                - high (float)
                - low (float)
                - close (float)
                - volume (int)

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass
