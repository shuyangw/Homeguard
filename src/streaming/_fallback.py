"""
Polling fallback for when WebSocket streaming is unavailable.

Uses Alpaca REST API directly (not AlpacaBroker) to maintain decoupling.

Reference: https://alpaca.markets/sdks/python/api_reference/data/stock/historical.html
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest,
    StockLatestBarRequest,
    StockLatestTradeRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame

from src.streaming.types import Bar, Quote, Trade
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FallbackPoller:
    """
    Polling fallback when WebSocket streaming is unavailable.

    Uses Alpaca REST API directly to maintain decoupling from
    the trading module's AlpacaBroker.

    Example:
        poller = FallbackPoller(api_key, secret_key)
        quote = poller.get_latest_quote("TQQQ")
        price = poller.get_latest_trade("TQQQ")
    """

    def __init__(self, api_key: str, secret_key: str):
        """
        Initialize fallback poller.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
        """
        self._client = StockHistoricalDataClient(api_key, secret_key)
        logger.info("FallbackPoller initialized")

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """
        Poll for latest quote via REST API.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Quote or None if request fails
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            response = self._client.get_stock_latest_quote(request)
            alpaca_quote = response[symbol]
            return Quote.from_alpaca(alpaca_quote)
        except Exception as e:
            logger.warning(f"Failed to poll quote for {symbol}: {e}")
            return None

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Poll for latest bar via REST API.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Bar or None if request fails
        """
        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            response = self._client.get_stock_latest_bar(request)
            alpaca_bar = response[symbol]
            return Bar.from_alpaca(alpaca_bar)
        except Exception as e:
            logger.warning(f"Failed to poll bar for {symbol}: {e}")
            return None

    def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """
        Poll for latest trade via REST API.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest Trade or None if request fails
        """
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            response = self._client.get_stock_latest_trade(request)
            alpaca_trade = response[symbol]
            return Trade.from_alpaca(alpaca_trade)
        except Exception as e:
            logger.warning(f"Failed to poll trade for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest trade price for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Last trade price or None if request fails
        """
        trade = self.get_latest_trade(symbol)
        if trade is None:
            return None
        return trade.price

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
    ) -> pd.DataFrame:
        """
        Fetch historical bars via REST API.

        Args:
            symbol: Ticker symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (default: 1 minute)

        Returns:
            DataFrame with OHLCV data, or empty DataFrame if request fails
        """
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
            )
            response = self._client.get_stock_bars(request)
            return response.df
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_multiple_latest_quotes(self, symbols: list) -> dict:
        """
        Poll for latest quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to Quote (missing symbols omitted)
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            response = self._client.get_stock_latest_quote(request)
            return {
                symbol: Quote.from_alpaca(alpaca_quote)
                for symbol, alpaca_quote in response.items()
            }
        except Exception as e:
            logger.warning(f"Failed to poll quotes for {symbols}: {e}")
            return {}

    def get_multiple_latest_trades(self, symbols: list) -> dict:
        """
        Poll for latest trades for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to Trade (missing symbols omitted)
        """
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbols)
            response = self._client.get_stock_latest_trade(request)
            return {
                symbol: Trade.from_alpaca(alpaca_trade)
                for symbol, alpaca_trade in response.items()
            }
        except Exception as e:
            logger.warning(f"Failed to poll trades for {symbols}: {e}")
            return {}
