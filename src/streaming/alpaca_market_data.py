"""
AlpacaMarketData -- unified Alpaca data provider (historical + streaming).

Why this exists
---------------
The production architecture separates execution from data:
    - Execution / portfolio: IBKR  (orders, positions, account)
    - Data acquisition:      Alpaca (historical bars + streaming quotes/bars)

Strategy adapters (RAMP, OMR, MP) take a `broker` and a `data_provider`. With
the broker routed to IBKR and the data_provider previously being `LiveDataProvider`
(streaming-only), historical fetches fell back to `self.broker` and unintentionally
pulled from IBKR. This class closes that gap by exposing a single data_provider
object that satisfies both `StreamingProviderInterface` (real-time buffer
reads) and `DataProviderInterface`-style historical fetches.

Internals
---------
- Historical: `AlpacaDataProvider` wrapping a dedicated `AlpacaBroker` instance
  used only for REST data calls (not execution).
- Streaming:  `LiveDataProvider` -- the existing Alpaca WebSocket client.

Scope
-----
Only Alpaca. Other data sources (yfinance, IBKR) are separate providers and
should be composed via `CompositeDataProvider` if fallback is needed.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.data.providers.alpaca import AlpacaDataProvider
from src.streaming.interface import StreamingProviderInterface
from src.streaming.live_data_provider import LiveDataProvider
from src.streaming.types import Bar, Quote, Trade
from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaMarketData(StreamingProviderInterface):
    """
    Single object exposing both streaming and historical market data, all from Alpaca.

    Satisfies StreamingProviderInterface (so RAMP's `isinstance(dp, StreamingProviderInterface)`
    check continues to work) and adds `get_historical_bars` / `get_historical_bars_batch`
    so historical fetches no longer fall back to the execution broker (IBKR).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: str = "iex",
        max_bars_per_symbol: int = 500,
        fallback_enabled: bool = True,
        paper: bool = True,
    ):
        """
        Initialize Alpaca market data.

        Credentials default to ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY
        from the environment if not explicitly provided.
        """
        if api_key is None:
            api_key = os.environ.get("ALPACA_PAPER_KEY_ID")
        if secret_key is None:
            secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca credentials not set. Provide api_key / secret_key or "
                "set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY."
            )

        # Historical surface: a dedicated AlpacaBroker used only for REST data calls.
        # Separate from any execution broker the strategy may also hold.
        data_broker = AlpacaBroker(
            api_key=api_key, secret_key=secret_key, paper=paper
        )
        self._historical = AlpacaDataProvider(data_broker)

        # Streaming surface: Alpaca WebSocket.
        self._streaming = LiveDataProvider(
            api_key=api_key,
            secret_key=secret_key,
            feed=feed,
            max_bars_per_symbol=max_bars_per_symbol,
            fallback_enabled=fallback_enabled,
        )

        self._feed = feed
        self._name = f"alpaca-market-{feed}"
        logger.info(f"AlpacaMarketData initialized (feed={feed.upper()})")

    # === Historical (DataProviderInterface-shaped) ===

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        return self._historical.get_historical_bars(
            symbol, start, end, timeframe, force_refresh
        )

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        return self._historical.get_historical_bars_batch(
            symbols, start, end, timeframe, force_refresh
        )

    # === StreamingProviderInterface (all delegated) ===

    @property
    def name(self) -> str:
        return self._name

    def start(self, symbols: Optional[List[str]] = None) -> None:
        self._streaming.start(symbols)

    def stop(self) -> None:
        self._streaming.stop()

    def is_connected(self) -> bool:
        return self._streaming.is_connected()

    def get_price(self, symbol: str) -> Optional[float]:
        return self._streaming.get_price(symbol)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._streaming.get_quote(symbol)

    def get_trade(self, symbol: str) -> Optional[Trade]:
        return self._streaming.get_trade(symbol)

    def get_bar(self, symbol: str) -> Optional[Bar]:
        return self._streaming.get_bar(symbol)

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        return self._streaming.get_bars(symbol, n)

    def get_vwap(self, symbol: str) -> Optional[float]:
        return self._streaming.get_vwap(symbol)

    def get_spread(self, symbol: str) -> Optional[float]:
        return self._streaming.get_spread(symbol)

    def on_bar(self, symbols: List[str], handler: Callable[[Bar], None]) -> str:
        return self._streaming.on_bar(symbols, handler)

    def on_quote(self, symbols: List[str], handler: Callable[[Quote], None]) -> str:
        return self._streaming.on_quote(symbols, handler)

    def on_trade(self, symbols: List[str], handler: Callable[[Trade], None]) -> str:
        return self._streaming.on_trade(symbols, handler)

    def unsubscribe(self, subscription_id: str) -> None:
        self._streaming.unsubscribe(subscription_id)

    def get_subscribed_symbols(self) -> set:
        return self._streaming.get_subscribed_symbols()
