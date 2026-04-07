"""
Streaming Data Platform - Real-time market data via Alpaca WebSocket.

This module provides a unified interface for streaming market data
that replaces the current polling-based system.

Public API:
    LiveDataProvider - Main interface (the only class strategies need)
    Bar, Quote, Trade - Data types

Example:
    from src.streaming import LiveDataProvider, Bar, Quote

    # Initialize with data feed
    provider = LiveDataProvider(feed='sip')  # 'iex' for free, 'sip' for paid

    # Start streaming
    provider.start(['TQQQ', 'SPY', 'SOXL'])

    # Get data instantly (from memory buffer, no API call)
    price = provider.get_price('TQQQ')        # Latest price
    quote = provider.get_quote('TQQQ')        # Latest bid/ask
    bars = provider.get_bars('TQQQ', n=15)    # Recent bars DataFrame

    # Real-time callbacks for intraday strategies
    async def handle_bar(bar: Bar):
        print(f"New bar: {bar.symbol} @ {bar.close}")

    provider.on_bar(['TQQQ'], handle_bar)

    # Cleanup
    provider.stop()

Replaces:
    - broker.get_latest_quote() -> provider.get_quote()
    - broker.get_latest_trade() -> provider.get_price()
    - broker.get_historical_bars() -> provider.get_bars()

Architecture:
    - Fully decoupled from src/trading/ and src/strategies/
    - Single WebSocket connection shared by all strategies
    - In-memory buffer for instant data access
    - Automatic fallback to polling if WebSocket fails

Reference:
    - Design doc: docs/architecture/20251209_STREAMING_DATA_PLATFORM.md
    - Alpaca SDK: https://alpaca.markets/sdks/python/
"""

from src.streaming.interface import StreamingProviderInterface
from src.streaming.live_data_provider import LiveDataProvider
from src.streaming.types import Bar, Quote, Trade
from src.streaming.binance_stream import BinanceStreamManager
from src.streaming.crypto_buffer import CryptoBarBuffer

__all__ = [
    "StreamingProviderInterface",
    "LiveDataProvider",
    "Bar",
    "Quote",
    "Trade",
    "BinanceStreamManager",
    "CryptoBarBuffer",
]
