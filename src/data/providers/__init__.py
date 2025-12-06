"""
Market Data Provider Package.

Provides a unified interface for fetching market data from multiple sources
with automatic fallback and caching.

Usage:
    from src.data.providers import create_data_provider, DataProviderInterface

    # Create composite provider with Alpaca primary, yfinance fallback
    provider = create_data_provider(broker=my_broker)

    # Fetch data
    df = provider.get_historical_bars('TQQQ', start, end, '1Min')
    data = provider.get_historical_bars_batch(symbols, start, end, '1D')
"""

from src.data.providers.base import (
    DataProviderInterface,
    DataProviderError,
    SymbolNotFoundError,
    DataUnavailableError,
)
from src.data.providers.alpaca import AlpacaDataProvider
from src.data.providers.yfinance import YFinanceDataProvider
from src.data.providers.composite import CompositeDataProvider
from src.data.providers.cache import DataCache
from src.data.providers.factory import create_data_provider


__all__ = [
    # Interface and errors
    'DataProviderInterface',
    'DataProviderError',
    'SymbolNotFoundError',
    'DataUnavailableError',
    # Providers
    'AlpacaDataProvider',
    'YFinanceDataProvider',
    'CompositeDataProvider',
    # Cache
    'DataCache',
    # Factory
    'create_data_provider',
]
