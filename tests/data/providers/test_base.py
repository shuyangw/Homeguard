"""Tests for DataProviderInterface contract."""

import pytest
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.data.providers.base import (
    DataProviderInterface,
    DataProviderError,
    SymbolNotFoundError,
    DataUnavailableError,
)


class MockProvider(DataProviderInterface):
    """Mock provider for testing interface contract."""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        self._data = data
        self._available = True
        self._supported_timeframes = ['1D', '1Min']

    @property
    def name(self) -> str:
        return "Mock"

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Optional[pd.DataFrame]:
        return self._data

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        if self._data is None:
            return {}
        return {s: self._data for s in symbols}

    def is_available(self) -> bool:
        return self._available

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in self._supported_timeframes


class TestDataProviderInterface:
    """Test DataProviderInterface contract."""

    def test_name_property(self):
        """Test provider name is accessible."""
        provider = MockProvider()
        assert provider.name == "Mock"

    def test_get_historical_bars_returns_none_on_no_data(self):
        """Test that None data returns None."""
        provider = MockProvider(data=None)
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())
        assert result is None

    def test_get_historical_bars_returns_dataframe(self):
        """Test that data is returned as DataFrame."""
        df = pd.DataFrame({'close': [100, 101]})
        provider = MockProvider(data=df)
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_historical_bars_batch_returns_dict(self):
        """Test batch returns dict."""
        df = pd.DataFrame({'close': [100, 101]})
        provider = MockProvider(data=df)
        result = provider.get_historical_bars_batch(
            ["AAPL", "MSFT"], datetime.now(), datetime.now()
        )
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_get_historical_bars_batch_empty_on_no_data(self):
        """Test batch returns empty dict on failure."""
        provider = MockProvider(data=None)
        result = provider.get_historical_bars_batch(
            ["AAPL"], datetime.now(), datetime.now()
        )
        assert result == {}

    def test_is_available_default_true(self):
        """Test default availability is True."""
        provider = MockProvider()
        assert provider.is_available() is True

    def test_supports_timeframe_default_true(self):
        """Test default timeframe support."""
        provider = MockProvider()
        assert provider.supports_timeframe('1D') is True
        assert provider.supports_timeframe('1Min') is True


class TestExceptions:
    """Test custom exceptions."""

    def test_data_provider_error(self):
        """Test base error."""
        with pytest.raises(DataProviderError):
            raise DataProviderError("Test error")

    def test_symbol_not_found_error(self):
        """Test symbol not found is subclass."""
        with pytest.raises(DataProviderError):
            raise SymbolNotFoundError("INVALID")

    def test_data_unavailable_error(self):
        """Test data unavailable is subclass."""
        with pytest.raises(DataProviderError):
            raise DataUnavailableError("Server down")
