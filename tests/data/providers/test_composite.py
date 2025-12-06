"""Tests for CompositeDataProvider."""

import pytest
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import Mock, patch
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.data.providers.composite import CompositeDataProvider


class MockProvider(DataProviderInterface):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str,
        data: Optional[pd.DataFrame] = None,
        available: bool = True,
        fail_symbols: Optional[List[str]] = None
    ):
        self._name = name
        self._data = data
        self._available = available
        self._fail_symbols = fail_symbols or []

    @property
    def name(self) -> str:
        return self._name

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Optional[pd.DataFrame]:
        if symbol in self._fail_symbols:
            return None
        return self._data

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for s in symbols:
            if s not in self._fail_symbols and self._data is not None:
                results[s] = self._data
        return results

    def is_available(self) -> bool:
        return self._available


class TestCompositeProvider:
    """Test CompositeDataProvider functionality."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'volume': [1000.0]
        }, index=pd.DatetimeIndex([datetime.now()]))

    def test_name(self, sample_data):
        """Test composite name."""
        provider = CompositeDataProvider(
            [MockProvider("Primary", sample_data)],
            cache_enabled=False
        )
        assert provider.name == "Composite"

    def test_primary_success(self, sample_data):
        """Test primary provider success."""
        primary = MockProvider("Primary", sample_data)
        fallback = MockProvider("Fallback", sample_data)

        provider = CompositeDataProvider([primary, fallback], cache_enabled=False)
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())

        assert result is not None
        assert provider.last_source == "Primary"

    def test_fallback_on_primary_failure(self, sample_data):
        """Test fallback when primary fails."""
        primary = MockProvider("Primary", data=None)
        fallback = MockProvider("Fallback", sample_data)

        provider = CompositeDataProvider([primary, fallback], cache_enabled=False)
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())

        assert result is not None
        assert provider.last_source == "Fallback"

    def test_returns_none_when_all_fail(self):
        """Test None returned when all providers fail."""
        primary = MockProvider("Primary", data=None)
        fallback = MockProvider("Fallback", data=None)

        provider = CompositeDataProvider([primary, fallback], cache_enabled=False)
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())

        assert result is None

    def test_skips_unavailable_provider(self, sample_data):
        """Test unavailable provider is skipped."""
        unavailable = MockProvider("Unavailable", sample_data, available=False)
        available = MockProvider("Available", sample_data)

        provider = CompositeDataProvider([unavailable, available], cache_enabled=False)
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())

        assert result is not None
        assert provider.last_source == "Available"

    def test_batch_per_symbol_fallback(self, sample_data):
        """Test batch uses per-symbol fallback."""
        # Primary fails for TECL, succeeds for TQQQ
        primary = MockProvider("Primary", sample_data, fail_symbols=["TECL"])
        fallback = MockProvider("Fallback", sample_data)

        provider = CompositeDataProvider([primary, fallback], cache_enabled=False)
        results = provider.get_historical_bars_batch(
            ["TQQQ", "TECL"], datetime.now(), datetime.now()
        )

        assert "TQQQ" in results
        assert "TECL" in results
        assert len(results) == 2

    def test_batch_reports_failed_symbols(self, sample_data):
        """Test batch reports symbols that couldn't be fetched."""
        primary = MockProvider("Primary", data=None)
        fallback = MockProvider("Fallback", data=None)

        provider = CompositeDataProvider([primary, fallback], cache_enabled=False)
        results = provider.get_historical_bars_batch(
            ["INVALID"], datetime.now(), datetime.now()
        )

        assert len(results) == 0

    def test_source_info(self, sample_data):
        """Test source info tracking."""
        provider = CompositeDataProvider(
            [MockProvider("Primary", sample_data)],
            cache_enabled=False
        )

        provider.get_historical_bars("AAPL", datetime.now(), datetime.now())
        source, fetch_time = provider.get_source_info()

        assert source == "Primary"
        assert fetch_time is not None


class TestCompositeWithCache:
    """Test CompositeDataProvider with caching."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'volume': [1000.0]
        }, index=pd.DatetimeIndex([datetime.now()]))

    @patch('src.data.providers.composite.DataCache')
    def test_caches_successful_result(self, mock_cache_class, sample_data):
        """Test successful results are cached."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        provider = CompositeDataProvider(
            [MockProvider("Primary", sample_data)],
            cache_enabled=True
        )
        provider.get_historical_bars("AAPL", datetime.now(), datetime.now(), "1D")

        mock_cache.store.assert_called_once()

    @patch('src.data.providers.composite.DataCache')
    def test_uses_cache_as_fallback(self, mock_cache_class, sample_data):
        """Test cache is used when all providers fail."""
        mock_cache = Mock()
        mock_cache.retrieve.return_value = sample_data
        mock_cache_class.return_value = mock_cache

        provider = CompositeDataProvider(
            [MockProvider("Primary", data=None)],
            cache_enabled=True
        )
        result = provider.get_historical_bars("AAPL", datetime.now(), datetime.now())

        assert result is not None
        assert provider.last_source == "cache"
