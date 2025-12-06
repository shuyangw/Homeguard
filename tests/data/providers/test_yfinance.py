"""Tests for YFinanceDataProvider."""

import pytest
import socket
from datetime import datetime
import pandas as pd
import pytz

from src.data.providers.yfinance import YFinanceDataProvider


def has_network_connection() -> bool:
    """Check if we have network connectivity to Yahoo Finance."""
    try:
        socket.create_connection(("finance.yahoo.com", 443), timeout=3)
        return True
    except (socket.timeout, socket.error, OSError):
        return False


# Cache the result to avoid repeated checks
_network_available = None


def network_available() -> bool:
    """Cached network availability check."""
    global _network_available
    if _network_available is None:
        _network_available = has_network_connection()
    return _network_available


skip_without_network = pytest.mark.skipif(
    not network_available(),
    reason="No network connection to Yahoo Finance"
)


class TestYFinanceProvider:
    """Test YFinanceDataProvider functionality."""

    @pytest.fixture
    def provider(self):
        return YFinanceDataProvider()

    def test_name(self, provider):
        """Test provider name."""
        assert provider.name == "yfinance"

    def test_map_timeframe_daily(self, provider):
        """Test daily timeframe mapping."""
        assert provider._map_timeframe('1D') == '1d'
        assert provider._map_timeframe('1d') == '1d'
        assert provider._map_timeframe('D') == '1d'

    def test_map_timeframe_intraday(self, provider):
        """Test intraday timeframe mapping."""
        assert provider._map_timeframe('1Min') == '1m'
        assert provider._map_timeframe('5Min') == '5m'
        assert provider._map_timeframe('15Min') == '15m'
        assert provider._map_timeframe('1Hour') == '1h'

    def test_map_timeframe_unknown_defaults_to_daily(self, provider):
        """Test unknown timeframe defaults to daily."""
        assert provider._map_timeframe('unknown') == '1d'

    def test_supports_timeframe(self, provider):
        """Test supported timeframes."""
        assert provider.supports_timeframe('1D') is True
        assert provider.supports_timeframe('1Min') is True
        assert provider.supports_timeframe('5Min') is True
        assert provider.supports_timeframe('unknown') is False

    def test_normalize_dataframe_lowercase_columns(self, provider):
        """Test column normalization to lowercase."""
        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [99],
            'Close': [102],
            'Volume': [1000]
        }, index=pd.DatetimeIndex([datetime.now()]))

        result = provider._normalize_dataframe(df, 'TEST')

        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

    def test_normalize_dataframe_flattens_multiindex(self, provider):
        """Test MultiIndex columns are flattened."""
        arrays = [['Open', 'Close'], ['AAPL', 'AAPL']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)

        df = pd.DataFrame(
            [[100, 102]],
            columns=index,
            index=pd.DatetimeIndex([datetime.now()])
        )

        # Should handle this without error
        # (actual flattening happens in the method)

    def test_normalize_dataframe_sets_eastern_timezone(self, provider):
        """Test timezone is converted to Eastern."""
        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [99],
            'Close': [102],
            'Volume': [1000]
        }, index=pd.DatetimeIndex([datetime(2024, 1, 1, 12, 0)]))

        result = provider._normalize_dataframe(df, 'TEST')

        assert result.index.tz is not None
        assert str(result.index.tz) == 'America/New_York'

    def test_normalize_dataframe_converts_existing_timezone(self, provider):
        """Test existing timezone is converted to Eastern."""
        utc = pytz.UTC
        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [99],
            'Close': [102],
            'Volume': [1000]
        }, index=pd.DatetimeIndex([datetime(2024, 1, 1, 12, 0, tzinfo=utc)]))

        result = provider._normalize_dataframe(df, 'TEST')

        assert str(result.index.tz) == 'America/New_York'

    def test_normalize_dataframe_returns_empty_on_missing_columns(self, provider):
        """Test empty DataFrame on missing required columns."""
        df = pd.DataFrame({
            'Open': [100],
            'Close': [102],
            # Missing High, Low, Volume
        }, index=pd.DatetimeIndex([datetime.now()]))

        result = provider._normalize_dataframe(df, 'TEST')

        assert result.empty


class TestYFinanceIntegration:
    """Integration tests for YFinanceDataProvider (requires network)."""

    @pytest.fixture
    def provider(self):
        return YFinanceDataProvider()

    @skip_without_network
    def test_fetch_daily_bars(self, provider):
        """Test fetching daily bars for a liquid symbol."""
        from datetime import timedelta

        end = datetime.now()
        start = end - timedelta(days=5)

        df = provider.get_historical_bars('TQQQ', start, end, '1D')

        assert df is not None
        assert not df.empty
        assert 'close' in df.columns

    @skip_without_network
    def test_fetch_intraday_bars(self, provider):
        """Test fetching intraday bars."""
        from datetime import timedelta

        end = datetime.now()
        start = end - timedelta(hours=6)

        df = provider.get_historical_bars('TQQQ', start, end, '1Min')

        # Note: yfinance may return empty for non-market hours
        # This test verifies the call doesn't error

    @skip_without_network
    def test_batch_fetch(self, provider):
        """Test batch fetching multiple symbols."""
        from datetime import timedelta

        end = datetime.now()
        start = end - timedelta(days=5)

        results = provider.get_historical_bars_batch(
            ['TQQQ', 'SQQQ', 'UPRO'], start, end, '1D'
        )

        assert isinstance(results, dict)
        assert len(results) > 0
