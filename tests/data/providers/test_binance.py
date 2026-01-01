"""
Unit tests for Binance data provider.

Tests:
- Symbol normalization
- Rate limiting
- Historical data fetching
- Current price fetching
- Failover behavior
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


class TestBinanceSymbolNormalization:
    """Test symbol normalization functions."""

    def test_to_binance_symbol_slash_format(self):
        """Test BTC/USD -> BTCUSDT conversion."""
        from src.data.providers.binance import _to_binance_symbol

        assert _to_binance_symbol('BTC/USD') == 'BTCUSDT'
        assert _to_binance_symbol('ETH/USD') == 'ETHUSDT'
        assert _to_binance_symbol('SOL/USD') == 'SOLUSDT'

    def test_to_binance_symbol_underscore_format(self):
        """Test BTC_USD -> BTCUSDT conversion."""
        from src.data.providers.binance import _to_binance_symbol

        assert _to_binance_symbol('BTC_USD') == 'BTCUSDT'
        assert _to_binance_symbol('ETH_USD') == 'ETHUSDT'

    def test_from_binance_symbol(self):
        """Test BTCUSDT -> BTC/USD conversion."""
        from src.data.providers.binance import _from_binance_symbol

        assert _from_binance_symbol('BTCUSDT') == 'BTC/USD'
        assert _from_binance_symbol('ETHUSDT') == 'ETH/USD'

    def test_from_binance_symbol_non_usdt(self):
        """Test non-USDT symbols are returned as-is."""
        from src.data.providers.binance import _from_binance_symbol

        assert _from_binance_symbol('BTCEUR') == 'BTCEUR'


class TestBinanceDataProvider:
    """Test BinanceDataProvider class."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        from src.data.providers.binance import BinanceDataProvider
        return BinanceDataProvider(timeout=5)

    def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.name == "Binance"

    def test_timeframe_support(self, provider):
        """Test timeframe support check."""
        assert provider.supports_timeframe('1D') is True
        assert provider.supports_timeframe('1H') is True
        assert provider.supports_timeframe('1Min') is True
        assert provider.supports_timeframe('invalid') is False

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_get_current_price(self, mock_request, provider):
        """Test getting current price."""
        mock_request.return_value = {'symbol': 'BTCUSDT', 'price': '50000.00'}

        price = provider.get_current_price('BTC/USD')

        assert price == 50000.0
        mock_request.assert_called_once()

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_get_current_price_failure(self, mock_request, provider):
        """Test getting current price when API fails."""
        mock_request.return_value = None

        price = provider.get_current_price('BTC/USD')

        assert price is None

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_get_current_prices_batch(self, mock_request, provider):
        """Test getting multiple current prices."""
        mock_request.return_value = [
            {'symbol': 'BTCUSDT', 'price': '50000.00'},
            {'symbol': 'ETHUSDT', 'price': '3000.00'},
            {'symbol': 'SOLUSDT', 'price': '100.00'},
        ]

        prices = provider.get_current_prices(['BTC/USD', 'ETH/USD', 'SOL/USD'])

        assert len(prices) == 3
        assert prices['BTC/USD'] == 50000.0
        assert prices['ETH/USD'] == 3000.0
        assert prices['SOL/USD'] == 100.0

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_get_historical_bars(self, mock_request, provider):
        """Test getting historical OHLCV bars."""
        # Mock kline data (Binance format)
        mock_klines = [
            [
                1640000000000,  # open_time
                '50000.00',     # open
                '51000.00',     # high
                '49000.00',     # low
                '50500.00',     # close
                '1000.00',      # volume
                1640003600000,  # close_time
                '50000000.00',  # quote_volume
                100,            # trades
                '500.00',       # taker_buy_base
                '25000000.00',  # taker_buy_quote
                '0'             # ignore
            ]
        ]
        mock_request.return_value = mock_klines

        start = datetime(2021, 12, 20)
        end = datetime(2021, 12, 21)
        df = provider.get_historical_bars('BTC/USD', start, end, '1D')

        assert df is not None
        assert len(df) == 1
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert df['close'].iloc[0] == 50500.0

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_get_historical_bars_empty(self, mock_request, provider):
        """Test getting historical bars when no data."""
        mock_request.return_value = []

        start = datetime(2021, 12, 20)
        end = datetime(2021, 12, 21)
        df = provider.get_historical_bars('INVALID/USD', start, end, '1D')

        assert df is None

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_is_available_success(self, mock_request, provider):
        """Test availability check when API is up."""
        mock_request.return_value = {}

        assert provider.is_available() is True

    @patch('src.data.providers.binance.BinanceDataProvider._request_with_retry')
    def test_is_available_failure(self, mock_request, provider):
        """Test availability check when API is down."""
        mock_request.return_value = None

        assert provider.is_available() is False


class TestCryptoDataProviderWithFallback:
    """Test composite provider with fallback."""

    @pytest.fixture
    def provider(self):
        """Create composite provider with mocked components."""
        from src.data.providers.binance import CryptoDataProviderWithFallback

        provider = CryptoDataProviderWithFallback()
        provider._binance = Mock()
        provider._alpaca = Mock()
        return provider

    def test_provider_name(self, provider):
        """Test composite provider name."""
        assert provider.name == "Binance+Alpaca"

    def test_uses_binance_by_default(self, provider):
        """Test that Binance is used by default."""
        provider._binance.get_current_price.return_value = 50000.0

        price = provider.get_current_price('BTC/USD')

        assert price == 50000.0
        provider._binance.get_current_price.assert_called_once()
        provider._alpaca.get_historical_bars.assert_not_called()

    def test_records_failure_on_binance_error(self, provider):
        """Test that failures are recorded."""
        provider._binance.get_current_price.side_effect = Exception("API Error")
        provider._alpaca.get_historical_bars.return_value = pd.DataFrame({
            'close': [49000.0]
        })

        provider.get_current_price('BTC/USD')

        assert provider._failure_count == 1

    def test_switches_to_fallback_after_threshold(self, provider):
        """Test fallback after 3 failures."""
        # Mock the singular get_current_price to throw an exception
        provider._binance.get_current_price.side_effect = Exception("Error")
        provider._alpaca.get_historical_bars.return_value = pd.DataFrame({'close': [50000.0]})

        # Trigger 3 failures by calling get_current_price
        for _ in range(3):
            provider.get_current_price('BTC/USD')

        assert provider._using_fallback is True
        assert provider._fallback_until is not None

    def test_fallback_window_expires(self, provider):
        """Test that fallback window expires."""
        # Set fallback window in the past
        provider._fallback_until = datetime.now() - timedelta(minutes=10)
        provider._failure_count = 5
        provider._using_fallback = True

        provider._binance.get_current_price.return_value = 50000.0

        price = provider.get_current_price('BTC/USD')

        # Should use Binance again
        assert price == 50000.0
        assert provider._failure_count == 0
        assert provider._using_fallback is False

    def test_get_historical_bars_with_fallback(self, provider):
        """Test historical bars fetch with fallback."""
        provider._binance.get_historical_bars.return_value = None
        provider._alpaca.get_historical_bars.return_value = pd.DataFrame({
            'open': [49000],
            'high': [50000],
            'low': [48000],
            'close': [49500],
            'volume': [1000]
        })

        start = datetime(2021, 12, 20)
        end = datetime(2021, 12, 21)
        df = provider.get_historical_bars('BTC/USD', start, end, '1D')

        assert df is not None
        # Should have fallen back to Alpaca
        provider._alpaca.get_historical_bars.assert_called_once()


class TestBinanceRateLimiting:
    """Test rate limiting behavior."""

    @patch('time.sleep')
    @patch('time.time')
    def test_rate_limit_enforced(self, mock_time, mock_sleep):
        """Test that rate limiting delay is enforced."""
        from src.data.providers.binance import BinanceDataProvider

        # Simulate time passing
        mock_time.side_effect = [0.0, 0.05, 0.15]  # Last request at 0, now at 0.05

        provider = BinanceDataProvider()
        provider._last_request_time = 0.0

        provider._rate_limit()

        # Should have slept for remaining time (0.1 - 0.05 = 0.05)
        mock_sleep.assert_called()


class TestBinanceRetryLogic:
    """Test retry logic for API requests."""

    @patch('src.data.providers.binance.requests.Session.get')
    @patch('time.sleep')
    def test_retry_on_server_error(self, mock_sleep, mock_get):
        """Test that server errors trigger retries."""
        from src.data.providers.binance import BinanceDataProvider

        # First two calls fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = Exception("Server Error")

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'price': '50000'}
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_fail, mock_response_fail, mock_response_success]

        provider = BinanceDataProvider()
        result = provider._request_with_retry('/api/v3/ticker/price')

        assert result == {'price': '50000'}
        assert mock_get.call_count == 3

    @patch('src.data.providers.binance.requests.Session.get')
    @patch('time.sleep')
    def test_returns_none_after_max_retries(self, mock_sleep, mock_get):
        """Test that None is returned after max retries."""
        from src.data.providers.binance import BinanceDataProvider

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server Error")

        mock_get.return_value = mock_response

        provider = BinanceDataProvider()
        result = provider._request_with_retry('/api/v3/ticker/price', max_retries=3)

        assert result is None
        assert mock_get.call_count == 3
