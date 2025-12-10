"""
Integration Tests for LiveDataProvider with Strategy Adapters.

Tests that strategies (OMR, RAMP) work correctly when receiving data
from LiveDataProvider (streaming) instead of broker API (polling).
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from src.trading.adapters.omr_live_adapter import OMRLiveAdapter
from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter


@pytest.fixture
def mock_broker():
    """Create mock broker interface."""
    broker = Mock()

    # Mock account info
    account = {
        'account_id': "test_account_123",
        'buying_power': 100000.0,
        'portfolio_value': 100000.0,
        'cash': 100000.0,
        'equity': 100000.0,
        'currency': 'USD'
    }
    broker.get_account.return_value = account

    # Mock market status
    broker.is_market_open.return_value = True

    # Mock historical data for SPY/VIX (still polled, not streamed)
    def get_historical_bars(symbol, start, end, timeframe):
        """Return mock historical data for non-streaming symbols."""
        dates = pd.date_range(start, end, freq='D')
        data = {
            'open': [100.0] * len(dates),
            'high': [101.0] * len(dates),
            'low': [99.0] * len(dates),
            'close': [100.5] * len(dates),
            'volume': [1000000] * len(dates)
        }
        return pd.DataFrame(data, index=dates)

    broker.get_historical_bars.side_effect = get_historical_bars

    # Mock positions
    broker.get_positions.return_value = []

    # Mock quotes
    broker.get_latest_quote.return_value = {'bid': 100.0, 'ask': 100.5}

    return broker


@pytest.fixture
def mock_streaming_provider():
    """
    Create mock LiveDataProvider that simulates streaming data.

    Simulates a buffer that contains today's intraday bars.
    """
    provider = Mock()

    # Give it the streaming provider's signature (get_bars method)
    def get_bars(symbol, n=None):
        """
        Simulate getting bars from streaming buffer.

        Returns today's intraday bars (9:30 AM - current time).
        """
        # Simulate current time at 3:50 PM
        today = pd.Timestamp.now().date()
        market_open = pd.Timestamp.combine(today, pd.Timestamp('09:30:00').time())
        current_time = pd.Timestamp.combine(today, pd.Timestamp('15:50:00').time())

        # Generate 1-minute bars from 9:30 AM to 3:50 PM (380 minutes)
        dates = pd.date_range(market_open, current_time, freq='1min')

        if n is not None and n < len(dates):
            dates = dates[-n:]  # Return last n bars

        # Simulate realistic intraday price movement
        base_price = {'TQQQ': 50.0, 'SQQQ': 25.0, 'UPRO': 60.0,
                     'AAPL': 175.0, 'MSFT': 380.0, 'GOOGL': 140.0}.get(symbol, 100.0)

        import numpy as np
        np.random.seed(hash(symbol) % 2**32)

        # Random walk
        returns = np.random.randn(len(dates)) * 0.001  # 0.1% volatility per minute
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'open': prices * 0.9995,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(10000, 100000, len(dates)),
            'vwap': prices * (1 + np.random.randn(len(dates)) * 0.0001),
            'trade_count': np.random.randint(100, 1000, len(dates))
        }, index=dates)

        return df

    provider.get_bars.side_effect = get_bars

    # Add get_quote for instant quote access
    def get_quote(symbol):
        """Simulate getting latest quote from buffer."""
        from src.streaming.types import Quote
        return Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=100.0,
            bid_size=100.0,
            ask_price=100.5,
            ask_size=100.0
        )

    provider.get_quote.side_effect = get_quote

    # Add get_price for instant price access
    def get_price(symbol):
        """Simulate getting latest price from buffer."""
        return 100.25  # Mid price

    provider.get_price.side_effect = get_price

    return provider


class TestOMRStreamingIntegration:
    """Test OMR adapter with LiveDataProvider streaming."""

    def test_omr_accepts_streaming_provider(self, mock_broker, mock_streaming_provider):
        """Test OMRLiveAdapter accepts LiveDataProvider."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ', 'SQQQ'],
            data_provider=mock_streaming_provider
        )

        assert adapter._data_provider is not None
        # Should have get_bars method (streaming signature)
        assert hasattr(adapter._data_provider, 'get_bars')

    def test_omr_fetches_intraday_from_streaming_buffer(self, mock_broker, mock_streaming_provider):
        """Test OMR fetches intraday data from streaming buffer instead of API."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ', 'SQQQ', 'UPRO'],
            data_provider=mock_streaming_provider
        )

        # Populate historical cache (for SPY/VIX, still polled)
        adapter._data_cache = {
            'SPY': mock_broker.get_historical_bars('SPY', '2024-01-01', '2024-12-31', '1D'),
            'VIX': mock_broker.get_historical_bars('VIX', '2024-01-01', '2024-12-31', '1D')
        }

        # Fetch market data (should use streaming for symbols, polling for SPY/VIX)
        market_data = adapter.fetch_market_data()

        # Should have all symbols
        assert 'TQQQ' in market_data
        assert 'SQQQ' in market_data
        assert 'UPRO' in market_data
        assert 'SPY' in market_data
        assert 'VIX' in market_data

        # Streaming provider should have been called for each symbol
        assert mock_streaming_provider.get_bars.call_count >= 3  # At least 3 symbols

        # Verify data is intraday (many bars, not daily)
        tqqq_data = market_data['TQQQ']
        assert len(tqqq_data) > 100, "Should have intraday bars (380+ for full day)"

    def test_omr_uses_streaming_not_broker_for_symbols(self, mock_broker, mock_streaming_provider):
        """Test OMR uses streaming provider, not broker API for symbol data."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=mock_streaming_provider
        )

        # Add SPY/VIX to cache (still polled)
        adapter._data_cache = {
            'SPY': mock_broker.get_historical_bars('SPY', '2024-01-01', '2024-12-31', '1D'),
            'VIX': mock_broker.get_historical_bars('VIX', '2024-01-01', '2024-12-31', '1D')
        }

        # Reset broker call count
        mock_broker.get_historical_bars.reset_mock()

        # Fetch market data
        market_data = adapter.fetch_market_data()

        # Streaming provider should be called for TQQQ
        assert mock_streaming_provider.get_bars.called
        assert any('TQQQ' in str(call) for call in mock_streaming_provider.get_bars.call_args_list)

        # Broker should NOT be called for TQQQ (only for SPY/VIX if needed)
        broker_calls = [str(call) for call in mock_broker.get_historical_bars.call_args_list]
        assert not any('TQQQ' in call for call in broker_calls), \
            "Broker should not be called for TQQQ when streaming provider exists"

    def test_omr_streaming_data_has_correct_columns(self, mock_broker, mock_streaming_provider):
        """Test streaming data returns correct OHLCV schema."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=mock_streaming_provider
        )

        adapter._data_cache = {
            'SPY': mock_broker.get_historical_bars('SPY', '2024-01-01', '2024-12-31', '1D'),
            'VIX': mock_broker.get_historical_bars('VIX', '2024-01-01', '2024-12-31', '1D')
        }

        market_data = adapter.fetch_market_data()

        # Verify TQQQ data has required columns
        tqqq_data = market_data['TQQQ']
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        for col in required_cols:
            assert col in tqqq_data.columns, f"Missing required column: {col}"


class TestRAMPStreamingIntegration:
    """Test RAMP adapter with LiveDataProvider streaming."""

    def test_ramp_accepts_streaming_provider(self, mock_broker, mock_streaming_provider):
        """Test RAMPLiveAdapter accepts LiveDataProvider."""
        adapter = RAMPLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            data_provider=mock_streaming_provider
        )

        assert adapter._data_provider is not None
        assert hasattr(adapter._data_provider, 'get_bars')

    def test_ramp_fetches_todays_closes_from_streaming_buffer(self, mock_broker, mock_streaming_provider):
        """Test RAMP fetches today's closes from streaming buffer instead of API."""
        from src.utils.timezone import tz

        adapter = RAMPLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            data_provider=mock_streaming_provider
        )

        # Create historical cache (missing today)
        yesterday = (tz.now() - timedelta(days=1)).date()
        dates = pd.date_range(end=yesterday, periods=30, freq='D')

        prices_df = pd.DataFrame({
            'AAPL': [175.0] * len(dates),
            'MSFT': [380.0] * len(dates),
            'GOOGL': [140.0] * len(dates)
        }, index=dates)

        adapter._data_cache = {
            'prices': prices_df,
            'SPY': pd.DataFrame({'close': [450.0] * len(dates)}, index=dates),
            'VIX': pd.DataFrame({'Close': [15.0] * len(dates)}, index=dates)
        }

        # Fetch today's closes (should use streaming)
        result = adapter.fetch_todays_closes()

        assert result == True

        # Streaming provider should have been called for each symbol
        assert mock_streaming_provider.get_bars.call_count >= 3

        # Verify each symbol was requested with n=1 (latest bar only)
        for call in mock_streaming_provider.get_bars.call_args_list:
            args, kwargs = call
            if len(args) > 1 or 'n' in kwargs:
                # Either positional arg or keyword arg for n
                n_value = args[1] if len(args) > 1 else kwargs.get('n', None)
                if n_value is not None:
                    assert n_value == 1, "RAMP should request only latest bar (n=1)"

    def test_ramp_uses_streaming_not_broker_for_todays_closes(self, mock_broker, mock_streaming_provider):
        """Test RAMP uses streaming provider, not broker API for today's closes."""
        from src.utils.timezone import tz

        adapter = RAMPLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            data_provider=mock_streaming_provider
        )

        # Setup cache
        yesterday = (tz.now() - timedelta(days=1)).date()
        dates = pd.date_range(end=yesterday, periods=30, freq='D')

        adapter._data_cache = {
            'prices': pd.DataFrame({
                'AAPL': [175.0] * len(dates),
                'MSFT': [380.0] * len(dates)
            }, index=dates),
            'SPY': pd.DataFrame({'close': [450.0] * len(dates)}, index=dates),
            'VIX': pd.DataFrame({'Close': [15.0] * len(dates)}, index=dates)
        }

        # Reset broker call count
        mock_broker.get_historical_bars.reset_mock()

        # Fetch today's closes
        result = adapter.fetch_todays_closes()

        assert result == True

        # Streaming provider should be called
        assert mock_streaming_provider.get_bars.called

        # Broker should NOT be called for AAPL/MSFT (only for SPY/VIX if needed)
        broker_calls = [str(call) for call in mock_broker.get_historical_bars.call_args_list]
        assert not any('AAPL' in call or 'MSFT' in call for call in broker_calls), \
            "Broker should not be called for symbols when streaming provider exists"

    def test_ramp_streaming_appends_to_historical_cache(self, mock_broker, mock_streaming_provider):
        """Test RAMP correctly appends streaming data to historical cache."""
        from src.utils.timezone import tz

        adapter = RAMPLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            data_provider=mock_streaming_provider
        )

        # Create cache missing today
        yesterday = (tz.now() - timedelta(days=1)).date()
        dates = pd.date_range(end=yesterday, periods=30, freq='D')

        original_prices = pd.DataFrame({
            'AAPL': [175.0 + i for i in range(len(dates))],
            'MSFT': [380.0 + i*2 for i in range(len(dates))]
        }, index=dates)

        adapter._data_cache = {
            'prices': original_prices.copy(),
            'SPY': pd.DataFrame({'close': [450.0] * len(dates)}, index=dates),
            'VIX': pd.DataFrame({'Close': [15.0] * len(dates)}, index=dates)
        }

        original_len = len(original_prices)

        # Fetch today's closes
        adapter.fetch_todays_closes()

        # Should have added one more row
        new_prices = adapter._data_cache['prices']
        assert len(new_prices) == original_len + 1, \
            f"Expected {original_len + 1} rows, got {len(new_prices)}"

        # Historical data values should be unchanged (ignore index frequency attribute)
        historical_rows = new_prices.iloc[:original_len]
        pd.testing.assert_frame_equal(historical_rows, original_prices, check_freq=False)

        # New row should have today's date
        last_date = new_prices.index[-1]
        if hasattr(last_date, 'date'):
            last_date = last_date.date()
        assert last_date == tz.now().date()


class TestStreamingVsPollingComparison:
    """Compare streaming vs polling behavior side-by-side."""

    def test_streaming_faster_than_polling(self, mock_broker, mock_streaming_provider):
        """
        Demonstrate that streaming is faster than polling.

        Note: In real world, streaming is 100-300x faster. In tests with mocks,
        both are fast, but we can verify the call patterns differ.
        """
        # Setup OMR with streaming
        omr_streaming = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ', 'SQQQ'],
            data_provider=mock_streaming_provider
        )

        # Setup OMR without streaming (polling via broker)
        omr_polling = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ', 'SQQQ'],
            data_provider=None  # No provider = use broker
        )

        # Add cache for both
        cache_data = {
            'SPY': mock_broker.get_historical_bars('SPY', '2024-01-01', '2024-12-31', '1D'),
            'VIX': mock_broker.get_historical_bars('VIX', '2024-01-01', '2024-12-31', '1D')
        }
        omr_streaming._data_cache = cache_data.copy()
        omr_polling._data_cache = cache_data.copy()

        # Reset mocks
        mock_broker.get_historical_bars.reset_mock()
        mock_streaming_provider.get_bars.reset_mock()

        # Fetch with streaming
        omr_streaming.fetch_market_data()

        # Fetch with polling
        omr_polling.fetch_market_data()

        # Streaming should use provider.get_bars
        assert mock_streaming_provider.get_bars.call_count >= 2, \
            "Streaming should call provider.get_bars for each symbol"

        # Polling should use broker.get_historical_bars
        broker_symbol_calls = [
            call for call in mock_broker.get_historical_bars.call_args_list
            if 'TQQQ' in str(call) or 'SQQQ' in str(call)
        ]
        assert len(broker_symbol_calls) >= 2, \
            "Polling should call broker.get_historical_bars for each symbol"

    def test_streaming_and_polling_return_same_schema(self, mock_broker, mock_streaming_provider):
        """Verify both streaming and polling return compatible data structures."""
        # Setup OMR with streaming
        omr_streaming = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=mock_streaming_provider
        )

        # Setup OMR with polling
        omr_polling = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=None
        )

        # Add cache
        cache_data = {
            'SPY': mock_broker.get_historical_bars('SPY', '2024-01-01', '2024-12-31', '1D'),
            'VIX': mock_broker.get_historical_bars('VIX', '2024-01-01', '2024-12-31', '1D')
        }
        omr_streaming._data_cache = cache_data.copy()
        omr_polling._data_cache = cache_data.copy()

        # Fetch data
        streaming_data = omr_streaming.fetch_market_data()
        polling_data = omr_polling.fetch_market_data()

        # Both should have TQQQ
        assert 'TQQQ' in streaming_data
        assert 'TQQQ' in polling_data

        # Both should have same columns
        streaming_cols = set(streaming_data['TQQQ'].columns)
        polling_cols = set(polling_data['TQQQ'].columns)

        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        assert required_cols.issubset(streaming_cols)
        assert required_cols.issubset(polling_cols)


class TestStreamingFallback:
    """Test streaming provider fallback behavior."""

    def test_streaming_provider_buffer_empty_fallback(self, mock_broker):
        """Test adapter falls back to broker when streaming buffer is empty."""
        # Create streaming provider that returns empty DataFrames
        empty_provider = Mock()
        empty_provider.get_bars.return_value = pd.DataFrame()  # Empty buffer

        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=empty_provider
        )

        adapter._data_cache = {
            'SPY': mock_broker.get_historical_bars('SPY', '2024-01-01', '2024-12-31', '1D'),
            'VIX': mock_broker.get_historical_bars('VIX', '2024-01-01', '2024-12-31', '1D')
        }

        # Fetch market data
        market_data = adapter.fetch_market_data()

        # Should have attempted to use provider
        assert empty_provider.get_bars.called

        # Since provider returned empty, adapter behavior depends on implementation
        # (May skip symbol or log warning - check adapter doesn't crash)
        assert isinstance(market_data, dict)


class TestStreamingProviderSignatureDetection:
    """Test detection of streaming vs polling provider."""

    def test_detects_streaming_provider_by_get_bars_method(self, mock_broker, mock_streaming_provider):
        """Test adapter detects LiveDataProvider by checking for get_bars method."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=mock_streaming_provider
        )

        # Adapter should detect streaming signature
        assert hasattr(adapter._data_provider, 'get_bars')

    def test_detects_polling_provider_by_get_historical_bars_batch(self, mock_broker):
        """Test adapter detects DataProviderInterface by checking for get_historical_bars_batch."""
        # Use spec to control which methods exist (Mock with spec won't auto-create attributes)
        from unittest.mock import create_autospec

        # Create a class with only polling methods
        class PollingProviderInterface:
            name = "polling"
            def get_historical_bars_batch(self, symbols, start, end, timeframe, force_refresh=False):
                return {}

        polling_provider = create_autospec(PollingProviderInterface, instance=True)
        polling_provider.name = "polling"
        polling_provider.get_historical_bars_batch.return_value = {}

        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=polling_provider
        )

        # Adapter should detect polling signature
        assert hasattr(adapter._data_provider, 'get_historical_bars_batch')
        # With autospec, get_bars should not exist
        assert not hasattr(adapter._data_provider, 'get_bars')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
