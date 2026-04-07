"""
Tests for smart fallback mechanism in streaming data platform.

Verifies that:
1. Hub detects insufficient buffer data
2. Hub correctly falls back to REST API from market open
3. OMR adapter validates data quality from streaming
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.streaming.interface import StreamingProviderInterface
from src.streaming._hub import MarketDataHub
from src.streaming.types import Bar
from src.utils.timezone import tz


@pytest.fixture
def mock_api_keys():
    """Mock API credentials."""
    return ("test_key", "test_secret")


@pytest.fixture
def hub(mock_api_keys):
    """Create a MarketDataHub for testing."""
    with patch('src.streaming._hub.StreamManager'), \
         patch('src.streaming._hub.FallbackPoller'):
        api_key, secret_key = mock_api_keys
        hub = MarketDataHub(api_key, secret_key, feed='iex', fallback_enabled=True)
        yield hub


class TestSmartFallback:
    """Test smart fallback mechanism when buffer has insufficient data."""

    def test_get_bars_with_sufficient_data_uses_buffer(self, hub):
        """When buffer has ≥90% of requested bars, use buffer data."""
        # Mock buffer with 360/390 bars (92% - sufficient)
        mock_bars = [
            Bar(
                symbol="TQQQ",
                timestamp=tz.now() - timedelta(minutes=390-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000.0,
                trade_count=1000.0,
                vwap=100.3 + i
            )
            for i in range(360)
        ]

        hub._bar_buffer.get_bars = Mock(return_value=mock_bars)
        hub._fallback.get_bars = Mock()

        # Request 390 bars
        result = hub.get_bars("TQQQ", n=390)

        # Should use buffer (92% > 90% threshold)
        assert len(result) == 360
        assert not hub._fallback.get_bars.called

    def test_get_bars_with_insufficient_data_triggers_fallback(self, hub):
        """When buffer has <90% of requested bars, trigger REST API fallback."""
        # Mock buffer with only 100/390 bars (26% - insufficient)
        mock_bars = [
            Bar(
                symbol="TQQQ",
                timestamp=tz.now() - timedelta(minutes=100-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000.0,
                trade_count=1000.0,
                vwap=100.3 + i
            )
            for i in range(100)
        ]

        hub._bar_buffer.get_bars = Mock(return_value=mock_bars)

        # Mock fallback to return full dataset
        mock_fallback_df = pd.DataFrame({
            'open': [100.0] * 390,
            'high': [101.0] * 390,
            'low': [99.0] * 390,
            'close': [100.5] * 390,
            'volume': [1000000.0] * 390,
        })
        hub._fallback.get_bars = Mock(return_value=mock_fallback_df)

        # Request 390 bars
        with patch('src.streaming._hub.logger') as mock_logger:
            result = hub.get_bars("TQQQ", n=390)

            # Should trigger fallback
            assert hub._fallback.get_bars.called
            assert mock_logger.warning.called
            # Should log percentage coverage (approximately 25-26%)
            log_message = str(mock_logger.warning.call_args)
            assert "100/390" in log_message or "25" in log_message

    def test_fallback_from_market_open_calculates_correct_times(self, hub):
        """Verify fallback correctly calculates market open to now time range."""
        from alpaca.data.timeframe import TimeFrame

        # Mock current time to 2:30 PM ET
        mock_now = tz.now().replace(hour=14, minute=30, second=0, microsecond=0)

        with patch('src.utils.timezone.tz.now', return_value=mock_now):
            hub._fallback.get_bars = Mock(return_value=pd.DataFrame())

            # Trigger fallback
            hub._fallback_from_market_open("TQQQ")

            # Verify REST API called with correct time range
            call_args = hub._fallback.get_bars.call_args
            assert call_args[1]['symbol'] == "TQQQ"
            # Check timeframe is Minute (compare string representation since object identity differs)
            assert str(call_args[1]['timeframe']) == str(TimeFrame.Minute)

            # Start should be 9:30 AM today
            start_time = call_args[1]['start']
            assert start_time.hour == 9
            assert start_time.minute == 30

            # End should be now (2:30 PM)
            end_time = call_args[1]['end']
            assert end_time.hour == 14
            assert end_time.minute == 30

    def test_fallback_before_market_open_uses_yesterday(self, hub):
        """If current time is before 9:30 AM, fallback uses yesterday's market open."""
        from alpaca.data.timeframe import TimeFrame

        # Mock current time to 9:00 AM ET (before market open)
        mock_now = tz.now().replace(hour=9, minute=0, second=0, microsecond=0)

        with patch('src.utils.timezone.tz.now', return_value=mock_now):
            hub._fallback.get_bars = Mock(return_value=pd.DataFrame())

            # Trigger fallback
            hub._fallback_from_market_open("TQQQ")

            # Verify REST API called with yesterday's date
            call_args = hub._fallback.get_bars.call_args
            start_time = call_args[1]['start']

            # Start should be yesterday's 9:30 AM
            expected_date = (mock_now - timedelta(days=1)).date()
            assert start_time.date() == expected_date
            assert start_time.hour == 9
            assert start_time.minute == 30

    def test_empty_buffer_triggers_fallback_immediately(self, hub):
        """When buffer is completely empty, trigger fallback immediately."""
        # Mock empty buffer
        hub._bar_buffer.get_bars = Mock(return_value=[])
        hub._fallback.get_bars = Mock(return_value=pd.DataFrame())

        with patch('src.streaming._hub.logger') as mock_logger:
            result = hub.get_bars("TQQQ", n=390)

            # Should trigger fallback
            assert hub._fallback.get_bars.called
            # Check that info was logged about fetching from market open
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("buffer empty" in call.lower() or "market open" in call.lower() for call in info_calls)

    def test_no_fallback_when_disabled(self, hub):
        """When fallback is disabled, return buffer data even if insufficient."""
        # Disable fallback
        hub._fallback = None

        # Mock buffer with insufficient data
        mock_bars = [
            Bar(
                symbol="TQQQ",
                timestamp=tz.now() - timedelta(minutes=50-i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000000.0,
                trade_count=1000.0,
                vwap=100.3
            )
            for i in range(50)
        ]
        hub._bar_buffer.get_bars = Mock(return_value=mock_bars)

        # Request 390 bars (only have 50)
        result = hub.get_bars("TQQQ", n=390)

        # Should return buffer data without fallback
        assert len(result) == 50

    def test_90_percent_threshold_boundary(self, hub):
        """Test behavior at exactly 90% threshold."""
        # Mock buffer with exactly 351/390 bars (90%)
        mock_bars = [
            Bar(
                symbol="TQQQ",
                timestamp=tz.now() - timedelta(minutes=351-i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000000.0,
                trade_count=1000.0,
                vwap=100.3
            )
            for i in range(351)
        ]

        hub._bar_buffer.get_bars = Mock(return_value=mock_bars)
        hub._fallback.get_bars = Mock()

        # Request 390 bars (have 351 = 90%)
        result = hub.get_bars("TQQQ", n=390)

        # Should use buffer (90% >= 90% threshold)
        assert len(result) == 351
        assert not hub._fallback.get_bars.called

    def test_fallback_returns_dataframe_format(self, hub):
        """Verify fallback returns data in correct DataFrame format."""
        # Mock insufficient buffer
        hub._bar_buffer.get_bars = Mock(return_value=[])

        # Mock fallback response
        mock_fallback_df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000.0, 1100000.0, 1200000.0],
        })
        mock_fallback_df.index = pd.date_range(
            start=tz.now() - timedelta(minutes=3),
            periods=3,
            freq='1min'
        )
        hub._fallback.get_bars = Mock(return_value=mock_fallback_df)

        # Get bars
        result = hub.get_bars("TQQQ", n=390)

        # Should return fallback DataFrame directly
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'open' in result.columns
        assert 'close' in result.columns


@pytest.mark.integration
class TestOMRDataValidation:
    """Test OMR adapter data quality validation with streaming."""

    def test_omr_logs_warning_for_insufficient_data(self):
        """OMR adapter should log warning when buffer has <90% of expected bars."""
        from src.trading.adapters.omr_live_adapter import OMRLiveAdapter

        # Mock broker and data provider
        mock_broker = Mock()
        mock_provider = Mock(spec=StreamingProviderInterface)

        # Mock get_bars to return only 100/390 bars (26%)
        mock_df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000000.0] * 100,
        })
        mock_df.index = pd.date_range(
            start=tz.now() - timedelta(minutes=100),
            periods=100,
            freq='1min'
        )
        mock_provider.get_bars = Mock(return_value=mock_df)
        mock_provider.name = "streaming-iex"

        # Create adapter
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=mock_provider
        )

        # Fetch market data
        with patch('src.trading.adapters.omr_live_adapter.logger') as mock_logger:
            adapter.fetch_market_data()

            # Should log warning about insufficient data
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("26%" in call or "100/390" in call for call in warning_calls)

    def test_omr_logs_success_for_complete_data(self):
        """OMR adapter should log info/debug for complete data."""
        from src.trading.adapters.omr_live_adapter import OMRLiveAdapter

        # Mock broker and data provider
        mock_broker = Mock()
        mock_provider = Mock(spec=StreamingProviderInterface)

        # Mock get_bars to return full 390 bars
        mock_df = pd.DataFrame({
            'open': [100.0] * 390,
            'high': [101.0] * 390,
            'low': [99.0] * 390,
            'close': [100.5] * 390,
            'volume': [1000000.0] * 390,
        })
        mock_df.index = pd.date_range(
            start=tz.now() - timedelta(minutes=390),
            periods=390,
            freq='1min'
        )
        mock_provider.get_bars = Mock(return_value=mock_df)
        mock_provider.name = "streaming-iex"

        # Create adapter
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ'],
            data_provider=mock_provider
        )

        # Fetch market data
        with patch('src.trading.adapters.omr_live_adapter.logger') as mock_logger:
            result = adapter.fetch_market_data()

            # Should have data for TQQQ
            assert 'TQQQ' in result
            assert len(result['TQQQ']) == 390

            # Should not log any warnings about insufficient data
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert not any("bars (" in call for call in warning_calls)
