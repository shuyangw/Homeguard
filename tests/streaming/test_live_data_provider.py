"""Unit tests for LiveDataProvider (mocked, no network calls)."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from src.streaming.types import Bar, Quote, Trade
from src.streaming._buffer import BarBuffer, QuoteBuffer, TradeBuffer


def make_bar(symbol: str = "AAPL", close: float = 150.0) -> Bar:
    """Helper to create test bars."""
    return Bar(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=10000.0,
        vwap=close,
    )


def make_quote(symbol: str = "AAPL", bid: float = 150.0, ask: float = 150.05) -> Quote:
    """Helper to create test quotes."""
    return Quote(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        bid_price=bid,
        bid_size=100.0,
        ask_price=ask,
        ask_size=100.0,
    )


def make_trade(symbol: str = "AAPL", price: float = 150.0) -> Trade:
    """Helper to create test trades."""
    return Trade(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        price=price,
        size=100.0,
    )


@pytest.fixture
def mock_env_credentials():
    """Set up mock environment variables for Alpaca credentials."""
    with patch.dict(os.environ, {
        "ALPACA_PAPER_KEY_ID": "test_key",
        "ALPACA_PAPER_SECRET_KEY": "test_secret",
    }):
        yield


@pytest.fixture
def mock_stream_manager():
    """Mock the StreamManager to avoid network calls."""
    with patch("src.streaming._hub.StreamManager") as mock:
        mock_instance = MagicMock()
        mock_instance.is_running.return_value = False
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_fallback_poller():
    """Mock the FallbackPoller to avoid network calls."""
    with patch("src.streaming._hub.FallbackPoller") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


class TestLiveDataProviderInit:
    """Tests for LiveDataProvider initialization."""

    def test_init_with_credentials(self, mock_stream_manager, mock_fallback_poller):
        """Test initialization with explicit credentials."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider(
            api_key="explicit_key",
            secret_key="explicit_secret",
            feed="sip",
        )

        assert provider.feed == "sip"
        assert not provider.is_connected()

    def test_init_loads_credentials_from_env(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test initialization loads credentials from environment variables."""
        from src.streaming import LiveDataProvider

        # Should not raise since env vars are set
        provider = LiveDataProvider(feed="iex")
        assert provider.feed == "iex"

    def test_init_raises_without_credentials(self, mock_stream_manager, mock_fallback_poller):
        """Test that initialization raises without credentials."""
        # Clear any existing env vars
        with patch.dict(os.environ, {}, clear=True):
            from src.streaming import LiveDataProvider

            with pytest.raises(ValueError, match="Alpaca API credentials not found"):
                LiveDataProvider(feed="iex")


class TestLiveDataProviderDataAccess:
    """Tests for data access methods."""

    def test_get_price(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test getting price from buffer."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        # Simulate data in buffer by directly adding to hub's trade buffer
        trade = make_trade("TQQQ", price=45.50)
        provider._hub._trade_buffer.update(trade)

        price = provider.get_price("TQQQ")
        assert price == 45.50

    def test_get_quote(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test getting quote from buffer."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        quote = make_quote("TQQQ", bid=45.00, ask=45.02)
        provider._hub._quote_buffer.update(quote)

        retrieved = provider.get_quote("TQQQ")
        assert retrieved.bid_price == 45.00
        assert retrieved.ask_price == 45.02
        assert retrieved.mid == pytest.approx(45.01, rel=1e-9)
        assert retrieved.spread == pytest.approx(0.02, rel=1e-9)

    def test_get_bars(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test getting bars DataFrame from buffer."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        # Add bars to buffer
        for i in range(5):
            bar = make_bar("TQQQ", close=45.0 + i)
            provider._hub._bar_buffer.add(bar)

        df = provider.get_bars("TQQQ")

        assert len(df) == 5
        assert "close" in df.columns
        assert df["close"].iloc[-1] == 49.0

    def test_get_bars_with_limit(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test getting limited bars."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        for i in range(10):
            bar = make_bar("TQQQ", close=45.0 + i)
            provider._hub._bar_buffer.add(bar)

        df = provider.get_bars("TQQQ", n=3)

        assert len(df) == 3

    def test_get_vwap(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test getting VWAP."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        bar = make_bar("TQQQ", close=45.0)
        bar = Bar(
            symbol="TQQQ",
            timestamp=datetime.now(timezone.utc),
            open=44.5,
            high=45.5,
            low=44.0,
            close=45.0,
            volume=10000.0,
            vwap=44.95,
        )
        provider._hub._bar_buffer.add(bar)

        vwap = provider.get_vwap("TQQQ")
        assert vwap == 44.95

    def test_get_spread(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test getting spread."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        quote = make_quote("TQQQ", bid=45.00, ask=45.05)
        provider._hub._quote_buffer.update(quote)

        spread = provider.get_spread("TQQQ")
        assert spread == pytest.approx(0.05, rel=1e-9)


class TestLiveDataProviderSubscriptions:
    """Tests for callback subscriptions."""

    def test_on_bar_returns_subscription_id(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test that on_bar returns a subscription ID."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        async def handler(bar):
            pass

        sub_id = provider.on_bar(["TQQQ"], handler)

        assert sub_id is not None
        assert len(sub_id) > 0

    def test_unsubscribe(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test unsubscribing from callbacks."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        async def handler(bar):
            pass

        sub_id = provider.on_bar(["TQQQ"], handler)
        provider.unsubscribe(sub_id)

        # Should not raise
        assert True


class TestLiveDataProviderFallback:
    """Tests for fallback behavior."""

    def test_falls_back_to_polling_when_buffer_empty(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test that fallback is used when buffer is empty."""
        from src.streaming import LiveDataProvider

        # Setup mock fallback to return a quote
        mock_fallback_poller.get_latest_quote.return_value = make_quote(
            "TQQQ", bid=45.0, ask=45.02
        )

        provider = LiveDataProvider()

        # Buffer is empty, should fall back to polling
        quote = provider.get_quote("TQQQ")

        mock_fallback_poller.get_latest_quote.assert_called_once_with("TQQQ")
        assert quote is not None
        assert quote.bid_price == 45.0


class TestLiveDataProviderLifecycle:
    """Tests for lifecycle methods."""

    def test_start_and_stop(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test start and stop lifecycle."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        provider.start(["TQQQ", "SPY"])
        mock_stream_manager.start.assert_called_once()

        provider.stop()
        mock_stream_manager.stop.assert_called_once()

    def test_clear_buffers(
        self, mock_env_credentials, mock_stream_manager, mock_fallback_poller
    ):
        """Test clearing buffers."""
        from src.streaming import LiveDataProvider

        provider = LiveDataProvider()

        # Add data
        provider._hub._bar_buffer.add(make_bar("TQQQ"))
        provider._hub._quote_buffer.update(make_quote("TQQQ"))

        assert provider._hub._bar_buffer.count("TQQQ") == 1

        provider.clear_buffers()

        assert provider._hub._bar_buffer.count("TQQQ") == 0
        assert provider._hub._quote_buffer.get("TQQQ") is None
