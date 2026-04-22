"""
Tests for AlpacaMarketData -- the composite that exposes Alpaca historical
data AND Alpaca streaming through a single object.

These tests verify the delegation contract: each public method routes to
the correct internal provider. The actual Alpaca API behavior is tested
elsewhere (AlpacaDataProvider and LiveDataProvider each have their own
unit tests).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.streaming.alpaca_market_data import AlpacaMarketData
from src.streaming.interface import StreamingProviderInterface


@pytest.fixture
def market_data():
    """Construct AlpacaMarketData with stubbed internals -- no network."""
    with patch("src.streaming.alpaca_market_data.AlpacaBroker") as MockBroker, \
         patch("src.streaming.alpaca_market_data.AlpacaDataProvider") as MockHist, \
         patch("src.streaming.alpaca_market_data.LiveDataProvider") as MockStream:
        MockBroker.return_value = MagicMock()
        hist_instance = MagicMock()
        MockHist.return_value = hist_instance
        stream_instance = MagicMock()
        MockStream.return_value = stream_instance

        md = AlpacaMarketData(api_key="key", secret_key="sec", feed="iex")
        # Expose internals for assertions
        md._test_historical = hist_instance
        md._test_streaming = stream_instance
        yield md


class TestInitialization:
    def test_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="Alpaca credentials"):
            AlpacaMarketData()

    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "env-key")
        monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "env-sec")
        with patch("src.streaming.alpaca_market_data.AlpacaBroker") as MockBroker, \
             patch("src.streaming.alpaca_market_data.AlpacaDataProvider"), \
             patch("src.streaming.alpaca_market_data.LiveDataProvider"):
            AlpacaMarketData()
            MockBroker.assert_called_once_with(
                api_key="env-key", secret_key="env-sec", paper=True
            )

    def test_is_streaming_provider(self, market_data):
        # Critical: RAMP's `isinstance(dp, StreamingProviderInterface)` must pass.
        assert isinstance(market_data, StreamingProviderInterface)

    def test_name_reflects_feed(self, market_data):
        assert market_data.name == "alpaca-market-iex"


class TestHistoricalDelegation:
    """Historical methods must route to AlpacaDataProvider, not LiveDataProvider."""

    def test_get_historical_bars_calls_historical(self, market_data):
        expected = pd.DataFrame({"close": [100.0, 101.0]})
        market_data._test_historical.get_historical_bars.return_value = expected
        result = market_data.get_historical_bars(
            "AAPL", start="2026-01-01", end="2026-01-10", timeframe="1D"
        )
        market_data._test_historical.get_historical_bars.assert_called_once_with(
            "AAPL", "2026-01-01", "2026-01-10", "1D", False
        )
        # And streaming provider must NOT be involved in historical fetches
        market_data._test_streaming.get_historical_bars.assert_not_called()
        assert result is expected

    def test_get_historical_bars_batch_calls_historical(self, market_data):
        market_data._test_historical.get_historical_bars_batch.return_value = {
            "AAPL": pd.DataFrame(), "SPY": pd.DataFrame()
        }
        result = market_data.get_historical_bars_batch(
            ["AAPL", "SPY"], start="2026-01-01", end="2026-01-10"
        )
        market_data._test_historical.get_historical_bars_batch.assert_called_once()
        assert set(result.keys()) == {"AAPL", "SPY"}


class TestStreamingDelegation:
    """StreamingProviderInterface methods must route to LiveDataProvider."""

    def test_start_stop_lifecycle(self, market_data):
        market_data.start(["AAPL", "SPY"])
        market_data._test_streaming.start.assert_called_once_with(["AAPL", "SPY"])
        market_data.stop()
        market_data._test_streaming.stop.assert_called_once()

    def test_is_connected_delegates(self, market_data):
        market_data._test_streaming.is_connected.return_value = True
        assert market_data.is_connected() is True

    def test_get_bars_delegates_to_stream(self, market_data):
        market_data._test_streaming.get_bars.return_value = pd.DataFrame({"close": [1.0]})
        result = market_data.get_bars("AAPL", n=1)
        market_data._test_streaming.get_bars.assert_called_once_with("AAPL", 1)
        # Historical must not be involved
        market_data._test_historical.get_bars.assert_not_called()
        assert not result.empty

    def test_get_price_delegates_to_stream(self, market_data):
        market_data._test_streaming.get_price.return_value = 450.25
        assert market_data.get_price("BRK.B") == 450.25

    def test_on_bar_registration_delegates(self, market_data):
        market_data._test_streaming.on_bar.return_value = "sub-id-42"
        handler = lambda bar: None
        sub_id = market_data.on_bar(["AAPL"], handler)
        market_data._test_streaming.on_bar.assert_called_once_with(["AAPL"], handler)
        assert sub_id == "sub-id-42"

    def test_get_subscribed_symbols_delegates(self, market_data):
        market_data._test_streaming.get_subscribed_symbols.return_value = {"AAPL", "SPY"}
        assert market_data.get_subscribed_symbols() == {"AAPL", "SPY"}


class TestSeparation:
    """Invariant: data and execution are fully separated.

    The data_broker constructed by AlpacaMarketData must never be exposed
    through public attributes -- callers must NOT use this for orders.
    """

    def test_no_public_broker_attribute(self, market_data):
        # `_historical` (internal) holds it; no `broker` / `execution_broker` public surface
        assert not hasattr(market_data, "broker")
        assert not hasattr(market_data, "execution_broker")
        assert not hasattr(market_data, "place_order")
        assert not hasattr(market_data, "get_account")
