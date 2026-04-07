"""Tests that LiveDataProvider conforms to StreamingProviderInterface."""

import pytest
from unittest.mock import MagicMock, patch

from src.streaming.interface import StreamingProviderInterface
from src.streaming.live_data_provider import LiveDataProvider


class TestStreamingProviderConformance:
    """Verify LiveDataProvider implements StreamingProviderInterface."""

    def test_is_subclass(self):
        assert issubclass(LiveDataProvider, StreamingProviderInterface)

    def test_has_name_property(self):
        assert isinstance(
            LiveDataProvider.name, property
        ), "name must be a @property, not a plain attribute"

    def test_all_abstract_methods_implemented(self):
        required = [
            'start', 'stop', 'is_connected',
            'get_price', 'get_quote', 'get_trade', 'get_bar', 'get_bars',
            'get_vwap', 'get_spread',
            'on_bar', 'on_quote', 'on_trade', 'unsubscribe',
            'get_subscribed_symbols',
        ]
        for method_name in required:
            assert hasattr(LiveDataProvider, method_name), (
                f"LiveDataProvider missing {method_name}"
            )


class TestStreamingProviderIsinstance:

    @patch('src.streaming.live_data_provider._get_alpaca_credentials',
           return_value=('fake_key', 'fake_secret'))
    @patch('src.streaming.live_data_provider.MarketDataHub')
    def test_isinstance_check(self, mock_hub, mock_creds):
        provider = LiveDataProvider()
        assert isinstance(provider, StreamingProviderInterface)

    def test_non_provider_fails_isinstance(self):
        assert not isinstance("not_a_provider", StreamingProviderInterface)
