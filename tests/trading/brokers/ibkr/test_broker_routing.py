"""Tests for broker routing configuration loader."""

import pytest
from unittest.mock import patch, MagicMock

from src.trading.config.broker_routing import load_broker_routing


class TestBrokerRouting:

    def test_load_returns_dict(self, tmp_path):
        config_file = tmp_path / "routing.yaml"
        config_file.write_text("""
brokers:
  alpaca:
    paper: true

strategies:
  omr:
    broker: alpaca

default_broker: alpaca
""")
        with patch(
            "src.trading.config.broker_routing.BrokerFactory"
        ) as mock_factory:
            mock_broker = MagicMock()
            mock_factory.create_broker.return_value = mock_broker

            result = load_broker_routing(str(config_file))

        assert "omr" in result
        assert result["omr"] is mock_broker

    def test_shared_broker_instances(self, tmp_path):
        config_file = tmp_path / "routing.yaml"
        config_file.write_text("""
brokers:
  alpaca:
    paper: true

strategies:
  omr:
    broker: alpaca
  ramp:
    broker: alpaca

default_broker: alpaca
""")
        with patch(
            "src.trading.config.broker_routing.BrokerFactory"
        ) as mock_factory:
            mock_broker = MagicMock()
            mock_factory.create_broker.return_value = mock_broker

            result = load_broker_routing(str(config_file))

        assert result["omr"] is result["ramp"]

    def test_default_broker_for_unlisted_strategy(self, tmp_path):
        config_file = tmp_path / "routing.yaml"
        config_file.write_text("""
brokers:
  alpaca:
    paper: true

strategies: {}

default_broker: alpaca
""")
        with patch(
            "src.trading.config.broker_routing.BrokerFactory"
        ) as mock_factory:
            mock_broker = MagicMock()
            mock_factory.create_broker.return_value = mock_broker

            result = load_broker_routing(str(config_file))

        assert result.get_default() is mock_broker
