"""
Tests for IBKRConfig and error handling.
"""

import pytest

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.errors import (
    PacingViolationError,
    classify_error,
    ibkr_code_to_exception,
    INFORMATIONAL_CODES,
)
from src.trading.brokers.interfaces.base import (
    BrokerError,
    BrokerConnectionError,
    InvalidOrderError,
    OrderNotFoundError,
    SymbolNotFoundError,
)


_IBKR_ENV_KEYS = (
    "IBKR_HOST",
    "IBKR_PORT",
    "IBKR_CLIENT_ID",
    "IBKR_ACCOUNT",
    "IBKR_READONLY",
    "IBKR_TIMEOUT",
    "IBKR_MARKET_DATA_TYPE",
)


def _clean_ibkr_env(monkeypatch):
    """Remove IBKR_* env vars so a sourced .env can't override explicit kwargs."""
    for env_key in _IBKR_ENV_KEYS:
        monkeypatch.delenv(env_key, raising=False)


class TestIBKRConfig:

    def test_defaults(self, monkeypatch, tmp_path):
        """Test class defaults when no YAML config is present."""
        _clean_ibkr_env(monkeypatch)
        monkeypatch.chdir(tmp_path)
        config = IBKRConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 4002
        assert config.client_id == 1
        assert config.readonly is False

    def test_paper_detection(self, monkeypatch):
        _clean_ibkr_env(monkeypatch)
        assert IBKRConfig(port=4002).is_paper is True
        assert IBKRConfig(port=7497).is_paper is True
        assert IBKRConfig(port=4001).is_paper is False
        assert IBKRConfig(port=7496).is_paper is False

    def test_gateway_type_label(self, monkeypatch):
        _clean_ibkr_env(monkeypatch)
        assert "Gateway" in IBKRConfig(port=4002).gateway_type
        assert "paper" in IBKRConfig(port=4002).gateway_type
        assert "TWS" in IBKRConfig(port=7496).gateway_type
        assert "live" in IBKRConfig(port=7496).gateway_type

    def test_pacing_defaults_reasonable(self):
        config = IBKRConfig()
        assert config.historical_pacing_max_per_10min < 60
        assert config.historical_pacing_max_per_10min > 50
        assert config.identical_request_cooldown_seconds >= 15.0


class TestErrorClassification:

    def test_informational_codes(self):
        for code in [2104, 2106, 2158]:
            assert classify_error(code) == "informational"

    def test_pacing_codes(self):
        assert classify_error(162) == "pacing"
        assert classify_error(366) == "pacing"

    def test_connection_codes(self):
        assert classify_error(1100) == "connection"
        assert classify_error(504) == "connection"

    def test_order_codes(self):
        assert classify_error(201) == "order"
        assert classify_error(202) == "order"
        assert classify_error(203) == "order"

    def test_unknown_codes_default_to_error(self):
        assert classify_error(99999) == "error"


class TestExceptionMapping:
    """Verify IBKR codes map to Homeguard exception types (not custom ones)."""

    def test_connection_error(self):
        exc = ibkr_code_to_exception(1100, "Lost connection")
        assert isinstance(exc, BrokerConnectionError)

    def test_symbol_not_found(self):
        exc = ibkr_code_to_exception(200, "No security definition")
        assert isinstance(exc, SymbolNotFoundError)

    def test_invalid_order(self):
        exc = ibkr_code_to_exception(201, "Order rejected")
        assert isinstance(exc, InvalidOrderError)

    def test_order_not_found(self):
        exc = ibkr_code_to_exception(135, "Can't find order")
        assert isinstance(exc, OrderNotFoundError)

    def test_unknown_code_gives_broker_error(self):
        exc = ibkr_code_to_exception(99999, "Unknown")
        assert isinstance(exc, BrokerError)

    def test_all_exceptions_are_homeguard_types(self):
        """No custom IBKR exceptions leak out."""
        for code in [504, 200, 201, 135, 99999]:
            exc = ibkr_code_to_exception(code, "test")
            assert isinstance(exc, BrokerError)
            assert not isinstance(exc, PacingViolationError)

    def test_error_message_includes_code(self):
        exc = ibkr_code_to_exception(201, "Insufficient margin")
        assert "201" in str(exc)
