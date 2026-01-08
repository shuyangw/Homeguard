"""
Unit tests for Binance WebSocket streaming.

Tests symbol normalization, message parsing, and stream management.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.streaming.binance_stream import BinanceStreamManager
from src.streaming.types import Bar


class TestSymbolNormalization:
    """Tests for symbol format conversion."""

    def test_to_binance_btc(self):
        """BTC/USD -> btcusdt"""
        result = BinanceStreamManager._to_binance_symbol("BTC/USD")
        assert result == "btcusdt"

    def test_to_binance_eth(self):
        """ETH/USD -> ethusdt"""
        result = BinanceStreamManager._to_binance_symbol("ETH/USD")
        assert result == "ethusdt"

    def test_to_binance_mkr(self):
        """MKR/USD -> mkrusdt"""
        result = BinanceStreamManager._to_binance_symbol("MKR/USD")
        assert result == "mkrusdt"

    def test_to_binance_lowercase(self):
        """Converts to lowercase"""
        result = BinanceStreamManager._to_binance_symbol("BTC/USD")
        assert result.islower()

    def test_from_binance_btcusdt(self):
        """BTCUSDT -> BTC/USD"""
        result = BinanceStreamManager._from_binance_symbol("BTCUSDT")
        assert result == "BTC/USD"

    def test_from_binance_lowercase(self):
        """btcusdt -> BTC/USD"""
        result = BinanceStreamManager._from_binance_symbol("btcusdt")
        assert result == "BTC/USD"

    def test_from_binance_ethusdt(self):
        """ETHUSDT -> ETH/USD"""
        result = BinanceStreamManager._from_binance_symbol("ETHUSDT")
        assert result == "ETH/USD"


class TestStreamManager:
    """Tests for BinanceStreamManager."""

    def test_init_single_symbol(self):
        """Initialize with single symbol."""
        manager = BinanceStreamManager(["BTC/USD"])
        assert manager._symbols == ["BTC/USD"]
        assert "btcusdt" in manager._binance_symbols

    def test_init_multiple_symbols(self):
        """Initialize with multiple symbols."""
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        manager = BinanceStreamManager(symbols)
        assert len(manager._symbols) == 3
        assert len(manager._binance_symbols) == 3

    def test_symbol_map_created(self):
        """Symbol map links Binance to our format."""
        manager = BinanceStreamManager(["BTC/USD", "ETH/USD"])
        assert manager._symbol_map["btcusdt"] == "BTC/USD"
        assert manager._symbol_map["ethusdt"] == "ETH/USD"

    def test_build_stream_url_single(self):
        """URL contains stream for single symbol."""
        manager = BinanceStreamManager(["BTC/USD"])
        url = manager._build_stream_url()
        assert "btcusdt@kline_1m" in url
        assert "stream.binance.com" in url

    def test_build_stream_url_multiple(self):
        """URL contains streams for multiple symbols."""
        manager = BinanceStreamManager(["BTC/USD", "ETH/USD"])
        url = manager._build_stream_url()
        assert "btcusdt@kline_1m" in url
        assert "ethusdt@kline_1m" in url
        assert "/" in url  # Streams separated by /

    def test_subscribe_klines_adds_handler(self):
        """subscribe_klines adds handler to list."""
        manager = BinanceStreamManager(["BTC/USD"])

        def handler(bar):
            pass

        manager.subscribe_klines(handler)
        assert handler in manager._kline_handlers

    def test_is_connected_initial_false(self):
        """is_connected returns False initially."""
        manager = BinanceStreamManager(["BTC/USD"])
        assert manager.is_connected() is False

    def test_symbols_property(self):
        """symbols property returns copy of symbol list."""
        original = ["BTC/USD", "ETH/USD"]
        manager = BinanceStreamManager(original)
        result = manager.symbols
        assert result == original
        # Verify it's a copy
        result.append("SOL/USD")
        assert len(manager._symbols) == 2

    def test_stats_initial(self):
        """Initial stats are zeroed."""
        manager = BinanceStreamManager(["BTC/USD"])
        stats = manager.get_stats()
        assert stats["messages_received"] == 0
        assert stats["bars_processed"] == 0
        assert stats["reconnect_count"] == 0


class TestKlineMessageParsing:
    """Tests for kline message parsing."""

    @pytest.fixture
    def sample_kline_message(self):
        """Sample Binance kline WebSocket message."""
        return {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "E": 1704700800123,
                "s": "BTCUSDT",
                "k": {
                    "t": 1704700800000,
                    "T": 1704700859999,
                    "s": "BTCUSDT",
                    "i": "1m",
                    "o": "50000.00",
                    "h": "50100.00",
                    "l": "49900.00",
                    "c": "50050.00",
                    "v": "100.5",
                    "n": 1234,
                    "x": False,
                    "q": "5025000.00",
                    "V": "50.25",
                    "Q": "2512500.00",
                },
            },
        }

    def test_process_kline_creates_bar(self, sample_kline_message):
        """process_kline creates Bar from message."""
        manager = BinanceStreamManager(["BTC/USD"])
        bars_received = []

        def handler(bar):
            bars_received.append(bar)

        manager._kline_handlers.append(handler)
        manager._process_kline(sample_kline_message["data"])

        assert len(bars_received) == 1
        bar = bars_received[0]
        assert bar.symbol == "BTC/USD"
        assert bar.close == 50050.00

    def test_on_message_parses_combined_stream(self, sample_kline_message):
        """on_message handles combined stream format."""
        manager = BinanceStreamManager(["BTC/USD"])
        bars_received = []

        def handler(bar):
            bars_received.append(bar)

        manager._kline_handlers.append(handler)
        manager._on_message(None, json.dumps(sample_kline_message))

        assert len(bars_received) == 1

    def test_on_message_ignores_non_kline(self):
        """on_message ignores non-kline messages."""
        manager = BinanceStreamManager(["BTC/USD"])
        bars_received = []

        def handler(bar):
            bars_received.append(bar)

        manager._kline_handlers.append(handler)

        # Send a non-kline message
        message = {"stream": "btcusdt@trade", "data": {"e": "trade"}}
        manager._on_message(None, json.dumps(message))

        assert len(bars_received) == 0

    def test_on_message_handles_invalid_json(self):
        """on_message handles invalid JSON gracefully."""
        manager = BinanceStreamManager(["BTC/USD"])
        # Should not raise
        manager._on_message(None, "not valid json{}")


class TestBarFromBinance:
    """Tests for Bar.from_binance() method."""

    @pytest.fixture
    def sample_kline_data(self):
        """Sample kline data dict."""
        return {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1704700800000,
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "100.5",
                "n": 1234,
            },
        }

    def test_from_binance_creates_bar(self, sample_kline_data):
        """Bar.from_binance creates Bar object."""
        bar = Bar.from_binance(sample_kline_data)
        assert isinstance(bar, Bar)

    def test_from_binance_symbol_conversion(self, sample_kline_data):
        """Symbol converted from BTCUSDT to BTC/USD."""
        bar = Bar.from_binance(sample_kline_data)
        assert bar.symbol == "BTC/USD"

    def test_from_binance_symbol_override(self, sample_kline_data):
        """symbol_override takes precedence."""
        bar = Bar.from_binance(sample_kline_data, symbol_override="BITCOIN/USD")
        assert bar.symbol == "BITCOIN/USD"

    def test_from_binance_ohlcv_values(self, sample_kline_data):
        """OHLCV values parsed correctly."""
        bar = Bar.from_binance(sample_kline_data)
        assert bar.open == 50000.00
        assert bar.high == 50100.00
        assert bar.low == 49900.00
        assert bar.close == 50050.00
        assert bar.volume == 100.5

    def test_from_binance_trade_count(self, sample_kline_data):
        """Trade count parsed."""
        bar = Bar.from_binance(sample_kline_data)
        assert bar.trade_count == 1234

    def test_from_binance_timestamp(self, sample_kline_data):
        """Timestamp converted from milliseconds."""
        bar = Bar.from_binance(sample_kline_data)
        assert isinstance(bar.timestamp, datetime)
        # 1704700800000 ms = 2024-01-08 08:00:00 UTC
        assert bar.timestamp.year == 2024

    def test_from_binance_exchange(self, sample_kline_data):
        """Exchange set to 'binance'."""
        bar = Bar.from_binance(sample_kline_data)
        assert bar.exchange == "binance"

    def test_from_binance_vwap_none(self, sample_kline_data):
        """VWAP is None (not provided by Binance)."""
        bar = Bar.from_binance(sample_kline_data)
        assert bar.vwap is None

    def test_from_binance_just_k_dict(self):
        """Works with just the 'k' dict (no outer wrapper)."""
        k_data = {
            "t": 1704700800000,
            "s": "ETHUSDT",
            "o": "3000.00",
            "h": "3050.00",
            "l": "2950.00",
            "c": "3025.00",
            "v": "500.0",
            "n": 100,
        }
        bar = Bar.from_binance({"k": k_data, "s": "ETHUSDT"})
        assert bar.close == 3025.00


class TestConnectionHandlers:
    """Tests for WebSocket connection event handlers."""

    def test_on_open_sets_connected(self):
        """on_open sets connected flag."""
        manager = BinanceStreamManager(["BTC/USD"])
        manager._on_open(None)
        assert manager._connected is True

    def test_on_open_resets_backoff(self):
        """on_open resets reconnect delay."""
        manager = BinanceStreamManager(["BTC/USD"])
        manager._reconnect_delay = 30.0  # Simulate previous reconnects
        manager._on_open(None)
        assert manager._reconnect_delay == manager.INITIAL_RECONNECT_DELAY

    def test_on_close_clears_connected(self):
        """on_close clears connected flag."""
        manager = BinanceStreamManager(["BTC/USD"])
        manager._connected = True
        manager._on_close(None, None, None)
        assert manager._connected is False

    def test_on_error_logs_but_doesnt_crash(self):
        """on_error handles errors gracefully."""
        manager = BinanceStreamManager(["BTC/USD"])
        # Should not raise
        manager._on_error(None, Exception("Test error"))


class TestStatsTracking:
    """Tests for connection statistics tracking."""

    def test_message_count_increments(self):
        """Message count increments on each message."""
        manager = BinanceStreamManager(["BTC/USD"])
        message = {"stream": "test", "data": {}}

        for _ in range(5):
            manager._on_message(None, json.dumps(message))

        stats = manager.get_stats()
        assert stats["messages_received"] == 5

    def test_uptime_calculated(self):
        """Uptime calculated when connected."""
        manager = BinanceStreamManager(["BTC/USD"])
        manager._on_open(None)

        stats = manager.get_stats()
        assert "uptime_seconds" in stats
        assert stats["uptime_seconds"] >= 0
