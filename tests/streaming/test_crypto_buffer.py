"""
Unit tests for CryptoBarBuffer.

Tests sliding window behavior, thread safety, and data retrieval.
"""

import pytest
import threading
import time
from datetime import datetime, timedelta

from src.streaming.crypto_buffer import CryptoBarBuffer
from src.streaming.types import Bar


class TestCryptoBarBuffer:
    """Tests for CryptoBarBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create a fresh buffer for each test."""
        return CryptoBarBuffer()

    @pytest.fixture
    def sample_bar(self):
        """Create a sample bar."""
        return Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            trade_count=1000,
            vwap=50025.0,
            exchange="binance",
        )

    def test_add_bar_creates_buffer(self, buffer, sample_bar):
        """Adding bar creates buffer for new symbol."""
        buffer.add_bar(sample_bar)
        assert buffer.has_symbol("BTC/USD")

    def test_add_bar_stores_bar(self, buffer, sample_bar):
        """Added bar can be retrieved."""
        buffer.add_bar(sample_bar)
        bars = buffer.get_bars("BTC/USD")
        assert len(bars) == 1
        assert bars[0].close == 50050.0

    def test_get_latest_returns_most_recent(self, buffer):
        """get_latest returns most recent bar."""
        for i in range(5):
            bar = Bar(
                symbol="BTC/USD",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=50000.0 + i,
                high=50100.0,
                low=49900.0,
                close=50000.0 + i * 10,
                volume=100.0,
            )
            buffer.add_bar(bar)

        latest = buffer.get_latest("BTC/USD")
        assert latest.close == 50040.0  # Last bar's close

    def test_get_latest_price(self, buffer, sample_bar):
        """get_latest_price returns close price."""
        buffer.add_bar(sample_bar)
        price = buffer.get_latest_price("BTC/USD")
        assert price == 50050.0

    def test_get_latest_price_missing_symbol(self, buffer):
        """get_latest_price returns None for missing symbol."""
        price = buffer.get_latest_price("UNKNOWN/USD")
        assert price is None


class TestSlidingWindow:
    """Tests for sliding window behavior."""

    @pytest.fixture
    def small_buffer(self):
        """Buffer with small capacity for testing eviction."""
        return CryptoBarBuffer(max_bars=10)

    def test_window_limit_enforced(self, small_buffer):
        """Buffer doesn't exceed max_bars."""
        for i in range(20):
            bar = Bar(
                symbol="BTC/USD",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50000.0 + i,
                volume=100.0,
            )
            small_buffer.add_bar(bar)

        assert small_buffer.count("BTC/USD") == 10

    def test_oldest_bar_evicted(self, small_buffer):
        """Oldest bar evicted when at capacity."""
        for i in range(15):
            bar = Bar(
                symbol="BTC/USD",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=float(i),  # Use index as close price
                volume=100.0,
            )
            small_buffer.add_bar(bar)

        bars = small_buffer.get_bars("BTC/USD")
        # Should have bars 5-14 (oldest 0-4 evicted)
        assert bars[0].close == 5.0
        assert bars[-1].close == 14.0

    def test_is_full(self, small_buffer):
        """is_full returns True when at capacity."""
        for i in range(10):
            bar = Bar(
                symbol="BTC/USD",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50000.0,
                volume=100.0,
            )
            small_buffer.add_bar(bar)

        assert small_buffer.is_full("BTC/USD") is True

    def test_not_full_initially(self, small_buffer):
        """is_full returns False for empty or partial buffer."""
        assert small_buffer.is_full("BTC/USD") is False

        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        small_buffer.add_bar(bar)
        assert small_buffer.is_full("BTC/USD") is False


class TestMultiSymbol:
    """Tests for multi-symbol handling."""

    @pytest.fixture
    def buffer(self):
        return CryptoBarBuffer()

    def test_multiple_symbols_isolated(self, buffer):
        """Bars for different symbols don't interfere."""
        btc_bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        eth_bar = Bar(
            symbol="ETH/USD",
            timestamp=datetime.now(),
            open=3000.0,
            high=3050.0,
            low=2950.0,
            close=3025.0,
            volume=500.0,
        )

        buffer.add_bar(btc_bar)
        buffer.add_bar(eth_bar)

        assert buffer.get_latest_price("BTC/USD") == 50050.0
        assert buffer.get_latest_price("ETH/USD") == 3025.0

    def test_get_all_prices(self, buffer):
        """get_all_prices returns all latest prices."""
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        prices = [50000.0, 3000.0, 100.0]

        for symbol, price in zip(symbols, prices):
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=100.0,
            )
            buffer.add_bar(bar)

        all_prices = buffer.get_all_prices()
        assert len(all_prices) == 3
        assert all_prices["BTC/USD"] == 50000.0
        assert all_prices["ETH/USD"] == 3000.0
        assert all_prices["SOL/USD"] == 100.0

    def test_symbols_list(self, buffer):
        """symbols() returns list of symbols in buffer."""
        symbols = ["BTC/USD", "ETH/USD"]
        for symbol in symbols:
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=1000.0,
                high=1010.0,
                low=990.0,
                close=1005.0,
                volume=100.0,
            )
            buffer.add_bar(bar)

        result = buffer.symbols()
        assert set(result) == set(symbols)


class TestDataFrame:
    """Tests for DataFrame conversion."""

    @pytest.fixture
    def buffer_with_data(self):
        buffer = CryptoBarBuffer()
        for i in range(5):
            bar = Bar(
                symbol="BTC/USD",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=50000.0 + i,
                high=50100.0 + i,
                low=49900.0 + i,
                close=50050.0 + i,
                volume=100.0 + i,
            )
            buffer.add_bar(bar)
        return buffer

    def test_to_dataframe_columns(self, buffer_with_data):
        """DataFrame has correct columns."""
        df = buffer_with_data.get_ohlcv_dataframe("BTC/USD")
        expected_cols = {"open", "high", "low", "close", "volume", "trade_count"}
        assert expected_cols.issubset(set(df.columns))

    def test_to_dataframe_indexed_by_timestamp(self, buffer_with_data):
        """DataFrame indexed by timestamp."""
        df = buffer_with_data.get_ohlcv_dataframe("BTC/USD")
        assert df.index.name == "timestamp"

    def test_to_dataframe_row_count(self, buffer_with_data):
        """DataFrame has correct number of rows."""
        df = buffer_with_data.get_ohlcv_dataframe("BTC/USD")
        assert len(df) == 5

    def test_to_dataframe_empty_symbol(self):
        """Empty DataFrame for missing symbol."""
        buffer = CryptoBarBuffer()
        df = buffer.get_ohlcv_dataframe("UNKNOWN/USD")
        assert len(df) == 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_writes(self):
        """Concurrent writes don't corrupt data."""
        buffer = CryptoBarBuffer()
        errors = []

        def add_bars(symbol, count):
            try:
                for i in range(count):
                    bar = Bar(
                        symbol=symbol,
                        timestamp=datetime.now() + timedelta(seconds=i),
                        open=1000.0,
                        high=1010.0,
                        low=990.0,
                        close=1005.0,
                        volume=100.0,
                    )
                    buffer.add_bar(bar)
            except Exception as e:
                errors.append(e)

        threads = []
        for symbol in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            t = threading.Thread(target=add_bars, args=(symbol, 100))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert buffer.count("BTC/USD") == 100
        assert buffer.count("ETH/USD") == 100
        assert buffer.count("SOL/USD") == 100

    def test_concurrent_read_write(self):
        """Concurrent reads and writes don't cause issues."""
        buffer = CryptoBarBuffer()
        errors = []
        stop_flag = threading.Event()

        def writer():
            try:
                for i in range(100):
                    bar = Bar(
                        symbol="BTC/USD",
                        timestamp=datetime.now() + timedelta(seconds=i),
                        open=1000.0,
                        high=1010.0,
                        low=990.0,
                        close=1005.0 + i,
                        volume=100.0,
                    )
                    buffer.add_bar(bar)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
            finally:
                stop_flag.set()

        def reader():
            try:
                while not stop_flag.is_set():
                    _ = buffer.get_latest("BTC/USD")
                    _ = buffer.get_bars("BTC/USD")
                    _ = buffer.get_ohlcv_dataframe("BTC/USD")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        for t in reader_threads:
            t.start()
        writer_thread.start()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert len(errors) == 0


class TestClearAndStats:
    """Tests for clear and stats methods."""

    @pytest.fixture
    def buffer_with_data(self):
        buffer = CryptoBarBuffer()
        for symbol in ["BTC/USD", "ETH/USD"]:
            for i in range(10):
                bar = Bar(
                    symbol=symbol,
                    timestamp=datetime.now() + timedelta(minutes=i),
                    open=1000.0,
                    high=1010.0,
                    low=990.0,
                    close=1005.0,
                    volume=100.0,
                )
                buffer.add_bar(bar)
        return buffer

    def test_clear_single_symbol(self, buffer_with_data):
        """Clear single symbol leaves others."""
        buffer_with_data.clear("BTC/USD")
        assert buffer_with_data.count("BTC/USD") == 0
        assert buffer_with_data.count("ETH/USD") == 10

    def test_clear_all(self, buffer_with_data):
        """Clear all removes everything."""
        buffer_with_data.clear()
        assert buffer_with_data.count("BTC/USD") == 0
        assert buffer_with_data.count("ETH/USD") == 0

    def test_total_bars(self, buffer_with_data):
        """total_bars counts across all symbols."""
        assert buffer_with_data.total_bars() == 20

    def test_get_stats(self, buffer_with_data):
        """get_stats returns comprehensive stats."""
        stats = buffer_with_data.get_stats()
        assert stats["symbols"] == 2
        assert stats["total_bars"] == 20
        assert "memory_mb" in stats
        assert "per_symbol" in stats

    def test_time_range(self, buffer_with_data):
        """time_range returns first and last timestamps."""
        time_range = buffer_with_data.time_range("BTC/USD")
        assert time_range is not None
        oldest, newest = time_range
        assert oldest < newest

    def test_time_range_missing_symbol(self, buffer_with_data):
        """time_range returns None for missing symbol."""
        assert buffer_with_data.time_range("UNKNOWN/USD") is None
