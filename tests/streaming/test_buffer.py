"""Unit tests for streaming data buffers."""

import threading
import time
from datetime import datetime, timezone, timedelta

import pytest

from src.streaming.types import Bar, Quote, Trade
from src.streaming._buffer import BarBuffer, QuoteBuffer, TradeBuffer


def make_bar(symbol: str, minute_offset: int = 0, close: float = 100.0) -> Bar:
    """Helper to create a bar for testing."""
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return Bar(
        symbol=symbol,
        timestamp=base_time + timedelta(minutes=minute_offset),
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=10000.0,
        vwap=close,
    )


def make_quote(symbol: str, bid: float = 100.0, ask: float = 100.05) -> Quote:
    """Helper to create a quote for testing."""
    return Quote(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        bid_price=bid,
        bid_size=100.0,
        ask_price=ask,
        ask_size=100.0,
    )


def make_trade(symbol: str, price: float = 100.0) -> Trade:
    """Helper to create a trade for testing."""
    return Trade(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        price=price,
        size=100.0,
    )


class TestBarBuffer:
    """Tests for BarBuffer."""

    def test_add_and_get_latest(self):
        """Test adding bars and getting the latest."""
        buffer = BarBuffer(max_bars_per_symbol=100)
        bar1 = make_bar("AAPL", minute_offset=0)
        bar2 = make_bar("AAPL", minute_offset=1)

        buffer.add(bar1)
        buffer.add(bar2)

        latest = buffer.get_latest("AAPL")
        assert latest == bar2

    def test_get_latest_empty(self):
        """Test getting latest from empty buffer."""
        buffer = BarBuffer()
        assert buffer.get_latest("AAPL") is None

    def test_get_bars(self):
        """Test getting multiple bars."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        for i in range(10):
            buffer.add(make_bar("AAPL", minute_offset=i, close=100 + i))

        bars = buffer.get_bars("AAPL")
        assert len(bars) == 10
        assert bars[0].close == 100
        assert bars[-1].close == 109

    def test_get_bars_with_limit(self):
        """Test getting limited number of bars."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        for i in range(10):
            buffer.add(make_bar("AAPL", minute_offset=i, close=100 + i))

        bars = buffer.get_bars("AAPL", n=3)
        assert len(bars) == 3
        assert bars[-1].close == 109  # Most recent

    def test_get_bars_empty(self):
        """Test getting bars from empty buffer."""
        buffer = BarBuffer()
        bars = buffer.get_bars("AAPL")
        assert bars == []

    def test_respects_max_size(self):
        """Test that buffer evicts old bars when at capacity."""
        buffer = BarBuffer(max_bars_per_symbol=5)

        for i in range(10):
            buffer.add(make_bar("AAPL", minute_offset=i, close=100 + i))

        bars = buffer.get_bars("AAPL")
        assert len(bars) == 5
        assert bars[0].close == 105  # First 5 evicted
        assert bars[-1].close == 109

    def test_multiple_symbols(self):
        """Test buffer handles multiple symbols independently."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        buffer.add(make_bar("AAPL", close=150))
        buffer.add(make_bar("GOOGL", close=140))
        buffer.add(make_bar("AAPL", close=151))

        assert buffer.get_latest("AAPL").close == 151
        assert buffer.get_latest("GOOGL").close == 140
        assert len(buffer.get_bars("AAPL")) == 2
        assert len(buffer.get_bars("GOOGL")) == 1

    def test_to_dataframe(self):
        """Test converting bars to DataFrame."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        for i in range(5):
            buffer.add(make_bar("AAPL", minute_offset=i, close=100 + i))

        df = buffer.to_dataframe("AAPL")

        assert len(df) == 5
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["close"].iloc[-1] == 104

    def test_to_dataframe_empty(self):
        """Test DataFrame from empty buffer."""
        buffer = BarBuffer()
        df = buffer.to_dataframe("AAPL")
        assert df.empty

    def test_clear_symbol(self):
        """Test clearing a specific symbol."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        buffer.add(make_bar("AAPL"))
        buffer.add(make_bar("GOOGL"))

        buffer.clear("AAPL")

        assert buffer.get_latest("AAPL") is None
        assert buffer.get_latest("GOOGL") is not None

    def test_clear_all(self):
        """Test clearing all symbols."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        buffer.add(make_bar("AAPL"))
        buffer.add(make_bar("GOOGL"))

        buffer.clear()

        assert buffer.get_latest("AAPL") is None
        assert buffer.get_latest("GOOGL") is None

    def test_symbols(self):
        """Test getting list of symbols."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        buffer.add(make_bar("AAPL"))
        buffer.add(make_bar("GOOGL"))
        buffer.add(make_bar("MSFT"))

        symbols = buffer.symbols()
        assert set(symbols) == {"AAPL", "GOOGL", "MSFT"}

    def test_count(self):
        """Test counting bars for a symbol."""
        buffer = BarBuffer(max_bars_per_symbol=100)

        for i in range(5):
            buffer.add(make_bar("AAPL", minute_offset=i))

        assert buffer.count("AAPL") == 5
        assert buffer.count("GOOGL") == 0

    def test_thread_safety(self):
        """Test buffer is thread-safe."""
        buffer = BarBuffer(max_bars_per_symbol=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    buffer.add(make_bar("AAPL", minute_offset=i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    buffer.get_latest("AAPL")
                    buffer.get_bars("AAPL", n=10)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestQuoteBuffer:
    """Tests for QuoteBuffer."""

    def test_update_and_get(self):
        """Test updating and getting quotes."""
        buffer = QuoteBuffer()
        quote = make_quote("AAPL", bid=150.0, ask=150.05)

        buffer.update(quote)
        retrieved = buffer.get("AAPL")

        assert retrieved == quote

    def test_get_empty(self):
        """Test getting from empty buffer."""
        buffer = QuoteBuffer()
        assert buffer.get("AAPL") is None

    def test_update_replaces(self):
        """Test that update replaces old quote."""
        buffer = QuoteBuffer()

        buffer.update(make_quote("AAPL", bid=100.0, ask=100.10))
        buffer.update(make_quote("AAPL", bid=101.0, ask=101.10))

        quote = buffer.get("AAPL")
        assert quote.bid_price == 101.0

    def test_get_price(self):
        """Test getting mid price."""
        buffer = QuoteBuffer()
        buffer.update(make_quote("AAPL", bid=100.0, ask=100.10))

        price = buffer.get_price("AAPL")
        assert price == 100.05

    def test_get_price_empty(self):
        """Test getting price from empty buffer."""
        buffer = QuoteBuffer()
        assert buffer.get_price("AAPL") is None

    def test_get_spread(self):
        """Test getting spread."""
        buffer = QuoteBuffer()
        buffer.update(make_quote("AAPL", bid=100.0, ask=100.10))

        spread = buffer.get_spread("AAPL")
        assert spread == pytest.approx(0.10, rel=1e-9)

    def test_get_spread_empty(self):
        """Test getting spread from empty buffer."""
        buffer = QuoteBuffer()
        assert buffer.get_spread("AAPL") is None

    def test_multiple_symbols(self):
        """Test buffer handles multiple symbols."""
        buffer = QuoteBuffer()

        buffer.update(make_quote("AAPL", bid=150.0, ask=150.05))
        buffer.update(make_quote("GOOGL", bid=140.0, ask=140.05))

        assert buffer.get("AAPL").bid_price == 150.0
        assert buffer.get("GOOGL").bid_price == 140.0

    def test_clear_symbol(self):
        """Test clearing a specific symbol."""
        buffer = QuoteBuffer()

        buffer.update(make_quote("AAPL"))
        buffer.update(make_quote("GOOGL"))

        buffer.clear("AAPL")

        assert buffer.get("AAPL") is None
        assert buffer.get("GOOGL") is not None

    def test_clear_all(self):
        """Test clearing all symbols."""
        buffer = QuoteBuffer()

        buffer.update(make_quote("AAPL"))
        buffer.update(make_quote("GOOGL"))

        buffer.clear()

        assert buffer.get("AAPL") is None
        assert buffer.get("GOOGL") is None

    def test_symbols(self):
        """Test getting list of symbols."""
        buffer = QuoteBuffer()

        buffer.update(make_quote("AAPL"))
        buffer.update(make_quote("GOOGL"))

        symbols = buffer.symbols()
        assert set(symbols) == {"AAPL", "GOOGL"}

    def test_thread_safety(self):
        """Test buffer is thread-safe."""
        buffer = QuoteBuffer()
        errors = []

        def writer():
            try:
                for i in range(100):
                    buffer.update(make_quote("AAPL", bid=100 + i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    buffer.get("AAPL")
                    buffer.get_price("AAPL")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestTradeBuffer:
    """Tests for TradeBuffer."""

    def test_update_and_get(self):
        """Test updating and getting trades."""
        buffer = TradeBuffer()
        trade = make_trade("AAPL", price=150.0)

        buffer.update(trade)
        retrieved = buffer.get("AAPL")

        assert retrieved == trade

    def test_get_empty(self):
        """Test getting from empty buffer."""
        buffer = TradeBuffer()
        assert buffer.get("AAPL") is None

    def test_get_price(self):
        """Test getting last trade price."""
        buffer = TradeBuffer()
        buffer.update(make_trade("AAPL", price=150.50))

        price = buffer.get_price("AAPL")
        assert price == 150.50

    def test_get_price_empty(self):
        """Test getting price from empty buffer."""
        buffer = TradeBuffer()
        assert buffer.get_price("AAPL") is None

    def test_clear_and_symbols(self):
        """Test clearing and symbol listing."""
        buffer = TradeBuffer()

        buffer.update(make_trade("AAPL"))
        buffer.update(make_trade("GOOGL"))

        assert set(buffer.symbols()) == {"AAPL", "GOOGL"}

        buffer.clear("AAPL")
        assert buffer.get("AAPL") is None
        assert buffer.get("GOOGL") is not None
