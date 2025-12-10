"""Unit tests for streaming data types."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.streaming.types import Bar, Quote, Trade


class TestBar:
    """Tests for Bar dataclass."""

    def test_create_bar(self):
        """Test creating a Bar with all fields."""
        bar = Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            open=150.0,
            high=152.5,
            low=149.5,
            close=151.0,
            volume=1000000.0,
            trade_count=5000.0,
            vwap=150.75,
            exchange="NASDAQ",
        )

        assert bar.symbol == "AAPL"
        assert bar.open == 150.0
        assert bar.high == 152.5
        assert bar.low == 149.5
        assert bar.close == 151.0
        assert bar.volume == 1000000.0
        assert bar.trade_count == 5000.0
        assert bar.vwap == 150.75
        assert bar.exchange == "NASDAQ"

    def test_create_bar_minimal(self):
        """Test creating a Bar with only required fields."""
        bar = Bar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=152.5,
            low=149.5,
            close=151.0,
            volume=1000000.0,
        )

        assert bar.symbol == "AAPL"
        assert bar.trade_count is None
        assert bar.vwap is None
        assert bar.exchange is None

    def test_from_alpaca(self):
        """Test conversion from Alpaca Bar model."""
        mock_alpaca_bar = MagicMock()
        mock_alpaca_bar.symbol = "TQQQ"
        mock_alpaca_bar.timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        mock_alpaca_bar.open = 45.0
        mock_alpaca_bar.high = 46.0
        mock_alpaca_bar.low = 44.5
        mock_alpaca_bar.close = 45.5
        mock_alpaca_bar.volume = 500000.0
        mock_alpaca_bar.trade_count = 2500.0
        mock_alpaca_bar.vwap = 45.25
        mock_alpaca_bar.exchange = "NYSE"

        bar = Bar.from_alpaca(mock_alpaca_bar)

        assert bar.symbol == "TQQQ"
        assert bar.open == 45.0
        assert bar.high == 46.0
        assert bar.low == 44.5
        assert bar.close == 45.5
        assert bar.volume == 500000.0
        assert bar.trade_count == 2500.0
        assert bar.vwap == 45.25
        assert bar.exchange == "NYSE"

    def test_from_alpaca_missing_optional_fields(self):
        """Test conversion when optional fields are missing."""
        mock_alpaca_bar = MagicMock(spec=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        mock_alpaca_bar.symbol = "SPY"
        mock_alpaca_bar.timestamp = datetime.now(timezone.utc)
        mock_alpaca_bar.open = 450.0
        mock_alpaca_bar.high = 452.0
        mock_alpaca_bar.low = 449.0
        mock_alpaca_bar.close = 451.0
        mock_alpaca_bar.volume = 10000000.0

        bar = Bar.from_alpaca(mock_alpaca_bar)

        assert bar.symbol == "SPY"
        assert bar.trade_count is None
        assert bar.vwap is None
        assert bar.exchange is None


class TestQuote:
    """Tests for Quote dataclass."""

    def test_create_quote(self):
        """Test creating a Quote with all fields."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            bid_price=150.0,
            bid_size=100.0,
            ask_price=150.05,
            ask_size=200.0,
            bid_exchange="NASDAQ",
            ask_exchange="NASDAQ",
            conditions=["R"],
            tape="C",
        )

        assert quote.symbol == "AAPL"
        assert quote.bid_price == 150.0
        assert quote.bid_size == 100.0
        assert quote.ask_price == 150.05
        assert quote.ask_size == 200.0
        assert quote.bid_exchange == "NASDAQ"
        assert quote.ask_exchange == "NASDAQ"
        assert quote.conditions == ["R"]
        assert quote.tape == "C"

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid_price=100.0,
            bid_size=100.0,
            ask_price=100.10,
            ask_size=100.0,
        )

        assert quote.mid == 100.05

    def test_spread(self):
        """Test spread calculation."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid_price=100.0,
            bid_size=100.0,
            ask_price=100.10,
            ask_size=100.0,
        )

        assert quote.spread == pytest.approx(0.10, rel=1e-9)

    def test_from_alpaca(self):
        """Test conversion from Alpaca Quote model."""
        mock_alpaca_quote = MagicMock()
        mock_alpaca_quote.symbol = "TQQQ"
        mock_alpaca_quote.timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        mock_alpaca_quote.bid_price = 45.00
        mock_alpaca_quote.bid_size = 500.0
        mock_alpaca_quote.ask_price = 45.02
        mock_alpaca_quote.ask_size = 300.0
        mock_alpaca_quote.bid_exchange = "NYSE"
        mock_alpaca_quote.ask_exchange = "NASDAQ"
        mock_alpaca_quote.conditions = ["R", "I"]
        mock_alpaca_quote.tape = "A"

        quote = Quote.from_alpaca(mock_alpaca_quote)

        assert quote.symbol == "TQQQ"
        assert quote.bid_price == 45.00
        assert quote.ask_price == 45.02
        assert quote.mid == pytest.approx(45.01, rel=1e-9)
        assert quote.spread == pytest.approx(0.02, rel=1e-9)
        assert quote.conditions == ["R", "I"]

    def test_from_alpaca_missing_optional_fields(self):
        """Test conversion when optional fields are missing."""
        mock_alpaca_quote = MagicMock(spec=["symbol", "timestamp", "bid_price", "bid_size", "ask_price", "ask_size"])
        mock_alpaca_quote.symbol = "SPY"
        mock_alpaca_quote.timestamp = datetime.now(timezone.utc)
        mock_alpaca_quote.bid_price = 450.0
        mock_alpaca_quote.bid_size = 1000.0
        mock_alpaca_quote.ask_price = 450.01
        mock_alpaca_quote.ask_size = 1500.0

        quote = Quote.from_alpaca(mock_alpaca_quote)

        assert quote.symbol == "SPY"
        assert quote.bid_exchange is None
        assert quote.ask_exchange is None
        assert quote.conditions is None
        assert quote.tape is None


class TestTrade:
    """Tests for Trade dataclass."""

    def test_create_trade(self):
        """Test creating a Trade with all fields."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            price=150.50,
            size=100.0,
            exchange="NASDAQ",
            id=12345,
            conditions=["@", "I"],
            tape="C",
        )

        assert trade.symbol == "AAPL"
        assert trade.price == 150.50
        assert trade.size == 100.0
        assert trade.exchange == "NASDAQ"
        assert trade.id == 12345
        assert trade.conditions == ["@", "I"]
        assert trade.tape == "C"

    def test_create_trade_minimal(self):
        """Test creating a Trade with only required fields."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=150.50,
            size=100.0,
        )

        assert trade.symbol == "AAPL"
        assert trade.exchange is None
        assert trade.id is None
        assert trade.conditions is None
        assert trade.tape is None

    def test_from_alpaca(self):
        """Test conversion from Alpaca Trade model."""
        mock_alpaca_trade = MagicMock()
        mock_alpaca_trade.symbol = "TQQQ"
        mock_alpaca_trade.timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        mock_alpaca_trade.price = 45.25
        mock_alpaca_trade.size = 1000.0
        mock_alpaca_trade.exchange = "NYSE"
        mock_alpaca_trade.id = 98765
        mock_alpaca_trade.conditions = ["@"]
        mock_alpaca_trade.tape = "A"

        trade = Trade.from_alpaca(mock_alpaca_trade)

        assert trade.symbol == "TQQQ"
        assert trade.price == 45.25
        assert trade.size == 1000.0
        assert trade.exchange == "NYSE"
        assert trade.id == 98765
        assert trade.conditions == ["@"]
        assert trade.tape == "A"

    def test_from_alpaca_missing_optional_fields(self):
        """Test conversion when optional fields are missing."""
        mock_alpaca_trade = MagicMock(spec=["symbol", "timestamp", "price", "size"])
        mock_alpaca_trade.symbol = "SPY"
        mock_alpaca_trade.timestamp = datetime.now(timezone.utc)
        mock_alpaca_trade.price = 451.50
        mock_alpaca_trade.size = 500.0

        trade = Trade.from_alpaca(mock_alpaca_trade)

        assert trade.symbol == "SPY"
        assert trade.exchange is None
        assert trade.id is None
        assert trade.conditions is None
        assert trade.tape is None
