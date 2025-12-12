"""Tests for BarAccumulator."""

from datetime import datetime

import pandas as pd
import pytest

from src.backtesting.adapters.nautilus.bar_accumulator import BarAccumulator
from tests.backtesting.adapters.nautilus.mocks import MockBar


class TestBarAccumulatorInit:
    """Tests for BarAccumulator initialization."""

    def test_init_valid_max_periods(self):
        """Test initialization with valid max_periods."""
        accumulator = BarAccumulator(max_periods=100)
        assert accumulator.max_periods == 100

    def test_init_minimum_periods(self):
        """Test initialization with minimum max_periods."""
        accumulator = BarAccumulator(max_periods=1)
        assert accumulator.max_periods == 1

    def test_init_invalid_periods_raises(self):
        """Test that zero or negative periods raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            BarAccumulator(max_periods=0)

        with pytest.raises(ValueError, match="at least 1"):
            BarAccumulator(max_periods=-1)


class TestBarAccumulatorAddBar:
    """Tests for adding bars to accumulator."""

    def test_add_single_bar(self):
        """Test adding a single bar."""
        accumulator = BarAccumulator(max_periods=100)
        bar = MockBar.from_ohlcv(
            symbol="AAPL",
            open_price=185.0,
            high=186.0,
            low=184.0,
            close=185.5,
            volume=10000.0,
            timestamp=datetime(2024, 1, 2, 9, 30),
        )

        accumulator.add_bar("AAPL", bar)

        assert accumulator.get_bar_count("AAPL") == 1
        assert "AAPL" in accumulator.get_symbols()

    def test_add_multiple_bars_same_symbol(self):
        """Test adding multiple bars for same symbol."""
        accumulator = BarAccumulator(max_periods=100)

        for i in range(10):
            bar = MockBar.from_ohlcv(
                symbol="AAPL",
                open_price=185.0 + i,
                high=186.0 + i,
                low=184.0 + i,
                close=185.5 + i,
                volume=10000.0 + i * 100,
                timestamp=datetime(2024, 1, 2, 9, 30 + i),
            )
            accumulator.add_bar("AAPL", bar)

        assert accumulator.get_bar_count("AAPL") == 10

    def test_add_bars_multiple_symbols(self):
        """Test adding bars for multiple symbols."""
        accumulator = BarAccumulator(max_periods=100)

        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            bar = MockBar.from_ohlcv(
                symbol=symbol,
                open_price=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30),
            )
            accumulator.add_bar(symbol, bar)

        assert len(accumulator.get_symbols()) == 3
        assert accumulator.get_bar_count("AAPL") == 1
        assert accumulator.get_bar_count("MSFT") == 1
        assert accumulator.get_bar_count("GOOGL") == 1

    def test_rolling_window_drops_old_bars(self):
        """Test that old bars are dropped when max_periods exceeded."""
        accumulator = BarAccumulator(max_periods=5)

        for i in range(10):
            bar = MockBar.from_ohlcv(
                symbol="AAPL",
                open_price=185.0 + i,
                high=186.0 + i,
                low=184.0 + i,
                close=185.5 + i,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30 + i),
            )
            accumulator.add_bar("AAPL", bar)

        # Should only keep last 5 bars
        assert accumulator.get_bar_count("AAPL") == 5

        # Latest bar should have close = 185.5 + 9 = 194.5
        latest = accumulator.get_latest_bar("AAPL")
        assert latest["close"] == 194.5


class TestBarAccumulatorDataFrame:
    """Tests for DataFrame generation."""

    def test_get_dataframe_empty_symbol(self):
        """Test getting DataFrame for unknown symbol returns None."""
        accumulator = BarAccumulator(max_periods=100)
        df = accumulator.get_dataframe("UNKNOWN")
        assert df is None

    def test_get_dataframe_schema(self):
        """Test DataFrame has correct schema."""
        accumulator = BarAccumulator(max_periods=100)

        for i in range(5):
            bar = MockBar.from_ohlcv(
                symbol="AAPL",
                open_price=185.0 + i,
                high=186.0 + i,
                low=184.0 + i,
                close=185.5 + i,
                volume=10000.0 + i * 100,
                timestamp=datetime(2024, 1, 2, 9, 30 + i),
            )
            accumulator.add_bar("AAPL", bar)

        df = accumulator.get_dataframe("AAPL")

        # Check columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check index is timestamp
        assert df.index.name == "timestamp"
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check length
        assert len(df) == 5

    def test_get_all_dataframes(self):
        """Test getting DataFrames for all symbols."""
        accumulator = BarAccumulator(max_periods=100)

        for symbol in ["AAPL", "MSFT"]:
            bar = MockBar.from_ohlcv(
                symbol=symbol,
                open_price=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30),
            )
            accumulator.add_bar(symbol, bar)

        all_dfs = accumulator.get_all_dataframes()

        assert len(all_dfs) == 2
        assert "AAPL" in all_dfs
        assert "MSFT" in all_dfs
        assert isinstance(all_dfs["AAPL"], pd.DataFrame)


class TestBarAccumulatorSufficientData:
    """Tests for checking sufficient data."""

    def test_has_sufficient_data_true(self):
        """Test sufficient data check returns True when enough bars."""
        accumulator = BarAccumulator(max_periods=100)

        for i in range(20):
            bar = MockBar.from_ohlcv(
                symbol="AAPL",
                open_price=185.0 + i,
                high=186.0 + i,
                low=184.0 + i,
                close=185.5 + i,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30 + i),
            )
            accumulator.add_bar("AAPL", bar)

        assert accumulator.has_sufficient_data("AAPL", required_periods=10)
        assert accumulator.has_sufficient_data("AAPL", required_periods=20)

    def test_has_sufficient_data_false(self):
        """Test sufficient data check returns False when not enough bars."""
        accumulator = BarAccumulator(max_periods=100)

        for i in range(5):
            bar = MockBar.from_ohlcv(
                symbol="AAPL",
                open_price=185.0 + i,
                high=186.0 + i,
                low=184.0 + i,
                close=185.5 + i,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30 + i),
            )
            accumulator.add_bar("AAPL", bar)

        assert not accumulator.has_sufficient_data("AAPL", required_periods=10)

    def test_has_sufficient_data_unknown_symbol(self):
        """Test sufficient data check returns False for unknown symbol."""
        accumulator = BarAccumulator(max_periods=100)
        assert not accumulator.has_sufficient_data("UNKNOWN", required_periods=1)


class TestBarAccumulatorClear:
    """Tests for clearing accumulated bars."""

    def test_clear_single_symbol(self):
        """Test clearing bars for a single symbol."""
        accumulator = BarAccumulator(max_periods=100)

        for symbol in ["AAPL", "MSFT"]:
            bar = MockBar.from_ohlcv(
                symbol=symbol,
                open_price=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30),
            )
            accumulator.add_bar(symbol, bar)

        accumulator.clear("AAPL")

        assert accumulator.get_bar_count("AAPL") == 0
        assert accumulator.get_bar_count("MSFT") == 1

    def test_clear_all(self):
        """Test clearing all bars."""
        accumulator = BarAccumulator(max_periods=100)

        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            bar = MockBar.from_ohlcv(
                symbol=symbol,
                open_price=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10000.0,
                timestamp=datetime(2024, 1, 2, 9, 30),
            )
            accumulator.add_bar(symbol, bar)

        accumulator.clear()

        assert len(accumulator.get_symbols()) == 0
