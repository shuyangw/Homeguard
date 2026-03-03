"""Tests for OptionsDataLoader - Hive-partitioned options parquet reader."""

from datetime import date, time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.strategies.options.data_loader import OptionsDataLoader


def _create_sample_parquet(base_dir, symbol: str, year: int, month: int):
    """Create minimal parquet files with 2 timestamps, each having 1 put and 1 call."""
    symbol_dir = base_dir / f"root={symbol}" / f"year={year}" / f"month={month}"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "timestamp": f"{year}-{month:02d}-15T15:55:00",
            "expiration": f"{year}-{month:02d}-28",
            "strike": 150.0,
            "right": "PUT",
            "open": 2.0,
            "high": 2.5,
            "low": 1.8,
            "close": 2.1,
            "volume": 100,
            "trade_count": 10,
            "vwap": 2.05,
            "bid_close": 2.0,
            "ask_close": 2.2,
            "implied_vol": 0.25,
            "delta": -0.30,
            "theta": -0.05,
            "vega": 0.15,
            "underlying_px": 155.0,
            "gamma_eod": 0.02,
            "open_interest_eod": 500,
        },
        {
            "timestamp": f"{year}-{month:02d}-15T15:55:00",
            "expiration": f"{year}-{month:02d}-28",
            "strike": 160.0,
            "right": "CALL",
            "open": 1.5,
            "high": 1.8,
            "low": 1.3,
            "close": 1.6,
            "volume": 200,
            "trade_count": 20,
            "vwap": 1.55,
            "bid_close": 1.5,
            "ask_close": 1.7,
            "implied_vol": 0.22,
            "delta": 0.45,
            "theta": -0.04,
            "vega": 0.12,
            "underlying_px": 155.0,
            "gamma_eod": 0.03,
            "open_interest_eod": 800,
        },
        {
            "timestamp": f"{year}-{month:02d}-15T16:00:00",
            "expiration": f"{year}-{month:02d}-28",
            "strike": 150.0,
            "right": "PUT",
            "open": 2.1,
            "high": 2.6,
            "low": 1.9,
            "close": 2.3,
            "volume": 150,
            "trade_count": 15,
            "vwap": 2.15,
            "bid_close": 2.2,
            "ask_close": 2.4,
            "implied_vol": 0.26,
            "delta": -0.32,
            "theta": -0.05,
            "vega": 0.15,
            "underlying_px": 154.5,
            "gamma_eod": 0.02,
            "open_interest_eod": 520,
        },
        {
            "timestamp": f"{year}-{month:02d}-15T16:00:00",
            "expiration": f"{year}-{month:02d}-28",
            "strike": 160.0,
            "right": "CALL",
            "open": 1.4,
            "high": 1.7,
            "low": 1.2,
            "close": 1.5,
            "volume": 250,
            "trade_count": 25,
            "vwap": 1.45,
            "bid_close": 1.4,
            "ask_close": 1.6,
            "implied_vol": 0.23,
            "delta": 0.43,
            "theta": -0.04,
            "vega": 0.12,
            "underlying_px": 154.5,
            "gamma_eod": 0.03,
            "open_interest_eod": 830,
        },
    ]

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, symbol_dir / "data.parquet")


@pytest.fixture
def loader_with_data(tmp_path):
    """Create sample data and return an OptionsDataLoader instance."""
    _create_sample_parquet(tmp_path, "AAPL", 2024, 6)
    _create_sample_parquet(tmp_path, "MSFT", 2024, 6)
    loader = OptionsDataLoader(data_dir=tmp_path)
    return loader


class TestGetAvailableSymbols:

    def test_get_available_symbols(self, loader_with_data):
        symbols = loader_with_data.get_available_symbols()
        assert isinstance(symbols, list)
        assert set(symbols) == {"AAPL", "MSFT"}


class TestGetEodChain:

    def test_get_eod_chain(self, loader_with_data):
        df = loader_with_data.get_eod_chain("AAPL", date(2024, 6, 15))

        assert not df.empty
        assert len(df) == 2

        expected_cols = {
            "datetime", "date", "time", "expiry", "days_to_expiry",
            "strike", "option_type", "open", "high", "low", "close",
            "volume", "trade_count", "vwap",
            "bid", "ask", "mid_price",
            "implied_vol", "delta", "theta", "vega", "gamma",
            "underlying_price", "open_interest",
        }
        assert expected_cols.issubset(set(df.columns))

        put_row = df[df["option_type"] == "P"].iloc[0]
        assert put_row["bid"] == 2.2
        assert put_row["ask"] == 2.4
        assert put_row["gamma"] == 0.02
        assert put_row["open_interest"] == 520
        assert put_row["underlying_price"] == 154.5

        call_row = df[df["option_type"] == "C"].iloc[0]
        assert call_row["option_type"] == "C"
        assert call_row["strike"] == 160.0

        assert put_row["expiry"] == date(2024, 6, 28)
        assert put_row["date"] == date(2024, 6, 15)
        assert put_row["time"] == time(16, 0)

        assert put_row["days_to_expiry"] == 13
        assert put_row["mid_price"] == pytest.approx(2.3)


class TestGetChainAtTime:

    def test_get_chain_at_time(self, loader_with_data):
        df = loader_with_data.get_chain_at_time(
            "AAPL", date(2024, 6, 15), time(15, 55)
        )

        assert not df.empty
        assert len(df) == 2

        put_row = df[df["option_type"] == "P"].iloc[0]
        assert put_row["bid"] == 2.0
        assert put_row["ask"] == 2.2
        assert put_row["underlying_price"] == 155.0
        assert put_row["time"] == time(15, 55)


class TestMissingData:

    def test_missing_symbol_returns_empty(self, loader_with_data):
        df = loader_with_data.get_eod_chain("UNKNOWN", date(2024, 6, 15))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_date_returns_empty(self, loader_with_data):
        df = loader_with_data.get_eod_chain("AAPL", date(2025, 1, 1))
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestCache:

    def test_month_cache_reused(self, loader_with_data):
        loader_with_data.get_eod_chain("AAPL", date(2024, 6, 15))
        assert ("AAPL", 2024, 6) in loader_with_data._month_cache


class TestGetDateRange:

    def test_get_date_range(self, loader_with_data):
        min_date, max_date = loader_with_data.get_date_range("AAPL")
        assert min_date == date(2024, 6, 15)
        assert max_date == date(2024, 6, 15)

    def test_get_date_range_unknown_symbol(self, loader_with_data):
        result = loader_with_data.get_date_range("UNKNOWN")
        assert result == (None, None)
