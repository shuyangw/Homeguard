"""Tests for Databento futures plugin."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA, FUTURES_TRADES_SCHEMA


class TestSymbolNormalization:
    def test_symbol_to_api_format(self):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        plugin = DatabentoFuturesPlugin.__new__(DatabentoFuturesPlugin)
        assert plugin._to_api_symbol("ES") == "ES.c.0"
        assert plugin._to_api_symbol("NQ") == "NQ.c.0"
        assert plugin._to_api_symbol("6E") == "6E.c.0"

    def test_already_continuous_format_passthrough(self):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        plugin = DatabentoFuturesPlugin.__new__(DatabentoFuturesPlugin)
        assert plugin._to_api_symbol("ES.c.0") == "ES.c.0"

    def test_normalize_symbol_for_filesystem(self):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        plugin = DatabentoFuturesPlugin.__new__(DatabentoFuturesPlugin)
        assert plugin._normalize_symbol("ES") == "ES"
        assert plugin._normalize_symbol("6E") == "6E"

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_storage_subdir_default_ohlcv(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(output_dir=Path(tmpdir))
            assert plugin._get_storage_subdir() == "futures/databento/1min"

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_storage_subdir_trades(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(
                output_dir=Path(tmpdir), schema="trades"
            )
            assert plugin._get_storage_subdir() == "futures/databento/trades"

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_schema_default_ohlcv(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(output_dir=Path(tmpdir))
            schema = plugin._get_schema()
            assert schema == CANONICAL_OHLCV_SCHEMA

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_schema_trades(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(
                output_dir=Path(tmpdir), schema="trades"
            )
            schema = plugin._get_schema()
            assert schema == FUTURES_TRADES_SCHEMA


class TestMissingApiKey:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
                DatabentoFuturesPlugin(output_dir=Path(tmpdir))

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_invalid_schema_raises(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="schema must be"):
                DatabentoFuturesPlugin(
                    output_dir=Path(tmpdir), schema="invalid"
                )


class TestFetchOhlcv:
    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_fetch_returns_ohlcv_dataframe(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client

        mock_data = MagicMock()
        mock_df = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(
                    [
                        "2024-01-02 09:30:00",
                        "2024-01-02 09:31:00",
                        "2024-01-02 09:32:00",
                    ],
                    utc=True,
                ),
                "open": [5000.0, 5001.0, 5002.0],
                "high": [5001.0, 5002.0, 5003.0],
                "low": [4999.0, 5000.0, 5001.0],
                "close": [5001.0, 5002.0, 5003.0],
                "volume": [100, 200, 150],
            }
        )
        mock_df.index = mock_df["ts_event"]
        mock_df.index.name = "ts_event"
        mock_data.to_df.return_value = mock_df
        mock_client.timeseries.get_range.return_value = mock_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(output_dir=Path(tmpdir))
            client = plugin._create_client()
            df = plugin._fetch_symbol_data(
                client, "ES", "2024-01-01", "2024-01-31"
            )

            assert list(df.columns) == CANONICAL_OHLCV_SCHEMA
            assert len(df) == 3
            assert df["open"].iloc[0] == 5000.0

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_download_ohlcv_stores_to_futures_1min(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client

        mock_data = MagicMock()
        mock_df = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(
                    [
                        "2024-01-02 09:30:00",
                        "2024-01-02 09:31:00",
                    ],
                    utc=True,
                ),
                "open": [5000.0, 5001.0],
                "high": [5001.0, 5002.0],
                "low": [4999.0, 5000.0],
                "close": [5001.0, 5002.0],
                "volume": [100, 200],
            }
        )
        mock_df.index = mock_df["ts_event"]
        mock_df.index.name = "ts_event"
        mock_data.to_df.return_value = mock_df
        mock_client.timeseries.get_range.return_value = mock_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(output_dir=Path(tmpdir))
            result = plugin.download(
                symbols=["ES"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            assert result.succeeded == 1

            # Verify stored in futures/databento/1min (not futures/databento/trades)
            ohlcv_parquet = (
                Path(tmpdir)
                / "futures" / "databento" / "1min"
                / "symbol=ES"
                / "year=2024"
                / "month=1"
                / "data.parquet"
            )
            assert ohlcv_parquet.exists()

            ohlcv_df = pd.read_parquet(ohlcv_parquet)
            assert list(ohlcv_df.columns) == CANONICAL_OHLCV_SCHEMA


class TestFetchTrades:
    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_fetch_returns_trades_dataframe(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client

        mock_data = MagicMock()
        mock_df = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(
                    [
                        "2024-01-02 09:30:00",
                        "2024-01-02 09:30:15",
                        "2024-01-02 09:31:05",
                    ],
                    utc=True,
                ),
                "price": [5000.0, 5001.0, 5002.0],
                "size": [10, 20, 15],
                "symbol": ["ES.c.0", "ES.c.0", "ES.c.0"],
                "action": ["T", "T", "T"],
                "side": ["A", "B", "A"],
            }
        )
        mock_df.index = mock_df["ts_event"]
        mock_df.index.name = "ts_event"
        mock_data.to_df.return_value = mock_df
        mock_client.timeseries.get_range.return_value = mock_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(
                output_dir=Path(tmpdir), schema="trades"
            )
            client = plugin._create_client()
            df = plugin._fetch_symbol_data(
                client, "ES", "2024-01-01", "2024-01-31"
            )

            assert "timestamp" in df.columns
            assert "price" in df.columns
            assert "size" in df.columns
            assert len(df) == 3

    @patch("src.data.acquisition.plugins.databento_futures.db")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"})
    def test_reconstruct_ohlcv_after_trades_download(self, mock_db):
        from src.data.acquisition.plugins.databento_futures import (
            DatabentoFuturesPlugin,
        )

        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client

        mock_data = MagicMock()
        mock_df = pd.DataFrame(
            {
                "ts_event": pd.to_datetime(
                    [
                        "2024-01-02 09:30:00",
                        "2024-01-02 09:30:15",
                        "2024-01-02 09:30:45",
                        "2024-01-02 09:31:05",
                    ],
                    utc=True,
                ),
                "price": [5000.0, 5001.0, 4999.0, 5002.0],
                "size": [10, 20, 15, 25],
                "symbol": ["ES.c.0"] * 4,
                "action": ["T"] * 4,
                "side": ["A", "B", "A", "B"],
            }
        )
        mock_df.index = mock_df["ts_event"]
        mock_df.index.name = "ts_event"
        mock_data.to_df.return_value = mock_df
        mock_client.timeseries.get_range.return_value = mock_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = DatabentoFuturesPlugin(
                output_dir=Path(tmpdir), schema="trades"
            )
            result = plugin.download(
                symbols=["ES"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            assert result.succeeded == 1

            # Verify raw trades stored
            trades_parquet = (
                Path(tmpdir)
                / "futures" / "databento" / "trades"
                / "symbol=ES"
                / "year=2024"
                / "month=1"
                / "data.parquet"
            )
            assert trades_parquet.exists()

            # Verify reconstructed OHLCV stored
            ohlcv_parquet = (
                Path(tmpdir)
                / "futures" / "databento" / "1min"
                / "symbol=ES"
                / "year=2024"
                / "month=1"
                / "data.parquet"
            )
            assert ohlcv_parquet.exists()

            ohlcv_df = pd.read_parquet(ohlcv_parquet)
            assert list(ohlcv_df.columns) == CANONICAL_OHLCV_SCHEMA


class TestDefaultSymbols:
    def test_default_futures_universe(self):
        from src.data.acquisition.plugins.databento_futures import (
            DEFAULT_FUTURES_UNIVERSE,
        )

        expected = ["ES", "NQ", "CL", "GC", "ZN", "6E", "ZC", "YM", "RTY"]
        assert DEFAULT_FUTURES_UNIVERSE == expected
