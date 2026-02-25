"""Tests for Alpaca equities plugin."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA


class TestEquitiesPlugin:
    @patch(
        "src.data.acquisition.plugins.alpaca_equities.StockHistoricalDataClient"
    )
    def test_fetch_returns_canonical_schema(self, mock_client_cls):
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_bars = MagicMock()
        mock_df = (
            pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "timestamp": pd.to_datetime(["2024-01-02 09:30:00"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                    "trade_count": [50.0],
                    "vwap": [100.2],
                }
            )
            .set_index(["symbol", "timestamp"])
        )
        mock_bars.df = mock_df
        mock_client.get_stock_bars.return_value = mock_bars

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = AlpacaEquitiesPlugin(output_dir=Path(tmpdir))
            client = plugin._create_client()
            df = plugin._fetch_symbol_data(
                client, "AAPL", "2024-01-01", "2024-01-31"
            )

            assert list(df.columns) == CANONICAL_OHLCV_SCHEMA
            assert len(df) == 1

    def test_storage_subdir(self):
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        plugin = AlpacaEquitiesPlugin.__new__(AlpacaEquitiesPlugin)
        assert plugin._get_storage_subdir() == "equities_1min"

    def test_normalize_symbol_passthrough(self):
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        plugin = AlpacaEquitiesPlugin.__new__(AlpacaEquitiesPlugin)
        assert plugin._normalize_symbol("AAPL") == "AAPL"
