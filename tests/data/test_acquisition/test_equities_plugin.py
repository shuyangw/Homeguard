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
        plugin._storage_subdir_override = None
        assert plugin._get_storage_subdir() == "equities_1min"

    def test_normalize_symbol_passthrough(self):
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        plugin = AlpacaEquitiesPlugin.__new__(AlpacaEquitiesPlugin)
        assert plugin._normalize_symbol("AAPL") == "AAPL"


class TestFeedAndAdjustment:
    @patch(
        "src.data.acquisition.plugins.alpaca_equities.StockHistoricalDataClient"
    )
    def test_feed_and_adjustment_passed_to_request(self, mock_client_cls):
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": pd.to_datetime(["2024-01-02 09:30:00"]),
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.5], "volume": [1000.0],
                "trade_count": [50.0], "vwap": [100.2],
            }
        ).set_index(["symbol", "timestamp"])
        mock_client.get_stock_bars.return_value = mock_bars

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = AlpacaEquitiesPlugin(
                output_dir=Path(tmpdir),
                feed=DataFeed.SIP,
                adjustment=Adjustment.SPLIT,
            )
            client = plugin._create_client()
            plugin._fetch_symbol_data(
                client, "AAPL", "2024-01-01", "2024-01-31"
            )

            assert mock_client.get_stock_bars.called
            request_arg = mock_client.get_stock_bars.call_args[0][0]
            assert isinstance(request_arg, StockBarsRequest)
            assert request_arg.feed == DataFeed.SIP
            assert request_arg.adjustment == Adjustment.SPLIT

    @patch(
        "src.data.acquisition.plugins.alpaca_equities.StockHistoricalDataClient"
    )
    def test_default_feed_and_adjustment_unchanged(self, mock_client_cls):
        """Default behavior must match existing callers: no feed, no adjustment."""
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame(columns=[
            "symbol", "timestamp", "open", "high", "low",
            "close", "volume", "trade_count", "vwap",
        ]).set_index(["symbol", "timestamp"])
        mock_client.get_stock_bars.return_value = mock_bars

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = AlpacaEquitiesPlugin(output_dir=Path(tmpdir))
            client = plugin._create_client()
            plugin._fetch_symbol_data(
                client, "AAPL", "2024-01-01", "2024-01-31"
            )

            request_arg = mock_client.get_stock_bars.call_args[0][0]
            # SDK uses None as "API default" sentinel for both
            assert request_arg.feed is None
            assert request_arg.adjustment is None


class TestStorageSubdirOverride:
    def test_override_routes_storage_subdir(self):
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        plugin = AlpacaEquitiesPlugin.__new__(AlpacaEquitiesPlugin)
        plugin._storage_subdir_override = "equities_1min_sip_raw"
        assert plugin._get_storage_subdir() == "equities_1min_sip_raw"

    def test_no_override_returns_default(self):
        from src.data.acquisition.plugins.alpaca_equities import (
            AlpacaEquitiesPlugin,
        )

        plugin = AlpacaEquitiesPlugin.__new__(AlpacaEquitiesPlugin)
        plugin._storage_subdir_override = None
        assert plugin._get_storage_subdir() == "equities_1min"
