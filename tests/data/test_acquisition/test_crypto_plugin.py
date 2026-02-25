"""Tests for Alpaca crypto plugin."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA


class TestCryptoPlugin:
    def test_normalize_symbol(self):
        from src.data.acquisition.plugins.alpaca_crypto import (
            AlpacaCryptoPlugin,
        )

        plugin = AlpacaCryptoPlugin.__new__(AlpacaCryptoPlugin)
        assert plugin._normalize_symbol("BTC/USD") == "BTC_USD"
        assert plugin._normalize_symbol("ETH/USD") == "ETH_USD"

    def test_storage_subdir(self):
        from src.data.acquisition.plugins.alpaca_crypto import (
            AlpacaCryptoPlugin,
        )

        plugin = AlpacaCryptoPlugin.__new__(AlpacaCryptoPlugin)
        assert plugin._get_storage_subdir() == "crypto_1min"

    @patch(
        "src.data.acquisition.plugins.alpaca_crypto.CryptoHistoricalDataClient"
    )
    def test_fetch_returns_canonical_schema(self, mock_client_cls):
        from src.data.acquisition.plugins.alpaca_crypto import (
            AlpacaCryptoPlugin,
        )

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_bars = MagicMock()
        mock_df = (
            pd.DataFrame(
                {
                    "symbol": ["BTC/USD"],
                    "timestamp": pd.to_datetime(["2024-01-02 00:00:00"]),
                    "open": [42000.0],
                    "high": [42100.0],
                    "low": [41900.0],
                    "close": [42050.0],
                    "volume": [100.0],
                    "trade_count": [500.0],
                    "vwap": [42025.0],
                }
            )
            .set_index(["symbol", "timestamp"])
        )
        mock_bars.df = mock_df
        mock_client.get_crypto_bars.return_value = mock_bars

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = AlpacaCryptoPlugin(output_dir=Path(tmpdir))
            client = plugin._create_client()
            df = plugin._fetch_symbol_data(
                client, "BTC/USD", "2024-01-01", "2024-01-31"
            )
            assert list(df.columns) == CANONICAL_OHLCV_SCHEMA
