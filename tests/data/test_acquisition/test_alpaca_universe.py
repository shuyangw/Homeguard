"""Tests for Alpaca universe snapshot helper."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


def make_asset(symbol, tradable=True, status="active"):
    a = MagicMock()
    a.symbol = symbol
    a.tradable = tradable
    a.status = MagicMock()
    a.status.value = status
    # Make status comparable
    from alpaca.trading.enums import AssetStatus
    a.status = AssetStatus.ACTIVE if status == "active" else AssetStatus.INACTIVE
    return a


class TestListActiveUsEquities:
    def test_filters_non_tradable(self):
        from src.data.acquisition.alpaca_universe import list_active_us_equities

        client = MagicMock()
        client.get_all_assets.return_value = [
            make_asset("AAPL", tradable=True),
            make_asset("XYZ", tradable=False),
            make_asset("MSFT", tradable=True),
        ]

        symbols = list_active_us_equities(client)
        assert symbols == ["AAPL", "MSFT"]

    def test_excludes_slash_and_dot_tickers(self):
        from src.data.acquisition.alpaca_universe import list_active_us_equities

        client = MagicMock()
        client.get_all_assets.return_value = [
            make_asset("AAPL"),
            make_asset("BRK/B"),
            make_asset("BF.B"),
            make_asset("SPY"),
        ]

        symbols = list_active_us_equities(client)
        assert symbols == ["AAPL", "SPY"]

    def test_excludes_inactive(self):
        from src.data.acquisition.alpaca_universe import list_active_us_equities

        client = MagicMock()
        client.get_all_assets.return_value = [
            make_asset("AAPL", status="active"),
            make_asset("DELISTED", status="inactive"),
        ]

        symbols = list_active_us_equities(client)
        assert symbols == ["AAPL"]

    def test_saves_csv_when_save_to_set(self):
        from src.data.acquisition.alpaca_universe import list_active_us_equities

        client = MagicMock()
        client.get_all_assets.return_value = [make_asset("AAPL")]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "universe.csv"
            symbols = list_active_us_equities(client, save_to=out_path)
            assert out_path.exists()
            df = pd.read_csv(out_path)
            assert df["Symbol"].tolist() == ["AAPL"]

    def test_no_csv_when_save_to_none(self):
        from src.data.acquisition.alpaca_universe import list_active_us_equities

        client = MagicMock()
        client.get_all_assets.return_value = [make_asset("AAPL")]

        list_active_us_equities(client, save_to=None)
        # No file to assert -- just ensure no exception
