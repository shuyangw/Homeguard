"""Tests for Alpaca news plugin."""


class TestNewsPlugin:
    def test_storage_subdir(self):
        from src.data.acquisition.plugins.alpaca_news import AlpacaNewsPlugin

        plugin = AlpacaNewsPlugin.__new__(AlpacaNewsPlugin)
        assert plugin._get_storage_subdir() == "news"

    def test_normalize_symbol_passthrough(self):
        from src.data.acquisition.plugins.alpaca_news import AlpacaNewsPlugin

        plugin = AlpacaNewsPlugin.__new__(AlpacaNewsPlugin)
        assert plugin._normalize_symbol("AAPL") == "AAPL"

    def test_schema(self):
        from src.data.acquisition.plugins.alpaca_news import AlpacaNewsPlugin

        plugin = AlpacaNewsPlugin.__new__(AlpacaNewsPlugin)
        schema = plugin._get_schema()
        assert "timestamp" in schema
        assert "headline" in schema
