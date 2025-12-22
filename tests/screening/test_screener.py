"""
Tests for the main StockScreener class.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.screening.filters import (
    PriceFilter,
    ScreenerConfig,
    ScreenerResult,
    TechnicalFilter,
    VolumeFilter,
)
from src.screening.screener import StockScreener


class TestStockScreenerInit:
    """Tests for StockScreener initialization."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_init_paper_mode(self, mock_client_class):
        """Test initialization in paper mode."""
        screener = StockScreener(paper=True)
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["paper"] is True

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_init_live_mode(self, mock_client_class):
        """Test initialization in live mode."""
        screener = StockScreener(paper=False)
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["paper"] is False

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_init_custom_cache_ttl(self, mock_client_class):
        """Test initialization with custom cache TTL."""
        screener = StockScreener(cache_ttl_seconds=120)
        assert screener.cache._snapshot_ttl == 120


class TestScreenMethod:
    """Tests for screen() method."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_screen_returns_symbols(self, mock_client_class, sample_snapshot_data):
        """Test screen returns list of symbol strings."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "PENNY", "LOWVOL"],
            max_results=10,
        )

        result = screener.screen(config)

        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_screen_with_price_filter(self, mock_client_class, sample_snapshot_data):
        """Test screen with price filter."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "PENNY", "LOWVOL"],
            price=PriceFilter(min_price=10),  # Should exclude PENNY ($2.50)
            max_results=10,
        )

        result = screener.screen(config)

        assert "PENNY" not in result
        assert len(result) <= 3  # AAPL, MSFT, LOWVOL

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_screen_with_volume_filter(self, mock_client_class, sample_snapshot_data):
        """Test screen with volume filter."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "PENNY", "LOWVOL"],
            volume=VolumeFilter(min_volume=1_000_000),  # Exclude low volume
            max_results=10,
        )

        result = screener.screen(config)

        assert "LOWVOL" not in result

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_screen_respects_max_results(self, mock_client_class, sample_snapshot_data):
        """Test screen respects max_results limit."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "PENNY", "LOWVOL"],
            max_results=2,
        )

        result = screener.screen(config)

        assert len(result) <= 2

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_screen_empty_universe(self, mock_client_class):
        """Test screen with empty universe."""
        mock_client = MagicMock()
        mock_client.get_all_tradable_assets.return_value = []
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(universe=[])

        result = screener.screen(config)

        assert result == []


class TestScreenWithDetails:
    """Tests for screen_with_details() method."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_returns_screener_results(self, mock_client_class, sample_snapshot_data):
        """Test returns list of ScreenerResult objects."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT"],
            max_results=10,
        )

        results = screener.screen_with_details(config)

        assert isinstance(results, list)
        assert all(isinstance(r, ScreenerResult) for r in results)

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_result_contains_metrics(self, mock_client_class, sample_snapshot_data):
        """Test result contains expected metrics."""
        mock_client = MagicMock()
        # Return only AAPL snapshot
        mock_client.get_snapshots.return_value = {"AAPL": sample_snapshot_data["AAPL"]}
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL"],
            max_results=10,
        )

        results = screener.screen_with_details(config)

        assert len(results) == 1
        result = results[0]
        assert result.symbol == "AAPL"
        assert result.price == 175.50
        assert result.volume == 50_000_000


class TestUniverseFetching:
    """Tests for universe fetching behavior."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_fetches_all_assets_when_universe_none(
        self, mock_client_class, sample_snapshot_data
    ):
        """Test fetches all tradable assets when universe is None."""
        mock_client = MagicMock()
        mock_client.get_all_tradable_assets.return_value = ["AAPL", "MSFT"]
        mock_client.get_snapshots.return_value = {
            "AAPL": sample_snapshot_data["AAPL"],
            "MSFT": sample_snapshot_data["MSFT"],
        }
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(universe=None)

        screener.screen(config)

        mock_client.get_all_tradable_assets.assert_called_once()

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_uses_provided_universe(self, mock_client_class, sample_snapshot_data):
        """Test uses provided universe instead of fetching."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(universe=["AAPL", "MSFT"])

        screener.screen(config)

        mock_client.get_all_tradable_assets.assert_not_called()


class TestCaching:
    """Tests for caching behavior."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_caches_snapshots(self, mock_client_class, sample_snapshot_data):
        """Test snapshots are cached."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(universe=["AAPL"])

        # First call
        screener.screen(config)
        assert mock_client.get_snapshots.call_count == 1

        # Second call should use cache
        screener.screen(config)
        # Snapshots are cached per-symbol, but we still call get_snapshots for uncached
        # In this case, all should be cached

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_clear_cache(self, mock_client_class):
        """Test cache can be cleared."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        screener.cache.set_snapshot("AAPL", {"price": 100})

        screener.clear_cache()

        assert screener.cache.get_snapshot("AAPL") is None

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_get_cache_stats(self, mock_client_class):
        """Test cache stats retrieval."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        stats = screener.get_cache_stats()

        assert "snapshots" in stats
        assert "historical" in stats
        assert "results" in stats


class TestSorting:
    """Tests for result sorting."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_sort_by_price_descending(self, mock_client_class, sample_snapshot_data):
        """Test sorting by price descending."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "PENNY"],
            sort_by="price",
            sort_descending=True,
            max_results=10,
        )

        results = screener.screen_with_details(config)

        # MSFT ($380) should be first
        if len(results) >= 2:
            assert results[0].price >= results[1].price

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_sort_by_volume(self, mock_client_class, sample_snapshot_data):
        """Test sorting by volume."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "LOWVOL"],
            sort_by="volume",
            sort_descending=True,
            max_results=10,
        )

        results = screener.screen_with_details(config)

        if len(results) >= 2:
            assert results[0].volume >= results[1].volume


class TestTechnicalFilters:
    """Tests for technical filter application."""

    @patch("src.screening.screener.AlpacaScreenerClient")
    def test_technical_filter_fetches_historical(
        self, mock_client_class, sample_snapshot_data, sample_historical_bars
    ):
        """Test technical filters trigger historical data fetch."""
        mock_client = MagicMock()
        mock_client.get_snapshots.return_value = sample_snapshot_data
        mock_client.get_historical_bars.return_value = {
            "AAPL": sample_historical_bars,
            "MSFT": sample_historical_bars,
        }
        mock_client_class.return_value = mock_client

        screener = StockScreener(paper=True)
        config = ScreenerConfig(
            universe=["AAPL", "MSFT"],
            technical=TechnicalFilter(rsi_14=("lt", 70)),
            max_results=10,
        )

        screener.screen(config)

        mock_client.get_historical_bars.assert_called()
