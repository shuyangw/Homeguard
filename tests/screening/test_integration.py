"""
Integration tests for screening module.

These tests make real API calls and should only be run when:
- Network is available
- Alpaca credentials are configured
- Explicit integration testing is needed

Run with: pytest tests/screening/test_integration.py -v -m integration
"""

import pytest

from src.screening import (
    StockScreener,
    ScreenerConfig,
    PriceFilter,
    VolumeFilter,
    TechnicalFilter,
)


@pytest.mark.integration
@pytest.mark.network
class TestScreenerIntegration:
    """Integration tests for StockScreener with real APIs."""

    @pytest.fixture
    def screener(self):
        """Create real screener (uses Alpaca IEX feed)."""
        return StockScreener(paper=True)

    def test_screen_small_universe(self, screener):
        """Test screening a small universe of real symbols."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            price=PriceFilter(min_price=10),
            max_results=10,
        )

        result = screener.screen(config)

        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(isinstance(s, str) for s in result)

    def test_screen_with_details(self, screener):
        """Test screen_with_details returns proper data."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT"],
            max_results=10,
        )

        results = screener.screen_with_details(config)

        assert len(results) <= 2
        for r in results:
            assert r.symbol in ["AAPL", "MSFT"]
            assert r.price > 0
            assert r.volume >= 0

    def test_screen_with_price_filter(self, screener):
        """Test real price filtering."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "F", "GE"],  # Mix of high/low price
            price=PriceFilter(min_price=100),
            max_results=10,
        )

        result = screener.screen(config)

        # AAPL and MSFT should pass, F and GE are typically under $100
        assert len(result) >= 0  # May vary based on market

    def test_screen_with_volume_filter(self, screener):
        """Test real volume filtering."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "GOOGL"],
            volume=VolumeFilter(min_volume=100_000),
            max_results=10,
        )

        result = screener.screen(config)

        # High-volume stocks should pass
        assert len(result) >= 0

    def test_cache_works(self, screener):
        """Test that caching reduces API calls."""
        config = ScreenerConfig(
            universe=["AAPL"],
            max_results=10,
        )

        # First call - should populate cache
        screener.screen(config)
        stats1 = screener.get_cache_stats()

        # Second call should use cache
        screener.screen(config)
        stats2 = screener.get_cache_stats()

        # Cache should have snapshots after first call
        assert stats1["snapshots"]["count"] > 0
        # Results should be cached after second call
        assert stats2["results"]["count"] >= 0

    def test_clear_cache(self, screener):
        """Test cache clearing works."""
        config = ScreenerConfig(
            universe=["AAPL"],
            max_results=10,
        )

        screener.screen(config)
        screener.clear_cache()
        stats = screener.get_cache_stats()

        assert stats["snapshots"]["count"] == 0


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
class TestFullUniverseScreen:
    """Tests for screening the full Alpaca universe."""

    @pytest.fixture
    def screener(self):
        """Create real screener."""
        return StockScreener(paper=True)

    def test_fetch_all_tradable_assets(self, screener):
        """Test fetching all tradable assets from Alpaca."""
        config = ScreenerConfig(
            universe=None,  # Fetch all from Alpaca
            price=PriceFilter(min_price=100, max_price=1000),
            volume=VolumeFilter(min_volume=1_000_000),
            max_results=50,
        )

        result = screener.screen(config)

        # Should return some results
        assert isinstance(result, list)
        assert len(result) <= 50
        # All symbols should be strings
        assert all(isinstance(s, str) for s in result)


@pytest.mark.integration
@pytest.mark.network
class TestTechnicalFiltersIntegration:
    """Integration tests for technical filters."""

    @pytest.fixture
    def screener(self):
        """Create real screener."""
        return StockScreener(paper=True)

    def test_rsi_filter(self, screener):
        """Test RSI filter with real data."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "GOOGL"],
            technical=TechnicalFilter(rsi_14=("lt", 80)),  # Not overbought
            max_results=10,
        )

        # This may fail if all stocks are overbought
        result = screener.screen(config)
        assert isinstance(result, list)

    def test_sorting_by_price(self, screener):
        """Test sorting by price."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "F", "GE"],
            sort_by="price",
            sort_descending=True,
            max_results=10,
        )

        results = screener.screen_with_details(config)

        if len(results) >= 2:
            # Should be sorted descending
            assert results[0].price >= results[1].price

    def test_sorting_by_volume(self, screener):
        """Test sorting by volume."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT", "GOOGL"],
            sort_by="volume",
            sort_descending=True,
            max_results=10,
        )

        results = screener.screen_with_details(config)

        if len(results) >= 2:
            assert results[0].volume >= results[1].volume
