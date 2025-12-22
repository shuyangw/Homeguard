"""
Tests for YFinanceFundamentalsProvider.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.data.yfinance.fundamentals import FundamentalData
from src.data.yfinance.provider import YFinanceFundamentalsProvider


class TestYFinanceFundamentalsProviderInit:
    """Tests for provider initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir)
            stats = provider.get_cache_stats()
            assert stats["ttl_hours"] == 24

    def test_init_custom_ttl(self):
        """Test initialization with custom TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(
                cache_ttl_hours=12,
                cache_dir=tmpdir,
            )
            stats = provider.get_cache_stats()
            assert stats["ttl_hours"] == 12


class TestGetFundamentals:
    """Tests for get_fundamentals method."""

    @pytest.fixture
    def provider(self):
        """Create provider with temp cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

    @patch("yfinance.Ticker")
    def test_get_fundamentals_single(self, mock_ticker_class, sample_yfinance_info):
        """Test fetching fundamentals for single symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_fundamentals(["AAPL"])

            assert "AAPL" in result
            assert result["AAPL"].symbol == "AAPL"
            assert result["AAPL"].market_cap == pytest.approx(3000.0, rel=0.01)

    @patch("yfinance.Ticker")
    def test_get_fundamentals_uses_cache(self, mock_ticker_class, sample_yfinance_info):
        """Test that cache is used on subsequent calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            # First call - fetches from API
            provider.get_fundamentals(["AAPL"])
            assert mock_ticker_class.call_count == 1

            # Second call - should use cache
            provider.get_fundamentals(["AAPL"])
            assert mock_ticker_class.call_count == 1  # No additional API call

    @patch("yfinance.Ticker")
    def test_get_fundamentals_skip_cache(self, mock_ticker_class, sample_yfinance_info):
        """Test skip_cache forces API fetch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            # First call
            provider.get_fundamentals(["AAPL"])
            assert mock_ticker_class.call_count == 1

            # Second call with skip_cache
            provider.get_fundamentals(["AAPL"], skip_cache=True)
            assert mock_ticker_class.call_count == 2

    @patch("yfinance.Ticker")
    def test_get_fundamentals_multiple_symbols(self, mock_ticker_class, sample_yfinance_info, sample_yfinance_info_minimal):
        """Test fetching fundamentals for multiple symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            def mock_ticker_factory(symbol):
                mock = MagicMock()
                if symbol == "AAPL":
                    mock.info = sample_yfinance_info
                else:
                    mock.info = sample_yfinance_info_minimal
                return mock

            mock_ticker_class.side_effect = mock_ticker_factory

            result = provider.get_fundamentals(["AAPL", "MSFT"])

            assert len(result) == 2
            assert result["AAPL"].market_cap == pytest.approx(3000.0, rel=0.01)
            assert result["MSFT"].market_cap == pytest.approx(2800.0, rel=0.01)

    @patch("yfinance.Ticker")
    def test_get_fundamentals_handles_errors(self, mock_ticker_class):
        """Test that errors are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker_class.side_effect = Exception("API Error")

            result = provider.get_fundamentals(["AAPL"])

            assert len(result) == 0


class TestGetSingle:
    """Tests for get_single method."""

    @patch("yfinance.Ticker")
    def test_get_single(self, mock_ticker_class, sample_yfinance_info):
        """Test get_single returns FundamentalData."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_single("AAPL")

            assert result is not None
            assert result.symbol == "AAPL"

    @patch("yfinance.Ticker")
    def test_get_single_not_found(self, mock_ticker_class):
        """Test get_single returns None for not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = {}
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_single("INVALID")

            assert result is None


class TestConvenienceMethods:
    """Tests for convenience methods."""

    @patch("yfinance.Ticker")
    def test_get_market_cap(self, mock_ticker_class, sample_yfinance_info):
        """Test get_market_cap returns market cap in billions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_market_cap("AAPL")

            assert result == pytest.approx(3000.0, rel=0.01)

    @patch("yfinance.Ticker")
    def test_get_sector(self, mock_ticker_class, sample_yfinance_info):
        """Test get_sector returns sector string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_sector("AAPL")

            assert result == "Technology"

    @patch("yfinance.Ticker")
    def test_get_pe_ratio(self, mock_ticker_class, sample_yfinance_info):
        """Test get_pe_ratio returns P/E ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_pe_ratio("AAPL")

            assert result == pytest.approx(28.5, rel=0.01)


class TestFilterMethods:
    """Tests for filter methods."""

    @pytest.fixture
    def populated_provider(self, sample_yfinance_info):
        """Provider with cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir)

            # Pre-populate cache
            provider._cache.set_batch({
                "AAPL": FundamentalData(
                    symbol="AAPL",
                    market_cap=3000.0,
                    sector="Technology",
                ),
                "JNJ": FundamentalData(
                    symbol="JNJ",
                    market_cap=400.0,
                    sector="Healthcare",
                ),
                "WMT": FundamentalData(
                    symbol="WMT",
                    market_cap=450.0,
                    sector="Consumer Defensive",
                ),
            })
            yield provider

    def test_filter_by_market_cap_min(self, populated_provider):
        """Test filtering by minimum market cap."""
        result = populated_provider.filter_by_market_cap(
            ["AAPL", "JNJ", "WMT"],
            min_cap=500.0,
        )

        assert result == ["AAPL"]

    def test_filter_by_market_cap_max(self, populated_provider):
        """Test filtering by maximum market cap."""
        result = populated_provider.filter_by_market_cap(
            ["AAPL", "JNJ", "WMT"],
            max_cap=500.0,
        )

        assert set(result) == {"JNJ", "WMT"}

    def test_filter_by_market_cap_range(self, populated_provider):
        """Test filtering by market cap range."""
        result = populated_provider.filter_by_market_cap(
            ["AAPL", "JNJ", "WMT"],
            min_cap=400.0,
            max_cap=500.0,
        )

        assert set(result) == {"JNJ", "WMT"}

    def test_filter_by_sector_include(self, populated_provider):
        """Test filtering by included sectors."""
        result = populated_provider.filter_by_sector(
            ["AAPL", "JNJ", "WMT"],
            include_sectors=["Technology", "Healthcare"],
        )

        assert set(result) == {"AAPL", "JNJ"}

    def test_filter_by_sector_exclude(self, populated_provider):
        """Test filtering by excluded sectors."""
        result = populated_provider.filter_by_sector(
            ["AAPL", "JNJ", "WMT"],
            exclude_sectors=["Technology"],
        )

        assert set(result) == {"JNJ", "WMT"}


class TestCacheManagement:
    """Tests for cache management methods."""

    def test_clear_cache(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir)
            provider._cache.set("AAPL", FundamentalData(symbol="AAPL"))

            provider.clear_cache()

            assert len(provider._cache) == 0

    def test_get_cache_stats(self):
        """Test getting cache stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir)
            stats = provider.get_cache_stats()

            assert "entries" in stats
            assert "hits" in stats
            assert "misses" in stats
            assert "hit_rate" in stats

    def test_evict_expired(self):
        """Test evicting expired entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(
                cache_dir=tmpdir,
                cache_ttl_hours=0.0001,  # Very short TTL
            )
            provider._cache.set("AAPL", FundamentalData(symbol="AAPL"))

            import time
            time.sleep(0.5)

            evicted = provider.evict_expired()
            assert evicted == 1


class TestGetInfo:
    """Tests for get_info method."""

    @patch("yfinance.Ticker")
    def test_get_info_returns_raw_dict(self, mock_ticker_class, sample_yfinance_info):
        """Test get_info returns raw yfinance info dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker = MagicMock()
            mock_ticker.info = sample_yfinance_info
            mock_ticker_class.return_value = mock_ticker

            result = provider.get_info("AAPL")

            assert result == sample_yfinance_info

    @patch("yfinance.Ticker")
    def test_get_info_handles_error(self, mock_ticker_class):
        """Test get_info returns empty dict on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = YFinanceFundamentalsProvider(cache_dir=tmpdir, rate_limit_delay=0)

            mock_ticker_class.side_effect = Exception("API Error")

            result = provider.get_info("AAPL")

            assert result == {}
