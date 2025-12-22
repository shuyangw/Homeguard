"""
Tests for screening filter models.
"""

import pytest
from pydantic import ValidationError

from src.screening.filters import (
    ComparisonOperator,
    FundamentalFilter,
    PriceFilter,
    ScreenerConfig,
    TechnicalFilter,
    VolumeFilter,
    evaluate_comparison,
)


class TestPriceFilter:
    """Tests for PriceFilter model."""

    def test_valid_price_range(self):
        """Test valid price range filter."""
        f = PriceFilter(min_price=10, max_price=100)
        assert f.min_price == 10
        assert f.max_price == 100

    def test_min_price_only(self):
        """Test filter with only min price."""
        f = PriceFilter(min_price=5)
        assert f.min_price == 5
        assert f.max_price is None

    def test_max_price_only(self):
        """Test filter with only max price."""
        f = PriceFilter(max_price=500)
        assert f.min_price is None
        assert f.max_price == 500

    def test_negative_price_rejected(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValidationError):
            PriceFilter(min_price=-10)

    def test_invalid_range_rejected(self):
        """Test that min > max is rejected."""
        with pytest.raises(ValidationError):
            PriceFilter(min_price=100, max_price=10)

    def test_change_filter_valid(self):
        """Test valid change percentage filter."""
        f = PriceFilter(change_1d_pct=("gt", 5.0))
        assert f.change_1d_pct == ("gt", 5.0)

    def test_change_filter_list_input(self):
        """Test change filter accepts list input (converts to tuple)."""
        f = PriceFilter(change_1d_pct=["lt", -3.0])
        assert f.change_1d_pct == ("lt", -3.0)

    def test_gap_filter(self):
        """Test gap percentage filter."""
        f = PriceFilter(gap_pct=("gt", 2.0))
        assert f.gap_pct == ("gt", 2.0)


class TestVolumeFilter:
    """Tests for VolumeFilter model."""

    def test_valid_volume_range(self):
        """Test valid volume filter."""
        f = VolumeFilter(min_volume=1_000_000, max_volume=100_000_000)
        assert f.min_volume == 1_000_000
        assert f.max_volume == 100_000_000

    def test_relative_volume_filter(self):
        """Test relative volume filter."""
        f = VolumeFilter(relative_volume=("gt", 2.0))
        assert f.relative_volume == ("gt", 2.0)

    def test_avg_volume_filter(self):
        """Test average volume filter."""
        f = VolumeFilter(min_avg_volume_20d=500_000)
        assert f.min_avg_volume_20d == 500_000

    def test_invalid_volume_range(self):
        """Test that min > max is rejected."""
        with pytest.raises(ValidationError):
            VolumeFilter(min_volume=10_000_000, max_volume=1_000_000)


class TestTechnicalFilter:
    """Tests for TechnicalFilter model."""

    def test_rsi_filter(self):
        """Test RSI filter."""
        f = TechnicalFilter(rsi_14=("lt", 30))
        assert f.rsi_14 == ("lt", 30)

    def test_sma_crossover_valid(self):
        """Test valid SMA crossover filter."""
        f = TechnicalFilter(sma_crossover=(50, 200, "above"))
        assert f.sma_crossover == (50, 200, "above")

    def test_sma_crossover_invalid_direction(self):
        """Test invalid SMA crossover direction."""
        with pytest.raises(ValidationError):
            TechnicalFilter(sma_crossover=(50, 200, "sideways"))

    def test_sma_crossover_fast_must_be_less(self):
        """Test that fast period must be less than slow."""
        with pytest.raises(ValidationError):
            TechnicalFilter(sma_crossover=(200, 50, "above"))

    def test_macd_signal_valid(self):
        """Test valid MACD signal filter."""
        f = TechnicalFilter(macd_signal="bullish")
        assert f.macd_signal == "bullish"

    def test_macd_signal_invalid(self):
        """Test invalid MACD signal value."""
        with pytest.raises(ValidationError):
            TechnicalFilter(macd_signal="sideways")

    def test_bollinger_position_valid(self):
        """Test valid Bollinger position filter."""
        f = TechnicalFilter(bollinger_position="below_lower")
        assert f.bollinger_position == "below_lower"

    def test_atr_filter(self):
        """Test ATR filter."""
        f = TechnicalFilter(atr_pct=("gt", 3.0))
        assert f.atr_pct == ("gt", 3.0)

    def test_52w_high_filter(self):
        """Test 52-week high distance filter."""
        f = TechnicalFilter(pct_from_52w_high=("gt", -10))
        assert f.pct_from_52w_high == ("gt", -10)

    def test_performance_filters(self):
        """Test performance period filters."""
        f = TechnicalFilter(perf_1m=("gt", 5), perf_3m=("gt", 10))
        assert f.perf_1m == ("gt", 5)
        assert f.perf_3m == ("gt", 10)


class TestFundamentalFilter:
    """Tests for FundamentalFilter model."""

    def test_market_cap_filter(self):
        """Test market cap filter."""
        f = FundamentalFilter(min_market_cap=10.0, max_market_cap=100.0)
        assert f.min_market_cap == 10.0
        assert f.max_market_cap == 100.0

    def test_pe_ratio_filter(self):
        """Test P/E ratio filter."""
        f = FundamentalFilter(min_pe_ratio=5, max_pe_ratio=25)
        assert f.min_pe_ratio == 5
        assert f.max_pe_ratio == 25

    def test_peg_ratio_filter(self):
        """Test PEG ratio filter."""
        f = FundamentalFilter(max_peg_ratio=1.5)
        assert f.max_peg_ratio == 1.5

    def test_profitability_filters(self):
        """Test profitability filters."""
        f = FundamentalFilter(
            min_profit_margin=10.0,
            min_roe=15.0,
            min_roa=5.0,
        )
        assert f.min_profit_margin == 10.0
        assert f.min_roe == 15.0
        assert f.min_roa == 5.0

    def test_sector_filter(self):
        """Test sector filter."""
        f = FundamentalFilter(sectors=["Technology", "Healthcare"])
        assert f.sectors == ["Technology", "Healthcare"]

    def test_exclude_sectors(self):
        """Test exclude sectors filter."""
        f = FundamentalFilter(exclude_sectors=["Utilities", "Energy"])
        assert f.exclude_sectors == ["Utilities", "Energy"]

    def test_dividend_filters(self):
        """Test dividend filters."""
        f = FundamentalFilter(min_dividend_yield=2.0, max_payout_ratio=60.0)
        assert f.min_dividend_yield == 2.0
        assert f.max_payout_ratio == 60.0

    def test_beta_filter(self):
        """Test beta filter."""
        f = FundamentalFilter(min_beta=0.5, max_beta=1.5)
        assert f.min_beta == 0.5
        assert f.max_beta == 1.5


class TestScreenerConfig:
    """Tests for ScreenerConfig model."""

    def test_empty_config(self):
        """Test empty config with defaults."""
        config = ScreenerConfig()
        assert config.universe is None
        assert config.max_results == 100
        assert config.sort_by == "relative_volume"
        assert config.sort_descending is True

    def test_with_universe(self):
        """Test config with explicit universe."""
        config = ScreenerConfig(universe=["AAPL", "MSFT", "GOOGL"])
        assert config.universe == ["AAPL", "MSFT", "GOOGL"]

    def test_with_all_filters(self):
        """Test config with all filter types."""
        config = ScreenerConfig(
            universe=["AAPL", "MSFT"],
            price=PriceFilter(min_price=10),
            volume=VolumeFilter(min_volume=1_000_000),
            technical=TechnicalFilter(rsi_14=("lt", 70)),
            fundamental=FundamentalFilter(min_market_cap=10.0),
            max_results=50,
        )
        assert config.price.min_price == 10
        assert config.volume.min_volume == 1_000_000
        assert config.technical.rsi_14 == ("lt", 70)
        assert config.fundamental.min_market_cap == 10.0

    def test_has_technical_filters(self):
        """Test has_technical_filters method."""
        # No technical filters
        config = ScreenerConfig(price=PriceFilter(min_price=10))
        assert config.has_technical_filters() is False

        # With technical filters
        config = ScreenerConfig(technical=TechnicalFilter(rsi_14=("lt", 30)))
        assert config.has_technical_filters() is True

    def test_has_fundamental_filters(self):
        """Test has_fundamental_filters method."""
        # No fundamental filters
        config = ScreenerConfig(price=PriceFilter(min_price=10))
        assert config.has_fundamental_filters() is False

        # With fundamental filters
        config = ScreenerConfig(fundamental=FundamentalFilter(min_market_cap=10))
        assert config.has_fundamental_filters() is True

    def test_requires_historical_data(self):
        """Test requires_historical_data method."""
        # Only price filter - no historical needed
        config = ScreenerConfig(price=PriceFilter(min_price=10))
        assert config.requires_historical_data() is False

        # RSI filter - needs historical
        config = ScreenerConfig(technical=TechnicalFilter(rsi_14=("lt", 30)))
        assert config.requires_historical_data() is True

        # Relative volume - needs historical
        config = ScreenerConfig(volume=VolumeFilter(relative_volume=("gt", 2)))
        assert config.requires_historical_data() is True


class TestEvaluateComparison:
    """Tests for evaluate_comparison function."""

    def test_greater_than(self):
        """Test greater than comparison."""
        assert evaluate_comparison(10, ("gt", 5)) is True
        assert evaluate_comparison(5, ("gt", 10)) is False

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        assert evaluate_comparison(10, ("gte", 10)) is True
        assert evaluate_comparison(9, ("gte", 10)) is False

    def test_less_than(self):
        """Test less than comparison."""
        assert evaluate_comparison(5, ("lt", 10)) is True
        assert evaluate_comparison(15, ("lt", 10)) is False

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        assert evaluate_comparison(10, ("lte", 10)) is True
        assert evaluate_comparison(11, ("lte", 10)) is False

    def test_equal(self):
        """Test equal comparison."""
        assert evaluate_comparison(10, ("eq", 10)) is True
        assert evaluate_comparison(10.0001, ("eq", 10)) is False

    def test_between(self):
        """Test between comparison."""
        assert evaluate_comparison(15, ("between", 10, 20)) is True
        assert evaluate_comparison(5, ("between", 10, 20)) is False
        assert evaluate_comparison(25, ("between", 10, 20)) is False

    def test_case_insensitive_operator(self):
        """Test that operator is case insensitive."""
        assert evaluate_comparison(10, ("GT", 5)) is True
        assert evaluate_comparison(10, ("Lt", 15)) is True
