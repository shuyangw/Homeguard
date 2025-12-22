"""
Tests for FundamentalData dataclass and parsing.
"""

import pytest

from src.data.yfinance.fundamentals import FundamentalData


class TestFundamentalDataBasic:
    """Basic tests for FundamentalData dataclass."""

    def test_create_minimal(self):
        """Test creating FundamentalData with minimal fields."""
        data = FundamentalData(symbol="AAPL")
        assert data.symbol == "AAPL"
        assert data.market_cap is None
        assert data.pe_ratio is None

    def test_create_with_values(self):
        """Test creating FundamentalData with values."""
        data = FundamentalData(
            symbol="AAPL",
            market_cap=3000.0,
            pe_ratio=28.5,
            sector="Technology",
        )
        assert data.symbol == "AAPL"
        assert data.market_cap == 3000.0
        assert data.pe_ratio == 28.5
        assert data.sector == "Technology"

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        data = FundamentalData(
            symbol="AAPL",
            market_cap=3000.0,
            pe_ratio=None,
        )
        d = data.to_dict()
        assert "symbol" in d
        assert "market_cap" in d
        assert "pe_ratio" not in d

    def test_to_dict_includes_all_set_values(self):
        """Test that to_dict includes all non-None values."""
        data = FundamentalData(
            symbol="MSFT",
            market_cap=2800.0,
            pe_ratio=32.0,
            sector="Technology",
            dividend_yield=0.8,
        )
        d = data.to_dict()
        assert d["symbol"] == "MSFT"
        assert d["market_cap"] == 2800.0
        assert d["pe_ratio"] == 32.0
        assert d["sector"] == "Technology"
        assert d["dividend_yield"] == 0.8


class TestFromYfinanceInfo:
    """Tests for from_yfinance_info class method."""

    def test_parse_full_info(self, sample_yfinance_info):
        """Test parsing full yfinance info dict."""
        data = FundamentalData.from_yfinance_info("AAPL", sample_yfinance_info)

        assert data.symbol == "AAPL"
        # Market cap should be converted to billions
        assert data.market_cap == pytest.approx(3000.0, rel=0.01)
        assert data.pe_ratio == pytest.approx(28.5, rel=0.01)
        assert data.forward_pe == pytest.approx(25.2, rel=0.01)
        assert data.peg_ratio == pytest.approx(2.1, rel=0.01)
        assert data.sector == "Technology"
        assert data.industry == "Consumer Electronics"

    def test_parse_minimal_info(self, sample_yfinance_info_minimal):
        """Test parsing minimal yfinance info."""
        data = FundamentalData.from_yfinance_info("MSFT", sample_yfinance_info_minimal)

        assert data.symbol == "MSFT"
        assert data.market_cap == pytest.approx(2800.0, rel=0.01)
        assert data.sector == "Technology"
        assert data.pe_ratio is None  # Not in minimal info

    def test_parse_dividend_info(self, sample_yfinance_info_dividend):
        """Test parsing dividend stock info."""
        data = FundamentalData.from_yfinance_info("JNJ", sample_yfinance_info_dividend)

        assert data.symbol == "JNJ"
        assert data.market_cap == pytest.approx(400.0, rel=0.01)
        assert data.pe_ratio == pytest.approx(15.2, rel=0.01)
        # Dividend yield should be converted to percentage
        assert data.dividend_yield == pytest.approx(2.9, rel=0.1)
        assert data.dividend_rate == pytest.approx(4.76, rel=0.01)
        assert data.sector == "Healthcare"

    def test_parse_profitability_margins(self, sample_yfinance_info):
        """Test parsing profitability margins (converted to %)."""
        data = FundamentalData.from_yfinance_info("AAPL", sample_yfinance_info)

        # yfinance returns ratios (0.25), we convert to % (25.0)
        assert data.profit_margin == pytest.approx(25.0, rel=0.1)
        assert data.operating_margin == pytest.approx(30.0, rel=0.1)
        assert data.gross_margin == pytest.approx(43.0, rel=0.1)

    def test_parse_returns(self, sample_yfinance_info):
        """Test parsing return metrics (ROE, ROA)."""
        data = FundamentalData.from_yfinance_info("AAPL", sample_yfinance_info)

        # yfinance returns ratios, we convert to %
        assert data.roe == pytest.approx(14.7, rel=0.1)
        assert data.roa == pytest.approx(21.0, rel=0.1)

    def test_parse_empty_info(self):
        """Test parsing empty info dict."""
        data = FundamentalData.from_yfinance_info("EMPTY", {})

        assert data.symbol == "EMPTY"
        assert data.market_cap is None
        assert data.sector is None

    def test_parse_info_with_invalid_values(self):
        """Test parsing info with invalid values."""
        info = {
            "symbol": "TEST",
            "marketCap": "invalid",
            "trailingPE": None,
            "sector": "",
        }
        data = FundamentalData.from_yfinance_info("TEST", info)

        assert data.symbol == "TEST"
        assert data.market_cap is None  # Invalid value
        assert data.pe_ratio is None  # None value
        assert data.sector is None  # Empty string


class TestFundamentalDataFields:
    """Tests for specific FundamentalData fields."""

    def test_valuation_fields(self):
        """Test all valuation ratio fields."""
        data = FundamentalData(
            symbol="TEST",
            pe_ratio=25.0,
            forward_pe=22.0,
            peg_ratio=1.5,
            pb_ratio=5.0,
            ps_ratio=3.0,
            price_to_cashflow=15.0,
        )
        assert data.pe_ratio == 25.0
        assert data.forward_pe == 22.0
        assert data.peg_ratio == 1.5
        assert data.pb_ratio == 5.0
        assert data.ps_ratio == 3.0
        assert data.price_to_cashflow == 15.0

    def test_profitability_fields(self):
        """Test profitability metric fields."""
        data = FundamentalData(
            symbol="TEST",
            profit_margin=15.0,
            operating_margin=20.0,
            gross_margin=40.0,
            ebitda_margin=25.0,
            roe=18.0,
            roa=10.0,
            roic=15.0,
        )
        assert data.profit_margin == 15.0
        assert data.operating_margin == 20.0
        assert data.gross_margin == 40.0
        assert data.roe == 18.0
        assert data.roa == 10.0

    def test_financial_health_fields(self):
        """Test financial health metric fields."""
        data = FundamentalData(
            symbol="TEST",
            debt_to_equity=0.5,
            current_ratio=2.0,
            quick_ratio=1.5,
            interest_coverage=10.0,
        )
        assert data.debt_to_equity == 0.5
        assert data.current_ratio == 2.0
        assert data.quick_ratio == 1.5
        assert data.interest_coverage == 10.0

    def test_dividend_fields(self):
        """Test dividend-related fields."""
        data = FundamentalData(
            symbol="TEST",
            dividend_yield=2.5,
            dividend_rate=1.0,
            payout_ratio=40.0,
            ex_dividend_date="2024-01-15",
        )
        assert data.dividend_yield == 2.5
        assert data.dividend_rate == 1.0
        assert data.payout_ratio == 40.0
        assert data.ex_dividend_date == "2024-01-15"

    def test_classification_fields(self):
        """Test classification fields."""
        data = FundamentalData(
            symbol="TEST",
            sector="Technology",
            industry="Software",
            country="United States",
            exchange="NASDAQ",
        )
        assert data.sector == "Technology"
        assert data.industry == "Software"
        assert data.country == "United States"
        assert data.exchange == "NASDAQ"

    def test_risk_fields(self):
        """Test risk metric fields."""
        data = FundamentalData(
            symbol="TEST",
            beta=1.2,
            volatility_52w=0.25,
            short_ratio=2.5,
            short_percent_of_float=5.0,
        )
        assert data.beta == 1.2
        assert data.volatility_52w == 0.25
        assert data.short_ratio == 2.5
        assert data.short_percent_of_float == 5.0

    def test_analyst_target_fields(self):
        """Test analyst target fields."""
        data = FundamentalData(
            symbol="TEST",
            target_mean_price=200.0,
            target_high_price=250.0,
            target_low_price=150.0,
            recommendation="buy",
        )
        assert data.target_mean_price == 200.0
        assert data.target_high_price == 250.0
        assert data.target_low_price == 150.0
        assert data.recommendation == "buy"
