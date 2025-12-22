"""
Fundamental data types for YFinance provider.

Defines dataclasses for fundamental financial data that Alpaca doesn't provide.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FundamentalData:
    """
    Fundamental financial data for a stock.

    All financial ratios and metrics that yfinance provides.
    Fields are Optional since not all stocks have all data available.
    """

    symbol: str

    # Market metrics
    market_cap: Optional[float] = None  # In billions
    enterprise_value: Optional[float] = None  # In billions
    shares_outstanding: Optional[float] = None  # In millions

    # Valuation ratios
    pe_ratio: Optional[float] = None  # Trailing P/E
    forward_pe: Optional[float] = None  # Forward P/E
    peg_ratio: Optional[float] = None  # PEG ratio
    pb_ratio: Optional[float] = None  # Price-to-book
    ps_ratio: Optional[float] = None  # Price-to-sales
    price_to_cashflow: Optional[float] = None

    # Earnings and growth
    eps: Optional[float] = None  # Trailing EPS
    forward_eps: Optional[float] = None
    eps_growth_yoy: Optional[float] = None  # YoY EPS growth %
    revenue_growth: Optional[float] = None  # Revenue growth %

    # Profitability
    profit_margin: Optional[float] = None  # %
    operating_margin: Optional[float] = None  # %
    gross_margin: Optional[float] = None  # %
    ebitda_margin: Optional[float] = None  # %

    # Returns
    roe: Optional[float] = None  # Return on equity %
    roa: Optional[float] = None  # Return on assets %
    roic: Optional[float] = None  # Return on invested capital %

    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None

    # Dividends
    dividend_yield: Optional[float] = None  # %
    dividend_rate: Optional[float] = None  # Annual dividend $
    payout_ratio: Optional[float] = None  # %
    ex_dividend_date: Optional[str] = None  # YYYY-MM-DD

    # Classification
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None

    # Risk metrics
    beta: Optional[float] = None
    volatility_52w: Optional[float] = None

    # Price targets
    target_mean_price: Optional[float] = None
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None
    recommendation: Optional[str] = None  # Buy, Hold, Sell

    # Additional
    short_ratio: Optional[float] = None
    short_percent_of_float: Optional[float] = None
    avg_analyst_rating: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_yfinance_info(cls, symbol: str, info: dict) -> "FundamentalData":
        """
        Create FundamentalData from yfinance Ticker.info dict.

        Args:
            symbol: Stock symbol
            info: Dict from yfinance.Ticker(symbol).info

        Returns:
            FundamentalData instance with populated fields
        """

        def safe_get(key: str, scale: float = 1.0) -> Optional[float]:
            """Safely get and scale a numeric value."""
            val = info.get(key)
            if val is None or val == "":
                return None
            try:
                return float(val) * scale
            except (TypeError, ValueError):
                return None

        def safe_get_str(key: str) -> Optional[str]:
            """Safely get a string value."""
            val = info.get(key)
            if val is None or val == "":
                return None
            return str(val)

        def safe_pct(key: str) -> Optional[float]:
            """Get value and convert to percentage if needed."""
            val = safe_get(key)
            if val is None:
                return None
            # yfinance returns ratios, not percentages
            if abs(val) < 1:
                return val * 100
            return val

        return cls(
            symbol=symbol,
            # Market metrics (convert from raw to billions/millions)
            market_cap=safe_get("marketCap", 1e-9),
            enterprise_value=safe_get("enterpriseValue", 1e-9),
            shares_outstanding=safe_get("sharesOutstanding", 1e-6),
            # Valuation
            pe_ratio=safe_get("trailingPE"),
            forward_pe=safe_get("forwardPE"),
            peg_ratio=safe_get("pegRatio"),
            pb_ratio=safe_get("priceToBook"),
            ps_ratio=safe_get("priceToSalesTrailing12Months"),
            price_to_cashflow=safe_get("priceToFreeCashFlows"),
            # Earnings
            eps=safe_get("trailingEps"),
            forward_eps=safe_get("forwardEps"),
            eps_growth_yoy=safe_pct("earningsQuarterlyGrowth"),
            revenue_growth=safe_pct("revenueGrowth"),
            # Profitability (convert from ratio to %)
            profit_margin=safe_pct("profitMargins"),
            operating_margin=safe_pct("operatingMargins"),
            gross_margin=safe_pct("grossMargins"),
            ebitda_margin=safe_pct("ebitdaMargins"),
            # Returns
            roe=safe_pct("returnOnEquity"),
            roa=safe_pct("returnOnAssets"),
            # Financial health
            debt_to_equity=safe_get("debtToEquity"),
            current_ratio=safe_get("currentRatio"),
            quick_ratio=safe_get("quickRatio"),
            # Dividends
            dividend_yield=safe_pct("dividendYield"),
            dividend_rate=safe_get("dividendRate"),
            payout_ratio=safe_pct("payoutRatio"),
            ex_dividend_date=safe_get_str("exDividendDate"),
            # Classification
            sector=safe_get_str("sector"),
            industry=safe_get_str("industry"),
            country=safe_get_str("country"),
            exchange=safe_get_str("exchange"),
            # Risk
            beta=safe_get("beta"),
            volatility_52w=safe_get("52WeekChange"),
            # Targets
            target_mean_price=safe_get("targetMeanPrice"),
            target_high_price=safe_get("targetHighPrice"),
            target_low_price=safe_get("targetLowPrice"),
            recommendation=safe_get_str("recommendationKey"),
            # Short interest
            short_ratio=safe_get("shortRatio"),
            short_percent_of_float=safe_pct("shortPercentOfFloat"),
            avg_analyst_rating=safe_get_str("averageAnalystRating"),
        )
