"""
Stock Screener Filter Models.

Pydantic models for type-safe filter configuration.
Supports price, volume, technical, and fundamental filters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ComparisonOperator(str, Enum):
    """Comparison operators for numeric filters."""

    GT = "gt"  # greater than
    GTE = "gte"  # greater than or equal
    LT = "lt"  # less than
    LTE = "lte"  # less than or equal
    EQ = "eq"  # equal
    BETWEEN = "between"  # between two values


class SortField(str, Enum):
    """Available fields for sorting screen results."""

    PRICE = "price"
    CHANGE_1D = "change_1d_pct"
    VOLUME = "volume"
    RELATIVE_VOLUME = "relative_volume"
    RSI = "rsi_14"
    ATR_PCT = "atr_pct"
    MARKET_CAP = "market_cap"


class CrossoverDirection(str, Enum):
    """Direction of moving average crossover."""

    ABOVE = "above"  # Fast MA above slow MA
    BELOW = "below"  # Fast MA below slow MA


# Type alias for comparison tuple: (operator, value) or (operator, low, high) for BETWEEN
ComparisonFilter = Union[Tuple[str, float], Tuple[str, float, float]]


class PriceFilter(BaseModel):
    """
    Filter stocks by price metrics.

    Examples:
        # Price between $10 and $100
        PriceFilter(min_price=10, max_price=100)

        # Stocks up more than 5% today
        PriceFilter(change_1d_pct=('gt', 5.0))

        # Gap up more than 2%
        PriceFilter(gap_pct=('gt', 2.0))
    """

    min_price: Optional[float] = Field(default=None, ge=0, description="Minimum stock price")
    max_price: Optional[float] = Field(default=None, ge=0, description="Maximum stock price")
    change_1d_pct: Optional[ComparisonFilter] = Field(
        default=None, description="1-day price change filter: ('gt', 5.0) for >5%"
    )
    change_5d_pct: Optional[ComparisonFilter] = Field(
        default=None, description="5-day price change filter"
    )
    change_1m_pct: Optional[ComparisonFilter] = Field(
        default=None, description="1-month price change filter"
    )
    gap_pct: Optional[ComparisonFilter] = Field(
        default=None, description="Gap percentage from previous close"
    )

    @field_validator("change_1d_pct", "change_5d_pct", "change_1m_pct", "gap_pct", mode="before")
    @classmethod
    def validate_comparison_filter(cls, v: Any) -> Optional[ComparisonFilter]:
        """Validate comparison filter format."""
        if v is None:
            return None
        if not isinstance(v, (tuple, list)):
            raise ValueError("Comparison filter must be a tuple: ('gt', 5.0)")
        if len(v) < 2:
            raise ValueError("Comparison filter must have at least 2 elements: ('gt', 5.0)")
        op = v[0].lower() if isinstance(v[0], str) else v[0]
        if op not in [e.value for e in ComparisonOperator]:
            raise ValueError(f"Invalid operator: {op}. Must be one of: {[e.value for e in ComparisonOperator]}")
        return tuple(v)

    @model_validator(mode="after")
    def validate_price_range(self) -> "PriceFilter":
        """Ensure min_price <= max_price if both are set."""
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError(f"min_price ({self.min_price}) must be <= max_price ({self.max_price})")
        return self


class VolumeFilter(BaseModel):
    """
    Filter stocks by volume metrics.

    Examples:
        # Minimum 1 million shares today
        VolumeFilter(min_volume=1_000_000)

        # Relative volume > 2x average
        VolumeFilter(relative_volume=('gt', 2.0))
    """

    min_volume: Optional[int] = Field(default=None, ge=0, description="Minimum daily volume")
    max_volume: Optional[int] = Field(default=None, ge=0, description="Maximum daily volume")
    min_avg_volume_20d: Optional[int] = Field(
        default=None, ge=0, description="Minimum 20-day average volume"
    )
    relative_volume: Optional[ComparisonFilter] = Field(
        default=None, description="Volume relative to 20-day average: ('gt', 2.0) for 2x"
    )

    @field_validator("relative_volume", mode="before")
    @classmethod
    def validate_comparison_filter(cls, v: Any) -> Optional[ComparisonFilter]:
        """Validate comparison filter format."""
        if v is None:
            return None
        if not isinstance(v, (tuple, list)):
            raise ValueError("Comparison filter must be a tuple: ('gt', 2.0)")
        if len(v) < 2:
            raise ValueError("Comparison filter must have at least 2 elements")
        return tuple(v)

    @model_validator(mode="after")
    def validate_volume_range(self) -> "VolumeFilter":
        """Ensure min_volume <= max_volume if both are set."""
        if self.min_volume is not None and self.max_volume is not None:
            if self.min_volume > self.max_volume:
                raise ValueError(f"min_volume ({self.min_volume}) must be <= max_volume ({self.max_volume})")
        return self


class TechnicalFilter(BaseModel):
    """
    Filter stocks by technical indicators (Finviz-like).

    Examples:
        # RSI below 30 (oversold)
        TechnicalFilter(rsi_14=('lt', 30))

        # Golden cross (50 SMA above 200 SMA)
        TechnicalFilter(sma_crossover=(50, 200, 'above'))

        # Near 52-week high (within 5%)
        TechnicalFilter(pct_from_52w_high=('gt', -5))

        # Strong 1-month performance
        TechnicalFilter(perf_1m=('gt', 10))
    """

    # RSI
    rsi_14: Optional[ComparisonFilter] = Field(
        default=None, description="14-period RSI filter: ('lt', 30) for oversold"
    )

    # Moving Averages
    sma_crossover: Optional[Tuple[int, int, str]] = Field(
        default=None, description="SMA crossover: (fast_period, slow_period, 'above'/'below')"
    )
    pct_from_sma_20: Optional[ComparisonFilter] = Field(
        default=None, description="% distance from 20-day SMA: ('gt', 5) for >5% above"
    )
    pct_from_sma_50: Optional[ComparisonFilter] = Field(
        default=None, description="% distance from 50-day SMA"
    )
    pct_from_sma_200: Optional[ComparisonFilter] = Field(
        default=None, description="% distance from 200-day SMA"
    )

    # MACD
    macd_signal: Optional[str] = Field(
        default=None, description="MACD signal: 'bullish', 'bearish', or 'any'"
    )

    # Bollinger Bands
    bollinger_position: Optional[str] = Field(
        default=None, description="Bollinger position: 'above_upper', 'below_lower', 'middle'"
    )

    # ATR / Volatility
    atr_pct: Optional[ComparisonFilter] = Field(
        default=None, description="ATR as percentage of price: ('gt', 3.0) for >3%"
    )
    historical_volatility_20d: Optional[ComparisonFilter] = Field(
        default=None, description="20-day historical volatility filter"
    )

    # 52-Week High/Low (Finviz-style)
    pct_from_52w_high: Optional[ComparisonFilter] = Field(
        default=None, description="% from 52-week high: ('gt', -10) for within 10% of high"
    )
    pct_from_52w_low: Optional[ComparisonFilter] = Field(
        default=None, description="% from 52-week low: ('gt', 20) for >20% above low"
    )
    new_52w_high: Optional[bool] = Field(
        default=None, description="True to filter for new 52-week highs"
    )
    new_52w_low: Optional[bool] = Field(
        default=None, description="True to filter for new 52-week lows"
    )

    # Performance (Finviz-style)
    perf_1w: Optional[ComparisonFilter] = Field(
        default=None, description="1-week performance %: ('gt', 5) for >5% gain"
    )
    perf_1m: Optional[ComparisonFilter] = Field(
        default=None, description="1-month performance %"
    )
    perf_3m: Optional[ComparisonFilter] = Field(
        default=None, description="3-month performance %"
    )
    perf_6m: Optional[ComparisonFilter] = Field(
        default=None, description="6-month performance %"
    )
    perf_1y: Optional[ComparisonFilter] = Field(
        default=None, description="1-year performance %"
    )
    perf_ytd: Optional[ComparisonFilter] = Field(
        default=None, description="Year-to-date performance %"
    )

    @field_validator(
        "rsi_14", "atr_pct", "historical_volatility_20d",
        "pct_from_sma_20", "pct_from_sma_50", "pct_from_sma_200",
        "pct_from_52w_high", "pct_from_52w_low",
        "perf_1w", "perf_1m", "perf_3m", "perf_6m", "perf_1y", "perf_ytd",
        mode="before"
    )
    @classmethod
    def validate_comparison_filter(cls, v: Any) -> Optional[ComparisonFilter]:
        """Validate comparison filter format."""
        if v is None:
            return None
        if not isinstance(v, (tuple, list)):
            raise ValueError("Comparison filter must be a tuple")
        if len(v) < 2:
            raise ValueError("Comparison filter must have at least 2 elements")
        return tuple(v)

    @field_validator("sma_crossover", mode="before")
    @classmethod
    def validate_sma_crossover(cls, v: Any) -> Optional[Tuple[int, int, str]]:
        """Validate SMA crossover format."""
        if v is None:
            return None
        if not isinstance(v, (tuple, list)) or len(v) != 3:
            raise ValueError("sma_crossover must be (fast_period, slow_period, 'above'/'below')")
        fast, slow, direction = v
        if not isinstance(fast, int) or not isinstance(slow, int):
            raise ValueError("SMA periods must be integers")
        if fast <= 0 or slow <= 0:
            raise ValueError("SMA periods must be positive")
        if fast >= slow:
            raise ValueError(f"Fast period ({fast}) must be less than slow period ({slow})")
        direction_lower = direction.lower() if isinstance(direction, str) else direction
        if direction_lower not in ["above", "below"]:
            raise ValueError("Direction must be 'above' or 'below'")
        return (fast, slow, direction_lower)

    @field_validator("macd_signal", mode="before")
    @classmethod
    def validate_macd_signal(cls, v: Any) -> Optional[str]:
        """Validate MACD signal value."""
        if v is None:
            return None
        v_lower = v.lower() if isinstance(v, str) else v
        if v_lower not in ["bullish", "bearish", "any"]:
            raise ValueError("macd_signal must be 'bullish', 'bearish', or 'any'")
        return v_lower

    @field_validator("bollinger_position", mode="before")
    @classmethod
    def validate_bollinger_position(cls, v: Any) -> Optional[str]:
        """Validate Bollinger Band position value."""
        if v is None:
            return None
        v_lower = v.lower() if isinstance(v, str) else v
        if v_lower not in ["above_upper", "below_lower", "middle"]:
            raise ValueError("bollinger_position must be 'above_upper', 'below_lower', or 'middle'")
        return v_lower


class FundamentalFilter(BaseModel):
    """
    Filter stocks by fundamental metrics (Finviz-like).

    Requires YFinance data provider integration.

    Examples:
        # Large cap only (>$10B)
        FundamentalFilter(min_market_cap=10.0)

        # Value stocks (P/E < 15)
        FundamentalFilter(max_pe_ratio=15.0)

        # Growth at reasonable price (PEG < 1.5)
        FundamentalFilter(max_peg_ratio=1.5)
    """

    # Market Cap
    min_market_cap: Optional[float] = Field(
        default=None, ge=0, description="Minimum market cap in billions USD"
    )
    max_market_cap: Optional[float] = Field(
        default=None, ge=0, description="Maximum market cap in billions USD"
    )

    # Valuation Ratios
    min_pe_ratio: Optional[float] = Field(default=None, description="Minimum P/E ratio (trailing)")
    max_pe_ratio: Optional[float] = Field(default=None, description="Maximum P/E ratio (trailing)")
    min_forward_pe: Optional[float] = Field(default=None, description="Minimum forward P/E ratio")
    max_forward_pe: Optional[float] = Field(default=None, description="Maximum forward P/E ratio")
    min_peg_ratio: Optional[float] = Field(default=None, description="Minimum PEG ratio")
    max_peg_ratio: Optional[float] = Field(default=None, description="Maximum PEG ratio")
    min_pb_ratio: Optional[float] = Field(default=None, description="Minimum Price/Book ratio")
    max_pb_ratio: Optional[float] = Field(default=None, description="Maximum Price/Book ratio")
    min_ps_ratio: Optional[float] = Field(default=None, description="Minimum Price/Sales ratio")
    max_ps_ratio: Optional[float] = Field(default=None, description="Maximum Price/Sales ratio")

    # Growth
    min_eps_growth_yoy: Optional[float] = Field(
        default=None, description="Minimum EPS growth YoY (%)"
    )
    min_revenue_growth_yoy: Optional[float] = Field(
        default=None, description="Minimum revenue growth YoY (%)"
    )

    # Profitability
    min_profit_margin: Optional[float] = Field(
        default=None, description="Minimum profit margin (%)"
    )
    min_operating_margin: Optional[float] = Field(
        default=None, description="Minimum operating margin (%)"
    )
    min_roe: Optional[float] = Field(default=None, description="Minimum Return on Equity (%)")
    min_roa: Optional[float] = Field(default=None, description="Minimum Return on Assets (%)")

    # Financial Health
    max_debt_equity: Optional[float] = Field(
        default=None, description="Maximum Debt/Equity ratio"
    )
    min_current_ratio: Optional[float] = Field(
        default=None, description="Minimum current ratio"
    )

    # Dividends
    min_dividend_yield: Optional[float] = Field(
        default=None, description="Minimum dividend yield (%)"
    )
    max_dividend_yield: Optional[float] = Field(
        default=None, description="Maximum dividend yield (%)"
    )
    min_payout_ratio: Optional[float] = Field(
        default=None, description="Minimum payout ratio (%)"
    )
    max_payout_ratio: Optional[float] = Field(
        default=None, description="Maximum payout ratio (%)"
    )

    # Beta
    min_beta: Optional[float] = Field(default=None, description="Minimum beta")
    max_beta: Optional[float] = Field(default=None, description="Maximum beta")

    # Sector/Industry
    sectors: Optional[List[str]] = Field(default=None, description="List of sectors to include")
    exclude_sectors: Optional[List[str]] = Field(
        default=None, description="List of sectors to exclude"
    )
    industries: Optional[List[str]] = Field(
        default=None, description="List of industries to include"
    )
    exclude_industries: Optional[List[str]] = Field(
        default=None, description="List of industries to exclude"
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "FundamentalFilter":
        """Ensure min <= max for range filters."""
        if self.min_market_cap is not None and self.max_market_cap is not None:
            if self.min_market_cap > self.max_market_cap:
                raise ValueError("min_market_cap must be <= max_market_cap")
        if self.min_pe_ratio is not None and self.max_pe_ratio is not None:
            if self.min_pe_ratio > self.max_pe_ratio:
                raise ValueError("min_pe_ratio must be <= max_pe_ratio")
        if self.min_forward_pe is not None and self.max_forward_pe is not None:
            if self.min_forward_pe > self.max_forward_pe:
                raise ValueError("min_forward_pe must be <= max_forward_pe")
        return self


class ScreenerConfig(BaseModel):
    """
    Complete screener configuration.

    Examples:
        # Screen all Alpaca tradable symbols with basic filters
        config = ScreenerConfig(
            universe=None,  # None = fetch all from Alpaca
            price=PriceFilter(min_price=10, max_price=500),
            volume=VolumeFilter(min_avg_volume_20d=500_000),
            max_results=50
        )

        # Screen from a specific list
        config = ScreenerConfig(
            universe=['AAPL', 'MSFT', 'GOOGL', ...],
            technical=TechnicalFilter(rsi_14=('lt', 30)),
            max_results=20
        )
    """

    universe: Optional[List[str]] = Field(
        default=None, description="List of symbols to screen. None = fetch all tradable from Alpaca"
    )
    price: Optional[PriceFilter] = Field(default=None, description="Price-based filters")
    volume: Optional[VolumeFilter] = Field(default=None, description="Volume-based filters")
    technical: Optional[TechnicalFilter] = Field(default=None, description="Technical indicator filters")
    fundamental: Optional[FundamentalFilter] = Field(
        default=None, description="Fundamental filters (requires YFinance)"
    )
    max_results: int = Field(default=100, ge=1, le=5000, description="Maximum results to return")
    sort_by: str = Field(default="relative_volume", description="Field to sort results by")
    sort_descending: bool = Field(default=True, description="Sort in descending order")

    @field_validator("sort_by", mode="before")
    @classmethod
    def validate_sort_by(cls, v: Any) -> str:
        """Validate sort field."""
        valid_fields = [e.value for e in SortField]
        if v not in valid_fields:
            # Allow custom fields but warn
            pass
        return v

    def has_technical_filters(self) -> bool:
        """Check if any technical filters are set."""
        if self.technical is None:
            return False
        return any(
            [
                self.technical.rsi_14,
                self.technical.sma_crossover,
                self.technical.macd_signal,
                self.technical.bollinger_position,
                self.technical.atr_pct,
                self.technical.historical_volatility_20d,
            ]
        )

    def has_fundamental_filters(self) -> bool:
        """Check if any fundamental filters are set."""
        if self.fundamental is None:
            return False
        return any(
            [
                self.fundamental.min_market_cap,
                self.fundamental.max_market_cap,
                self.fundamental.min_pe_ratio,
                self.fundamental.max_pe_ratio,
                self.fundamental.min_forward_pe,
                self.fundamental.max_forward_pe,
                self.fundamental.sectors,
                self.fundamental.exclude_sectors,
            ]
        )

    def requires_historical_data(self) -> bool:
        """Check if historical data is needed for filtering."""
        if self.has_technical_filters():
            return True
        # Volume filters that need historical data
        if self.volume and (self.volume.min_avg_volume_20d or self.volume.relative_volume):
            return True
        # Price change filters that need historical data
        if self.price and (self.price.change_5d_pct or self.price.change_1m_pct):
            return True
        return False


@dataclass
class ScreenerResult:
    """
    Result for a single screened symbol.

    Contains the symbol and all computed metrics that passed filters.
    """

    symbol: str
    price: float
    change_1d_pct: float
    volume: int
    relative_volume: Optional[float] = None
    avg_volume_20d: Optional[int] = None
    gap_pct: Optional[float] = None
    rsi_14: Optional[float] = None
    atr_pct: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    sector: Optional[str] = None
    passed_filters: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_1d_pct": self.change_1d_pct,
            "volume": self.volume,
            "relative_volume": self.relative_volume,
            "avg_volume_20d": self.avg_volume_20d,
            "gap_pct": self.gap_pct,
            "rsi_14": self.rsi_14,
            "atr_pct": self.atr_pct,
            "market_cap": self.market_cap,
            "pe_ratio": self.pe_ratio,
            "sector": self.sector,
            "passed_filters": self.passed_filters,
        }


def evaluate_comparison(
    value: float, comparison: ComparisonFilter
) -> bool:
    """
    Evaluate a comparison filter against a value.

    Args:
        value: The value to compare
        comparison: Tuple of (operator, threshold) or (operator, low, high) for BETWEEN

    Returns:
        True if the comparison passes, False otherwise
    """
    op = comparison[0].lower() if isinstance(comparison[0], str) else comparison[0]

    if op == ComparisonOperator.GT.value:
        return value > comparison[1]
    elif op == ComparisonOperator.GTE.value:
        return value >= comparison[1]
    elif op == ComparisonOperator.LT.value:
        return value < comparison[1]
    elif op == ComparisonOperator.LTE.value:
        return value <= comparison[1]
    elif op == ComparisonOperator.EQ.value:
        return abs(value - comparison[1]) < 1e-9
    elif op == ComparisonOperator.BETWEEN.value:
        if len(comparison) < 3:
            raise ValueError("BETWEEN operator requires 3 elements: ('between', low, high)")
        return comparison[1] <= value <= comparison[2]
    else:
        raise ValueError(f"Unknown operator: {op}")
