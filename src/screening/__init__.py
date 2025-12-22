"""
Stock Screener Module.

Provides Finviz-like stock screening using Alpaca's APIs.
Supports price, volume, technical, and fundamental filters.

Usage:
    from src.screening import StockScreener, ScreenerConfig, PriceFilter

    # Initialize screener (IEX feed for paper trading)
    screener = StockScreener(paper=True)

    # Screen all tradable symbols with filters
    config = ScreenerConfig(
        universe=None,  # Fetch all from Alpaca
        price=PriceFilter(min_price=10, max_price=500),
        volume=VolumeFilter(min_avg_volume_20d=500_000),
        technical=TechnicalFilter(rsi_14=('lt', 30)),
        max_results=50
    )

    # Get matching symbols
    symbols = screener.screen(config)

    # Get detailed results
    results = screener.screen_with_details(config)

For fundamental filters (market cap, P/E, etc.), integrate with YFinance:
    from src.data.yfinance import YFinanceProvider

    yf_provider = YFinanceProvider()
    screener = StockScreener(paper=True, yfinance_provider=yf_provider)

    config = ScreenerConfig(
        fundamental=FundamentalFilter(min_market_cap=10.0, max_pe_ratio=20),
        max_results=50
    )
"""

# Main screener class
from src.screening.screener import StockScreener

# Filter models
from src.screening.filters import (
    ComparisonOperator,
    CrossoverDirection,
    FundamentalFilter,
    PriceFilter,
    ScreenerConfig,
    ScreenerResult,
    SortField,
    TechnicalFilter,
    VolumeFilter,
    evaluate_comparison,
)

# Alpaca client
from src.screening.alpaca_client import (
    AlpacaScreenerClient,
    SnapshotData,
)

# Cache
from src.screening.cache import (
    CacheStats,
    ScreenerCache,
    hash_config,
)

# Technical indicators
from src.screening.indicators import (
    IndicatorResult,
    TechnicalIndicators,
    compute_indicators_for_screening,
)

__all__ = [
    # Main class
    "StockScreener",
    # Filters
    "ScreenerConfig",
    "PriceFilter",
    "VolumeFilter",
    "TechnicalFilter",
    "FundamentalFilter",
    "ScreenerResult",
    "ComparisonOperator",
    "CrossoverDirection",
    "SortField",
    "evaluate_comparison",
    # Alpaca client
    "AlpacaScreenerClient",
    "SnapshotData",
    # Cache
    "ScreenerCache",
    "CacheStats",
    "hash_config",
    # Indicators
    "TechnicalIndicators",
    "IndicatorResult",
    "compute_indicators_for_screening",
]
