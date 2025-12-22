"""
YFinance Data Module - Fundamental data provider using Yahoo Finance.

This module provides access to fundamental stock data that Alpaca doesn't offer,
including market cap, P/E ratios, sector, dividend info, and more.

Usage:
    from src.data.yfinance import YFinanceFundamentalsProvider, FundamentalData

    provider = YFinanceFundamentalsProvider()

    # Get fundamentals for multiple symbols
    fundamentals = provider.get_fundamentals(['AAPL', 'MSFT', 'GOOGL'])

    # Access specific data
    apple = fundamentals['AAPL']
    print(f"Market Cap: ${apple.market_cap:.1f}B")
    print(f"P/E Ratio: {apple.pe_ratio:.1f}")
    print(f"Sector: {apple.sector}")

    # Filter by criteria
    large_caps = provider.filter_by_market_cap(symbols, min_cap=10.0)
    tech_stocks = provider.filter_by_sector(symbols, include_sectors=['Technology'])

Integration with Screener:
    from src.screening import StockScreener
    from src.data.yfinance import YFinanceFundamentalsProvider

    provider = YFinanceFundamentalsProvider()
    screener = StockScreener(paper=True, yfinance_provider=provider)

    # Now fundamental filters work
    config = ScreenerConfig(
        fundamental=FundamentalFilter(min_market_cap=10.0, max_pe_ratio=25.0)
    )
    symbols = screener.screen(config)
"""

from src.data.yfinance.fundamentals import FundamentalData
from src.data.yfinance.cache import FundamentalsCache
from src.data.yfinance.provider import YFinanceFundamentalsProvider

__all__ = [
    "FundamentalData",
    "FundamentalsCache",
    "YFinanceFundamentalsProvider",
]
