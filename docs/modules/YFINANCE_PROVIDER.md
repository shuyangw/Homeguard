# YFinance Fundamentals Provider User Guide

A comprehensive guide to using the YFinance Fundamentals Provider for accessing stock fundamental data like market cap, P/E ratios, dividends, and sector information.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Available Data Fields](#available-data-fields)
4. [Common Use Cases](#common-use-cases)
5. [Integration with Screener](#integration-with-screener)
6. [Caching](#caching)
7. [API Reference](#api-reference)

## Overview

The YFinance Fundamentals Provider fetches fundamental stock data from Yahoo Finance that Alpaca doesn't provide. This includes:

- Market metrics (market cap, enterprise value)
- Valuation ratios (P/E, PEG, P/B, P/S)
- Profitability metrics (margins, ROE, ROA)
- Dividend information (yield, payout ratio)
- Classification (sector, industry)
- Risk metrics (beta, short interest)

**Key Features:**
- 40+ fundamental data fields
- 24-hour persistent cache (parquet)
- Rate limiting to avoid API blocks
- Batch operations with progress logging

## Getting Started

### Basic Usage

```python
from src.data.yfinance import YFinanceFundamentalsProvider

# Create provider
provider = YFinanceFundamentalsProvider()

# Get fundamentals for a symbol
data = provider.get_single("AAPL")

if data:
    print(f"Symbol: {data.symbol}")
    print(f"Market Cap: ${data.market_cap:.1f}B")
    print(f"P/E Ratio: {data.pe_ratio:.1f}")
    print(f"Sector: {data.sector}")
    print(f"Dividend Yield: {data.dividend_yield:.2f}%")
```

### Batch Fetching

```python
# Fetch multiple symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
fundamentals = provider.get_fundamentals(symbols)

for symbol, data in fundamentals.items():
    print(f"{symbol}: ${data.market_cap:.1f}B, P/E: {data.pe_ratio or 'N/A'}")
```

### Quick Metrics

```python
# Convenience methods for specific fields
market_cap = provider.get_market_cap("AAPL")  # In billions
sector = provider.get_sector("AAPL")
pe = provider.get_pe_ratio("AAPL")
```

## Available Data Fields

### Market Metrics

| Field | Type | Description |
|-------|------|-------------|
| `market_cap` | float | Market capitalization in billions |
| `enterprise_value` | float | Enterprise value in billions |
| `shares_outstanding` | float | Shares outstanding in millions |

### Valuation Ratios

| Field | Type | Description |
|-------|------|-------------|
| `pe_ratio` | float | Trailing P/E ratio |
| `forward_pe` | float | Forward P/E ratio |
| `peg_ratio` | float | PEG ratio (P/E to growth) |
| `pb_ratio` | float | Price-to-book ratio |
| `ps_ratio` | float | Price-to-sales ratio |
| `price_to_cashflow` | float | Price-to-free-cashflow |

### Earnings & Growth

| Field | Type | Description |
|-------|------|-------------|
| `eps` | float | Trailing EPS |
| `forward_eps` | float | Forward EPS estimate |
| `eps_growth_yoy` | float | Year-over-year EPS growth % |
| `revenue_growth` | float | Revenue growth % |

### Profitability

| Field | Type | Description |
|-------|------|-------------|
| `profit_margin` | float | Net profit margin % |
| `operating_margin` | float | Operating margin % |
| `gross_margin` | float | Gross margin % |
| `ebitda_margin` | float | EBITDA margin % |
| `roe` | float | Return on equity % |
| `roa` | float | Return on assets % |
| `roic` | float | Return on invested capital % |

### Financial Health

| Field | Type | Description |
|-------|------|-------------|
| `debt_to_equity` | float | Debt-to-equity ratio |
| `current_ratio` | float | Current ratio |
| `quick_ratio` | float | Quick ratio |
| `interest_coverage` | float | Interest coverage ratio |

### Dividends

| Field | Type | Description |
|-------|------|-------------|
| `dividend_yield` | float | Dividend yield % |
| `dividend_rate` | float | Annual dividend per share |
| `payout_ratio` | float | Payout ratio % |
| `ex_dividend_date` | str | Ex-dividend date (YYYY-MM-DD) |

### Classification

| Field | Type | Description |
|-------|------|-------------|
| `sector` | str | Sector (e.g., "Technology") |
| `industry` | str | Industry (e.g., "Consumer Electronics") |
| `country` | str | Country (e.g., "United States") |
| `exchange` | str | Exchange (e.g., "NASDAQ") |

### Risk Metrics

| Field | Type | Description |
|-------|------|-------------|
| `beta` | float | Beta vs S&P 500 |
| `volatility_52w` | float | 52-week price volatility |
| `short_ratio` | float | Days to cover short interest |
| `short_percent_of_float` | float | Short interest % of float |

### Analyst Data

| Field | Type | Description |
|-------|------|-------------|
| `target_mean_price` | float | Mean analyst target price |
| `target_high_price` | float | Highest analyst target |
| `target_low_price` | float | Lowest analyst target |
| `recommendation` | str | Consensus (buy, hold, sell) |

## Common Use Cases

### Value Stock Analysis

```python
from src.data.yfinance import YFinanceFundamentalsProvider

provider = YFinanceFundamentalsProvider()
symbols = ["JNJ", "PG", "KO", "PEP", "WMT"]

fundamentals = provider.get_fundamentals(symbols)

print("Value Stock Analysis:")
print("-" * 60)
for symbol, data in fundamentals.items():
    print(f"{symbol}:")
    print(f"  P/E: {data.pe_ratio:.1f}" if data.pe_ratio else "  P/E: N/A")
    print(f"  Dividend: {data.dividend_yield:.2f}%" if data.dividend_yield else "  Dividend: N/A")
    print(f"  Payout: {data.payout_ratio:.1f}%" if data.payout_ratio else "  Payout: N/A")
    print()
```

### Filter by Market Cap

```python
provider = YFinanceFundamentalsProvider()

symbols = ["AAPL", "MSFT", "GOOGL", "F", "GE", "WMT", "TGT"]

# Large cap only (> $100B)
large_caps = provider.filter_by_market_cap(symbols, min_cap=100.0)
print(f"Large caps: {large_caps}")

# Mid cap ($10B - $100B)
mid_caps = provider.filter_by_market_cap(symbols, min_cap=10.0, max_cap=100.0)
print(f"Mid caps: {mid_caps}")

# Small cap (< $10B)
small_caps = provider.filter_by_market_cap(symbols, max_cap=10.0)
print(f"Small caps: {small_caps}")
```

### Filter by Sector

```python
provider = YFinanceFundamentalsProvider()

symbols = ["AAPL", "MSFT", "JNJ", "PFE", "XOM", "CVX", "WMT", "TGT"]

# Tech stocks only
tech = provider.filter_by_sector(
    symbols,
    include_sectors=["Technology"]
)
print(f"Tech: {tech}")

# Exclude energy
non_energy = provider.filter_by_sector(
    symbols,
    exclude_sectors=["Energy"]
)
print(f"Non-Energy: {non_energy}")
```

### Quality Score Calculation

```python
from src.data.yfinance import YFinanceFundamentalsProvider

provider = YFinanceFundamentalsProvider()

def calculate_quality_score(data):
    """Simple quality score based on profitability."""
    score = 0

    # ROE > 15% = +1
    if data.roe and data.roe > 15:
        score += 1

    # Profit margin > 10% = +1
    if data.profit_margin and data.profit_margin > 10:
        score += 1

    # Low debt (D/E < 1) = +1
    if data.debt_to_equity and data.debt_to_equity < 100:  # D/E is in percentage
        score += 1

    # Current ratio > 1.5 = +1
    if data.current_ratio and data.current_ratio > 1.5:
        score += 1

    return score

symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
fundamentals = provider.get_fundamentals(symbols)

scores = {}
for symbol, data in fundamentals.items():
    scores[symbol] = calculate_quality_score(data)

# Sort by quality score
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for symbol, score in ranked:
    print(f"{symbol}: Quality Score {score}/4")
```

### Raw Data Access

```python
# Get full yfinance info dict for advanced use
provider = YFinanceFundamentalsProvider()
info = provider.get_info("AAPL")

# Access any yfinance field
print(f"Full Name: {info.get('shortName')}")
print(f"Employees: {info.get('fullTimeEmployees')}")
print(f"52-Week High: ${info.get('fiftyTwoWeekHigh')}")
```

## Integration with Screener

### Enable Fundamental Filters

```python
from src.screening import StockScreener, ScreenerConfig, FundamentalFilter
from src.data.yfinance import YFinanceFundamentalsProvider

# Create provider
provider = YFinanceFundamentalsProvider()

# Pass to screener
screener = StockScreener(paper=True, yfinance_provider=provider)

# Now fundamental filters work
config = ScreenerConfig(
    universe=None,  # Screen all Alpaca symbols
    fundamental=FundamentalFilter(
        min_market_cap=10.0,  # > $10B
        max_pe_ratio=25.0,
        min_roe=15.0,
        sectors=["Technology", "Healthcare"],
    ),
    max_results=50,
)

symbols = screener.screen(config)
```

### Value Investing Screen

```python
config = ScreenerConfig(
    universe=None,
    fundamental=FundamentalFilter(
        min_market_cap=5.0,  # > $5B
        max_pe_ratio=15.0,  # Cheap
        min_dividend_yield=2.0,  # Pays dividends
        max_debt_to_equity=100.0,  # Low leverage
    ),
    max_results=30,
)

value_stocks = screener.screen(config)
```

### Growth Stock Screen

```python
config = ScreenerConfig(
    universe=None,
    fundamental=FundamentalFilter(
        min_market_cap=10.0,
        min_revenue_growth=10.0,  # > 10% growth
        min_profit_margin=10.0,  # Profitable
        sectors=["Technology", "Healthcare", "Consumer Cyclical"],
    ),
    max_results=30,
)

growth_stocks = screener.screen(config)
```

## Caching

### Default Behavior

- Cache TTL: 24 hours (fundamentals change slowly)
- Storage: Parquet file in `.cache/` directory
- Persistent across sessions

### Cache Management

```python
provider = YFinanceFundamentalsProvider()

# View cache statistics
stats = provider.get_cache_stats()
print(f"Cached entries: {stats['entries']}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"TTL: {stats['ttl_hours']} hours")

# Clear cache
provider.clear_cache()

# Evict only expired entries
evicted = provider.evict_expired()
print(f"Evicted {evicted} expired entries")
```

### Custom Cache Settings

```python
# Shorter TTL (12 hours)
provider = YFinanceFundamentalsProvider(cache_ttl_hours=12)

# Custom cache location
provider = YFinanceFundamentalsProvider(
    cache_dir="/path/to/cache",
    cache_ttl_hours=48,  # 2 days
)
```

### Force Refresh

```python
# Skip cache for fresh data
data = provider.get_single("AAPL", skip_cache=True)

# Or for batch
data = provider.get_fundamentals(symbols, skip_cache=True)
```

## API Reference

### YFinanceFundamentalsProvider

```python
class YFinanceFundamentalsProvider:
    def __init__(
        self,
        cache_ttl_hours: float = 24,
        rate_limit_delay: float = 0.25,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            cache_ttl_hours: Cache TTL in hours (default: 24)
            rate_limit_delay: Delay between requests in seconds (default: 0.25)
            cache_dir: Directory for cache file (default: .cache)
        """

    def get_fundamentals(
        self,
        symbols: List[str],
        skip_cache: bool = False,
    ) -> Dict[str, FundamentalData]:
        """Fetch fundamentals for multiple symbols."""

    def get_single(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> Optional[FundamentalData]:
        """Fetch fundamentals for a single symbol."""

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap in billions."""

    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector string."""

    def get_pe_ratio(self, symbol: str) -> Optional[float]:
        """Get trailing P/E ratio."""

    def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get raw yfinance info dict."""

    def filter_by_market_cap(
        self,
        symbols: List[str],
        min_cap: Optional[float] = None,
        max_cap: Optional[float] = None,
    ) -> List[str]:
        """Filter symbols by market cap."""

    def filter_by_sector(
        self,
        symbols: List[str],
        include_sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
    ) -> List[str]:
        """Filter symbols by sector."""

    def clear_cache(self) -> None:
        """Clear all cached data."""

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""

    def evict_expired(self) -> int:
        """Evict expired cache entries."""
```

### FundamentalData

```python
@dataclass
class FundamentalData:
    symbol: str
    market_cap: Optional[float]  # Billions
    pe_ratio: Optional[float]
    # ... (40+ fields, see Available Data Fields)

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
```

## Troubleshooting

### Rate Limiting

If you see "Rate limited" warnings, increase the delay:

```python
provider = YFinanceFundamentalsProvider(rate_limit_delay=0.5)  # 500ms
```

### Missing Data

Not all stocks have all fundamental data. Handle None values:

```python
data = provider.get_single("AAPL")
if data and data.pe_ratio:
    print(f"P/E: {data.pe_ratio:.1f}")
else:
    print("P/E: Not available")
```

### Cache Not Working

Check cache stats and location:

```python
stats = provider.get_cache_stats()
print(f"Cache location: {provider._cache._cache_path}")
print(f"Entries: {stats['entries']}")
```

## See Also

- [Module README](../../src/data/yfinance/README.md) - Technical documentation
- [Stock Screener](./SCREENING.md) - Integration guide
- [yfinance Documentation](https://github.com/ranaroussi/yfinance) - Upstream library
