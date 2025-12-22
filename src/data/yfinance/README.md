# YFinance Fundamentals Module

Provides fundamental stock data from Yahoo Finance that Alpaca doesn't offer, including market cap, P/E ratios, sector, dividends, and profitability metrics.

## Overview

This module fetches and caches fundamental data for use with the Stock Screener or standalone analysis. It includes:

- **FundamentalData**: Dataclass with 40+ financial metrics
- **FundamentalsCache**: Persistent parquet caching (24-hour TTL)
- **YFinanceFundamentalsProvider**: Main provider class with rate limiting

## Quick Start

```python
from src.data.yfinance import YFinanceFundamentalsProvider

provider = YFinanceFundamentalsProvider()

# Get fundamentals for multiple symbols
fundamentals = provider.get_fundamentals(['AAPL', 'MSFT', 'GOOGL'])

# Access specific data
apple = fundamentals['AAPL']
print(f"Market Cap: ${apple.market_cap:.1f}B")
print(f"P/E Ratio: {apple.pe_ratio:.1f}")
print(f"Sector: {apple.sector}")
print(f"Dividend Yield: {apple.dividend_yield:.2f}%")
```

## Installation

The module uses yfinance which is already a project dependency:

```python
# Dependencies (in requirements.txt)
yfinance>=0.2.0
pandas>=2.0.0
```

## FundamentalData Fields

### Market Metrics
| Field | Type | Description |
|-------|------|-------------|
| `market_cap` | float | Market cap in billions |
| `enterprise_value` | float | Enterprise value in billions |
| `shares_outstanding` | float | Shares outstanding in millions |

### Valuation Ratios
| Field | Type | Description |
|-------|------|-------------|
| `pe_ratio` | float | Trailing P/E ratio |
| `forward_pe` | float | Forward P/E ratio |
| `peg_ratio` | float | PEG ratio |
| `pb_ratio` | float | Price-to-book ratio |
| `ps_ratio` | float | Price-to-sales ratio |
| `price_to_cashflow` | float | Price-to-cashflow ratio |

### Earnings & Growth
| Field | Type | Description |
|-------|------|-------------|
| `eps` | float | Trailing EPS |
| `forward_eps` | float | Forward EPS |
| `eps_growth_yoy` | float | YoY EPS growth % |
| `revenue_growth` | float | Revenue growth % |

### Profitability (% values)
| Field | Type | Description |
|-------|------|-------------|
| `profit_margin` | float | Profit margin % |
| `operating_margin` | float | Operating margin % |
| `gross_margin` | float | Gross margin % |
| `ebitda_margin` | float | EBITDA margin % |

### Returns (% values)
| Field | Type | Description |
|-------|------|-------------|
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
| `dividend_rate` | float | Annual dividend $ |
| `payout_ratio` | float | Payout ratio % |
| `ex_dividend_date` | str | Ex-dividend date (YYYY-MM-DD) |

### Classification
| Field | Type | Description |
|-------|------|-------------|
| `sector` | str | Sector (e.g., "Technology") |
| `industry` | str | Industry (e.g., "Software") |
| `country` | str | Country (e.g., "United States") |
| `exchange` | str | Exchange (e.g., "NASDAQ") |

### Risk Metrics
| Field | Type | Description |
|-------|------|-------------|
| `beta` | float | Beta vs S&P 500 |
| `volatility_52w` | float | 52-week volatility |
| `short_ratio` | float | Short ratio |
| `short_percent_of_float` | float | Short % of float |

### Analyst Targets
| Field | Type | Description |
|-------|------|-------------|
| `target_mean_price` | float | Mean analyst target |
| `target_high_price` | float | High analyst target |
| `target_low_price` | float | Low analyst target |
| `recommendation` | str | Recommendation (buy, hold, sell) |

## Usage Examples

### Get Single Symbol Data

```python
from src.data.yfinance import YFinanceFundamentalsProvider

provider = YFinanceFundamentalsProvider()

# Get single symbol (uses cache if available)
data = provider.get_single("AAPL")

if data:
    print(f"Symbol: {data.symbol}")
    print(f"Market Cap: ${data.market_cap:.1f}B")
    print(f"P/E Ratio: {data.pe_ratio:.1f}")
    print(f"Sector: {data.sector}")
```

### Convenience Methods

```python
provider = YFinanceFundamentalsProvider()

# Quick access to specific metrics
market_cap = provider.get_market_cap("AAPL")  # Returns billions
sector = provider.get_sector("AAPL")
pe = provider.get_pe_ratio("AAPL")
```

### Filter by Market Cap

```python
provider = YFinanceFundamentalsProvider()

symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]

# Filter to large-cap only (>$500B)
large_caps = provider.filter_by_market_cap(symbols, min_cap=500.0)

# Filter to mid-cap ($10B-$100B)
mid_caps = provider.filter_by_market_cap(symbols, min_cap=10.0, max_cap=100.0)
```

### Filter by Sector

```python
provider = YFinanceFundamentalsProvider()

symbols = ["AAPL", "MSFT", "JNJ", "PG", "XOM"]

# Include only Technology
tech = provider.filter_by_sector(
    symbols,
    include_sectors=["Technology"]
)

# Exclude Energy and Utilities
filtered = provider.filter_by_sector(
    symbols,
    exclude_sectors=["Energy", "Utilities"]
)
```

### Integration with Stock Screener

```python
from src.screening import StockScreener, ScreenerConfig, FundamentalFilter
from src.data.yfinance import YFinanceFundamentalsProvider

# Create provider and screener
provider = YFinanceFundamentalsProvider()
screener = StockScreener(paper=True, yfinance_provider=provider)

# Screen with fundamental filters
config = ScreenerConfig(
    universe=None,  # Screen all Alpaca tradable symbols
    fundamental=FundamentalFilter(
        min_market_cap=10.0,  # >$10B
        max_pe_ratio=25.0,
        min_roe=15.0,
        sectors=["Technology", "Healthcare"],
    ),
    max_results=50,
)

symbols = screener.screen(config)
```

### Batch Fetch with Progress

```python
provider = YFinanceFundamentalsProvider()

# Fetches ~500 symbols with progress logging
symbols = ["AAPL", "MSFT", ...]  # Large list
fundamentals = provider.get_fundamentals(symbols)

# Progress logged every 50 symbols:
# [YFinanceFundamentals] Progress: 50/500
# [YFinanceFundamentals] Progress: 100/500
# ...
```

### Force Refresh (Skip Cache)

```python
provider = YFinanceFundamentalsProvider()

# Force fresh data from yfinance
data = provider.get_single("AAPL", skip_cache=True)

# Or for batch
data = provider.get_fundamentals(["AAPL", "MSFT"], skip_cache=True)
```

## Caching

### Cache Behavior

| Setting | Default | Description |
|---------|---------|-------------|
| TTL | 24 hours | Time before entries expire |
| Storage | Parquet | Persistent across sessions |
| Location | `.cache/` | Project root `.cache` directory |

### Cache Management

```python
provider = YFinanceFundamentalsProvider()

# Get cache statistics
stats = provider.get_cache_stats()
print(f"Entries: {stats['entries']}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")

# Clear all cached data
provider.clear_cache()

# Evict only expired entries
evicted = provider.evict_expired()
print(f"Evicted {evicted} expired entries")
```

### Custom Cache Location

```python
provider = YFinanceFundamentalsProvider(
    cache_dir="/path/to/custom/cache",
    cache_ttl_hours=12,  # Shorter TTL
)
```

## Rate Limiting

The provider includes automatic rate limiting to avoid Yahoo Finance blocks:

| Setting | Default | Description |
|---------|---------|-------------|
| `rate_limit_delay` | 0.25s | Delay between requests |

```python
# Custom rate limiting
provider = YFinanceFundamentalsProvider(
    rate_limit_delay=0.5,  # 500ms between requests
)
```

## Raw Data Access

For advanced use cases, access the raw yfinance info dict:

```python
provider = YFinanceFundamentalsProvider()

# Get raw info dict (bypasses cache)
info = provider.get_info("AAPL")

# Access any yfinance field
print(info.get("52WeekHigh"))
print(info.get("shortName"))
print(info.get("fullTimeEmployees"))
```

## Error Handling

The provider handles errors gracefully:

```python
provider = YFinanceFundamentalsProvider()

# Returns None for invalid symbols
data = provider.get_single("INVALID_SYMBOL")
if data is None:
    print("Symbol not found")

# Batch operations skip failed symbols
result = provider.get_fundamentals(["AAPL", "INVALID"])
# result = {"AAPL": FundamentalData(...)}  # INVALID skipped
```

## Module Structure

```
src/data/yfinance/
    __init__.py          # Public exports
    fundamentals.py      # FundamentalData dataclass
    cache.py             # FundamentalsCache with parquet storage
    provider.py          # YFinanceFundamentalsProvider class
    README.md            # This file
```

## Testing

```bash
# Run all yfinance tests
pytest tests/data/yfinance/ -v

# Run specific test file
pytest tests/data/yfinance/test_provider.py -v
```

## Known Limitations

1. **Rate Limiting**: Yahoo Finance may block requests if called too frequently. The default 0.25s delay helps prevent this.

2. **Data Availability**: Not all stocks have all fundamental data. Fields return `None` when unavailable.

3. **Data Freshness**: yfinance data may have slight delays compared to real-time sources.

4. **Batch Performance**: Large batches (1000+ symbols) may take several minutes due to rate limiting.

## See Also

- [Stock Screener](../../screening/README.md) - Uses this provider for fundamental filters
- [yfinance Documentation](https://github.com/ranaroussi/yfinance) - Upstream library
