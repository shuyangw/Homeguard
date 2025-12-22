# Stock Screener User Guide

A comprehensive guide to using the Stock Screener module for filtering stocks based on price, volume, technical, and fundamental criteria.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Filter Types](#filter-types)
4. [Common Screening Recipes](#common-screening-recipes)
5. [Strategy Integration](#strategy-integration)
6. [Performance Tips](#performance-tips)
7. [API Reference](#api-reference)

## Overview

The Stock Screener module provides Finviz-like stock screening capabilities using Alpaca's market data APIs. It can:

- Screen ALL ~10,000 tradable US equities from Alpaca
- Apply price, volume, technical, and fundamental filters
- Cache data to minimize API calls
- Sort and limit results
- Return symbol lists or detailed metrics

**Key Design Decisions:**
- Screener fetches universe from Alpaca by default (strategies don't provide symbols)
- Strategies call the screener (not vice versa)
- IEX feed for paper trading, SIP for live trading
- Synchronous API only (no async)

## Getting Started

### Basic Usage

```python
from src.screening import StockScreener, ScreenerConfig, PriceFilter

# Create screener (paper=True uses IEX feed)
screener = StockScreener(paper=True)

# Screen all tradable symbols with basic price filter
config = ScreenerConfig(
    universe=None,  # None = fetch all from Alpaca
    price=PriceFilter(min_price=10, max_price=500),
    max_results=50,
)

# Get matching symbols
symbols = screener.screen(config)
print(f"Found {len(symbols)} matching stocks")
```

### Screening a Specific Universe

```python
# Screen only from provided list
config = ScreenerConfig(
    universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    price=PriceFilter(min_price=100),
    max_results=10,
)

symbols = screener.screen(config)
```

### Getting Detailed Results

```python
# Get full metrics for each match
results = screener.screen_with_details(config)

for r in results:
    print(f"{r.symbol}: ${r.price:.2f}, Vol: {r.volume:,}")
```

## Filter Types

### PriceFilter

Filters based on stock price and price changes.

```python
from src.screening import PriceFilter

# Price range
PriceFilter(min_price=10, max_price=500)

# Daily change > 5%
PriceFilter(change_1d_pct=("gt", 5.0))

# Gap down > 3%
PriceFilter(gap_pct=("lt", -3.0))

# Combined
PriceFilter(
    min_price=5,
    max_price=200,
    change_1d_pct=("gt", 2.0),
)
```

### VolumeFilter

Filters based on trading volume.

```python
from src.screening import VolumeFilter

# Minimum volume
VolumeFilter(min_volume=1_000_000)

# Average volume (20-day)
VolumeFilter(min_avg_volume_20d=500_000)

# Relative volume (vs 20-day average)
VolumeFilter(relative_volume=("gt", 2.0))  # 2x normal volume

# Combined high activity
VolumeFilter(
    min_volume=1_000_000,
    relative_volume=("gt", 1.5),
)
```

### TechnicalFilter

Filters based on technical indicators.

```python
from src.screening import TechnicalFilter

# RSI oversold
TechnicalFilter(rsi_14=("lt", 30))

# Golden cross (50 SMA above 200 SMA)
TechnicalFilter(sma_crossover=(50, 200, "above"))

# Death cross
TechnicalFilter(sma_crossover=(50, 200, "below"))

# MACD bullish signal
TechnicalFilter(macd_signal="bullish")

# Below lower Bollinger Band
TechnicalFilter(bollinger_position="below_lower")

# High volatility (ATR > 3% of price)
TechnicalFilter(atr_pct=("gt", 3.0))

# Near 52-week high (within 10%)
TechnicalFilter(pct_from_52w_high=("gt", -10))

# Strong monthly performance
TechnicalFilter(perf_1m=("gt", 10))

# Combined momentum
TechnicalFilter(
    rsi_14=("gt", 50),
    sma_crossover=(50, 200, "above"),
    perf_1m=("gt", 5),
)
```

### FundamentalFilter

Filters based on fundamental data (requires YFinance provider).

```python
from src.screening import FundamentalFilter
from src.data.yfinance import YFinanceFundamentalsProvider

# Create provider and pass to screener
provider = YFinanceFundamentalsProvider()
screener = StockScreener(paper=True, yfinance_provider=provider)

# Market cap filter
FundamentalFilter(min_market_cap=10.0, max_market_cap=100.0)  # $10B - $100B

# Value stocks
FundamentalFilter(
    max_pe_ratio=15.0,
    min_dividend_yield=2.0,
)

# Quality growth
FundamentalFilter(
    min_roe=15.0,
    min_profit_margin=10.0,
    min_revenue_growth=5.0,
)

# Sector filtering
FundamentalFilter(
    sectors=["Technology", "Healthcare"],
)

# Exclude sectors
FundamentalFilter(
    exclude_sectors=["Utilities", "Energy"],
)
```

### Comparison Operators

All tuple-based filters support these operators:

| Operator | Example | Description |
|----------|---------|-------------|
| `gt` | `("gt", 5.0)` | Greater than |
| `gte` | `("gte", 5.0)` | Greater than or equal |
| `lt` | `("lt", 30.0)` | Less than |
| `lte` | `("lte", 30.0)` | Less than or equal |
| `eq` | `("eq", 100)` | Equal to |
| `between` | `("between", 10, 50)` | Between two values |

## Common Screening Recipes

### Momentum Strategy

```python
from src.screening import (
    StockScreener,
    ScreenerConfig,
    PriceFilter,
    VolumeFilter,
    TechnicalFilter,
)

screener = StockScreener(paper=True)

config = ScreenerConfig(
    universe=None,  # All tradable symbols
    price=PriceFilter(min_price=10, max_price=500),
    volume=VolumeFilter(min_avg_volume_20d=500_000),
    technical=TechnicalFilter(
        rsi_14=("gt", 50),
        sma_crossover=(50, 200, "above"),
        perf_1m=("gt", 5),
    ),
    max_results=50,
    sort_by="relative_volume",
    sort_descending=True,
)

momentum_stocks = screener.screen(config)
```

### Mean Reversion / Oversold

```python
config = ScreenerConfig(
    universe=None,
    price=PriceFilter(
        min_price=5,
        max_price=200,
        change_1d_pct=("lt", -3.0),  # Down > 3% today
    ),
    volume=VolumeFilter(
        min_volume=1_000_000,
        relative_volume=("gt", 1.5),  # Higher than normal
    ),
    technical=TechnicalFilter(
        rsi_14=("lt", 30),  # Oversold
        bollinger_position="below_lower",
    ),
    max_results=20,
)

oversold_stocks = screener.screen(config)
```

### Value Investing

```python
from src.data.yfinance import YFinanceFundamentalsProvider

provider = YFinanceFundamentalsProvider()
screener = StockScreener(paper=True, yfinance_provider=provider)

config = ScreenerConfig(
    universe=None,
    price=PriceFilter(min_price=10),
    fundamental=FundamentalFilter(
        min_market_cap=10.0,  # > $10B
        max_pe_ratio=15.0,
        min_dividend_yield=2.0,
        min_roe=12.0,
    ),
    max_results=30,
    sort_by="dividend_yield",
    sort_descending=True,
)

value_stocks = screener.screen(config)
```

### Gap Up Scanners

```python
config = ScreenerConfig(
    universe=None,
    price=PriceFilter(
        min_price=10,
        gap_pct=("gt", 3.0),  # Gap up > 3%
    ),
    volume=VolumeFilter(
        relative_volume=("gt", 2.0),
    ),
    max_results=20,
)

gap_ups = screener.screen(config)
```

### Breakout Scanner

```python
config = ScreenerConfig(
    universe=None,
    price=PriceFilter(min_price=10, max_price=500),
    volume=VolumeFilter(relative_volume=("gt", 2.0)),
    technical=TechnicalFilter(
        pct_from_52w_high=("gt", -5),  # Within 5% of 52-week high
        rsi_14=("between", 50, 70),  # Strong but not overbought
    ),
    max_results=30,
)

breakouts = screener.screen(config)
```

## Strategy Integration

### Adding Screener to a Strategy

```python
from src.strategies.base import BaseStrategy
from src.screening import StockScreener, ScreenerConfig, PriceFilter, VolumeFilter

class MomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.screener = StockScreener(paper=True)

    def select_universe(self) -> list[str]:
        """Dynamically select trading universe."""
        config = ScreenerConfig(
            universe=None,  # Screen all Alpaca symbols
            price=PriceFilter(min_price=10, max_price=500),
            volume=VolumeFilter(min_avg_volume_20d=500_000),
            max_results=100,
        )
        return self.screener.screen(config)

    def generate_signals(self, data):
        # Use dynamically selected universe
        symbols = self.select_universe()
        # ... rest of strategy logic
```

### Caching Considerations

```python
class OptimizedStrategy:
    def __init__(self):
        # Use longer cache TTL for less frequent screening
        self.screener = StockScreener(
            paper=True,
            cache_ttl_seconds=300,  # 5-minute cache
        )

    def run(self):
        # Cache is shared across calls
        # Second screen with same config uses cached data
        pass
```

## Performance Tips

### 1. Provide Universe When Possible

```python
# Slower: Screen all ~10,000 symbols
config = ScreenerConfig(universe=None, ...)

# Faster: Screen specific list
sp500 = load_sp500_symbols()
config = ScreenerConfig(universe=sp500, ...)
```

### 2. Order Filters by Cost

Fast filters (price, volume) run before slow filters (technical, fundamental).
The screener automatically optimizes this, but you can help by:

```python
# Good: Fast filters narrow down before technical calculations
config = ScreenerConfig(
    price=PriceFilter(min_price=10),  # Fast
    volume=VolumeFilter(min_volume=1_000_000),  # Fast
    technical=TechnicalFilter(rsi_14=("lt", 30)),  # Slow (needs historical)
)
```

### 3. Use Cache Stats to Debug

```python
# Check cache performance
stats = screener.get_cache_stats()
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")
```

### 4. Increase Cache TTL for Infrequent Screening

```python
# For daily screening, longer cache is fine
screener = StockScreener(paper=True, cache_ttl_seconds=3600)  # 1 hour
```

## API Reference

### StockScreener

```python
class StockScreener:
    def __init__(
        self,
        paper: bool = True,
        cache_ttl_seconds: int = 60,
        yfinance_provider: Optional[YFinanceFundamentalsProvider] = None,
    ):
        """
        Args:
            paper: Use IEX feed (True) or SIP feed (False)
            cache_ttl_seconds: TTL for snapshot cache
            yfinance_provider: Provider for fundamental data
        """

    def screen(self, config: ScreenerConfig) -> List[str]:
        """Screen and return matching symbol list."""

    def screen_with_details(self, config: ScreenerConfig) -> List[ScreenerResult]:
        """Screen and return detailed results."""

    def clear_cache(self) -> None:
        """Clear all cached data."""

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
```

### ScreenerConfig

```python
class ScreenerConfig(BaseModel):
    universe: Optional[List[str]] = None  # None = all Alpaca symbols
    price: Optional[PriceFilter] = None
    volume: Optional[VolumeFilter] = None
    technical: Optional[TechnicalFilter] = None
    fundamental: Optional[FundamentalFilter] = None
    max_results: int = 100
    sort_by: str = "relative_volume"
    sort_descending: bool = True
```

### ScreenerResult

```python
@dataclass
class ScreenerResult:
    symbol: str
    price: float
    volume: int
    change_pct: Optional[float]
    gap_pct: Optional[float]
    relative_volume: Optional[float]
    indicators: Optional[IndicatorResult]
```

## See Also

- [Module README](../../src/screening/README.md) - Technical documentation
- [YFinance Provider](./YFINANCE_PROVIDER.md) - Fundamental data integration
- [Strategy Base](../../src/strategies/base/) - Strategy integration patterns
