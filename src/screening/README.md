# Stock Screener Module

A Finviz-like stock screener using Alpaca's market data APIs. Screens stocks based on price, volume, technical, and fundamental filters.

## Overview

The screener fetches all tradable US equity symbols from Alpaca by default, then applies configurable filters to narrow down the universe. Strategies call the screener to get symbols matching their criteria.

**Key Features:**
- Fetches ~10,000 tradable US equities from Alpaca
- Price, volume, technical, and fundamental filters
- In-memory caching with configurable TTLs
- IEX feed (free) with SIP migration path
- Thread-safe for concurrent usage
- Pydantic models for type-safe configuration

## Quick Start

```python
from src.screening import StockScreener, ScreenerConfig, PriceFilter, VolumeFilter

# Create screener (paper=True uses IEX feed)
screener = StockScreener(paper=True)

# Screen ALL tradable symbols with basic filters
config = ScreenerConfig(
    universe=None,  # None = fetch all from Alpaca
    price=PriceFilter(min_price=10, max_price=500),
    volume=VolumeFilter(min_volume=1_000_000),
    max_results=50,
)

# Get matching symbols
symbols = screener.screen(config)
print(f"Found {len(symbols)} matching stocks")
```

## Installation

The module uses Alpaca's SDK which is already a project dependency:

```python
# Dependencies (in requirements.txt)
alpaca-py>=0.10.0
pandas>=2.0.0
pydantic>=2.0.0
```

## Filter Reference

### PriceFilter

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `min_price` | float | Minimum stock price | `10.0` |
| `max_price` | float | Maximum stock price | `500.0` |
| `change_1d_pct` | tuple | 1-day change percentage | `("gt", 5.0)` |
| `gap_pct` | tuple | Gap from previous close | `("gt", 2.0)` |

```python
# Stocks between $10-$500 with >5% daily gain
PriceFilter(min_price=10, max_price=500, change_1d_pct=("gt", 5.0))
```

### VolumeFilter

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `min_volume` | int | Minimum daily volume | `1_000_000` |
| `max_volume` | int | Maximum daily volume | `100_000_000` |
| `min_avg_volume_20d` | int | Min 20-day average volume | `500_000` |
| `relative_volume` | tuple | Volume vs 20-day average | `("gt", 2.0)` |

```python
# High volume stocks with 2x normal volume
VolumeFilter(min_volume=1_000_000, relative_volume=("gt", 2.0))
```

### TechnicalFilter

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `rsi_14` | tuple | RSI(14) condition | `("lt", 30)` |
| `sma_crossover` | tuple | SMA crossover (fast, slow, dir) | `(50, 200, "above")` |
| `macd_signal` | str | MACD signal type | `"bullish"` |
| `bollinger_position` | str | Position vs Bollinger Bands | `"below_lower"` |
| `atr_pct` | tuple | ATR as % of price | `("gt", 3.0)` |
| `pct_from_52w_high` | tuple | Distance from 52-week high | `("gt", -10)` |
| `pct_from_52w_low` | tuple | Distance from 52-week low | `("lt", 20)` |
| `perf_1w` | tuple | 1-week performance % | `("gt", 5)` |
| `perf_1m` | tuple | 1-month performance % | `("gt", 10)` |
| `perf_3m` | tuple | 3-month performance % | `("gt", 15)` |
| `perf_6m` | tuple | 6-month performance % | `("gt", 20)` |
| `perf_1y` | tuple | 1-year performance % | `("gt", 25)` |
| `perf_ytd` | tuple | Year-to-date performance % | `("gt", 10)` |
| `sma_20_distance` | tuple | Distance from SMA(20) % | `("lt", 5)` |
| `sma_50_distance` | tuple | Distance from SMA(50) % | `("lt", 10)` |
| `sma_200_distance` | tuple | Distance from SMA(200) % | `("gt", 0)` |

```python
# Oversold stocks with golden cross
TechnicalFilter(
    rsi_14=("lt", 30),
    sma_crossover=(50, 200, "above"),
)
```

### FundamentalFilter

Requires YFinance data provider integration.

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `min_market_cap` | float | Min market cap (billions) | `1.0` |
| `max_market_cap` | float | Max market cap (billions) | `100.0` |
| `min_pe_ratio` | float | Min P/E ratio | `5.0` |
| `max_pe_ratio` | float | Max P/E ratio | `25.0` |
| `max_peg_ratio` | float | Max PEG ratio | `1.5` |
| `min_pb_ratio` | float | Min price-to-book | `0.5` |
| `max_pb_ratio` | float | Max price-to-book | `3.0` |
| `min_ps_ratio` | float | Min price-to-sales | `0.5` |
| `max_ps_ratio` | float | Max price-to-sales | `5.0` |
| `min_eps_growth_yoy` | float | Min YoY EPS growth % | `10.0` |
| `min_revenue_growth` | float | Min revenue growth % | `5.0` |
| `min_profit_margin` | float | Min profit margin % | `10.0` |
| `min_operating_margin` | float | Min operating margin % | `15.0` |
| `min_roe` | float | Min return on equity % | `15.0` |
| `min_roa` | float | Min return on assets % | `5.0` |
| `max_debt_to_equity` | float | Max debt/equity ratio | `1.0` |
| `min_current_ratio` | float | Min current ratio | `1.5` |
| `sectors` | list | Include only these sectors | `["Technology"]` |
| `exclude_sectors` | list | Exclude these sectors | `["Utilities"]` |
| `industries` | list | Include only these industries | `["Software"]` |
| `min_dividend_yield` | float | Min dividend yield % | `2.0` |
| `max_dividend_yield` | float | Max dividend yield % | `6.0` |
| `max_payout_ratio` | float | Max payout ratio % | `60.0` |
| `min_beta` | float | Min beta | `0.5` |
| `max_beta` | float | Max beta | `1.5` |

```python
# Value stocks with good fundamentals
FundamentalFilter(
    min_market_cap=10.0,  # >$10B
    max_pe_ratio=15.0,
    min_roe=15.0,
    min_dividend_yield=2.0,
)
```

### Comparison Operators

All tuple filters support these operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `gt` | Greater than | `("gt", 5.0)` |
| `gte` | Greater than or equal | `("gte", 5.0)` |
| `lt` | Less than | `("lt", 30.0)` |
| `lte` | Less than or equal | `("lte", 30.0)` |
| `eq` | Equal to | `("eq", 100)` |
| `between` | Between two values | `("between", 10, 50)` |

## Usage Examples

### Screen from Provided Universe

```python
from src.screening import StockScreener, ScreenerConfig, PriceFilter

screener = StockScreener(paper=True)

# Screen from specific list instead of all Alpaca symbols
config = ScreenerConfig(
    universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    price=PriceFilter(min_price=100),
    max_results=10,
)

symbols = screener.screen(config)
```

### Screen with Details

```python
from src.screening import StockScreener, ScreenerConfig, VolumeFilter

screener = StockScreener(paper=True)

config = ScreenerConfig(
    universe=["AAPL", "MSFT", "NVDA"],
    volume=VolumeFilter(relative_volume=("gt", 1.5)),
    sort_by="relative_volume",
    sort_descending=True,
)

# Get detailed results with all metrics
results = screener.screen_with_details(config)

for r in results:
    print(f"{r.symbol}: ${r.price:.2f}, Vol: {r.volume:,}")
```

### Momentum Strategy Screener

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
    universe=None,  # Screen ALL tradable symbols
    price=PriceFilter(min_price=10, max_price=500),
    volume=VolumeFilter(min_avg_volume_20d=500_000),
    technical=TechnicalFilter(
        rsi_14=("gt", 50),
        sma_crossover=(50, 200, "above"),
        perf_1m=("gt", 5),
    ),
    max_results=50,
    sort_by="volume",
    sort_descending=True,
)

symbols = screener.screen(config)
```

### Mean Reversion Screener

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
    universe=None,
    price=PriceFilter(
        min_price=5,
        max_price=200,
        change_1d_pct=("lt", -3.0),  # Down >3% today
    ),
    volume=VolumeFilter(
        min_volume=1_000_000,
        relative_volume=("gt", 1.5),  # Higher than normal volume
    ),
    technical=TechnicalFilter(
        rsi_14=("lt", 30),  # Oversold
        bollinger_position="below_lower",
    ),
    max_results=20,
)

symbols = screener.screen(config)
```

## Caching

The screener uses in-memory caching to reduce API calls:

| Data Type | Default TTL | Description |
|-----------|-------------|-------------|
| Snapshots | 60 seconds | Current price/volume data |
| Historical | 1 hour | Daily bars for technicals |
| Results | 30 seconds | Cached screen results |
| Assets | 1 hour | Tradable asset list |

```python
# Custom cache TTL
screener = StockScreener(paper=True, cache_ttl_seconds=120)

# Clear cache manually
screener.clear_cache()

# Get cache statistics
stats = screener.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1f}%")
```

## IEX vs SIP Feed

| Feed | Subscription | Delay | Use Case |
|------|--------------|-------|----------|
| IEX | Free | 15-minute | Paper trading, testing |
| SIP | Paid | Real-time | Live trading |

```python
# IEX feed (paper trading)
screener = StockScreener(paper=True)

# SIP feed (live trading - requires subscription)
screener = StockScreener(paper=False)
```

The screener automatically uses IEX for paper mode and SIP for live mode, matching the existing AlpacaBroker pattern.

## API Reference

### StockScreener

```python
class StockScreener:
    def __init__(
        self,
        paper: bool = True,
        cache_ttl_seconds: int = 60,
        yfinance_provider: Optional[Any] = None,
    ):
        """
        Initialize the stock screener.

        Args:
            paper: Use IEX feed (True) or SIP feed (False)
            cache_ttl_seconds: TTL for snapshot cache
            yfinance_provider: Optional YFinanceProvider for fundamentals
        """

    def screen(self, config: ScreenerConfig) -> List[str]:
        """
        Screen stocks and return matching symbol list.

        Args:
            config: Screening configuration

        Returns:
            List of matching symbol strings
        """

    def screen_with_details(
        self, config: ScreenerConfig
    ) -> List[ScreenerResult]:
        """
        Screen stocks and return detailed results.

        Args:
            config: Screening configuration

        Returns:
            List of ScreenerResult with full metrics
        """

    def clear_cache(self) -> None:
        """Clear all cached data."""

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics."""
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

## Error Handling

The screener handles API errors gracefully:

```python
try:
    symbols = screener.screen(config)
except Exception as e:
    logger.error(f"Screening failed: {e}")
    symbols = []
```

Common errors:
- Rate limiting: Automatic retry with exponential backoff
- Missing data: Symbols with missing snapshots are skipped
- API errors: Logged and re-raised for caller handling

## Thread Safety

The screener is thread-safe and can be used from multiple threads:

```python
from concurrent.futures import ThreadPoolExecutor

screener = StockScreener(paper=True)
configs = [config1, config2, config3]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(screener.screen, configs))
```

## Performance Tips

1. **Provide a universe when possible** - Screening 10,000 symbols is slower than screening 500
2. **Use fast filters first** - Price/volume filters reduce candidates before technical calculations
3. **Cache aggressively** - Increase cache TTL for less time-sensitive applications
4. **Batch requests** - The client batches API requests automatically (200 symbols per request)

## Module Structure

```
src/screening/
    __init__.py          # Public exports
    screener.py          # Main StockScreener class
    filters.py           # Pydantic filter models
    cache.py             # In-memory TTL cache
    alpaca_client.py     # Alpaca API wrapper
    indicators.py        # Technical indicator calculations
    README.md            # This file
```

## Testing

```bash
# Run all screening tests
pytest tests/screening/ -v

# Run specific test file
pytest tests/screening/test_screener.py -v
```

## See Also

- [YFinance Provider](../data/yfinance/README.md) - Fundamental data integration
- [Alpaca Documentation](https://docs.alpaca.markets/) - Alpaca API reference
- [Strategy Integration](../../docs/modules/SCREENING.md) - User guide for strategy integration
