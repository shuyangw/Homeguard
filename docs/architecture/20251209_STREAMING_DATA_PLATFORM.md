# Real-Time Streaming Data Platform

**Date**: 2025-12-09
**Status**: Implementation Complete
**Author**: Claude Code

---

## Executive Summary

This document describes a **fully decoupled** streaming data platform that **replaces** the current polling-based live data system. The platform uses Alpaca's WebSocket API to provide real-time market data to all current and future trading strategies through a single, unified interface.

### Key Principles

1. **Complete Decoupling** - Zero dependencies on existing trading/strategy code
2. **Single Interface** - One `LiveDataProvider` class serves all strategies
3. **Transparent Replacement** - Strategies don't know if data is polled or streamed
4. **Future-Proof** - New strategies plug in without platform changes

---

## Alpaca API Reference

*Official documentation: [alpaca-py SDK](https://alpaca.markets/sdks/python/), [Streaming Market Data](https://docs.alpaca.markets/docs/streaming-market-data)*

### StockDataStream (WebSocket Client)

```python
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

# Constructor
StockDataStream(
    api_key: str,
    secret_key: str,
    raw_data: bool = False,
    feed: DataFeed = DataFeed.IEX,  # DataFeed.SIP for paid
    websocket_params: Optional[Dict] = None,
    url_override: Optional[str] = None
)

# Lifecycle
client.run()   # Starts WebSocket event loop (BLOCKING - run in background thread)
client.stop()  # Stops the WebSocket connection
client.close() # Closes the connection

# Subscription Methods (handler signature: async def handler(data) -> None)
client.subscribe_bars(handler, *symbols)         # Minute bars
client.subscribe_daily_bars(handler, *symbols)   # Daily bars
client.subscribe_updated_bars(handler, *symbols) # Updated minute bars (late trades)
client.subscribe_quotes(handler, *symbols)       # Bid/ask quotes
client.subscribe_trades(handler, *symbols)       # Trade ticks

# Unsubscription
client.unsubscribe_bars(*symbols)
client.unsubscribe_quotes(*symbols)
client.unsubscribe_trades(*symbols)

# Wildcard: Use "*" to subscribe to all symbols
client.subscribe_bars(handler, "*")
```

**IMPORTANT**: `client.run()` is **BLOCKING**. The implementation runs it in a daemon thread:
```python
self._thread = threading.Thread(target=self._run_client, daemon=True)
self._thread.start()
```

### Available Data Feeds

| Feed | Enum | Description | Access |
|------|------|-------------|--------|
| SIP | `DataFeed.SIP` | Securities Information Processor (100% trades) | Paid subscription |
| IEX | `DataFeed.IEX` | Investors Exchange (~2-10% trades) | Free |

### Alpaca Model Schemas

**Bar** (`alpaca.data.models.bars.Bar`):
| Attribute | Type | Description |
|-----------|------|-------------|
| `symbol` | `str` | Ticker identifier |
| `timestamp` | `datetime` | Opening timestamp |
| `open` | `float` | Opening price |
| `high` | `float` | High price |
| `low` | `float` | Low price |
| `close` | `float` | Closing price |
| `volume` | `float` | Volume traded |
| `trade_count` | `Optional[float]` | Number of trades |
| `vwap` | `Optional[float]` | Volume weighted average price |
| `exchange` | `Optional[str]` | Exchange where bar formed |

**Quote** (`alpaca.data.models.quotes.Quote`):
| Attribute | Type | Description |
|-----------|------|-------------|
| `symbol` | `str` | Ticker identifier |
| `timestamp` | `datetime` | Quote submission time |
| `bid_price` | `float` | Bid price |
| `bid_size` | `float` | Bid size |
| `bid_exchange` | `Optional[str]` | Bid exchange |
| `ask_price` | `float` | Ask price |
| `ask_size` | `float` | Ask size |
| `ask_exchange` | `Optional[str]` | Ask exchange |
| `conditions` | `Optional[List[str]]` | Quote conditions |
| `tape` | `Optional[str]` | Quote tape |

**Trade** (`alpaca.data.models.trades.Trade`):
| Attribute | Type | Description |
|-----------|------|-------------|
| `symbol` | `str` | Ticker identifier |
| `timestamp` | `datetime` | Trade submission time |
| `price` | `float` | Trade price |
| `size` | `float` | Trade size |
| `exchange` | `Optional[str]` | Exchange where trade occurred |
| `id` | `Optional[int]` | Trade identifier |
| `conditions` | `Optional[List[str]]` | Trade conditions |
| `tape` | `Optional[str]` | Trade tape |

### REST Fallback (StockHistoricalDataClient)

Used when WebSocket is unavailable:

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest

client = StockHistoricalDataClient('api-key', 'secret-key')

# Latest quote
quote = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols="SPY"))

# Latest trade
trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols="SPY"))

# Historical bars
bars = client.get_stock_bars(StockBarsRequest(
    symbol_or_symbols=["SPY", "TQQQ"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 31)
))
```

---

## Problem Statement

### Current System (Polling)

```python
# Every strategy does this independently
quote = broker.get_latest_quote('TQQQ')    # API call
bars = broker.get_historical_bars(...)      # API call
price = broker.get_latest_trade('TQQQ')     # API call
```

**Issues:**
- Each call = separate API request
- No data sharing between strategies
- Latency: 100-500ms per request
- Rate limits: Risk of throttling with multiple strategies
- Intraday strategies (ORB) would need 360+ calls/day/symbol

### Proposed System (Streaming)

```python
# All strategies share one connection
from src.streaming import LiveDataProvider

provider = LiveDataProvider(api_key, secret_key, feed='sip')
await provider.start(['TQQQ', 'SOXL', 'SPY'])

# Instant access (from memory buffer)
price = provider.get_price('TQQQ')        # No API call
quote = provider.get_quote('TQQQ')        # No API call
bars = provider.get_bars('TQQQ', n=15)    # No API call

# Real-time callbacks for intraday strategies
provider.on_bar(['TQQQ'], my_handler)     # Called on each new bar
```

---

## Architecture

### Dependency Direction

```
┌─────────────────────────────────────────────────────────────┐
│                     src/streaming/                          │
│                                                             │
│  STANDALONE MODULE - No imports from:                       │
│    - src/trading/                                           │
│    - src/strategies/                                        │
│    - src/backtesting/                                       │
│                                                             │
│  Only depends on:                                           │
│    - alpaca-py (StockDataStream)                           │
│    - pandas, asyncio (stdlib)                              │
│    - src/utils/logger (logging only)                       │
│    - src/settings (API keys only)                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Exports
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              LiveDataProvider + Bar, Quote, Trade           │
│              (Public API - this is all strategies see)      │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Used by
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Trading Strategies                         │
│  OMRLiveAdapter, RAMPLiveAdapter, ORBLiveAdapter, Future... │
└─────────────────────────────────────────────────────────────┘
```

### Internal Components

```
┌──────────────────────────────────────────────────────────────────┐
│                      LiveDataProvider                             │
│  (Public Interface - only thing strategies import)               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       MarketDataHub                               │
│  - Coordinates all internal components                            │
│  - Routes data to subscribers                                     │
│  - Manages symbol subscriptions                                   │
└───────┬────────────────────┬────────────────────┬────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ StreamManager │    │   BarBuffer   │    │FallbackPoller │
│  (WebSocket)  │    │   (Memory)    │    │   (Backup)    │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                 Alpaca StockDataStream                            │
│            wss://stream.data.alpaca.markets/v2/{feed}            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Public API

### LiveDataProvider

This is the **only class** that trading code imports.

```python
from src.streaming import LiveDataProvider, Bar, Quote

class LiveDataProvider:
    """
    Single interface for all live market data.
    Replaces broker.get_latest_quote() and broker.get_historical_bars().
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = 'iex'  # 'iex' (free) or 'sip' (paid, full coverage)
    ):
        ...

    # === Lifecycle ===

    async def start(self, symbols: List[str]) -> None:
        """Start streaming and subscribe to symbols."""

    async def stop(self) -> None:
        """Stop streaming and cleanup."""

    def is_connected(self) -> bool:
        """Check if WebSocket is active."""

    # === On-Demand Data (Replaces Polling) ===

    def get_price(self, symbol: str) -> float:
        """Get latest price. Replaces broker.get_latest_trade()."""

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest bid/ask. Replaces broker.get_latest_quote()."""

    def get_bars(self, symbol: str, n: int = None) -> pd.DataFrame:
        """Get recent bars from buffer. Replaces broker.get_historical_bars()."""

    def get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP."""

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current bid-ask spread."""

    # === Real-Time Callbacks ===

    def on_bar(self, symbols: List[str], handler: Callable) -> str:
        """Register callback for new minute bars. Returns subscription ID."""

    def on_quote(self, symbols: List[str], handler: Callable) -> str:
        """Register callback for new quotes."""

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a callback subscription."""
```

### Data Types

*Aligned with Alpaca SDK schemas - see Alpaca API Reference above.*

```python
@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: Optional[float] = None
    vwap: Optional[float] = None
    exchange: Optional[str] = None

    @classmethod
    def from_alpaca(cls, alpaca_bar) -> 'Bar':
        """Convert Alpaca Bar model to our Bar dataclass."""
        ...

@dataclass
class Quote:
    symbol: str
    timestamp: datetime
    bid_price: float    # Note: bid_price, not bid
    bid_size: float
    ask_price: float    # Note: ask_price, not ask
    ask_size: float
    bid_exchange: Optional[str] = None
    ask_exchange: Optional[str] = None
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @classmethod
    def from_alpaca(cls, alpaca_quote) -> 'Quote':
        """Convert Alpaca Quote model to our Quote dataclass."""
        ...

@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    price: float
    size: float
    exchange: Optional[str] = None
    id: Optional[int] = None
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None

    @classmethod
    def from_alpaca(cls, alpaca_trade) -> 'Trade':
        """Convert Alpaca Trade model to our Trade dataclass."""
        ...
```

---

## What This Replaces

| Current (Polling) | New (Streaming) | Notes |
|-------------------|-----------------|-------|
| `broker.get_latest_quote(symbol)` | `provider.get_quote(symbol)` | From memory, no API call |
| `broker.get_latest_trade(symbol)` | `provider.get_price(symbol)` | From memory |
| `broker.get_historical_bars(...)` | `provider.get_bars(symbol, n)` | From buffer |
| Multiple API calls per strategy | Single WebSocket connection | Shared across all strategies |
| 100-500ms latency | <10ms latency | Data already in memory |
| Rate limit concerns | No rate limits | WebSocket is push-based |

---

## Strategy Integration Examples

### OMR (Event-Driven, 2 times/day)

```python
class OMRLiveAdapter:
    def __init__(self, provider: LiveDataProvider, broker, symbols):
        self.provider = provider
        self.broker = broker
        self.symbols = symbols

    async def run_entry_check(self):
        """Called at 3:50 PM"""
        for symbol in self.symbols:
            # Instant - no API call
            price = self.provider.get_price(symbol)
            bars = self.provider.get_bars(symbol, n=390)  # Full day
            # ... generate signals
```

### RAMP (Daily Rebalance, 1 time/day)

```python
class RAMPLiveAdapter:
    async def run_rebalance(self):
        """Called at 3:55 PM"""
        prices = {
            symbol: self.provider.get_price(symbol)
            for symbol in self.symbols
        }
        # ... calculate allocations
```

### ORB (Continuous Intraday)

```python
class ORBLiveAdapter:
    async def start(self):
        """Subscribe to real-time bars"""
        self.provider.on_bar(self.symbols, self._handle_bar)

    async def _handle_bar(self, symbol: str, bar: Bar):
        """Called automatically on each new minute bar"""
        current_time = bar.timestamp.time()

        if time(9, 30) <= current_time < time(9, 45):
            self._update_opening_range(symbol, bar)
        elif time(9, 45) <= current_time < time(15, 30):
            await self._check_breakout(symbol, bar)
```

### Future Strategy (Plug-and-Play)

```python
class FutureStrategyAdapter:
    def __init__(self, provider: LiveDataProvider, ...):
        self.provider = provider
        # No additional setup needed - just use provider

    def get_market_data(self, symbol):
        return {
            'price': self.provider.get_price(symbol),
            'quote': self.provider.get_quote(symbol),
            'vwap': self.provider.get_vwap(symbol),
        }
```

---

## File Structure

```
src/streaming/
├── __init__.py              # Public exports only
├── types.py                 # Bar, Quote, Trade dataclasses
├── live_data_provider.py    # PUBLIC: Main interface
├── _hub.py                  # INTERNAL: Central coordinator
├── _stream.py               # INTERNAL: WebSocket handler
├── _buffer.py               # INTERNAL: In-memory cache
├── _fallback.py             # INTERNAL: Polling backup
└── _utils.py                # INTERNAL: Helpers

config/
└── streaming.yaml           # Configuration

tests/streaming/
├── test_live_data_provider.py
├── test_buffer.py
├── test_stream.py
└── test_integration.py
```

### Package Exports

```python
# src/streaming/__init__.py
"""
Streaming Data Platform - Real-time market data via Alpaca WebSocket.

Public API:
    LiveDataProvider - Main interface (only class strategies need)
    Bar, Quote, Trade - Data types
"""

from src.streaming.live_data_provider import LiveDataProvider
from src.streaming.types import Bar, Quote, Trade

__all__ = ['LiveDataProvider', 'Bar', 'Quote', 'Trade']
```

**Convention:** Internal modules prefixed with underscore (`_hub.py`) are implementation details.

---

## Configuration

```yaml
# config/streaming.yaml
streaming:
  # Data feed
  feed: sip  # 'iex' (free, ~2% coverage) or 'sip' (paid, full)

  # WebSocket
  websocket:
    reconnect_attempts: 5
    reconnect_delay_seconds: 5
    heartbeat_interval_seconds: 30

  # In-memory buffer
  buffer:
    max_bars_per_symbol: 500  # ~8 hours of 1-min bars
    max_quotes_per_symbol: 100

  # Fallback to polling if WebSocket fails
  fallback:
    enabled: true
    poll_interval_seconds: 60
```

---

## Resilience Features

### Auto-Reconnect

```
WebSocket Disconnects
        │
        ▼
┌─────────────────────┐
│ Attempt Reconnect   │──────┐
│ (up to 5 times)     │      │ Success
└─────────┬───────────┘      │
          │ Fail             ▼
          ▼            ┌───────────┐
┌─────────────────────┐│ Resume    │
│ Activate Fallback   ││ Streaming │
│ (Polling Mode)      │└───────────┘
└─────────────────────┘
```

### Fallback Polling

If WebSocket fails completely:
- Strategies continue to work via polling
- `provider.get_price()` falls back to `broker.get_latest_trade()`
- Strategies are unaware of the switch
- Automatic recovery when WebSocket reconnects

### Buffer Persistence

- 500 bars per symbol kept in memory (~8 hours)
- Strategies can access recent history without API calls
- Buffer survives brief disconnects

---

## Alpaca Feed Comparison

| Feature | IEX (Free) | SIP (Paid) |
|---------|------------|------------|
| Coverage | ~2-10% of trades | 100% of trades |
| Symbols | All US equities | All US equities |
| Latency | Same | Same |
| Cost | Free | Subscription required |
| Best For | Paper trading, testing | Production trading |

**Recommendation:** Use IEX for development/paper, SIP for production.

---

## Implementation Phases

### Phase 1: Core Infrastructure (Complete)
- [x] Create `src/streaming/` package structure
- [x] Implement `types.py` (Bar, Quote, Trade)
- [x] Implement `_buffer.py` (in-memory cache)
- [x] Implement `_stream.py` (WebSocket wrapper)
- [x] Unit tests for buffer and types (58 tests passing)

### Phase 2: Public Interface (Complete)
- [x] Implement `_hub.py` (coordinator)
- [x] Implement `_fallback.py` (polling backup)
- [x] Implement `live_data_provider.py` (public API)
- [x] Integration tests with mocked dependencies
- [x] Create `config/streaming.yaml`

### Phase 3: Strategy Migration (Pending)
- [ ] Update `OMRLiveAdapter` to use `LiveDataProvider`
- [ ] Update `RAMPLiveAdapter` to use `LiveDataProvider`
- [ ] Create `ORBLiveAdapter` using real-time callbacks
- [ ] Paper trade all strategies

### Phase 4: Production (Pending)
- [ ] Deploy to EC2
- [ ] Add health monitoring
- [ ] Performance tuning

---

## Success Criteria

| Criteria | Target |
|----------|--------|
| Data latency | <10ms from buffer |
| WebSocket uptime | >99.9% during market hours |
| Reconnect time | <30 seconds |
| Memory usage | <50MB for 100 symbols |
| Strategy changes | Zero (drop-in replacement) |

---

## Dependencies

**Required (already installed):**
- `alpaca-py` - includes `StockDataStream`
- `pandas` - DataFrame support
- `asyncio` - async/await support

**No new packages needed.**

---

## Summary

The streaming data platform provides:

1. **Single WebSocket** serving all strategies
2. **Unified interface** (`LiveDataProvider`) hiding complexity
3. **Memory buffer** for instant data access
4. **Auto-failover** to polling if needed
5. **Zero coupling** with existing trading code
6. **Future-proof** design for new strategies

This replaces scattered `broker.get_latest_*()` calls with a centralized, efficient, real-time data system.
