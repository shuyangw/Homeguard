# Streaming Data Platform

**Purpose**: Real-time market data streaming via WebSocket for live trading strategies.

**Performance**: 32x faster than REST polling (167s -> 5s for RAMP's 500 symbols)

---

## Architecture

```
WebSocket Connection (Alpaca)
    |
StreamManager (_stream.py)
    |-> Manages WebSocket lifecycle
    |-> Subscribe/unsubscribe symbols
    |-> Auto-reconnect on disconnect
    |
MarketDataHub (_hub.py)
    |-> Coordinates multiple streams
    |-> Routes messages to buffers
    |
BarBuffer (_buffer.py)
    |-> In-memory cache (500 bars/symbol)
    |-> LRU eviction for memory management
    |
FallbackPoller (_fallback.py)
    |-> REST API backup
    |-> Triggers when buffer < 90% filled
    |
LiveDataProvider (live_data_provider.py)
    |-> Public API for strategies
    |-> get_price(), get_bars(), get_quote()
```

---

## Module Reference

### `live_data_provider.py` - Public API

The only class that strategies should import.

```python
from src.streaming import LiveDataProvider

provider = LiveDataProvider(
    symbols=['TQQQ', 'SOXL', 'UPRO'],
    feed='iex'  # or 'sip' for paid feed
)
provider.start()

# Get current price
price = provider.get_price('TQQQ')  # -> 45.50

# Get recent bars
bars = provider.get_bars('TQQQ', 10)  # -> DataFrame with 10 bars

# Get bid/ask quote
quote = provider.get_quote('TQQQ')  # -> Quote(bid=45.49, ask=45.51)

# Get VWAP
vwap = provider.get_vwap('TQQQ')  # -> 45.48

# Register callback for real-time updates
provider.on_bar(lambda bar: print(f"New bar: {bar.symbol} {bar.close}"))
```

### `_stream.py` - WebSocket Manager

Internal class for WebSocket connection management.

- Connects to `wss://stream.data.alpaca.markets/v2/{feed}`
- Handles authentication and subscription
- Auto-reconnect with exponential backoff
- Message parsing and routing

### `_buffer.py` - In-Memory Cache

Stores recent bars for each symbol.

- Default: 500 bars per symbol
- Thread-safe operations
- LRU eviction when memory limit reached
- Coverage tracking (% of buffer filled)

### `_hub.py` - Stream Coordinator

Coordinates multiple data streams.

- Single point of control for all subscriptions
- Message routing to appropriate buffers
- Connection health monitoring

### `_fallback.py` - REST API Fallback

Automatically falls back to REST API when:
- Buffer coverage < 90%
- WebSocket disconnected
- Before market open (no streaming data yet)
- Mid-day restart (buffer not populated)

### `types.py` - Type Definitions

Data classes aligned with Alpaca models:
- `Bar`: OHLCV bar data
- `Quote`: Bid/ask quotes
- `Trade`: Individual trades

---

## Configuration

### Environment Variables

```bash
# Enable streaming (default: false)
USE_STREAMING=true

# Feed type: 'iex' (free, ~2% coverage) or 'sip' (paid, 100%)
STREAMING_FEED=iex
```

### Usage in Live Trading

```python
import os

if os.getenv('USE_STREAMING', 'false').lower() == 'true':
    from src.streaming import LiveDataProvider
    provider = LiveDataProvider(symbols=universe, feed=os.getenv('STREAMING_FEED', 'iex'))
    provider.start()
else:
    from src.data.providers import CompositeDataProvider
    provider = CompositeDataProvider()
```

---

## Performance Comparison

| Operation | REST Polling | Streaming |
|-----------|--------------|-----------|
| Single price | ~200ms | <10ms |
| 10 bars | ~500ms | <10ms |
| 500 symbols (RAMP) | 150s | 0.5s |
| Real-time updates | N/A | Instant |

---

## Related Documentation

- [DATA_FLOW.md](../../docs/architecture/DATA_FLOW.md#streaming-data-flow) - Streaming data flow diagram
- [20251209_STREAMING_DATA_PLATFORM.md](../../docs/architecture/20251209_STREAMING_DATA_PLATFORM.md) - Full implementation details
- [20251209_STREAMING_FEATURE_FLAG.md](../../docs/architecture/20251209_STREAMING_FEATURE_FLAG.md) - Deployment guide

---

**Last Updated**: 2025-12-15
