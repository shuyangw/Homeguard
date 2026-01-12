# CSCM REST API Refactor

**Date:** 2026-01-12
**Status:** Deployed to Production (EC2)

## Summary

Refactored CSCM Demo Trading System to use Binance.US REST API as the primary data source for historical klines, replacing the previous streaming-only approach that was insufficient for 40-day SMA calculation.

## Problem

The original implementation relied on WebSocket streaming to populate a bar buffer, which only held ~24 hours of 1-minute data. This was insufficient for:
- 40-day BTC SMA calculation (regime detection)
- 28-day momentum ranking

When a one-time rebalance was triggered, the system defaulted to "bearish" regime due to insufficient historical data, preventing any positions from being opened.

## Solution

### Architecture Changes

| Component | Before | After |
|-----------|--------|-------|
| Historical Data | Streaming buffer only | REST API primary |
| BTC SMA (40-day) | Aggregated from 1-min bars | Direct daily klines from REST |
| Momentum (28-day) | Aggregated from 1-min bars | Direct daily klines from REST |
| Real-time Quotes | Streaming only | Streaming + REST fallback |
| API Endpoint | binance.com (blocked) | binance.us |

### Files Modified

| File | Changes |
|------|---------|
| `src/data/providers/binance.py` | Changed `BASE_URL` from `api.binance.com` to `api.binance.us` |
| `src/streaming/binance_stream.py` | Changed WebSocket URL to `stream.binance.us:9443` |
| `src/trading/adapters/cscm_demo_adapter.py` | Added `BinanceDataProvider` for REST historical data |
| `src/trading/demo/demo_broker.py` | Added `_rest_provider` and REST fallback for quotes/portfolio values |

### Code Changes

**CSCMDemoAdapter._fetch_historical_data()** - Now fetches 60 days of daily klines via REST:
```python
def _fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
    end = datetime.now()
    start = end - timedelta(days=60)
    return self._data_provider.get_historical_bars_batch(
        self.universe, start, end, '1D'
    )
```

**DemoBroker.get_crypto_quote()** - REST fallback when streaming unavailable:
```python
def get_crypto_quote(self, symbol: str) -> Dict:
    # Try streaming first
    bar = self._bar_buffer.get_latest(symbol)
    if bar is not None:
        return {..., "source": "streaming"}

    # Fall back to REST
    price = self._rest_provider.get_current_price(symbol)
    if price is not None:
        return {..., "source": "rest"}
```

## Verification

### Local Test (macOS)
```
Fetched: 14/14 symbols from Binance REST API
BTC: $92,259.36
SMA: $89,553.37
Regime: bullish
Top 5: ['BCH/USD', 'CRV/USD', 'AVAX/USD', 'DOT/USD', 'SOL/USD']
```

### EC2 Test
```
Fetched: 14/14 symbols from Binance REST API
BTC: $92,254.12
SMA: $89,553.24
Regime: bullish
Top 5: ['BCH/USD', 'CRV/USD', 'AVAX/USD', 'DOT/USD', 'SOL/USD']
```

## Commits

| Hash | Message |
|------|---------|
| `a0824ef` | Switch Binance WebSocket to Binance.US endpoint |
| `5f39750` | Add REST API fallback for CSCM historical data |

## API Rate Limits

Binance.US public endpoints are FREE with generous limits:
- **1,200 requests/minute** for raw requests
- CSCM needs only **14 requests/week** for rebalance (one per symbol)
- Even hourly trailing stop checks would use <0.1% of limit

## EC2 Cost Implications

The REST-only approach enables potential cost savings:

| Schedule | Hours/Month | Cost/Month |
|----------|-------------|------------|
| Current (9 AM - 4:30 PM Mon-Fri) | ~162 | ~$2.72 |
| 24/7 (if needed for trailing stops) | ~730 | ~$12.26 |

With REST API, CSCM can operate without 24/7 streaming. Weekly rebalance can run during market hours window.

## Current Portfolio Status

- **Total Value:** $100,000.00
- **Cash:** $100,000.00
- **Positions:** 0 (all cash)
- **Regime:** Bullish
- **Next Rebalance:** Sunday 00:00 UTC

## Future Considerations

1. **Drop streaming entirely for CSCM** - REST is sufficient for weekly strategy
2. **Keep streaming code** - Useful for future high-frequency strategies
3. **Hourly trailing stop checks** - Could use REST API during market hours

## Related Documentation

- [CSCM Demo Trading System](../strategies/20260108_CSCM_DEMO_TRADING.md) - Updated with new architecture
- [Infrastructure Overview](../INFRASTRUCTURE_OVERVIEW.md)
