# Live Trading Guidelines

This document covers critical issues and best practices when working with the live trading system.

## Common Issues and Pitfalls

### 1. Type Mismatches

**CRITICAL**: Live trading data comes from external APIs with unpredictable types.

```python
# BAD - Assumes numeric
price = data['price']
shares = int(capital / price)  # Fails if price is string!

# GOOD - Explicit type conversion
price = float(data.get('price', 0))
if price > 0:
    shares = int(capital / price)
```

Common type mismatches to watch for:
- `str` vs `int`/`float` for prices and quantities
- `float` vs `Decimal` for financial calculations
- `datetime` vs `str` timestamps (API returns ISO strings)
- `None` vs `0` for missing values

### 2. VIX Data Fetching

**CRITICAL**: VIX data is required for regime detection. Always implement fallbacks.

The system uses a 3-source VIX fallback chain (see `src/trading/adapters/omr_live_adapter.py`):

1. **Primary**: Yahoo Finance (`^VIX`)
2. **Fallback 1**: Alternative Yahoo endpoint
3. **Fallback 2**: Cached/default VIX value

```python
# Example: VIX fetch with fallbacks
def get_vix_with_fallback() -> float:
    """Always returns a VIX value, never fails."""
    # Try primary source
    vix = fetch_vix_yahoo()
    if vix is not None:
        return vix

    # Try alternative source
    vix = fetch_vix_alternative()
    if vix is not None:
        return vix

    # Use cached value as last resort
    logger.warning("Using cached VIX value - all sources failed")
    return get_cached_vix()
```

**Rules**:
- Never block trading if VIX fetch fails - use fallbacks
- Log all VIX fetch failures for monitoring
- Cache successful VIX values for fallback use
- Test VIX fetching independently before deployment

### 3. Bayesian Model Symbol Coverage

**CRITICAL**: The Bayesian model must be trained with ALL symbols in the trading universe.

When the model encounters a symbol it wasn't trained on:
- It cannot generate predictions for that symbol
- The symbol is silently skipped (no trades)
- This causes confusion when "no signals" are generated

**Before Deploying**:
```bash
# Retrain model with current production universe
python scripts/trading/retrain_bayesian_model.py

# Verify model coverage
python -c "
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
model = BayesianReversionModel()
print(f'Model trained on {len(model.trained_symbols)} symbols')
print(f'Symbols: {model.trained_symbols}')
"
```

**Configuration Alignment**:
The trading universe is defined in `config/trading/production.yaml`:
```yaml
symbols:
  - SPY
  - QQQ
  - IWM
  # ... all 20 production symbols
```

The model must be retrained whenever:
1. Adding new symbols to the universe
2. Removing symbols from the universe
3. Updating the model architecture

### 4. Timezone Handling (Broker Data Contract)

**CRITICAL**: AlpacaBroker always returns data in Eastern Time (ET).

Alpaca's API returns all timestamps in UTC. To avoid confusion when filtering by market hours (e.g., `between_time(9:30, 9:35)`), the broker layer converts all returned data to ET before returning.

**Contract**:
```python
# AlpacaBroker.get_bars() and get_historical_bars() return ET data
df = broker.get_historical_bars(symbol, start, end, timeframe='1Min')
# df.index is now in America/New_York timezone
# Safe to use: df.between_time(time(9, 30), time(9, 35))
```

**Why This Matters**:
```python
# WITHOUT broker conversion (old behavior):
# API returns UTC timestamps
df.index  # 2024-01-15 14:30:00+00:00 (UTC)
# between_time(9:30) looks for 9:30 UTC = 4:30 AM ET - WRONG!

# WITH broker conversion (current behavior):
# Broker converts to ET before returning
df.index  # 2024-01-15 09:30:00-05:00 (ET)
# between_time(9:30) correctly finds 9:30 AM ET - CORRECT!
```

**Validation Helper**:
```python
from src.utils.timezone import assert_et_timezone

# Verify data is in ET (raises if not)
assert_et_timezone(df, context="broker data")
```

**Key Files**:
- Broker conversion: `src/trading/brokers/alpaca_broker.py:_ensure_et_timezone()`
- Timezone utility: `src/utils/timezone.py` (`ensure_et_index`, `assert_et_timezone`)

### 5. Market Hours and Schedule

Live trading only executes during market hours:
- **Entry time**: 3:50 PM ET (configurable)
- **Exit time**: 9:35 AM ET (next trading day)
- **Pre-fetch time**: 3:45 PM ET (data caching)

The system automatically:
- Detects market open/closed status
- Skips weekends and holidays
- Uses NYSE calendar for trading days

**Logs to Monitor**:
```
Market: OPEN | Checks: 1640 | Runs: 1 | Signals: 0 | Orders: 0/0
```
- `Market: OPEN/CLOSED` - Current market status
- `Runs` - Number of strategy executions (should be 1+ when market is open)
- `Signals` - Generated trading signals
- `Orders` - Submitted/filled orders

## Deployment Checklist

Before deploying live trading updates:

### 1. Model Verification
- [ ] Bayesian model trained with current universe symbols
- [ ] Model file committed to repository (`models/bayesian_reversion_model.pkl`)
- [ ] Model loads without errors

### 2. Data Source Testing
- [ ] VIX data fetches successfully from all 3 sources
- [ ] Intraday data fetches for all universe symbols
- [ ] Historical data available for regime detection

### 3. Type Safety
- [ ] All API responses have explicit type conversion
- [ ] All calculations handle potential None values
- [ ] Timestamps correctly parsed from strings

### 4. Error Handling
- [ ] All exceptions logged with full context
- [ ] Graceful fallbacks for data source failures
- [ ] Circuit breakers for repeated failures

## EC2 Connection

Use connection scripts in `scripts/ec2/`: `connect.bat` (Windows), `local_connect.sh` (Linux/Mac).
Username is `ec2-user` (not `ubuntu`). See `docs/INFRASTRUCTURE_OVERVIEW.md` for full details.

## Common "No Signals" Causes

When the system generates 0 signals, check:

1. **Strategy Filters Too Strict**
   - `Signal evaluation: X symbols checked, 0 passed filters`
   - Check `min_probability` and `min_return` thresholds

2. **Model Not Covering Symbols**
   - `Bayesian model: LOADED (N symbols)`
   - Verify N matches your universe size

3. **Regime Detection Issues**
   - `Insufficient data for regime classification`
   - Check historical data availability

4. **Market Conditions**
   - Low volatility = few mean reversion opportunities
   - High VIX = strategy may be more selective

5. **Data Quality**
   - Missing intraday data for symbols
   - Stale cache from previous session

## CSCM (Cross-Sectional Crypto Momentum) Strategy

### Overview
CSCM is a weekly crypto momentum strategy that:
- Rebalances on Sunday 0:00 UTC
- Trades 24/7 (crypto markets never close)
- Uses BTC regime filter (BTC > 40-day SMA = bullish)
- Holds top 7 coins by 28-day momentum
- Goes to cash in bearish regime
- **25% trailing stop** for drawdown protection

### Backtest Results (2021-2024)
| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.41 |
| Total Return | 1566% |
| Max Drawdown | 19% |
| Win Rate | ~52% |

### Trailing Stop
The strategy uses a 25% trailing stop to protect gains:
- Tracks peak portfolio value
- If portfolio drops 25% from peak, exits all positions
- Resets on next rebalance day (allows re-entry)
- Reduces max drawdown from 57% to 19%

### Market Hours
Unlike stock strategies, CSCM runs continuously:
- **No market hours check** - Crypto trades 24/7
- **Weekly rebalance** - Sunday 0:00 UTC
- **Check interval** - 1 hour (configurable)

### Broker Architecture
CSCM uses a different broker stack than stock strategies:
- **Primary**: Coinbase Advanced Trade API
- **Secondary**: Alpaca Crypto API (failover)
- **Router**: `CryptoBrokerRouter` handles automatic failover

### Deployment
Separate systemd service for independent operation:
```bash
# Start CSCM strategy
sudo systemctl start homeguard-cscm

# Check status
sudo systemctl status homeguard-cscm

# View logs
journalctl -u homeguard-cscm -f
```

### Key Files
- Entry point: `scripts/trading/run_cscm_live.py`
- Live adapter: `src/trading/adapters/cscm_live_adapter.py`
- Signals: `src/strategies/advanced/cscm_signals.py`
- Config: `config/trading/cscm_live.yaml`
- Service: `scripts/ec2/services/homeguard-cscm.service`

### Common Issues

1. **Coinbase API Credentials**
   - Set `COINBASE_API_KEY` and `COINBASE_API_SECRET` in `.env`
   - Get from https://www.coinbase.com/settings/api

2. **Failover to Alpaca**
   - If Coinbase fails, router switches to Alpaca Crypto
   - Alpaca uses same credentials as stock trading

3. **No Rebalance on Sunday**
   - Check if `_last_rebalance` date matches today
   - Verify UTC timezone handling

## Related Documentation

- Infrastructure: `docs/INFRASTRUCTURE_OVERVIEW.md`
- Health Checks: `docs/HEALTH_CHECK_CHEATSHEET.md`
- Trading Script: `scripts/trading/run_live_paper_trading.py`
- OMR Adapter: `src/trading/adapters/omr_live_adapter.py`
- CSCM Script: `scripts/trading/run_cscm_live.py`
- CSCM Adapter: `src/trading/adapters/cscm_live_adapter.py`
