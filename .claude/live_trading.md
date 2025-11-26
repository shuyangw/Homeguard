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

### 4. Market Hours and Schedule

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

## EC2 Connection Settings

**CRITICAL**: Use the correct SSH settings when connecting to the trading EC2 instance.

### Connection Details
- **Instance IP**: `100.30.95.146` (check AWS console if changed)
- **Username**: `ec2-user` (Amazon Linux 2 - NOT `ubuntu`!)
- **Key file**: `~/.ssh/homeguard-trading.pem`
- **Key name in AWS**: `homeguard-trading`

### Quick Connect (Windows)
```batch
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146
```

### Quick Connect (Linux/Mac)
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146
```

### Connection Script
Use the provided scripts for convenience:
- Windows: `scripts/ec2/connect.bat`
- Linux/Mac: `scripts/ec2/local_connect.sh`

### Common Connection Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Permission denied (publickey)` | Wrong username | Use `ec2-user`, not `ubuntu` |
| `Permission denied (publickey)` | Wrong key file | Check `~/.ssh/homeguard-trading.pem` exists |
| `Connection refused` | Instance stopped | Start instance via AWS console |
| `Connection timed out` | Wrong IP | Check current IP in AWS console |

## Monitoring and Debugging

### Check EC2 Service Status
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo systemctl status homeguard-trading.service"
```

### View Recent Logs
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading.service --since '1 hour ago'"
```

### Check Specific Trading Window (e.g., 3:50 PM ET = 20:50 UTC)
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading.service --since '2025-11-25 20:45' --until '2025-11-25 21:00'"
```

### Restart Service After Updates
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "cd ~/Homeguard && git pull && sudo systemctl restart homeguard-trading.service"
```

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

## Related Documentation

- Infrastructure: `docs/INFRASTRUCTURE_OVERVIEW.md`
- Health Checks: `docs/HEALTH_CHECK_CHEATSHEET.md`
- Trading Script: `scripts/trading/run_live_paper_trading.py`
- OMR Adapter: `src/trading/adapters/omr_live_adapter.py`
