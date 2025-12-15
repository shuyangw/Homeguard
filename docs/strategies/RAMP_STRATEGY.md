# RAMP Strategy Documentation

**Regime-Aware Momentum Protection (RAMP)** - A production trading strategy that adapts momentum parameters based on detected market regimes.

**Status**: Production (Deployed 2025-12-08)
**Last Updated**: 2025-12-08

---

## Overview

RAMP extends the basic momentum protection strategy with market regime detection, dynamically adapting parameters based on whether the market is in a bull, bear, sideways, or unpredictable state.

### Key Features

- **Regime Detection**: Uses VIX, SPY momentum, and moving averages to classify 5 market regimes
- **Adaptive Parameters**: Different momentum periods, weights, and position counts per regime
- **Walk-Forward Validated**: Trained on 2017-2021, tested out-of-sample on 2022-2024
- **Crash Protection**: Reduces exposure when VIX > 25 or SPY drawdown > 5%
- **Dynamic Position Sizing**: 1/N equal-weight allocation across top_n positions

### Performance (Walk-Forward Validation)

| Metric | In-Sample (2017-2021) | Out-of-Sample (2022-2024) |
|--------|----------------------|---------------------------|
| **Sharpe Ratio** | 0.784 | **0.846** |
| **CAGR** | 16.2% | 16.3% |
| **Max Drawdown** | -46.9% | -15.0% |
| **Win Rate** | 53.3% | 52.9% |

**Validation**: 2025-12-12 using Yahoo Finance split-adjusted data.
See full validation report: [`20251212_RAMP_WALK_FORWARD_VALIDATION.md`](20251212_RAMP_WALK_FORWARD_VALIDATION.md)

---

## Strategy Logic

### Universe
- S&P 500 stocks (approximately 500 symbols)
- Excludes leveraged ETFs (to avoid conflict with OMR strategy)

### Momentum Formula
```
momentum_score = (long_weight * return_long_period) - (penalty_weight * return_short_period)
```

The penalty term penalizes recent gains, avoiding buying stocks that have already run up significantly (avoids "buying at the top").

### Regime Detection

RAMP uses `MarketRegimeDetector` to classify the current market into 5 states:

| Regime | Description | Characteristics |
|--------|-------------|-----------------|
| **STRONG_BULL** | Strong uptrend | SPY above 50/200 MA, low VIX, positive momentum |
| **WEAK_BULL** | Weakening uptrend | SPY above MAs but momentum fading |
| **SIDEWAYS** | Range-bound | SPY between MAs, low directional momentum |
| **UNPREDICTABLE** | High uncertainty | Mixed signals, elevated VIX |
| **BEAR** | Downtrend | SPY below MAs, negative momentum |

### Regime-Specific Parameters

| Regime | Long Period | Short Period | Long Weight | Penalty Weight | Top N |
|--------|-------------|--------------|-------------|----------------|-------|
| **STRONG_BULL** | 21 | 5 | 0.3 | 5.0 | 20 |
| **WEAK_BULL** | 21 | 5 | 0.3 | 5.0 | 10 |
| **SIDEWAYS** | 21 | 5 | 0.5 | 2.0 | 5 |
| **UNPREDICTABLE** | 42 | 21 | 0.5 | 4.0 | 10 |
| **BEAR** | 21 | 5 | 0.3 | 3.0 | 10 |

**Rationale**:
- **STRONG_BULL**: More positions (20) for diversification in rising markets
- **SIDEWAYS**: Fewer positions (5) with larger size, lower penalty (2.0) for more stable signals
- **UNPREDICTABLE**: Longer lookback (42/21) to smooth noisy signals
- **BEAR**: Lower penalty (3.0) to catch oversold bounces

---

## Position Sizing

### Dynamic 1/N Allocation

RAMP uses equal-weight position sizing based on current regime's `top_n`:

```
position_pct = max_capital_allocation / current_top_n
target_value = portfolio_value * position_pct
```

### Capital Allocation Examples

With `max_capital_allocation = 1.0` (100%) and **$100,000** portfolio:

| Regime | top_n | Position Size | Per Position | Total Allocated |
|--------|-------|---------------|--------------|-----------------|
| STRONG_BULL | 20 | 5% | $5,000 | $100,000 |
| WEAK_BULL | 10 | 10% | $10,000 | $100,000 |
| SIDEWAYS | 5 | 20% | $20,000 | $100,000 |
| UNPREDICTABLE | 10 | 10% | $10,000 | $100,000 |
| BEAR | 10 | 10% | $10,000 | $100,000 |

**Key Points**:
- RAMP is always fully invested when enough buy signals exist
- Number of positions varies by regime (5-20)
- Each position is equally weighted within the regime
- In uncertain markets (SIDEWAYS), fewer but larger positions reduce noise

---

## Risk Management

### Crash Protection Triggers

| Signal | Threshold | Action |
|--------|-----------|--------|
| High VIX | > 25 | Reduce exposure to 50% |
| SPY Drawdown | > 5% from recent high | Reduce exposure to 50% |

When protection triggers, `exposure_pct = 0.5` (50%), so:
- All position sizes are halved
- Maintains portfolio diversification but with less capital at risk

### Portfolio Health Checks

Before each execution, RAMP verifies:
- Minimum buying power: $5,000
- Minimum portfolio value: $10,000
- Maximum positions: 25 (buffer above max 20)
- Position age: < 48 hours (drift detection)

---

## Schedule

| Time (EST) | Event |
|------------|-------|
| **9:30 AM** | Market opens, preload historical data |
| **3:55 PM** | Execute rebalancing (near close for accurate prices) |
| **4:00 PM** | Market closes |

**Why 3:55 PM?**
- Prices are more stable near close
- Reduces overnight gap risk from end-of-day volatility
- Allows time for order execution before close

---

## Configuration

### Live Adapter Parameters

```python
RAMPLiveAdapter(
    broker=broker,
    symbols=None,                    # Default: S&P 500
    max_capital_allocation=1.0,      # 100% of portfolio
    reduced_exposure=0.5,            # 50% when protection triggers
    vix_threshold=25.0,              # VIX level for protection
    spy_dd_threshold=-0.05,          # 5% SPY drawdown threshold
    slippage_per_share=0.01,         # Expected slippage
    data_provider=None               # Uses broker if not provided
)
```

### Strategy Toggle Configuration

Location: `config/trading/strategy_toggle.yaml`

```yaml
strategies:
  ramp:
    enabled: true
    shutdown_requested: false
  omr:
    enabled: true
    shutdown_requested: false
  mp:
    enabled: false  # Replaced by RAMP
    shutdown_requested: false
```

---

## Deployment

### EC2 Service

Service file: `scripts/ec2/services/homeguard-ramp.service`

```bash
# Start RAMP service
sudo systemctl start homeguard-ramp

# Check status
sudo systemctl status homeguard-ramp

# View logs
journalctl -u homeguard-ramp -f

# Restart after code changes
sudo systemctl restart homeguard-ramp
```

### Multi-Strategy Coordination

RAMP runs alongside OMR with proper isolation:
- Separate systemd services (`homeguard-ramp`, `homeguard-omr`)
- Separate position tracking in `strategy_positions.json`
- Execution locks prevent simultaneous order execution
- Different universes (S&P 500 vs leveraged ETFs)

---

## Code Architecture

### File Locations

| File | Purpose |
|------|---------|
| `src/strategies/advanced/ramp_strategy.py` | Pure signal generation logic |
| `src/trading/adapters/ramp_live_adapter.py` | Live trading adapter |
| `src/strategies/advanced/market_regime_detector.py` | Regime classification |
| `scripts/ec2/services/homeguard-ramp.service` | Systemd service definition |
| `config/trading/strategy_toggle.yaml` | Enable/disable toggle |

### Class Hierarchy

```
StrategyAdapter (base)
    └── RAMPLiveAdapter
            │
            ├── RAMPSignalWrapper (StrategySignals)
            │       └── RAMPSignals (core logic)
            │
            ├── StrategyStateManager (position tracking)
            └── PortfolioHealthChecker (risk checks)
```

### Signal Flow

```
Market Data (Alpaca API)
        ↓
RAMPSignals.generate_signals()
        ↓
MarketRegimeDetector.detect_regime()
        ↓
Calculate momentum with regime-specific params
        ↓
Rank stocks, select top_n
        ↓
Compare with current positions
        ↓
Generate BUY/SELL signals
        ↓
RAMPLiveAdapter.execute_signals()
        ↓
AlpacaBroker.place_stock_order()
        ↓
StrategyStateManager.add_position()
```

---

## Decision History

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-12 | Re-validated with clean data | OOS Sharpe 0.846 (YF split-adjusted data), confirms robustness |
| 2025-12-08 | Deployed RAMP, disabled MP | Regime detection adds value over static momentum |
| 2025-12-08 | Changed to 1/N position sizing | Simpler, matches backtest methodology |
| 2025-12-07 | Walk-forward validation | Confirmed regime detection adds value OOS |
| 2025-12-06 | Optimized regime parameters | Grid search on 2017-2021 data |

---

## Monitoring

### Log Messages

Key log prefixes to watch:
- `[RAMP]` - General RAMP messages
- `[RAMP] Regime:` - Current detected regime
- `[RAMP] Position sizing:` - Per-position allocation
- `[RAMP] Risk signals:` - Protection triggers

### Health Indicators

| Indicator | Good | Warning |
|-----------|------|---------|
| Regime detection | Stable regime for 2+ days | Rapid regime changes |
| Position count | Matches regime's top_n | Significantly different |
| Fill rate | > 95% | < 90% |
| Buying power | > $5,000 | < $5,000 |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No signals generated | Data fetch failed | Check Alpaca API connection |
| Wrong position count | Regime changed | Normal - will rebalance next day |
| Protection triggered | VIX > 25 or SPY drawdown | Expected behavior, positions halved |
| Execution lock timeout | Other strategy running | Wait for release or check logs |

### Debugging Commands

```bash
# Check current regime and positions
journalctl -u homeguard-ramp --since "1 hour ago" | grep -E "Regime:|Position"

# Check for errors
journalctl -u homeguard-ramp --since "1 hour ago" | grep -i error

# View full execution log
journalctl -u homeguard-ramp --since "3:50 PM" --until "4:05 PM"
```

---

## Related Documentation

- [Strategy Framework](../../src/strategies/STRATEGY_FRAMEWORK.md)
- [Live Trading System](../../src/trading/LIVE_TRADING_SYSTEM.md)
- [Multi-Strategy Position Management](../architecture/MULTI_STRATEGY_POSITION_MANAGEMENT.md)
- [Market Regime Detector](../../src/strategies/advanced/market_regime_detector.py)

---

## Changelog

- **2025-12-08**: Initial production deployment, replaced MP strategy
- **2025-12-08**: Documentation created
