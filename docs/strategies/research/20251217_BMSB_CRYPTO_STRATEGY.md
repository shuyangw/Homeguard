# Bull Market Support Band (BMSB) Strategy

**Date:** 2025-12-17
**Status:** Backtested
**Asset Class:** Crypto (BTC, ETH)

## Overview

The Bull Market Support Band (BMSB) is a trend-following strategy originally popularized in crypto trading by zkdev on TradingView. It uses a band formed by two moving averages to identify macro trend direction and generate trading signals.

### Core Concept

The strategy creates a "support band" using:
- **SMA** (Simple Moving Average) - default 20 periods
- **EMA** (Exponential Moving Average) - default 21 periods

The band acts as:
- **Support** during bull markets (price bounces off band)
- **Resistance** during bear markets (price rejects at band)

### Signal Logic

| Condition | Signal |
|-----------|--------|
| Price closes above both MAs | Long entry / Short exit |
| Price closes below both MAs | Long exit / Short entry |
| Price inside band | Hold current position |

## Implementation

### Files

| File | Purpose |
|------|---------|
| `src/strategies/advanced/bmsb_indicators.py` | Indicator calculations (SMA, EMA, RSI, ATR, band width) |
| `src/strategies/advanced/bmsb_strategy.py` | Strategy class inheriting from LongShortStrategy |
| `src/strategies/registry.py` | Strategy registration |
| `tests/strategies/test_bmsb_strategy.py` | Unit tests (15 tests) |

### Configuration Parameters

```yaml
strategy:
  name: BMSBStrategy
  parameters:
    # Core parameters
    sma_period: 50          # SMA period (optimized from 20)
    ema_period: 55          # EMA period (optimized from 21)
    timeframe: daily        # 'weekly', 'daily', or 'raw'

    # Trading mode
    long_only: false        # Enable shorts for best performance

    # Signal timing
    signal_on_close: true   # Generate signals on bar close
    require_both_above: true
    require_both_below: true

    # Risk management
    use_trailing_stop: true
    trailing_stop_pct: 0.10  # 10% trailing stop

    # Optional filters (not recommended - see findings)
    use_htf_filter: false    # Weekly trend filter
    use_rsi_filter: false    # RSI momentum filter
    use_atr_stop: false      # ATR-based stops
```

### Registry Names

The strategy can be referenced by any of these names:
- `BMSBStrategy`
- `Bull Market Support Band`
- `BMSB`
- `BMSB Strategy`
- `Bull Market Band`

## Backtest Results

**Test Period:** 2020-01-01 to 2024-12-31
**Symbols:** BTC_USD, ETH_USD
**Initial Capital:** $100,000
**Position Size:** 20% per trade
**Max Positions:** 2

### Configuration Comparison

| Configuration | Total Return | Sharpe | Max DD | Win Rate | Trades |
|---------------|--------------|--------|--------|----------|--------|
| **Longer MA (50/55)** | **+288.65%** | **0.47** | -44.30% | 47.60% | 292 |
| Daily Baseline (20/21) | +135.29% | 0.42 | -43.03% | 42.73% | 330 |
| Weekly (20/21) | +83.44% | 0.35 | -50.00% | 40.00% | 15 |
| Wider Stop (20%) | +132.97% | 0.33 | -50.63% | 36.67% | 180 |
| Long-Only | +30.66% | 0.17 | -43.61% | 32.96% | 179 |
| All Filters (HTF+RSI+Width) | -40.63% | -0.15 | -52.00% | 35.00% | 89 |
| ATR Stops Only | -87.77% | -0.45 | -90.00% | 28.00% | 1,221 |

### Best Configuration

**Winner: Longer MA periods (50/55)**

Config file: `config/backtesting/bmsb_crypto_longer_ma.yaml`

```yaml
strategy:
  name: BMSBStrategy
  parameters:
    sma_period: 50
    ema_period: 55
    timeframe: daily
    long_only: false
    use_trailing_stop: true
    trailing_stop_pct: 0.10
    use_htf_filter: false
    use_rsi_filter: false
    use_atr_stop: false
```

## Key Findings

### What Worked

1. **Longer MA periods (50/55 vs 20/21)**
   - Reduced whipsaws significantly
   - Fewer but higher quality trades
   - More than doubled returns (+288% vs +135%)
   - Best Sharpe ratio (0.47)

2. **Daily timeframe**
   - Good balance of signal frequency and quality
   - Weekly was too slow (only 15 trades in 5 years)

3. **Long/Short mode**
   - Shorts contributed significant alpha during bear markets (2022)
   - Removing shorts cut returns by ~75%

### What Didn't Work

1. **Complex filters (HTF, RSI, band width)**
   - Too restrictive for crypto's strong trending behavior
   - Filtered out profitable signals
   - Turned +135% baseline into -41% loss

2. **ATR-based stops**
   - Crypto volatility made ATR stops too tight
   - Caused massive over-trading (1,221 trades)
   - -88% return - worst performer

3. **Wider trailing stops (20%)**
   - Let losses run too long
   - Slightly worse than 10% stops
   - Higher drawdown (-50% vs -43%)

4. **Long-only mode**
   - Missed bear market profits from shorts
   - Only +31% return vs +135% baseline

## Lessons Learned

1. **Simplicity wins for crypto**: The original indicator concept worked well. Adding equity-style filters hurt performance.

2. **Crypto trends strongly**: Don't filter out signals - the strong trending behavior means most breakouts are real.

3. **Shorts matter**: Unlike equities where "time in market" favors longs, crypto bear markets are severe enough that shorts add significant value.

4. **Fixed % stops beat ATR stops in crypto**: The extreme volatility of crypto makes ATR-based stops impractical - they trigger too frequently.

5. **Longer lookback periods reduce noise**: 50/55 day MAs filter out daily noise while still capturing major trend changes.

## Available Config Files

| File | Description |
|------|-------------|
| `bmsb_crypto_longer_ma.yaml` | **Recommended** - 50/55 MA, best performance |
| `bmsb_crypto_daily.yaml` | Daily baseline with 20/21 MA |
| `bmsb_crypto.yaml` | Weekly timeframe (original indicator) |
| `bmsb_crypto_long_only.yaml` | Long-only weekly |
| `bmsb_crypto_daily_long_only.yaml` | Long-only daily |
| `bmsb_crypto_wide_stop.yaml` | 20% trailing stop |
| `bmsb_crypto_optimized.yaml` | All filters enabled (poor performance) |
| `bmsb_crypto_atr_stop.yaml` | ATR-based stops (poor performance) |

## Usage

### Running a Backtest

```bash
# Recommended configuration
python -m src.backtest_runner --config config/backtesting/bmsb_crypto_longer_ma.yaml

# Daily baseline
python -m src.backtest_runner --config config/backtesting/bmsb_crypto_daily.yaml
```

### Programmatic Usage

```python
from src.strategies.registry import get_strategy_class

# Get strategy class
BMSBStrategy = get_strategy_class("BMSB")

# Create instance with custom parameters
strategy = BMSBStrategy(
    sma_period=50,
    ema_period=55,
    timeframe='daily',
    use_trailing_stop=True,
    trailing_stop_pct=0.10
)
```

## References

- Original TradingView indicator: Bull Market Support Band by zkdev
- Implementation based on weekly 20 SMA / 21 EMA concept
- Adapted for daily timeframe with optimized parameters
