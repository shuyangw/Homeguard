# Opening Range Breakout (ORB) Strategy

**Status**: Research/Backtesting
**Last Updated**: 2025-12-08

---

## Overview

The Opening Range Breakout (ORB) strategy trades breakouts from the first 15 minutes of trading (9:30-9:45 AM ET). It uses multiple confirmation filters to improve signal quality and is optimized for high-liquidity leveraged ETFs.

### Key Features

- **15-minute opening range** (9:30-9:45 AM ET) - consensus best timeframe
- **Multi-filter confirmation**: RVOL, VWAP, EMA trend, regime
- **Intraday only**: All positions closed by 3:45 PM ET (no overnight risk)
- **Long-short capable**: Can trade both directions based on breakout type

### Expected Performance (Research Benchmarks)

Per academic research:
- **Filtered ORB on stocks**: 2.81 Sharpe, 36% annualized alpha (Zarattini et al. 2024)
- **NQ futures 15-min ORB**: 74.56% win rate, 2.51 profit factor
- **TQQQ ORB**: 1,485% returns vs 26.7% passive (2016-2023)

Realistic post-cost expectations: **1.0-1.5 Sharpe** for systematic implementation.

---

## Strategy Logic

### Opening Range Calculation

```
Time Window: 9:30 AM - 9:45 AM ET (first 15 minutes)
OR High:     Max of all highs during window
OR Low:      Min of all lows during window
OR Height:   OR High - OR Low
```

### Entry Conditions

#### Long Entry (All must be true)
1. Time > 9:45 AM ET (OR complete)
2. Close > OR High (price breakout)
3. RVOL > 1.5x (volume confirmation)
4. Close > VWAP (buyer control)
5. 9 EMA > 21 EMA (bullish trend)
6. Regime != BEAR (regime filter)

#### Short Entry (All must be true)
1. Time > 9:45 AM ET
2. Close < OR Low (price breakdown)
3. RVOL > 1.5x
4. Close < VWAP (seller control)
5. 9 EMA < 21 EMA (bearish trend)
6. Regime != STRONG_BULL

### Exit Conditions

| Exit Type | Long | Short |
|-----------|------|-------|
| Stop-Loss | OR Low | OR High |
| Target | Entry + 1x OR Height | Entry - 1x OR Height |
| Time Exit | 3:45 PM ET | 3:45 PM ET |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opening_range_minutes` | 15 | Minutes for OR calculation (5, 10, 15, 30, 60) |
| `rvol_threshold` | 1.5 | Minimum RVOL for entry |
| `rvol_lookback` | 20 | Bars for RVOL average |
| `fast_ema` | 9 | Fast EMA period |
| `slow_ema` | 21 | Slow EMA period |
| `target_multiplier` | 1.0 | Target = Entry + (mult * OR Height) |
| `long_only` | False | Skip short trades |
| `use_regime` | True | Enable regime filtering |
| `use_gap_filter` | False | Align with gap direction |

---

## Best Instruments

### Primary Universe (3x Leveraged ETFs)

| Symbol | Description | Volume |
|--------|-------------|--------|
| TQQQ | 3x Nasdaq | Highest liquidity |
| SQQQ | 3x Nasdaq Bear | |
| SOXL | 3x Semiconductor | High volatility |
| SOXS | 3x Semiconductor Bear | |
| UPRO | 3x S&P 500 | |
| TNA | 3x Small Cap | |
| TECL | 3x Technology | |

### Why Leveraged ETFs?

1. **High liquidity** - Clean breakouts, tight spreads
2. **High beta** - Stronger breakout moves
3. **Predictable patterns** - Well-defined opening ranges
4. **No catalyst screening** - Unlike individual stocks

---

## Regime Behavior

Uses `MarketRegimeDetector` for adaptive filtering:

| Regime | Long Trades | Short Trades | Notes |
|--------|-------------|--------------|-------|
| STRONG_BULL | Yes | **No** | Skip counter-trend shorts |
| WEAK_BULL | Yes | Yes | Full capability |
| SIDEWAYS | Yes | Yes | Full capability |
| UNPREDICTABLE | Yes | Yes | Consider reducing size |
| BEAR | **No** | Yes | Skip counter-trend longs |

---

## Schedule

| Time (ET) | Action |
|-----------|--------|
| 9:30 AM | Market opens, OR formation begins |
| 9:45 AM | OR complete, entry scanning begins |
| 3:30 PM | Entry cutoff (no new trades) |
| 3:45 PM | Force exit all positions |
| 4:00 PM | Market close |

---

## Usage

### Backtest (Single)

```bash
python -m src.backtest_runner --config config/backtesting/orb_single.yaml
```

### Parameter Optimization

```bash
python -m src.backtest_runner --config config/backtesting/orb_sweep.yaml
```

### Walk-Forward Validation

```bash
python -m src.backtest_runner --config config/backtesting/orb_walk_forward.yaml
```

### Programmatic Usage

```python
from src.strategies.advanced.orb_strategy import ORBStrategy

strategy = ORBStrategy(
    opening_range_minutes=15,
    rvol_threshold=1.5,
    target_multiplier=1.0,
    long_only=False,
    use_regime=True
)

# Get parameters
params = strategy.get_parameters()

# Set regime (from external detector)
strategy.set_regime('STRONG_BULL')

# Generate signals (called by backtest engine)
long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(minute_data)
```

---

## Data Requirements

- **Timeframe**: 1-minute bars
- **Required columns**: open, high, low, close, volume, vwap
- **Schema**: Canonical 8-column parquet format
- **Historical**: 20+ days for RVOL calculation

---

## Risk Management

### Per-Trade Risk
- Stop-loss = Opposite side of opening range
- Risk = Entry - Stop (for longs)
- Default R:R = 1:1 (target_multiplier=1.0)

### Position Sizing
- Default: 10% of capital per trade
- Max positions: 5 concurrent

### Why Intraday Only?
- Avoids overnight gap risk
- No conflict with OMR strategy (overnight)
- Cleaner risk management

---

## Troubleshooting

### No Signals Generated

1. **Low RVOL**: Volume not exceeding threshold
2. **EMA Filter**: Trend not aligned with breakout direction
3. **VWAP Filter**: Price on wrong side of VWAP
4. **Regime Filter**: BEAR blocking longs or STRONG_BULL blocking shorts
5. **Entry Cutoff**: After 3:30 PM - no new entries

### Too Many False Breakouts

1. Increase `rvol_threshold` (try 2.0)
2. Enable `use_gap_filter`
3. Increase `opening_range_minutes` (try 30)

### Stops Too Tight

1. Increase `target_multiplier` (improves R:R)
2. Use 30-minute OR (wider range)

---

## Related Documentation

- [Strategy Framework](../../src/strategies/STRATEGY_FRAMEWORK.md)
- [Backtesting Engine](../../src/backtesting/BACKTESTING_ENGINE.md)
- [RAMP Strategy](./RAMP_STRATEGY.md) - Regime detection reference
- [Data Handling](../../CLAUDE.md#data-handling) - Data requirements

---

## Changelog

- **2025-12-08**: Initial implementation
  - Core ORBStrategy with LongShortStrategy base
  - ORBIndicators for opening range and RVOL
  - ORBUniverse for leveraged ETF lists
  - 43 unit tests passing
  - YAML configs for single, sweep, walk-forward
