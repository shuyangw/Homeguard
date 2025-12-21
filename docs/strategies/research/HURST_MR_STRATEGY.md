# Hurst Exponent Mean Reversion Strategy

**Status**: Active Research
**Asset Class**: Crypto
**Source**: `src/strategies/advanced/hurst_mr_strategy.py`

---

## Overview

A mean reversion strategy that uses the Hurst exponent as a regime filter to identify mean-reverting market conditions. Simpler and more interpretable than ML-based approaches.

---

## Core Logic

### Hurst Exponent Interpretation

| Hurst Value | Regime | Strategy Action |
|-------------|--------|-----------------|
| H < 0.5 | Mean-reverting (anti-persistent) | Trade allowed |
| H = 0.5 | Random walk | Avoid |
| H > 0.5 | Trending (persistent) | Avoid |

### Entry Conditions

**Long Entry**:
- Z-score < -threshold (oversold)
- Hurst < hurst_threshold (mean-reverting regime)

**Short Entry**:
- Z-score > +threshold (overbought)
- Hurst < hurst_threshold (mean-reverting regime)

### Exit Conditions

1. **Mean Reversion Exit**: Z-score crosses toward zero
2. **Stop Loss**: ATR-based or fixed percentage
3. **Take Profit**: ATR-based or fixed percentage (~3:1 R:R)
4. **Time Stop**: Exit after max_hold_bars

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hurst_window` | 100 | Bars for Hurst calculation |
| `hurst_threshold` | 0.45 | H < threshold allows trades |
| `zscore_window` | 20 | Z-score calculation window |
| `zscore_entry_threshold` | 2.0 | Entry Z-score magnitude |
| `zscore_exit_threshold` | 0.5 | Mean reversion exit level |
| `atr_period` | 14 | ATR calculation period |
| `atr_stop_multiplier` | 1.5 | Stop = ATR x mult |
| `atr_target_multiplier` | 4.5 | Target = ATR x mult |
| `use_fixed_pct_exits` | False | Use fixed % instead of ATR |
| `fixed_stop_pct` | 0.10 | Fixed 10% stop loss |
| `fixed_target_pct` | 0.318 | Fixed 31.8% take profit |
| `long_only` | False | Long-only mode |
| `max_hold_bars` | 10 | Maximum holding period |

---

## Usage

### Backtest Configuration

```yaml
strategy:
  name: HurstMR
  params:
    hurst_window: 100
    hurst_threshold: 0.45
    zscore_window: 20
    zscore_entry_threshold: 2.0
    max_hold_bars: 10
```

### Programmatic Usage

```python
from src.strategies.advanced.hurst_mr_strategy import HurstMRStrategy

strategy = HurstMRStrategy(
    hurst_window=100,
    hurst_threshold=0.45,
    zscore_entry_threshold=2.0
)

# Get current signal
signal = strategy.get_current_signal(data)
print(f"Hurst: {signal.hurst:.3f}, Z-score: {signal.zscore:.2f}")

# Get regime statistics
stats = strategy.get_hurst_stats(data)
print(f"Mean-reverting: {stats['mean_reverting_pct']:.1f}%")
```

---

## Advantages vs ML Approach

1. **Interpretability**: Hurst exponent has clear statistical meaning
2. **Simplicity**: No ML model training required
3. **Stability**: No retraining, consistent behavior
4. **Debugging**: Easy to understand why trades are taken/skipped

---

## Related Documentation

- [ML_CRYPTO_MR_STRATEGY.md](ML_CRYPTO_MR_STRATEGY.md) - ML-based alternative
- [20251217_HURST_MR_OPTIMIZATION_RESULTS.md](../../20251217_HURST_MR_OPTIMIZATION_RESULTS.md) - Optimization results

---

**Last Updated**: 2025-12-21
