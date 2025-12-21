# ML Crypto Mean Reversion Strategy

**Status**: Active Research
**Asset Class**: Crypto
**Source**: `src/strategies/advanced/ml_crypto_mr_strategy.py`

---

## Overview

A mean reversion strategy for cryptocurrency markets enhanced with a machine learning regime filter. The ML model distinguishes ranging vs trending conditions, only trading during favorable mean-reverting regimes.

Based on Reddit r/algotrading post: "2 years building, 3 months live"

---

## Core Logic

### Regime Detection

The ML classifier (GradientBoosting by default) predicts market regime:

| Regime | Frequency | Strategy Action |
|--------|-----------|-----------------|
| Ranging | ~70-80% | Mean reversion trades allowed |
| Trending | ~20-30% | Signals avoided (trend trap prevention) |

### Entry Conditions

**Long Entry**:
- Z-score < -threshold (oversold)
- ML filter predicts "ranging"
- RSI < oversold level (optional confirmation)

**Short Entry**:
- Z-score > +threshold (overbought)
- ML filter predicts "ranging"
- RSI > overbought level (optional confirmation)

### Exit Conditions

1. **Mean Reversion Exit**: Z-score crosses toward zero
2. **Stop Loss**: ATR-based or fixed percentage
3. **Take Profit**: ATR-based or fixed percentage (~3:1 R:R)
4. **Time Stop**: Exit after max_hold_bars

---

## Parameters

### Z-Score Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zscore_window` | 20 | Z-score calculation window |
| `zscore_entry_threshold` | 2.0 | Entry Z-score magnitude |
| `zscore_exit_threshold` | 0.5 | Mean reversion exit level |

### RSI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_period` | 14 | RSI calculation period |
| `rsi_oversold` | 30 | RSI oversold threshold |
| `rsi_overbought` | 70 | RSI overbought threshold |
| `use_rsi_confirmation` | True | Require RSI confirmation |

### Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR calculation period |
| `atr_stop_multiplier` | 1.5 | Stop = ATR x mult |
| `atr_target_multiplier` | 4.5 | Target = ATR x mult |
| `use_fixed_pct_exits` | False | Use fixed % instead of ATR |
| `fixed_stop_pct` | 0.10 | Fixed 10% stop loss |
| `fixed_target_pct` | 0.318 | Fixed 31.8% take profit |

### ML Filter Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ml_filter` | True | Enable ML regime filter |
| `ml_lookback_days` | 252 | ML training lookback |
| `ml_retrain_frequency` | 20 | Retrain every N bars |
| `adx_threshold` | 25.0 | ADX trending threshold |
| `choppiness_threshold` | 61.8 | Choppiness ranging threshold |
| `model_type` | gradient_boosting | ML model type |

### Trade Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `long_only` | False | Long-only mode |
| `max_hold_bars` | 10 | Maximum holding period |

---

## ML Model Types

| Model | Description |
|-------|-------------|
| `gradient_boosting` | Default. GradientBoostingClassifier |
| `xgboost` | XGBoost (requires xgboost package) |
| `random_forest` | RandomForestClassifier |
| `ensemble` | Voting ensemble of all models |

---

## Features Used by ML Model

- Z-score and Z-score rate of change
- RSI
- ADX (trend strength)
- Choppiness Index
- Price efficiency ratio
- Volatility metrics
- Bollinger Band width

---

## Usage

### Backtest Configuration

```yaml
strategy:
  name: MLCryptoMR
  params:
    zscore_window: 20
    zscore_entry_threshold: 2.0
    use_ml_filter: true
    model_type: gradient_boosting
    max_hold_bars: 10
```

### Programmatic Usage

```python
from src.strategies.advanced.ml_crypto_mr_strategy import MLCryptoMRStrategy

strategy = MLCryptoMRStrategy(
    zscore_entry_threshold=2.0,
    use_ml_filter=True,
    model_type='gradient_boosting'
)

# Get current signal
signal = strategy.get_current_signal(data)
print(f"Regime: {'Ranging' if signal.is_ranging else 'Trending'}")
print(f"Z-score: {signal.zscore:.2f}, RSI: {signal.rsi:.1f}")

# Get regime statistics
stats = strategy.get_regime_stats(data)
print(f"Ranging: {stats['ranging_pct']:.1f}%")
```

---

## Related Documentation

- [HURST_MR_STRATEGY.md](HURST_MR_STRATEGY.md) - Simpler non-ML alternative
- [20251216_ML_CRYPTO_MR_FINDINGS.md](20251216_ML_CRYPTO_MR_FINDINGS.md) - Research findings

---

**Last Updated**: 2025-12-21
