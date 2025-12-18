# ML Crypto Mean Reversion Strategy

**Status**: Research/Implementation
**Date**: 2025-12-16
**Source**: Reddit r/algotrading "2 years building, 3 months live"

---

## Strategy Overview

A mean reversion strategy for cryptocurrency markets (BTC/ETH) enhanced with a machine learning regime filter to identify favorable market conditions. The strategy capitalizes on the market's natural tendency to spend 70-80% of time in ranging/choppy conditions where price oscillates around a mean.

| Attribute | Value |
|-----------|-------|
| **Asset Class** | Cryptocurrency (BTC, ETH) |
| **Timeframe** | Daily (swing trading, multi-day holds) |
| **Edge Type** | Statistical mean reversion + ML regime classification |
| **Direction** | Long + Short |
| **Validation** | Walk-forward testing |

---

## Core Market Hypothesis

Markets exhibit two dominant regimes:

1. **Ranging Markets (~70-80% of time)**: Price oscillates within a channel, reverting to the mean after becoming overextended. Natural support and resistance levels form where buyers and sellers consistently step in.

2. **Trending Markets (~20-30% of time)**: Directional moves break established ranges, and mean reversion signals become traps as "overextended" conditions continue extending.

The strategy exploits the first regime while using ML classification to avoid the second.

---

## Entry Logic

### Long Entry
All conditions must be met:
1. Z-score < -2.0 (price is oversold, 2 std below rolling mean)
2. RSI < 30 (momentum confirmation)
3. ML Filter predicts "Ranging" regime (not trending)

### Short Entry
All conditions must be met:
1. Z-score > +2.0 (price is overbought, 2 std above rolling mean)
2. RSI > 70 (momentum confirmation)
3. ML Filter predicts "Ranging" regime (not trending)

---

## Exit Logic

Positions are exited on ANY of the following conditions:

| Exit Type | Long Exit | Short Exit |
|-----------|-----------|------------|
| **Mean Reversion** | Z-score >= -0.5 | Z-score <= 0.5 |
| **Stop Loss** | Price < Entry - (ATR x 1.5) | Price > Entry + (ATR x 1.5) |
| **Take Profit** | Price > Entry + (ATR x 4.5) | Price < Entry - (ATR x 4.5) |
| **Time Stop** | Held > 10 bars | Held > 10 bars |

The ATR-based stops provide approximately 3:1 risk/reward ratio.

---

## ML Regime Filter

### Purpose
The ML model acts as a regime classifier, answering: *"Is this a range-bound move likely to revert, or a breakout that will continue trending?"*

### Model Details

| Attribute | Value |
|-----------|-------|
| **Algorithm** | GradientBoostingClassifier |
| **Estimators** | 100 |
| **Max Depth** | 3 |
| **Training** | Inline during backtest (walk-forward) |

### Features

| Feature | Description | Ranging Signal |
|---------|-------------|----------------|
| ADX | Average Directional Index | Low ADX (< 25) |
| Choppiness Index | Range-bound indicator | High (> 61.8) |
| Efficiency Ratio | Trend efficiency | Low (choppy) |
| Bollinger Width | Volatility measure | Narrow |
| Absolute Z-score | Deviation from mean | High (reversion opportunity) |

### Labels
- **Ranging (1)**: Forward returns < 2% absolute (mean reversion worked)
- **Trending (0)**: Forward returns >= 2% (trend continued)

### Rule-Based Fallback
When ML is disabled or insufficient training data:
- **Ranging**: ADX < 25 AND Choppiness Index > 61.8
- **Trending**: Otherwise

---

## Indicators

### Z-Score (Mean Reversion Signal)
```
Z-score = (Close - Rolling_Mean) / Rolling_Std
```
- Window: 20 bars (daily)
- Entry threshold: |Z| > 2.0
- Exit threshold: |Z| < 0.5

### RSI (Momentum Confirmation)
- Period: 14
- Oversold: < 30
- Overbought: > 70

### ATR (Risk Management)
- Period: 14
- Stop multiplier: 1.5x ATR
- Target multiplier: 4.5x ATR

### ADX (Trend Strength)
- Period: 14
- Trending threshold: > 25
- Ranging threshold: < 20

### Choppiness Index
- Period: 14
- Ranging threshold: > 61.8
- Trending threshold: < 38.2

### Efficiency Ratio (Kaufman)
```
ER = |Direction| / Volatility
   = |Close - Close[n]| / Sum(|Close - Close[1]|, n)
```
- Period: 10
- High ER = trending, Low ER = choppy

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zscore_window` | 20 | Z-score calculation window |
| `zscore_entry_threshold` | 2.0 | Entry threshold (oversold/overbought) |
| `zscore_exit_threshold` | 0.5 | Mean reversion exit threshold |
| `rsi_period` | 14 | RSI period |
| `rsi_oversold` | 30 | RSI oversold level |
| `rsi_overbought` | 70 | RSI overbought level |
| `use_rsi_confirmation` | true | Require RSI confirmation |
| `atr_period` | 14 | ATR period |
| `atr_stop_multiplier` | 1.5 | Stop loss = ATR x mult |
| `atr_target_multiplier` | 4.5 | Take profit = ATR x mult |
| `use_ml_filter` | true | Enable ML regime filter |
| `ml_lookback_days` | 252 | ML training lookback |
| `adx_threshold` | 25.0 | ADX trending threshold |
| `choppiness_threshold` | 61.8 | Choppiness ranging threshold |
| `long_only` | false | Long-only mode |
| `max_hold_bars` | 10 | Maximum holding period |

---

## Reported Performance (Source)

### Backtest Results (1 Year, No Leverage)

| Metric | Value |
|--------|-------|
| Total Return | 767% |
| Win Rate | 38.17% |
| Risk/Reward | 3.18 |
| Max Drawdown | 27.32% |
| Sharpe Ratio | 4.64 |
| Sortino Ratio | 9.46 |
| Total Trades | 131 |

### Live Trading (3 Months)

| Metric | Value |
|--------|-------|
| Return | 59% |
| Max Drawdown | 12.7% |
| Slippage Impact | Minimal |

---

## Known Weaknesses

### 1. Trending Market Vulnerability
**Severity**: High

Strong, sustained trends cause mean reversion signals to repeatedly fail. Each "overextended" signal becomes more overextended.

**Mitigation**: ML filter identifies and avoids many trending conditions, but cannot eliminate this structural weakness.

### 2. Low Volatility Periods
**Severity**: Medium

Tight consolidation produces insufficient deviations from the mean. No trading losses, but no opportunities either.

**Mitigation**: Accept reduced activity during these periods.

### 3. Gap Risk (Black Swan)
**Severity**: High (low probability)

Fixed stop losses are vulnerable to price gapping through the stop level during extreme events.

**Mitigation**: Size positions appropriately for tail scenarios.

### 4. Full Position Sizing Amplification
**Severity**: Medium-High

Full position sizing per trade amplifies both gains and losses.

**Mitigation**: Homeguard implementation uses 20% position sizing rather than 100%.

---

## Validation Approach

### Walk-Forward Testing
1. Train on 12 months of data
2. Test on next 3 months (out-of-sample)
3. Step forward 3 months and repeat
4. Combine all OOS results

**Expected OOS Degradation**: 10-15% from in-sample (acceptable)

### Parameter Sensitivity
- Test key parameters +/- 20%
- Strategy should remain profitable across parameter space
- Performance degrades at extremes but doesn't break

### Regime Analysis
- Ranging markets: Strong performance
- Trending markets: Degraded performance (expected)

---

## Usage

### Config File
```yaml
# config/backtesting/ml_crypto_mr_baseline.yaml
mode: single

strategy:
  name: MLCryptoMRStrategy
  parameters:
    zscore_window: 20
    zscore_entry_threshold: 2.0
    use_ml_filter: true
    atr_stop_multiplier: 1.5
    atr_target_multiplier: 4.5

symbols:
  list: [BTC_USD, ETH_USD]

backtest:
  timeframe: crypto_1day
  market_hours_only: false
  allow_shorts: true
```

### Command
```bash
python -m src.backtest_runner --config config/backtesting/ml_crypto_mr_baseline.yaml
```

---

## References

- Source: Reddit r/algotrading "2 years building, 3 months live"
- Implementation: `src/strategies/advanced/ml_crypto_mr_strategy.py`
- Indicators: `src/strategies/advanced/ml_crypto_mr_indicators.py`

---

**Last Updated**: 2025-12-16
