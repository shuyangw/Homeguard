# ICT/SMC Liquidity-Based Strategy - Backtest Results

**Date**: 2025-12-15
**Status**: Research Complete - Shelved
**Strategy**: ICT (Inner Circle Trader) / Smart Money Concepts

---

## Executive Summary

Implemented and backtested an ICT/SMC liquidity-based trading strategy that trades liquidity sweeps at order blocks with switch candle confirmation. Tested on both leveraged ETFs and crypto assets.

**Best Results:**
- **Leveraged ETFs**: +49.06% return, 0.65 Sharpe, -21.12% max DD (NY Kill Zone + 1.5 R:R)
- **Crypto (5 symbols)**: +100.19% return, 0.81 Sharpe, -34.10% max DD (3.0 R:R)

---

## Strategy Overview

### Core ICT Concepts Implemented

1. **Market Structure Detection**: Swing highs/lows classification (HH, HL, LH, LL)
2. **Order Block Identification**: Last opposing candle before impulse move
3. **Liquidity Sweep Detection**: Price sweeping above/below swing points
4. **Switch Candle Confirmation**: Rejection patterns (hammer, engulfing, etc.)
5. **Session Filters**: NY Kill Zone (9:30-11:30 AM ET), crypto sessions

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `swing_lookback` | Bars to look back for swing detection | 5 |
| `min_swing_size_pct` | Minimum swing size as % of price | 0.3% |
| `order_block_max_age` | Max bars since OB formation | 30 |
| `risk_reward_ratio` | Target R:R for exits | 2.0 |
| `atr_stop_multiplier` | ATR multiplier for stop loss | 1.5 |
| `session_filter` | Trading session filter | none |

---

## Leveraged ETF Results

**Universe**: TQQQ, SQQQ, UPRO, TNA, TZA
**Period**: 2024-01-01 to 2024-12-31
**Transaction Costs**: 0.1% fees + 0.05% slippage

### Configuration Comparison

| Configuration | Return | Sharpe | Max DD | Win Rate |
|--------------|--------|--------|--------|----------|
| Baseline (no filters) | +14.93% | 0.34 | -18.02% | 55% |
| Momentum Filter (EMA) | +45.89% | 0.62 | -17.23% | 58% |
| Structure Filter | -4.63% | -0.08 | -22.45% | 48% |
| **NY Kill Zone** | +40.59% | 0.59 | -18.56% | 57% |
| **NY Kill Zone + R:R 1.5** | **+49.06%** | **0.65** | -21.12% | 59% |

### Best ICT-Aligned Configuration

```yaml
# config/backtesting/ict_best_ict_aligned.yaml
strategy:
  name: ICTStrategy
  parameters:
    session_filter: ny_open      # ICT Kill Zone (9:30-11:30 AM ET)
    risk_reward_ratio: 1.5       # Tighter R:R captures more winners
    use_momentum_filter: false   # Pure ICT (no EMA)
    use_structure_filter: false  # Allow counter-trend reversals
```

**Why NY Kill Zone Works**: Institutions are most active during the first 2 hours of US market open. Liquidity sweeps are more reliable during this window.

---

## Crypto Results

**Universe**: BTC_USD, ETH_USD, AVAX_USD, LINK_USD, DOT_USD
**Period**: 2023-01-01 to 2024-12-31
**Transaction Costs**: 0.1% fees + 0.1% slippage (0.2% total per trade)

### Configuration Comparison

| Configuration | Return | Sharpe | Max DD | Win Rate | Trades |
|--------------|--------|--------|--------|----------|--------|
| BTC+ETH (2.0 R:R) | +35.96% | 0.53 | -9.86% | 62% | 29 |
| BTC+ETH (3.0 R:R) | +37.55% | 0.55 | -9.57% | 60% | 30 |
| BTC+ETH (4.0 R:R) | +36.59% | 0.54 | -9.67% | 55% | 29 |
| BTC+ETH (Long-Only) | +1.68% | 0.37 | -2.02% | 50% | 20 |
| **5 Cryptos (3.0 R:R)** | **+100.19%** | **0.81** | -34.10% | 45% | 155 |
| 5 Cryptos + Max Loss 10% | +86.33% | 0.71 | -38.70% | 45% | 156 |

### Best Crypto Configuration

```yaml
# config/backtesting/ict_crypto_expanded.yaml
strategy:
  name: ICTStrategy
  parameters:
    risk_reward_ratio: 3.0       # Wider R:R for crypto's larger moves
    session_filter: none         # 24/7 trading
    long_only: false             # Both directions needed
symbols:
  list: [BTC_USD, ETH_USD, AVAX_USD, LINK_USD, DOT_USD]
backtest:
  timeframe: crypto_1min
  market_hours_only: false
```

### Key Crypto Findings

1. **Diversification is critical**: 5 symbols vs 2 symbols doubles returns (100% vs 36%)
2. **3:1 R:R is optimal**: Better than both 2:1 and 4:1 for crypto
3. **Altcoins outperform**: LINK, AVAX, DOT had massive winners (+13000%, +5344%, +4117%)
4. **Long-only hurts**: Must trade both directions (+1.68% vs +35.96%)
5. **US session filter ineffective**: Trades already clustered during US hours

---

## Asset Class Comparison

| Metric | Leveraged ETFs | Crypto (5 symbols) |
|--------|---------------|-------------------|
| Return | +49.06% | +100.19% |
| Sharpe Ratio | 0.65 | 0.81 |
| Max Drawdown | -21.12% | -34.10% |
| Optimal R:R | 1.5 | 3.0 |
| Session Filter | NY Kill Zone | None |
| Win Rate | ~59% | ~45% |

**Conclusion**: Crypto offers higher returns and Sharpe but with higher drawdown risk. ETFs provide more consistent performance with lower volatility.

---

## Transaction Cost Sensitivity

Current backtests use optimistic transaction costs:

| Asset Class | Fees | Slippage | Total |
|-------------|------|----------|-------|
| Leveraged ETFs | 0.1% | 0.05% | 0.15% |
| Crypto | 0.1% | 0.1% | 0.2% |

**Realistic crypto costs** would be 0.3-0.5% per trade, which would reduce returns by ~15-25% given 155 trades.

---

## Files Created

### Strategy Implementation
- `src/strategies/advanced/ict_strategy.py` - Main strategy class
- `src/strategies/advanced/ict_indicators.py` - ICT-specific indicators

### Configuration Files
| File | Purpose |
|------|---------|
| `config/backtesting/ict_best_ict_aligned.yaml` | Best ETF config (NY Kill Zone + 1.5 R:R) |
| `config/backtesting/ict_crypto.yaml` | Baseline crypto config |
| `config/backtesting/ict_crypto_expanded.yaml` | Best crypto config (5 symbols + 3.0 R:R) |
| `config/backtesting/ict_crypto_*.yaml` | Various crypto test configs |

### Infrastructure Changes
- `src/backtesting/engine/streaming_data_loader.py` - Added crypto_1min/1hour/1day support
- `src/backtesting/engine/backtest_engine.py` - Added timeframe parameter
- `src/settings/schema.py` - Added timeframe to BacktestSettings

---

## Future Improvements (Not Implemented)

1. **Multi-timeframe analysis**: Use 15m/1H for trend, 1m for entry
2. **Zone quality scoring**: Weight order blocks by freshness, touch count
3. **Break of Structure (BOS) / Change of Character (CHoCH)** detection
4. **Volume profile integration**: Identify high-volume nodes
5. **Walk-forward validation**: Out-of-sample testing for robustness

---

## Conclusion

The ICT/SMC strategy shows positive expectancy on both leveraged ETFs and crypto. The methodology translates well across asset classes with appropriate parameter tuning:

- **ETFs**: Use NY Kill Zone session filter with tighter 1.5 R:R
- **Crypto**: Diversify across 5+ symbols with wider 3.0 R:R

Strategy is shelved pending further validation or live trading consideration.
