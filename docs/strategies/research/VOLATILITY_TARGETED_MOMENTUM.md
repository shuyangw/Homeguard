# Volatility Targeted Momentum Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/volatility_targeted_momentum.py`

---

## Overview

Momentum strategy with volatility-scaled position sizing. Uses MA filter for trend direction and scales position size inversely with volatility to target consistent risk.

---

## Logic

### Entry Conditions

**Long Entry**: Price > Long-term MA (uptrend confirmed)

**Position Size**: Scaled by target volatility / realized volatility

### Volatility Targeting

```
Position Size = (Target Vol / Realized Vol) x Base Size

Example:
- Target Vol: 15%
- Realized Vol: 30%
- Position Size: 50% of base (half sized due to high vol)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ma_period` | 200 | Trend filter MA period |
| `vol_window` | 20 | Volatility calculation window |
| `target_vol` | 0.15 | Target annual volatility (15%) |
| `max_leverage` | 2.0 | Maximum position size multiplier |

---

## Advantages

- Risk-adjusted returns more stable
- Reduces exposure in high volatility
- Increases exposure in low volatility

---

## Limitations

- Lag in volatility estimation
- May underperform in low-vol rallies
- Complex position sizing logic

---

**Last Updated**: 2025-12-21
