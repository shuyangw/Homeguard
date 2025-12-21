# 52-Week High Breakout Strategy

**Status**: Research/Legacy
**Asset Class**: Equities (Multi-Symbol)
**Source**: `src/strategies/research/high52_breakout_strategy.py`

---

## Overview

Monthly rebalancing strategy that invests in stocks near their 52-week high. Based on research showing stocks near highs tend to continue outperforming.

---

## Logic

### Stock Selection

1. Calculate distance from 52-week high for all stocks
2. Rank by proximity to high (closest = best)
3. Select top N stocks nearest to 52-week high

### Rebalancing

**Frequency**: Monthly

**Position Sizing**: Equal-weight among selected stocks

### Signal Generation

```
1. For each stock: Proximity = Close / 52-Week High
2. Rank all stocks by Proximity (highest first)
3. Select top N stocks
4. Equal-weight portfolio
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 10 | Number of stocks to hold |
| `min_proximity` | 0.95 | Minimum proximity to high (95%) |
| `lookback` | 252 | 52-week high lookback (trading days) |

---

## Usage

```python
from src.strategies.research.high52_breakout_strategy import High52WeekBreakout

strategy = High52WeekBreakout(top_n=10, min_proximity=0.95)
signals = strategy.generate_signals(universe_data)
```

---

## Research Basis

- George and Hwang (2004): 52-week high predicts returns
- Stocks near highs have momentum
- Less reversal risk than pure momentum

---

## Limitations

- Requires broad universe
- Monthly turnover costs
- May concentrate in few sectors

---

**Last Updated**: 2025-12-21
