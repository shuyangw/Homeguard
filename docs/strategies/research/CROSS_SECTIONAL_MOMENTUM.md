# Cross-Sectional Momentum Strategy

**Status**: Research/Legacy
**Asset Class**: Equities (Multi-Symbol)
**Source**: `src/strategies/research/cross_sectional_momentum.py`

---

## Overview

Ranks stocks by momentum and goes long the top performers. Classic Fama-French style momentum factor strategy that rotates into recent winners.

---

## Logic

### Ranking

Stocks ranked by trailing returns over multiple lookback periods:
- 3-month return
- 6-month return
- 12-month return (excluding last month)

### Position Selection

**Long**: Top N% of ranked stocks

**Rebalance**: Monthly

### Signal Generation

```
1. Calculate trailing returns for all stocks
2. Rank stocks by composite momentum score
3. Select top N% for portfolio
4. Equal-weight or momentum-weight positions
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_3m` | 63 | 3-month lookback (trading days) |
| `lookback_6m` | 126 | 6-month lookback |
| `lookback_12m` | 252 | 12-month lookback |
| `skip_recent` | 21 | Skip last month (reversal effect) |
| `top_pct` | 0.10 | Top 10% for long portfolio |
| `rebalance_freq` | monthly | Rebalance frequency |

---

## Usage

```python
from src.strategies.research.cross_sectional_momentum import CrossSectionalMomentum

strategy = CrossSectionalMomentum(top_pct=0.10, rebalance_freq='monthly')
signals = strategy.generate_signals(universe_data)
```

---

## Considerations

- Requires multi-symbol data
- Transaction costs from monthly turnover
- Works best with large liquid universe

---

**Last Updated**: 2025-12-21
