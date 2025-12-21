# Pairs Trading Strategy

**Status**: Research/Legacy
**Asset Class**: Equities (Pairs)
**Source**: `src/strategies/research/pairs_trading.py`

---

## Overview

Statistical arbitrage strategy trading the spread between two cointegrated stocks. Goes long the underperformer and short the outperformer when spread diverges.

---

## Logic

### Pair Selection

1. Test for cointegration (Engle-Granger or Johansen)
2. Calculate hedge ratio (beta)
3. Construct spread: Spread = Stock_A - beta x Stock_B

### Trading Rules

**Long Spread**: Z-score < -entry_threshold (spread too cheap)
- Long Stock_A, Short Stock_B

**Short Spread**: Z-score > +entry_threshold (spread too expensive)
- Short Stock_A, Long Stock_B

**Exit**: Z-score crosses zero (mean reversion complete)

### Signal Generation

```
Spread Z-score = (Spread - Mean) / StdDev

Z < -2.0 -> Long spread (buy A, sell B)
Z > +2.0 -> Short spread (sell A, buy B)
Z crosses 0 -> Exit
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 60 | Window for spread statistics |
| `entry_threshold` | 2.0 | Z-score entry threshold |
| `exit_threshold` | 0.0 | Z-score exit threshold |
| `max_hold_days` | 20 | Maximum holding period |

---

## Usage

```python
from src.strategies.research.pairs_trading import PairsTrading

strategy = PairsTrading(
    pair=('AAPL', 'MSFT'),
    entry_threshold=2.0
)
signals = strategy.generate_signals(data)
```

---

## Risks

- Cointegration can break down
- Spread can diverge further before converging
- Requires margin for short leg
- Execution timing critical

---

**Last Updated**: 2025-12-21
