# Moving Average Crossover Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/moving_average.py`

---

## Overview

Classic trend-following strategy using two moving averages. Generates buy signals when the fast MA crosses above the slow MA, and sell signals on the reverse.

---

## Logic

### Entry Conditions

**Long Entry**: Fast MA > Slow MA (bullish crossover)

**Exit**: Fast MA < Slow MA (bearish crossover)

### Signal Generation

```
Fast MA (e.g., 20-day) crosses above Slow MA (e.g., 50-day) -> BUY
Fast MA crosses below Slow MA -> SELL
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 20 | Fast moving average period |
| `slow_window` | 50 | Slow moving average period |
| `ma_type` | SMA | Moving average type (SMA/EMA) |

---

## Usage

```python
from src.strategies.research.moving_average import MovingAverageCrossover

strategy = MovingAverageCrossover(fast_window=20, slow_window=50)
signals = strategy.generate_signals(data)
```

---

## Limitations

- Lagging indicator (late entries/exits)
- Whipsaws in sideways markets
- No position sizing built-in

---

**Last Updated**: 2025-12-21
