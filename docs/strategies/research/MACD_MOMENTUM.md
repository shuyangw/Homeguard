# MACD Momentum Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/momentum.py`

---

## Overview

Momentum strategy using MACD (Moving Average Convergence Divergence) indicator. Trades crossovers of the MACD line and signal line.

---

## Logic

### Entry Conditions

**Long Entry**: MACD line crosses above Signal line

**Exit**: MACD line crosses below Signal line

### MACD Calculation

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD - Signal
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_period` | 12 | Fast EMA period |
| `slow_period` | 26 | Slow EMA period |
| `signal_period` | 9 | Signal line EMA period |

---

## Usage

```python
from src.strategies.research.momentum import MACDMomentum

strategy = MACDMomentum(fast_period=12, slow_period=26, signal_period=9)
signals = strategy.generate_signals(data)
```

---

## Limitations

- Lagging indicator
- False signals in ranging markets
- No position sizing

---

**Last Updated**: 2025-12-21
