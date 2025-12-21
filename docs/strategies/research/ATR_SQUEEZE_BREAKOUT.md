# ATR Squeeze Breakout Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/breakout_strategies.py`

---

## Overview

Volatility contraction/expansion strategy. Identifies low volatility "squeeze" conditions (low ATR percentile) and trades the subsequent breakout when volatility expands.

---

## Logic

### Squeeze Detection

ATR falls to low percentile of historical range (consolidation phase)

### Breakout Entry

**Long Entry**: After squeeze, price breaks above channel

**Short Entry**: After squeeze, price breaks below channel

### Signal Generation

```
1. Calculate ATR percentile over lookback
2. Identify squeeze: ATR < 20th percentile
3. Wait for breakout from price channel
4. Enter on breakout direction
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR calculation period |
| `atr_lookback` | 100 | ATR percentile lookback |
| `squeeze_pctl` | 20 | ATR percentile for squeeze |
| `channel_period` | 20 | Price channel lookback |

---

## Usage

```python
from src.strategies.research.breakout_strategies import ATRSqueezeBreakout

strategy = ATRSqueezeBreakout(squeeze_pctl=20, channel_period=20)
signals = strategy.generate_signals(data)
```

---

## Theory

- Low volatility precedes high volatility
- Consolidation builds energy for breakout
- Similar to Bollinger Band squeeze concept

---

## Limitations

- Squeeze can last indefinitely
- Breakout direction uncertain
- False breakouts common

---

**Last Updated**: 2025-12-21
