# RSI Mean Reversion Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/mean_reversion.py`

---

## Overview

Mean reversion strategy using RSI (Relative Strength Index). Enters when RSI indicates oversold conditions and exits on overbought.

---

## Logic

### Entry Conditions

**Long Entry**: RSI < 30 (oversold)

**Exit**: RSI > 70 (overbought)

### Signal Generation

```
RSI drops below 30 -> BUY (oversold bounce expected)
RSI rises above 70 -> SELL (take profit)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_period` | 14 | RSI calculation period |
| `oversold` | 30 | Oversold threshold |
| `overbought` | 70 | Overbought threshold |

---

## Usage

```python
from src.strategies.research.mean_reversion import RSIMeanReversion

strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
signals = strategy.generate_signals(data)
```

---

## Limitations

- RSI can stay oversold/overbought in strong trends
- No stop loss logic
- Long-only

---

**Last Updated**: 2025-12-21
