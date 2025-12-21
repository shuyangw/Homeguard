# Mean Reversion (Bollinger Bands) Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/mean_reversion.py`

---

## Overview

Classic mean reversion strategy using Bollinger Bands. Enters long when price touches the lower band (oversold) and exits when price reaches the middle band (mean).

---

## Logic

### Entry Conditions

**Long Entry**: Close < Lower Bollinger Band

**Exit**: Close >= Middle Band (20-day SMA)

### Signal Generation

```
Price touches lower band (2 std dev below mean) -> BUY
Price reverts to mean (middle band) -> SELL
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 20 | Bollinger Band period |
| `num_std` | 2.0 | Number of standard deviations |

---

## Usage

```python
from src.strategies.research.mean_reversion import BollingerMeanReversion

strategy = BollingerMeanReversion(window=20, num_std=2.0)
signals = strategy.generate_signals(data)
```

---

## Limitations

- Fails in strong trends (price stays outside bands)
- No stop loss logic built-in
- Long-only (no short positions)

---

**Last Updated**: 2025-12-21
