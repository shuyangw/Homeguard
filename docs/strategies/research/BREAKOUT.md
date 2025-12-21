# Breakout Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/momentum.py`

---

## Overview

Classic breakout strategy that enters when price breaks above N-period high. Can include volume and ATR filters for confirmation.

---

## Logic

### Entry Conditions

**Long Entry**: Close > Highest High of last N bars

**Exit**: Close < Lowest Low of last N bars (or time-based)

### Signal Generation

```
Price breaks above 20-day high -> BUY
Price breaks below 20-day low -> SELL (or stop)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Breakout lookback period |
| `volume_filter` | True | Require volume confirmation |
| `atr_filter` | True | Require ATR expansion |

---

## Filters

- **Volume Filter**: Entry only if volume > average volume
- **ATR Filter**: Entry only if ATR expanding (volatility breakout)

---

## Limitations

- Many false breakouts in choppy markets
- Late entries (price already moved)
- Requires good stop loss management

---

**Last Updated**: 2025-12-21
