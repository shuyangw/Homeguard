# Mean Reversion Long/Short Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/mean_reversion_long_short.py`

---

## Overview

Flip-flop mean reversion strategy that maintains a position at all times. Goes long at lower Bollinger Band and short at upper band, always positioned in one direction.

---

## Logic

### Position Management

**Long Position**: When price < Lower BB, flip to long

**Short Position**: When price > Upper BB, flip to short

**Always Positioned**: Never flat, always long or short

### Signal Generation

```
Price < Lower BB -> Go LONG (close any short)
Price > Upper BB -> Go SHORT (close any long)
No exit to flat - always in market
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 20 | Bollinger Band period |
| `num_std` | 2.0 | Number of standard deviations |

---

## Risk Considerations

- Always exposed to market (no cash position)
- Can suffer extended drawdowns in trends
- Requires margin for short positions

---

**Last Updated**: 2025-12-21
