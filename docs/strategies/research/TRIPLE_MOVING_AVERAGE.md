# Triple Moving Average Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/moving_average.py`

---

## Overview

Enhanced trend-following strategy using three moving averages. Requires all three to be aligned (fast > medium > slow) for entry, reducing false signals.

---

## Logic

### Entry Conditions

**Long Entry**: Fast MA > Medium MA > Slow MA (full bullish alignment)

**Exit**: Any crossover violation

### Signal Generation

```
Full alignment check:
  Fast (10) > Medium (20) > Slow (50) -> BUY
  Any violation -> SELL
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 10 | Fast MA period |
| `medium_window` | 20 | Medium MA period |
| `slow_window` | 50 | Slow MA period |

---

## Advantages vs Dual MA

- Fewer false signals in choppy markets
- Confirms trend strength before entry
- More conservative approach

---

## Limitations

- Even later entries than dual MA
- May miss short-term opportunities

---

**Last Updated**: 2025-12-21
