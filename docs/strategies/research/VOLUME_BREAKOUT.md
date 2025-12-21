# Volume Breakout Strategy

**Status**: Research/Legacy
**Asset Class**: Equities
**Source**: `src/strategies/research/breakout_strategies.py`

---

## Overview

Breakout strategy with volume confirmation. Requires volume spike (4x average) combined with price closing near the high to confirm breakout.

---

## Logic

### Entry Conditions

**Long Entry**:
1. Volume > 4x average volume (volume spike)
2. Close near high of day (bullish candle)
3. Price above recent range

### Confirmation

```
Volume Ratio = Today's Volume / Avg Volume (20-day)
Close Position = (Close - Low) / (High - Low)

Entry when:
- Volume Ratio > 4.0
- Close Position > 0.75 (close in top 25% of range)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_mult` | 4.0 | Volume multiplier threshold |
| `vol_avg_period` | 20 | Average volume lookback |
| `close_pct` | 0.75 | Close must be in top 25% of range |

---

## Usage

```python
from src.strategies.research.breakout_strategies import VolumeBreakout

strategy = VolumeBreakout(volume_mult=4.0, close_pct=0.75)
signals = strategy.generate_signals(data)
```

---

## Advantages

- Volume confirms institutional interest
- Filters out low-conviction breakouts

---

## Limitations

- Volume spikes can be manipulation
- Late entry if waiting for confirmation
- Requires intraday volume data

---

**Last Updated**: 2025-12-21
