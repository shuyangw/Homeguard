# ICT/SMC Liquidity-Based Trading Strategy

**Created:** 2024-12-13
**Status:** Production-Ready
**Last Updated:** 2024-12-13

## Overview

The ICT (Inner Circle Trader) / SMC (Smart Money Concepts) strategy is an intraday trading approach that identifies institutional order flow by detecting liquidity sweeps at key price levels, confirmed by rejection patterns at unmitigated order blocks.

**Core Premise:** Retail traders place stop losses at obvious swing highs/lows, creating "liquidity pools." Institutions sweep these stops to fill large orders, then price reverses. This strategy identifies these sweep-and-reverse patterns.

---

## Core Concepts

### 1. Market Structure

Market structure defines the current trend through swing point classification:

```
Bullish Structure:        Bearish Structure:
    HH                        LH
   /  \                      /  \
  /    \  HH                /    \  LH
 /      \/  \              /      \/  \
HL       HL  \            LL       LL  \
              HL                        LL

HH = Higher High          LH = Lower High
HL = Higher Low           LL = Lower Low
```

**Classification Rules:**
- **Bullish**: Consecutive HH and HL pattern
- **Bearish**: Consecutive LH and LL pattern
- **Ranging**: Mixed or unclear structure

**Parameters:**
- `swing_lookback`: Bars on each side for swing detection (default: 5)
- `min_swing_size_pct`: Minimum swing size as % of price (default: 0.25%)

### 2. Order Blocks

Order blocks are the last opposing candle before a strong impulsive move. They represent zones where institutions placed orders.

```
Bullish Order Block:          Bearish Order Block:

     |                              ___
     |  Strong move up             |   | Last up candle
    _|_                            |___|
   |   | <-- Last down candle          |
   |___|     before impulse            |  Strong move down
     |                                 |
```

**Identification Criteria:**
- Must precede an impulse move (configurable % threshold)
- Impulse must occur within N bars (`order_block_max_age`)
- Zone defined by the candle's high/low range

**Mitigation:**
- Order block becomes "mitigated" when price returns and trades through the zone
- Only unmitigated order blocks are valid for entries

**Parameters:**
- `min_impulse_move_pct`: Minimum impulse size (default: 0.4%)
- `order_block_max_age`: Maximum bars for OB validity (default: 50)
- `impulse_bars`: Bars to measure impulse after OB (default: 10)

### 3. Liquidity Levels

Liquidity accumulates at predictable locations:

- **Buy-side liquidity**: Above swing highs (stop losses for shorts)
- **Sell-side liquidity**: Below swing lows (stop losses for longs)

```
Buy-side liquidity (stops above highs):
                 $$$  <-- Stop losses clustered here
    ----HH----   $$$
   /          \
  /            \

Sell-side liquidity (stops below lows):
  \            /
   \          /
    ----LL----
                 $$$  <-- Stop losses clustered here
                 $$$
```

### 4. Liquidity Sweep

A sweep occurs when price temporarily breaks through a liquidity level, triggering stops, then reverses.

```
Liquidity Sweep (Bullish):

Previous Low --------
                    \
                     \  Sweep below
                      \/
                      /\  Reversal
                     /
                    /
              Close above low = Confirmed sweep
```

**Confirmation Requirements:**
- Price must exceed the liquidity level
- Price must close back inside the previous range
- Indicates stops were triggered but buying absorbed the selling

**Parameters:**
- `sweep_threshold_pct`: Minimum sweep depth (default: 0.12%)

### 5. Switch Candle

A switch candle is a reversal pattern that confirms the sweep. It shows rejection of the swept level.

**Bullish Switch Candles:**
```
Hammer:              Bullish Engulfing:
    |                     ___
    |                    |   |
   _|_                   |   |
  |   |                 _|___|
  |___|                |     |
    |                  |_____|
  Long lower wick      Body engulfs prior
```

**Bearish Switch Candles:**
```
Shooting Star:       Bearish Engulfing:
    |                  _____
   _|_                |     |
  |   |               |_____|
  |___|                 |   |
    |                   |___|
  Long upper wick      Body engulfs prior
```

**Requirements:**
- `min_wick_ratio`: Minimum wick-to-range ratio (default: 0.55)
- `min_body_ratio`: Minimum body-to-range ratio (default: 0.4)
- Direction must match expected reversal

---

## Strategy Variants

### Reversal Setup

Trade against the current market structure after a liquidity sweep:

```
1. Identify bearish structure (LH/LL)
2. Wait for sweep of sell-side liquidity (below lows)
3. Confirm with bullish switch candle
4. Check for unmitigated bullish order block nearby
5. Enter LONG (counter-trend reversal)

     LH
    /  \
   /    \  LH
  /      \/
 LL       LL
           \
            \ Sweep
             \/
             /\ Switch candle
            /
       ENTRY LONG
```

**Characteristics:**
- Counter-trend entries
- Higher risk/reward potential
- Lower win rate (compensated by larger winners)

### Continuation Setup

Trade with the current market structure on a pullback:

```
1. Identify bullish structure (HH/HL)
2. Wait for pullback to order block zone
3. Confirm with bullish switch candle at OB
4. Enter LONG (trend continuation)

       HH
      /
     /
    HL
   /  \
  /    \ Pullback to OB
 /      \/
         /\ Switch candle at OB
        /
   ENTRY LONG
```

**Characteristics:**
- Trend-following entries
- More frequent setups
- Higher win rate with moderate risk/reward

---

## Entry Filters

### HTF (Higher Timeframe) Filter

Only take trades aligned with higher timeframe trend:

| Trade Direction | Required HTF Bias |
|-----------------|-------------------|
| Long | Bullish or Neutral |
| Short | Bearish or Neutral |

**Parameters:**
- `use_htf_filter`: Enable/disable (default: true)
- `htf_lookback`: Bars for HTF bias calculation (default: 40)

### Volume Filter

Require above-average volume for entry confirmation:

```
RVOL = Current Volume / Average Volume (20-bar)

Entry allowed if: RVOL > rvol_threshold
```

**Parameters:**
- `use_volume_filter`: Enable/disable (default: true)
- `rvol_threshold`: Minimum relative volume (default: 1.5)

### Regime Filter

Adjust strategy based on market regime:

| Regime | Long Entries | Short Entries |
|--------|--------------|---------------|
| STRONG_BULL | Allowed | Blocked |
| WEAK_BULL | Allowed | Allowed |
| SIDEWAYS | Allowed | Allowed |
| UNPREDICTABLE | Allowed | Allowed |
| BEAR | Blocked | Allowed |

**Parameters:**
- `use_regime`: Enable/disable (default: true)

### Time Filters

```
Entry Window:   9:45 AM - 3:30 PM ET
Exit Cutoff:    3:45 PM ET (force close all)
Market Close:   4:00 PM ET
```

**Parameters:**
- `exit_time_hour`: Force exit hour (default: 15)
- `exit_time_minute`: Force exit minute (default: 45)

---

## Exit Logic

### Stop Loss Placement

Stop loss placed below/above the switch candle wick with ATR buffer:

```
Long Entry:
Entry ------>  O
               |
               |
Switch low --> |___

Stop loss ---> X  (Switch low - ATR * multiplier)
```

**Parameters:**
- `atr_period`: ATR calculation period (default: 14)
- `atr_stop_multiplier`: Stop buffer in ATRs (default: 1.8)

### Take Profit

Target calculated using risk:reward ratio:

```
Risk = |Entry - Stop Loss|
Target = Entry + (Risk * R:R ratio)  [for longs]
Target = Entry - (Risk * R:R ratio)  [for shorts]

Example with 1.5 R:R (long):
Entry:     $100.00
Stop:      $98.00   (Risk = $2.00)
Target:    $103.00  (Reward = $3.00)
```

**Parameters:**
- `risk_reward_ratio`: Target R:R (default: 1.5)

### Time Exit

All positions closed at exit time to avoid overnight exposure.

---

## Signal Flow Diagram

```
                    +------------------+
                    |   Load 1m Data   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Detect Swing     |
                    | Points (HH/HL/   |
                    | LH/LL)           |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v---------+          +--------v---------+
     | Identify Order   |          | Map Liquidity    |
     | Blocks           |          | Levels           |
     +--------+---------+          +--------+---------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v---------+
                    | Detect Liquidity |
                    | Sweeps           |
                    | (VECTORIZED)     |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Detect Switch    |
                    | Candles          |
                    | (VECTORIZED)     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v----+  +------v------+  +----v--------+
     | HTF Filter  |  | Volume      |  | Regime      |
     |             |  | Filter      |  | Filter      |
     +--------+----+  +------+------+  +----+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    | Generate Entry   |
                    | Signal           |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Monitor for      |
                    | Exit (SL/TP/Time)|
                    +------------------+
```

---

## Implementation Details

### Vectorized Computation

For performance (7.5x speedup), the following are pre-computed for the entire dataset:

| Computation | Method |
|-------------|--------|
| Liquidity sweeps | `detect_liquidity_sweeps_vectorized()` |
| Switch candles | `detect_switch_candles_vectorized()` |
| OB mitigation | `precompute_order_block_mitigation()` |
| Entry window | `compute_entry_window_mask()` |
| Volume filter | `compute_high_volume_mask()` |

### Sequential Processing

Required for stateful logic:
- Active position tracking
- Daily trade count limits
- Real-time exit monitoring

### Position Management

- `max_positions_per_day`: Limit daily trade count (default: 2)
- `long_only`: Restrict to long-only trading (default: false)

---

## Parameters Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trade_type` | str | 'both' | 'reversal', 'continuation', or 'both' |
| `swing_lookback` | int | 5 | Bars for swing detection |
| `min_swing_size_pct` | float | 0.0025 | Minimum swing size (0.25%) |
| `min_impulse_move_pct` | float | 0.004 | Minimum OB impulse (0.4%) |
| `order_block_max_age` | int | 50 | Max OB age in bars |
| `impulse_bars` | int | 10 | Bars to measure impulse |
| `risk_reward_ratio` | float | 1.5 | Target R:R |

### Filter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_htf_filter` | bool | true | Enable HTF alignment |
| `htf_lookback` | int | 40 | Bars for HTF bias |
| `use_regime` | bool | true | Enable regime filtering |
| `use_volume_filter` | bool | true | Enable volume confirmation |
| `rvol_threshold` | float | 1.5 | Minimum relative volume |

### Entry Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_wick_ratio` | float | 0.55 | Switch candle wick ratio |
| `min_body_ratio` | float | 0.4 | Switch candle body ratio |
| `sweep_threshold_pct` | float | 0.0012 | Minimum sweep depth |
| `max_positions_per_day` | int | 2 | Daily trade limit |
| `long_only` | bool | false | Long-only mode |

### Exit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atr_period` | int | 14 | ATR calculation period |
| `atr_stop_multiplier` | float | 1.8 | Stop buffer in ATRs |
| `exit_time_hour` | int | 15 | Force exit hour (ET) |
| `exit_time_minute` | int | 45 | Force exit minute |

---

## Usage

### Running Backtest

```bash
# Production config (balanced risk/return)
python -m src.backtest_runner --config config/backtesting/ict_production.yaml

# High return config (concentrated positions)
python -m src.backtest_runner --config config/backtesting/ict_high_return_final.yaml
```

### Programmatic Usage

```python
from src.strategies.advanced.ict_strategy import ICTStrategy
from src.backtesting.engine.backtest_engine import BacktestEngine

# Create strategy with robust parameters
strategy = ICTStrategy(
    trade_type='both',
    risk_reward_ratio=1.5,
    use_htf_filter=True,
    use_volume_filter=True,
    rvol_threshold=1.5,
    min_wick_ratio=0.55,
    min_body_ratio=0.4,
    max_positions_per_day=2
)

# Run backtest
engine = BacktestEngine(initial_capital=100000)
portfolio = engine.run(
    strategy=strategy,
    symbols=['NVDA', 'TSLA', 'AMD', 'META', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-12-31'
)
```

---

## Files

| File | Purpose |
|------|---------|
| `src/strategies/advanced/ict_indicators.py` | Core ICT indicators (vectorized) |
| `src/strategies/advanced/ict_strategy.py` | Strategy class |
| `config/backtesting/ict_production.yaml` | Production config |
| `config/backtesting/ict_high_return_final.yaml` | High return config |
| `tests/strategies/test_ict_indicators.py` | Indicator unit tests |
| `tests/strategies/test_ict_strategy.py` | Strategy unit tests |

---

## Theoretical Foundation

### Market Microstructure

The strategy is based on observable institutional trading patterns:

1. **Accumulation/Distribution:** Large orders leave footprints as order blocks
2. **Stop Hunting:** Price moves to trigger clustered stops before reversing
3. **Liquidity Cycles:** Price alternates between liquidity collection and trending

### Why It Works

- Retail traders place stops at obvious levels (swing highs/lows)
- Institutions need liquidity to fill large orders
- Sweeping stops provides this liquidity
- Once filled, institutions defend their positions (price reverses)

---

## Limitations and Risks

1. **1-Minute Noise:** High noise level requires robust filtering
2. **Slippage:** Fast entries may experience significant slippage
3. **Parameter Sensitivity:** Performance varies with parameter choices
4. **Market Dependency:** Works better in liquid, trending markets
5. **Drawdown Risk:** Concentrated positions increase drawdown

---

## See Also

- [ICT Optimization Results](20251213_ICT_OPTIMIZATION_RESULTS.md) - Backtesting results
- [RAMP Strategy](RAMP_STRATEGY.md) - Alternative momentum strategy
