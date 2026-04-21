# OpEx Pinning Strategy - Current Status

**Date**: 2025-12-30
**Status**: SHELVED - Pending proper options data

---

## Summary

The OpEx Pinning & Release Strategy exploits gamma hedging dynamics near monthly options expiration. Implementation is complete but **blocked by data limitations** - ThetaData's 1-minute options feed lacks gamma and open interest, which are essential for GEX calculations.

---

## What Was Built

### Phase 1: Options Data Pipeline [COMPLETE]

| Component | Location | Description |
|-----------|----------|-------------|
| ThetaData Client | `src/data/options/thetadata_client.py` | REST API client for Theta Terminal |
| Options Schema | `src/data/options/options_schema.py` | Dataclasses for OptionSnapshot, OptionsChain, DailyGEXSummary |
| Options Store | `src/data/options/options_store.py` | Parquet-based persistent storage |
| ThetaData Adapter | `src/data/options/thetadata_adapter.py` | Reads existing ThetaData parquet files |

### Phase 2: GEX Calculation Engine [COMPLETE]

| Component | Location | Description |
|-----------|----------|-------------|
| GEX Calculator | `src/strategies/opex/gex_calculator.py` | Calculates strike-level and total GEX |
| OpEx Calendar | `src/strategies/opex/calendar.py` | Identifies OpEx phases (PINNING, POST_OPEX, etc.) |

### Phase 3: Signal Generation & Strategy [COMPLETE]

| Component | Location | Description |
|-----------|----------|-------------|
| Signal Generator | `src/strategies/opex/signal_generator.py` | Generates trading signals from GEX + phase |
| Strategy Class | `src/strategies/advanced/opex_pinning_strategy.py` | BaseStrategy implementation |
| Config | `config/backtesting/opex_pinning.yaml` | Backtest configuration |

### Phase 4: Backtesting & Validation [COMPLETE]

| Component | Location | Description |
|-----------|----------|-------------|
| Backtest Runner | `src/strategies/opex/backtest.py` | Specialized runner for options data |
| Validation | `src/strategies/opex/validation.py` | Statistical validation tests |
| Tests | `tests/strategies/test_opex/` | 152 unit tests passing |

---

## The Data Problem

### What We Have

ThetaData 1-minute options data stored at `H:\Stock_Data\options\options_1min\`:
- **Symbols**: SPY, QQQ, IWM, AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA, AMD, NFLX, BA, DIS, JPM, V, XOM
- **Date Range**: Nov 2024 - Dec 2025 (~13 months)
- **Format**: Partitioned parquet (`root={symbol}/year={YYYY}/month={MM}/day_{DD}.parquet`)

### What's Missing

| Field | 1-Min Data | Needed for GEX |
|-------|------------|----------------|
| delta | 98.1% populated | Yes (have it) |
| theta | 95.9% populated | No |
| vega | 96.2% populated | No |
| implied_vol | 97.2% populated | Yes (have it) |
| **gamma** | **0% - ALL NULL** | **YES - CRITICAL** |
| **open_interest** | **0% - ALL NULL** | **YES - CRITICAL** |

### Why This Matters

GEX (Gamma Exposure) formula:
```
GEX = gamma x OI x 100 x spot x direction
```

Without gamma and OI, we cannot:
- Calculate actual GEX levels
- Identify true pin strikes (max GEX)
- Determine if GEX is positive/negative (stabilizing vs destabilizing)

### Workaround Attempted

Created `ThetaDataAdapter` with estimation:
- **Gamma**: Estimated via Black-Scholes formula
- **OI**: Estimated as volume x 5 (rough proxy)

**Results**: 18 trades over 12 months, 11.1% win rate, -$415 PnL. Estimations are too inaccurate for reliable signals.

---

## What's Needed to Continue

### Option 1: ThetaData EOD API (Recommended)

The `ThetaDataClient` already has EOD endpoints that return gamma and OI:
- `get_eod_data()` - Historical EOD for single contract
- `get_greeks_snapshot()` - v3 API for Greeks
- `get_open_interest_snapshot()` - v3 API for OI

**Requirements**:
1. Start Theta Terminal (desktop app)
2. Run download script to fetch EOD data
3. Store in separate `options_eod/` directory

**Script needed**: `scripts/download_options_eod.py`

### Option 2: Alternative Data Source

| Source | Gamma | OI | Cost | Notes |
|--------|-------|-----|------|-------|
| ThetaData EOD | Yes | Yes | Included in subscription | Requires Theta Terminal |
| CBOE DataShop | Yes | Yes | $$ | Official exchange data |
| Orats | Yes | Yes | $$$ | Premium analytics |
| SpotGamma | Pre-computed GEX | N/A | $$$$ | Direct GEX levels |

---

## Strategy Logic (For Reference)

### OpEx Phases

| Phase | Days | Logic |
|-------|------|-------|
| NORMAL | >5 days before | No special edge |
| PRE_OPEX | T-5 to T-3 | Building gamma exposure |
| PINNING | T-2 to T-1 | **Fade moves** - price gravitates to pin |
| OPEX_DAY | T | High uncertainty, skip |
| POST_OPEX | T+1 to T+2 | **Follow momentum** - gamma released |

### Signal Generation

**PINNING Phase** (positive GEX):
- Price above pin -> SHORT (fade the move)
- Price below pin -> LONG (fade the move)

**POST_OPEX Phase**:
- Price above pin -> LONG (momentum continuation)
- Price below pin -> SHORT (momentum continuation)

### Entry Criteria

- Phase is PINNING or POST_OPEX
- Net GEX is positive (stabilizing environment)
- Distance to pin: 0.5% to 3% (not too close, not too far)

---

## File Structure

```
src/data/options/
    __init__.py
    options_schema.py      # OptionSnapshot, OptionsChain, DailyGEXSummary
    options_store.py       # Parquet storage
    thetadata_client.py    # REST API client
    thetadata_adapter.py   # Reads existing parquet files

src/strategies/opex/
    __init__.py
    calendar.py            # OpEx phases, 3rd Friday detection
    gex_calculator.py      # GEX calculations
    signal_generator.py    # Signal generation
    backtest.py            # Specialized backtest runner
    validation.py          # Statistical validation

src/strategies/advanced/
    opex_pinning_strategy.py  # Strategy class

config/backtesting/
    opex_pinning.yaml      # Backtest configuration

tests/strategies/test_opex/
    test_calendar.py
    test_gex_calculator.py
    test_signal_generator.py
    test_opex_strategy.py
    test_validation.py
```

---

## Test Coverage

All 152 tests passing:

```
tests/strategies/test_opex/test_calendar.py         - 15 tests
tests/strategies/test_opex/test_gex_calculator.py   - 25 tests
tests/strategies/test_opex/test_signal_generator.py - 20 tests
tests/strategies/test_opex/test_opex_strategy.py    - 18 tests
tests/strategies/test_opex/test_validation.py       - 14 tests
tests/data/test_options/                            - 60 tests
```

---

## To Resume This Work

1. **Start Theta Terminal** on local machine
2. **Create EOD download script**:
   ```bash
   python scripts/download_options_eod.py --symbols SPY,QQQ,IWM --start 2020-01-01 --end 2024-12-31
   ```
3. **Update ThetaDataAdapter** to read EOD data (or create new adapter)
4. **Re-run backtest** with proper gamma/OI data
5. **Validate** with walk-forward testing

---

## Related Documentation

- Original plan: `.claude/plans/stateful-waddling-charm.md`
- Options data schema: `src/data/options/options_schema.py`
- GEX calculation: `src/strategies/opex/gex_calculator.py`
