# ORB Strategy - Current Status & Roadmap

**Date**: 2025-12-09
**Status**: Backtesting Complete, Live Trading Not Ready

---

## Executive Summary

The Opening Range Breakout (ORB) strategy has been fully implemented for backtesting with 43 passing unit tests. Initial backtest results on leveraged ETFs show poor performance (-9.23% return, 0.35 Sharpe) due to testing during a bear market period and single-symbol limitation in the backtest engine.

**Key Blockers for Production:**
1. No real-time streaming data infrastructure
2. Backtest engine limited to single symbol
3. Parameters not optimized

---

## Implementation Status

### Completed

| Component | File | Status |
|-----------|------|--------|
| Core Strategy | `src/strategies/advanced/orb_strategy.py` | Done |
| Indicators | `src/strategies/advanced/orb_indicators.py` | Done |
| Universe | `src/strategies/universe/orb_universe.py` | Done |
| Registry | `src/strategies/registry.py` | Integrated |
| Unit Tests | `tests/strategies/test_orb_*.py` | 43 tests passing |
| Documentation | `docs/strategies/ORB_STRATEGY.md` | Done |
| Single Backtest Config | `config/backtesting/orb_single.yaml` | Done |
| Parameter Sweep Config | `config/backtesting/orb_sweep.yaml` | Done |
| Walk-Forward Config | `config/backtesting/orb_walk_forward.yaml` | Done |

### Not Started

| Component | Description | Blocker |
|-----------|-------------|---------|
| `ORBLiveAdapter` | Live trading adapter | Needs streaming data |
| Real-time streaming | WebSocket market data | Architecture not built |
| Multi-symbol backtest | Test across all 5 ETFs | Engine limitation |

---

## Backtest Results

### Test 1: Leveraged ETFs (2022-2024)

**Config:** `config/backtesting/orb_single.yaml`

```
Symbols: TQQQ, SOXL, UPRO, TNA, TECL (only TQQQ tested due to engine limitation)
Period: 2022-01-01 to 2024-12-31
Initial Capital: $100,000
Position Size: 10% per trade
```

**Results:**

| Metric | Value |
|--------|-------|
| Total Return | -9.23% |
| Annual Return | -3.19% |
| Sharpe Ratio | 0.35 |
| Max Drawdown | -20.18% |
| Win Rate | 37.07% |
| Total Trades | 437 |
| Final Value | $90,773 |

**Why Results Are Poor:**

1. **Bear Market (2022)**: Strategy tested during significant market downturn
2. **Single Symbol Only**: Engine warning: "Multi-symbol backtesting simplified to first symbol only"
3. **Default Parameters**: No optimization performed
4. **Long-biased in downtrend**: ORB generates more long signals; 2022 was brutal for longs

### Test 2: Index ETFs (2022-2024) - Earlier Test

**Symbols:** QQQ, SPY, AAPL, NVDA, AMD (substitutes when leveraged data missing)

| Metric | Value |
|--------|-------|
| Total Return | 11.66% |
| Sharpe Ratio | 0.31 |

**Note:** Better returns but still poor Sharpe due to same limitations.

---

## Data Infrastructure

### Available Data

| Symbol | Timeframe | Location | Status |
|--------|-----------|----------|--------|
| TQQQ | 1-minute | `F:\Stock_Data\equities_1min\symbol=TQQQ` | Downloaded |
| SOXL | 1-minute | `F:\Stock_Data\equities_1min\symbol=SOXL` | Downloaded |
| UPRO | 1-minute | `F:\Stock_Data\equities_1min\symbol=UPRO` | Downloaded |
| TNA | 1-minute | `F:\Stock_Data\equities_1min\symbol=TNA` | Downloaded |
| TECL | 1-minute | `F:\Stock_Data\equities_1min\symbol=TECL` | Downloaded |
| SPY | 1-minute | `F:\Stock_Data\equities_1min\symbol=SPY` | Available |
| QQQ | 1-minute | `F:\Stock_Data\equities_1min\symbol=QQQ` | Available |

**Total:** 4.1M bars of leveraged ETF minute data (2020-2024)

### Live Data Gap

Currently, live trading uses **polling** via `AlpacaBroker.get_latest_quote()`. ORB requires:

- Real-time minute bars (9:30 AM - 4:00 PM)
- Continuous monitoring for breakout detection
- Sub-second latency for entry/exit execution

**Proposed Solution:** Streaming data platform using Alpaca's `StockDataStream` WebSocket API (plan created but not approved).

---

## What ORB Needs to Succeed

### 1. Fix Multi-Symbol Backtesting (High Priority)

The backtest engine currently simplifies to single symbol. Need to:
- Investigate `src/backtesting/engine/` for limitation
- Enable true portfolio backtesting across 5 ETFs
- Aggregate signals across symbols

### 2. Parameter Optimization (High Priority)

Run parameter sweep to find optimal settings:

```bash
python -m src.backtest_runner --config config/backtesting/orb_sweep.yaml
```

**Parameters to optimize:**
| Parameter | Default | Search Range |
|-----------|---------|--------------|
| `opening_range_minutes` | 15 | [5, 10, 15, 30] |
| `rvol_threshold` | 1.5 | [1.0, 1.5, 2.0] |
| `target_multiplier` | 1.0 | [0.5, 1.0, 1.5, 2.0] |

### 3. Walk-Forward Validation (Medium Priority)

Prevent overfitting with out-of-sample testing:

```bash
python -m src.backtest_runner --config config/backtesting/orb_walk_forward.yaml
```

- Train: 12 months
- Test: 6 months
- Roll forward every 6 months

### 4. Regime-Aware Trading (Medium Priority)

ORB performs differently across market regimes:

| Regime | Expected Performance |
|--------|---------------------|
| STRONG_BULL | Good (follow momentum) |
| WEAK_BULL | Moderate |
| SIDEWAYS | Poor (false breakouts) |
| BEAR | Poor for longs |

Consider:
- Skip trading in SIDEWAYS regime
- Long-only in STRONG_BULL
- Short-only in BEAR

### 5. Live Data Infrastructure (Required for Production)

Build streaming data platform:
- `src/streaming/LiveDataProvider` - single interface for strategies
- WebSocket connection to Alpaca
- In-memory bar buffer for quick lookups
- Auto-reconnect and fallback to polling

### 6. Live Trading Adapter (Required for Production)

Create `ORBLiveAdapter`:
- Subscribe to minute bars at market open
- Build opening range (9:30-9:45)
- Monitor for breakouts (9:45-3:30)
- Execute entries/exits via broker
- Force close at 3:45 PM

---

## Recommended Next Steps

### Immediate (This Week)

1. **Investigate multi-symbol backtest limitation**
   - Find why engine simplifies to single symbol
   - Fix or work around

2. **Run parameter optimization**
   - Use `orb_sweep.yaml` config
   - Find best parameters for TQQQ

3. **Test on bull market period**
   - Backtest 2020-2021 to compare performance
   - ORB should perform better in trending markets

### Short-Term (Next 2 Weeks)

4. **Walk-forward validation**
   - Validate parameters are robust
   - Measure out-of-sample Sharpe

5. **Add regime filtering**
   - Integrate `MarketRegimeDetector`
   - Skip SIDEWAYS/BEAR for longs

### Medium-Term (Before Production)

6. **Build streaming infrastructure**
   - Implement `LiveDataProvider`
   - Test WebSocket reliability

7. **Create `ORBLiveAdapter`**
   - Connect strategy to live trading
   - Paper trade for 2+ weeks

---

## Files Reference

```
src/strategies/advanced/
├── orb_strategy.py          # Core strategy (356 lines)
├── orb_indicators.py        # Indicator calculations (240 lines)

src/strategies/universe/
├── orb_universe.py          # Symbol lists

config/backtesting/
├── orb_single.yaml          # Single backtest
├── orb_sweep.yaml           # Parameter optimization
├── orb_walk_forward.yaml    # Walk-forward validation

tests/strategies/
├── test_orb_strategy.py     # 21 tests
├── test_orb_indicators.py   # 22 tests

docs/strategies/
├── ORB_STRATEGY.md          # Full documentation
├── 20251209_ORB_STATUS.md   # This file
```

---

## Conclusion

ORB is **implementation complete** but **not production ready**. The core strategy logic is sound with comprehensive tests. Poor backtest results are likely due to:

1. Testing in bear market (2022)
2. Single-symbol limitation
3. Unoptimized parameters

With proper optimization, regime filtering, and streaming data infrastructure, ORB could complement OMR (overnight) by capturing intraday momentum during market hours.

**Estimated effort to production:** 2-3 weeks of development + 2 weeks paper trading
