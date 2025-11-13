# Architecture Refactoring Roadmap

## Executive Summary

### Current State ✅
- **Base trading infrastructure is already asset-agnostic**
  - `BrokerInterface`, `ExecutionEngine`, `PositionManager` work with any security
  - Well-designed foundation requiring minimal changes

### What Needs Refactoring ⚠️
- **Strategy layer has duplicated logic** between backtesting and live trading
- **Hardcoded ETF lists** in signal generators
- **No clear separation** between pure strategy logic and execution infrastructure

### Proposed Solution 🎯
- **Three-layer architecture**:
  1. **Pure Strategy Layer**: Asset-agnostic signal generation (reusable)
  2. **Adapter Layer**: Connects strategies to backtesting or live trading
  3. **Infrastructure Layer**: Already generic (minimal changes needed)

## Architecture Comparison

### Before (Current)
```
Backtesting                          Live Trading
═══════════════════════════════════  ═══════════════════════════════════
src/strategies/                      src/trading/strategies/
├── moving_average.py ──────┐        ├── omr_live_strategy.py ──────┐
├── momentum.py             │        │   (duplicate OMR logic!)    │
├── mean_reversion.py       │        │                             │
└── overnight_mean_reversion.py ────┼────────────────────────────────┤
    (OMR for backtesting)   │        │                             │
                            │        │                             │
    DUPLICATED LOGIC! ──────┴────────┴──────────────────────────────┘

    Hardcoded ETF lists in multiple places
    No reuse between backtest and live trading
```

### After (Proposed)
```
                Pure Strategy Logic (Reusable)
    ┌───────────────────────────────────────────────────┐
    │  src/strategies/implementations/                  │
    │  ├── moving_average/ma_crossover_signals.py       │
    │  ├── momentum/momentum_signals.py                 │
    │  ├── mean_reversion/rsi_signals.py                │
    │  └── overnight/omr_signals.py                     │
    │                                                    │
    │  Pure signal generation - no infrastructure deps  │
    └───────────────────────────────────────────────────┘
                            ▲
                            │ (used by)
         ┌──────────────────┴──────────────────┐
         │                                     │
         │                                     │
┌────────▼──────────┐              ┌───────────▼────────┐
│ Backtest Adapters │              │ Live Trading       │
│ ═════════════════ │              │ Adapters           │
│ src/backtesting/  │              │ ════════════════   │
│   adapters/       │              │ src/trading/       │
│   ├── ma.py       │              │   adapters/        │
│   ├── momentum.py │              │   ├── ma.py        │
│   ├── omr.py ────────────────────────► omr.py         │
│   └── ...        │   Same logic!│   └── ...           │
└───────────────────┘              └────────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐              ┌─────────────────────┐
│ Portfolio       │              │ BrokerInterface     │
│ (backtest)      │              │ ExecutionEngine     │
└─────────────────┘              │ (live trading)      │
                                 └─────────────────────┘
```

## Benefits

### 1. **Single Source of Truth**
- Strategy logic written **once**, used everywhere
- Bug fixes benefit both backtesting and live trading
- Easier to maintain and test

### 2. **Asset Agnostic**
- Same strategy can trade stocks, ETFs, options, crypto
- Symbol lists injected via configuration (not hardcoded)
- Easy to add new asset classes

### 3. **Clean Separation of Concerns**
```
Strategy Logic       →  "What to trade" (signals)
Backtest Adapter     →  "How to simulate" (portfolio mechanics)
Live Trading Adapter →  "How to execute" (broker integration)
```

### 4. **Testability**
- Pure strategies = pure functions (easy to unit test)
- No mocking needed (no external dependencies)
- Integration tests focus on adapters

### 5. **Reusability**
- Leverage existing backtest strategies for live trading
- Mix and match strategies with different brokers
- Compose strategies (combine signals from multiple strategies)

## Migration Phases

### Phase 1: Foundation (Week 1) - **No Breaking Changes**

**Goal**: Create core abstractions without breaking existing code

```
✅ Tasks:
1. Create src/strategies/core/
   ├── base_strategy.py    # Abstract StrategySignals interface
   ├── signal.py           # Signal data structure
   └── indicator.py        # Common indicators

2. Create src/strategies/universe/
   ├── etf_universe.py     # ETF lists (LEVERAGED_3X, etc.)
   ├── equity_universe.py  # Stock screeners
   └── crypto_universe.py

3. Create adapter directories:
   - src/backtesting/adapters/
   - src/trading/adapters/

✅ Status: Existing code still works (no changes to imports)
```

### Phase 2: Extract Pure Strategies (Week 2) - **Low Risk**

**Goal**: Move strategy logic to pure implementations

```
✅ Tasks:
1. Extract MA Crossover:
   - Create: src/strategies/implementations/moving_average/ma_crossover_signals.py
   - Pure logic only (no backtest/live dependencies)
   - Unit tests

2. Extract Momentum:
   - Create: src/strategies/implementations/momentum/momentum_signals.py
   - Pure logic only
   - Unit tests

3. Extract Mean Reversion:
   - Create: src/strategies/implementations/mean_reversion/rsi_signals.py
   - Pure logic only
   - Unit tests

✅ Status: New files created, old files still exist (parallel implementation)
```

### Phase 3: Create Adapters (Week 3) - **Medium Risk**

**Goal**: Connect pure strategies to infrastructure

```
✅ Tasks:
1. Backtest adapters:
   - src/backtesting/adapters/ma_backtest_adapter.py
   - src/backtesting/adapters/momentum_backtest_adapter.py
   - src/backtesting/adapters/rsi_backtest_adapter.py

2. Live trading adapters:
   - src/trading/adapters/ma_live_adapter.py
   - src/trading/adapters/momentum_live_adapter.py
   - src/trading/adapters/rsi_live_adapter.py

3. Update configs to use new adapters

✅ Status: Both old and new implementations available (gradual migration)
```

### Phase 4: Migrate OMR Strategy (Week 4) - **High Value**

**Goal**: Decouple and reuse OMR components

```
✅ Tasks:
1. Refactor existing components (already mostly pure!):
   - market_regime_detector.py → Already reusable ✅
   - bayesian_reversion_model.py → Already reusable ✅
   - overnight_signal_generator.py → Remove hardcoded ETF list

2. Create pure OMR signals:
   src/strategies/implementations/overnight/
   ├── omr_signals.py              # Pure signal logic
   ├── regime_detector.py          # Move from src/strategies/advanced/
   └── bayesian_model.py           # Move from src/strategies/advanced/

3. Create adapters:
   - src/backtesting/adapters/omr_backtest_adapter.py
   - src/trading/adapters/omr_live_adapter.py

4. Update overnight_mean_reversion.py to use adapter

✅ Status: OMR logic reusable for both backtest and live trading
```

### Phase 5: Refactor Trading Bot (Week 5) - **Infrastructure**

**Goal**: Make TradingBot strategy-agnostic

```
✅ Tasks:
1. Rename: paper_trading_bot.py → trading_bot.py

2. Update TradingBot.__init__():
   Before:
   def __init__(self, broker_config, strategy_config):
       self.strategy = OMRLiveStrategy(strategy_config)  # Hardcoded!

   After:
   def __init__(self, broker_config, strategy: TradingStrategy):
       self.strategy = strategy  # Injected!

3. Update _fetch_current_data():
   - Use strategy.get_data_requirements()
   - Generic data fetching based on requirements

4. Update tests to use new interface

✅ Status: TradingBot can use ANY strategy
```

### Phase 6: Testing & Validation (Week 6)

**Goal**: Ensure everything works correctly

```
✅ Tasks:
1. Integration tests:
   - Test each pure strategy with backtest adapter
   - Test each pure strategy with live trading adapter
   - Validate signal consistency between backtest and live

2. Performance tests:
   - Ensure no performance degradation
   - Validate memory usage

3. End-to-end tests:
   - Full backtest with new adapters
   - Paper trading with new adapters

✅ Status: All tests passing, ready for production
```

### Phase 7: Cleanup & Documentation (Week 7)

**Goal**: Remove old code, update docs

```
✅ Tasks:
1. Deprecate old strategy files:
   - Add deprecation warnings
   - Update import paths in existing scripts

2. Update documentation:
   - Architecture diagrams
   - Strategy development guide
   - Migration guide for custom strategies

3. Remove deprecated code (after grace period)

✅ Status: Clean codebase with modern architecture
```

## File Changes Summary

### New Files (Created)
```
src/strategies/core/
├── base_strategy.py           [NEW]
├── signal.py                  [NEW]
└── indicator.py               [NEW]

src/strategies/implementations/
├── moving_average/
│   └── ma_crossover_signals.py    [NEW]
├── momentum/
│   └── momentum_signals.py        [NEW]
├── mean_reversion/
│   └── rsi_signals.py            [NEW]
└── overnight/
    ├── omr_signals.py            [NEW]
    ├── regime_detector.py        [MOVED from src/strategies/advanced/]
    └── bayesian_model.py         [MOVED from src/strategies/advanced/]

src/strategies/universe/
├── etf_universe.py            [NEW]
├── equity_universe.py         [NEW]
└── crypto_universe.py         [NEW]

src/backtesting/adapters/
├── ma_backtest_adapter.py     [NEW]
├── momentum_backtest_adapter.py   [NEW]
├── rsi_backtest_adapter.py    [NEW]
└── omr_backtest_adapter.py    [NEW]

src/trading/adapters/
├── ma_live_adapter.py         [NEW]
├── momentum_live_adapter.py   [NEW]
├── rsi_live_adapter.py        [NEW]
└── omr_live_adapter.py        [NEW]
```

### Modified Files
```
src/trading/core/
├── paper_trading_bot.py → trading_bot.py  [RENAMED]
│   - Accept TradingStrategy interface instead of hardcoded OMR
│   - Use strategy.get_data_requirements() for data fetching
│   - Remove ETF-specific comments

src/strategies/advanced/
├── overnight_signal_generator.py  [MODIFIED]
│   - Remove hardcoded LEVERAGED_ETFS list
│   - Accept symbols via constructor
```

### Deprecated Files (Eventually Remove)
```
src/strategies/base_strategies/
├── moving_average.py         [DEPRECATED → Use adapters]
├── momentum.py               [DEPRECATED → Use adapters]
└── mean_reversion.py         [DEPRECATED → Use adapters]

src/strategies/advanced/
├── overnight_mean_reversion.py   [DEPRECATED → Use adapters]
└── ...

src/trading/strategies/
└── omr_live_strategy.py      [DEPRECATED → Use omr_live_adapter.py]
```

### No Changes Needed ✅
```
src/trading/brokers/
├── broker_interface.py       [NO CHANGE]
└── alpaca_broker.py          [NO CHANGE]

src/trading/core/
├── execution_engine.py       [NO CHANGE]
└── position_manager.py       [NO CHANGE]
```

## Success Metrics

### Week 1-2
- ✅ Core abstractions created
- ✅ Unit tests for pure strategies
- ✅ No regressions in existing backtests

### Week 3-4
- ✅ All adapters implemented
- ✅ Integration tests passing
- ✅ OMR working with both backtest and live trading

### Week 5-6
- ✅ TradingBot refactored
- ✅ All tests passing
- ✅ Documentation updated

### Week 7
- ✅ Old code deprecated
- ✅ Migration guide complete
- ✅ Ready for new strategies

## Risk Mitigation

### Low Risk
- Creating new files (Phase 1-2)
- Adding parallel implementations (Phase 3)
- Unit testing pure strategies

### Medium Risk
- Refactoring OMR components (Phase 4)
- Updating TradingBot (Phase 5)
- Mitigation: Keep old code until new code proven

### High Risk
- Removing deprecated code (Phase 7)
- Mitigation: Long deprecation period, thorough testing

## Next Steps

1. **Review architecture docs**: Read all three architecture documents
2. **Approve approach**: Confirm this architecture meets your needs
3. **Start Phase 1**: Create core abstractions (no breaking changes)
4. **Implement incrementally**: One phase at a time, validate each step

## Questions to Consider

1. **Asset classes**: Which assets do you want to support initially?
   - ETFs (already supported)
   - Stocks
   - Options
   - Crypto
   - Futures

2. **Strategy priorities**: Which strategies to migrate first?
   - OMR (highest value - already partially reusable)
   - MA Crossover (simplest - good starting point)
   - Momentum
   - Mean Reversion
   - Pairs Trading

3. **Timeline**: Is 7 weeks reasonable, or prefer faster/slower?

4. **Breaking changes**: Accept gradual migration or want clean break?

Let me know if you'd like me to start implementing any phase!
