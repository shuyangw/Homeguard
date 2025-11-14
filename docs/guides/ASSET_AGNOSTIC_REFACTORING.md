# Asset-Agnostic Trading Infrastructure Refactoring

## Current State Analysis

### ✅ Already Generic (Base Infrastructure)

These components are **already asset-agnostic** and work with any tradable security:

1. **`BrokerInterface`** (broker_interface.py)
   - Abstract interface for any broker
   - Methods work with any symbol: `place_order()`, `get_positions()`, `get_bars()`
   - No hardcoded asset types

2. **`ExecutionEngine`** (execution_engine.py)
   - Executes orders for any symbol
   - Retry logic, fill confirmation, batch operations
   - Completely generic

3. **`PositionManager`** (position_manager.py)
   - Tracks positions for any symbol
   - P&L calculation, risk limits, stop-loss monitoring
   - Asset-agnostic

4. **`AlpacaBroker`** (alpaca_broker.py)
   - Implements BrokerInterface for Alpaca
   - Can trade stocks, ETFs, crypto (whatever Alpaca supports)
   - No asset-specific logic

### ❌ Needs Decoupling (Strategy Layer)

These components have **ETF-specific hardcoded logic**:

1. **`OvernightSignalGenerator`** (overnight_signal_generator.py:27)
   ```python
   LEVERAGED_ETFS = [
       'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'TMF', 'TMV',
       'TECL', 'TECS', 'FAS', 'FAZ', 'TNA', 'TZA',
       # ... hardcoded list
   ]

   for symbol in self.LEVERAGED_ETFS:  # Line 111
       # Generate signals only for these ETFs
   ```

2. **`OMRLiveStrategy`** (omr_live_strategy.py)
   - Designed specifically for leveraged ETF overnight patterns
   - Assumes overnight mean reversion behavior
   - Comments mention "leveraged ETFs"

3. **`PaperTradingBot`** (paper_trading_bot.py:411, 484)
   - Comments mention "Leveraged ETFs"
   - Coupled to OMRLiveStrategy

## Proposed Architecture

### Layer 1: Generic Trading Infrastructure (✅ Keep As-Is)

```
src/trading/
├── brokers/
│   ├── broker_interface.py      # Abstract broker interface
│   ├── alpaca_broker.py          # Alpaca implementation
│   └── broker_factory.py         # Factory for creating brokers
├── core/
│   ├── execution_engine.py       # Generic order execution
│   ├── position_manager.py       # Generic position tracking
│   └── trading_bot.py            # Generic bot orchestrator (renamed)
```

**No changes needed** - this layer is already perfectly generic!

### Layer 2: Strategy Interface (New Abstract Layer)

```
src/trading/strategies/
├── base_strategy.py              # NEW: Abstract strategy interface
├── data_requirements.py          # NEW: Data fetching requirements
└── signal.py                     # NEW: Signal data structure
```

**New abstract interface** that all strategies must implement:

```python
# src/trading/strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Train strategy models with historical data."""
        pass

    @abstractmethod
    def generate_entry_signals(
        self,
        current_data: Dict[str, pd.DataFrame],
        broker: BrokerInterface
    ) -> List[Dict]:
        """Generate entry signals."""
        pass

    @abstractmethod
    def generate_exit_signals(
        self,
        broker: BrokerInterface
    ) -> List[Dict]:
        """Generate exit signals."""
        pass

    @abstractmethod
    def get_data_requirements(self) -> 'DataRequirements':
        """Return data requirements for this strategy."""
        pass
```

### Layer 3: Asset-Specific Strategy Implementations

```
src/trading/strategies/
├── etf/
│   ├── omr_etf_strategy.py           # OMR for ETFs (renamed)
│   ├── etf_universe.py               # ETF symbol lists
│   └── etf_signal_generator.py       # ETF-specific signals
├── equity/
│   ├── momentum_equity_strategy.py   # Momentum for stocks
│   ├── mean_reversion_equity.py      # Mean reversion for stocks
│   └── equity_universe.py            # Stock screeners
├── options/
│   ├── wheel_strategy.py             # Options wheel strategy
│   ├── iron_condor_strategy.py       # Iron condor
│   └── options_chain_analyzer.py     # Options-specific logic
└── crypto/
    ├── arbitrage_crypto_strategy.py  # Crypto arbitrage
    └── crypto_universe.py            # Crypto pairs
```

Each asset class has:
- Specific strategies (OMR for ETFs, momentum for stocks, etc.)
- Universe definitions (which symbols to trade)
- Asset-specific signal generators

### Layer 4: Generic Trading Bot (Refactored)

```python
# src/trading/core/trading_bot.py (renamed from paper_trading_bot.py)
class TradingBot:
    """
    Generic trading bot that works with ANY strategy.

    Completely asset-agnostic - delegates all asset-specific logic
    to the strategy implementation.
    """

    def __init__(
        self,
        broker_config_path: str,
        strategy: TradingStrategy  # Inject any strategy
    ):
        self.broker = BrokerFactory.create_from_yaml(broker_config_path)
        self.execution_engine = ExecutionEngine(self.broker)
        self.position_manager = PositionManager(...)
        self.strategy = strategy  # Strategy determines asset type

    def _fetch_current_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data based on strategy's requirements.

        Completely generic - asks strategy what data it needs.
        """
        # Get data requirements from strategy
        requirements = self.strategy.get_data_requirements()

        market_data = {}

        # Fetch daily data for symbols that need it
        for symbol, timeframe, lookback in requirements.daily_data:
            bars = self.broker.get_bars(
                symbols=[symbol],
                timeframe='1Day',
                start=datetime.now() - timedelta(days=lookback),
                end=datetime.now()
            )
            market_data[symbol] = bars

        # Fetch minute data for symbols that need it
        for symbol, timeframe, lookback in requirements.intraday_data:
            bars = self.broker.get_bars(
                symbols=[symbol],
                timeframe=timeframe,
                start=datetime.now() - timedelta(hours=lookback),
                end=datetime.now()
            )
            market_data[symbol] = bars

        return market_data
```

## Implementation Example

### Before (Coupled):

```python
# Hardcoded ETF list in signal generator
class OvernightSignalGenerator:
    LEVERAGED_ETFS = ['TQQQ', 'SQQQ', ...]  # Hardcoded!

    def generate_signals(self, market_data, timestamp):
        for symbol in self.LEVERAGED_ETFS:  # Only ETFs
            ...
```

```python
# PaperTradingBot coupled to OMR strategy
class PaperTradingBot:
    def __init__(self, broker_config, strategy_config):
        self.strategy = OMRLiveStrategy(strategy_config)  # Hardcoded!
```

### After (Decoupled):

```python
# Generic signal generator - symbols from config
class SignalGenerator:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols  # Injected, not hardcoded

    def generate_signals(self, market_data, timestamp):
        for symbol in self.symbols:  # Any symbols
            ...
```

```python
# Generic trading bot - strategy injected
class TradingBot:
    def __init__(self, broker_config, strategy: TradingStrategy):
        self.strategy = strategy  # Any strategy!
```

```python
# ETF-specific strategy implementation
class OMRETFStrategy(TradingStrategy):
    def __init__(self, config: Dict):
        # Get ETF universe from config or universe manager
        self.symbols = config.get('symbols', ETFUniverse.LEVERAGED_3X)
        self.signal_generator = SignalGenerator(self.symbols)
        ...

    def get_data_requirements(self):
        return DataRequirements(
            daily_data=[('SPY', '1Day', 250), ('VIX', '1Day', 365)],
            intraday_data=[(s, '1Min', 1) for s in self.symbols]
        )
```

```python
# Stock momentum strategy implementation
class MomentumEquityStrategy(TradingStrategy):
    def __init__(self, config: Dict):
        self.symbols = config.get('symbols', EquityUniverse.SP500)
        ...

    def get_data_requirements(self):
        return DataRequirements(
            daily_data=[(s, '1Day', 200) for s in self.symbols],
            intraday_data=[]  # No intraday data needed
        )
```

## Usage Examples

### ETF Trading:
```python
# config/strategies/omr_etf.yaml
strategy_type: "etf.OMRETFStrategy"
symbols: ['TQQQ', 'SQQQ', 'UPRO', 'SPXU']
min_win_rate: 0.58
...

# Create bot with ETF strategy
strategy = OMRETFStrategy(config)
bot = TradingBot(broker_config, strategy)
bot.start_trading()
```

### Stock Trading:
```python
# config/strategies/momentum_equity.yaml
strategy_type: "equity.MomentumEquityStrategy"
symbols: ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
lookback_period: 20
...

# Create bot with stock strategy
strategy = MomentumEquityStrategy(config)
bot = TradingBot(broker_config, strategy)
bot.start_trading()
```

### Options Trading:
```python
# config/strategies/wheel_options.yaml
strategy_type: "options.WheelStrategy"
underlying_symbols: ['SPY', 'QQQ', 'IWM']
delta_target: 0.30
...

# Create bot with options strategy
strategy = WheelStrategy(config)
bot = TradingBot(broker_config, strategy)
bot.start_trading()
```

## Migration Plan

### Phase 1: Create Abstract Interfaces (Low Risk)
1. Create `TradingStrategy` abstract base class
2. Create `DataRequirements` data class
3. Create `Signal` data class
4. **No changes to existing code** - just new files

### Phase 2: Refactor Strategy Layer (Medium Risk)
1. Rename `OMRLiveStrategy` → `OMRETFStrategy`
2. Make it inherit from `TradingStrategy`
3. Move hardcoded `LEVERAGED_ETFS` to `ETFUniverse` class
4. Inject symbols via config instead of hardcoding
5. Implement `get_data_requirements()` method

### Phase 3: Refactor Trading Bot (Medium Risk)
1. Rename `PaperTradingBot` → `TradingBot`
2. Accept `TradingStrategy` instead of hardcoded OMR
3. Use `strategy.get_data_requirements()` for data fetching
4. Remove ETF-specific comments

### Phase 4: Add New Asset Classes (Low Risk)
1. Implement `MomentumEquityStrategy` for stocks
2. Implement `WheelStrategy` for options
3. All use same `TradingBot` infrastructure!

## Benefits

1. **Extensibility**: Add new asset classes without touching base infrastructure
2. **Testability**: Test strategies independently of infrastructure
3. **Reusability**: Same broker, execution, position management for all assets
4. **Maintainability**: Clear separation of concerns
5. **Flexibility**: Mix and match strategies with different brokers

## File Changes Summary

### New Files:
- `src/trading/strategies/base_strategy.py`
- `src/trading/strategies/data_requirements.py`
- `src/trading/strategies/signal.py`
- `src/trading/strategies/etf/etf_universe.py`
- `src/trading/strategies/equity/` (future)
- `src/trading/strategies/options/` (future)

### Renamed Files:
- `paper_trading_bot.py` → `trading_bot.py`
- `omr_live_strategy.py` → `etf/omr_etf_strategy.py`

### Modified Files:
- `overnight_signal_generator.py` - remove hardcoded ETF list
- `trading_bot.py` - accept any strategy, use data requirements
- Strategy configs - add `symbols` parameter

### No Changes Needed:
- ✅ `broker_interface.py`
- ✅ `execution_engine.py`
- ✅ `position_manager.py`
- ✅ `alpaca_broker.py`

## Conclusion

The **base trading infrastructure is already asset-agnostic** - it's a well-designed foundation. The refactoring mainly involves:

1. Creating abstract strategy interfaces
2. Moving hardcoded ETF symbols to configuration
3. Allowing strategy injection into the trading bot

This gives you a **flexible, extensible architecture** where adding support for new asset classes (stocks, options, crypto) is just a matter of implementing the `TradingStrategy` interface - no changes to the core infrastructure needed!
