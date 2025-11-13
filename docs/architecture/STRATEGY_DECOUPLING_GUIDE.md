# Strategy Decoupling and Reuse Guide

## Overview

This guide shows how to:
1. **Decouple strategy logic from trading infrastructure**
2. **Reuse existing backtest strategies for live trading**
3. **Create a unified strategy interface** for both backtesting and live trading

## Current State: Two Separate Strategy Layers

### Backtest Strategies (src/strategies/)
```
src/strategies/
├── base_strategies/
│   ├── moving_average.py           # MA crossover
│   ├── momentum.py                 # Momentum and breakout
│   ├── mean_reversion.py           # RSI mean reversion
│   └── mean_reversion_long_short.py
├── advanced/
│   ├── overnight_mean_reversion.py # OMR for backtesting
│   ├── volatility_targeted_momentum.py
│   ├── cross_sectional_momentum.py
│   ├── pairs_trading.py
│   ├── market_regime_detector.py   # ✅ Already reused in live trading
│   ├── bayesian_reversion_model.py # ✅ Already reused in live trading
│   └── overnight_signal_generator.py # ✅ Already reused in live trading
```

### Live Trading Strategies (src/trading/strategies/)
```
src/trading/strategies/
└── omr_live_strategy.py            # OMR for live trading (duplicate logic!)
```

**Problem**: Duplicated strategy logic between backtesting and live trading layers!

## Proposed Architecture: Unified Strategy Layer

### Core Principle: Separation of Concerns

**Strategy Logic** (signal generation) should be **completely independent** from:
- Backtesting engine (Portfolio class, data access)
- Live trading infrastructure (BrokerInterface, ExecutionEngine)

### New Directory Structure

```
src/
├── strategies/                      # PURE STRATEGY LOGIC (no trading/backtest dependencies)
│   ├── core/                       # Abstract interfaces
│   │   ├── base_strategy.py        # Abstract strategy interface
│   │   ├── signal.py               # Signal data structure
│   │   └── indicator_calculator.py  # Technical indicators
│   │
│   ├── implementations/             # Concrete strategy implementations
│   │   ├── moving_average/
│   │   │   ├── ma_crossover.py     # Pure signal generation logic
│   │   │   └── triple_ma.py
│   │   ├── momentum/
│   │   │   ├── momentum.py         # Pure momentum signals
│   │   │   └── breakout.py
│   │   ├── mean_reversion/
│   │   │   ├── rsi_reversion.py    # Pure mean reversion signals
│   │   │   └── bollinger_reversion.py
│   │   ├── overnight/
│   │   │   ├── omr_signals.py      # Pure OMR signal logic
│   │   │   ├── regime_detector.py  # Market regime classification
│   │   │   └── bayesian_model.py   # Probability calculations
│   │   └── pairs/
│   │       └── pairs_signals.py
│   │
│   └── universe/                    # Symbol universe definitions
│       ├── etf_universe.py         # ETF lists (leveraged 3x, etc.)
│       ├── equity_universe.py      # Stock screeners (S&P 500, etc.)
│       └── crypto_universe.py
│
├── backtesting/                     # BACKTESTING ADAPTERS
│   ├── engine/
│   │   └── portfolio.py            # Backtesting engine
│   └── adapters/
│       ├── strategy_adapter.py     # Adapts pure strategies to backtest engine
│       ├── ma_backtest.py          # MA strategy adapter
│       ├── momentum_backtest.py
│       └── omr_backtest.py         # Uses src/strategies/implementations/overnight/
│
└── trading/                         # LIVE TRADING ADAPTERS
    ├── core/
    │   ├── trading_bot.py          # Generic trading orchestrator
    │   ├── execution_engine.py     # Order execution
    │   └── position_manager.py     # Position tracking
    ├── brokers/
    │   ├── broker_interface.py
    │   └── alpaca_broker.py
    └── adapters/
        ├── strategy_adapter.py     # Adapts pure strategies to live trading
        ├── ma_live.py              # MA strategy adapter
        ├── momentum_live.py
        └── omr_live.py             # Uses src/strategies/implementations/overnight/
```

## Key Abstraction: Pure Strategy Interface

### src/strategies/core/base_strategy.py

```python
"""
Pure strategy interface - no dependencies on backtesting or live trading.

Strategies implement pure signal generation logic. Adapters handle
integration with backtesting engines or live trading infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from dataclasses import dataclass


@dataclass
class Signal:
    """Pure signal data structure."""
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    metadata: Dict  # Strategy-specific data


class StrategySignals(ABC):
    """
    Abstract base class for pure signal generation strategies.

    No dependencies on:
    - Backtesting engine (Portfolio, data access methods)
    - Live trading infrastructure (BrokerInterface, ExecutionEngine)

    Only depends on:
    - pandas DataFrames (standard data structure)
    - datetime (standard library)
    - indicators (pure calculation functions)
    """

    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate trading signals based on market data.

        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
            timestamp: Current timestamp for signal generation

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def get_required_data(self) -> Dict[str, Dict]:
        """
        Specify what data this strategy needs.

        Returns:
            Dict like:
            {
                'SPY': {'timeframe': '1Day', 'lookback': 200},
                'TQQQ': {'timeframe': '1Min', 'lookback': 1},
                ...
            }
        """
        pass
```

## Example: Reusing MA Crossover Strategy

### Step 1: Extract Pure Signal Logic

**Before** (tightly coupled to backtesting engine):
```python
# src/strategies/base_strategies/moving_average.py
from src.engine.base_strategy import Strategy  # ❌ Coupled to backtest engine

class MovingAverageCrossover(Strategy):
    def generate_signals(self, data, timestamp):
        # Uses self.portfolio, self.get_indicator()
        # Tightly coupled to backtesting infrastructure

        prices = self.portfolio.get_latest_price(...)  # ❌ Backtest-specific
        ...
```

**After** (pure signal generation):
```python
# src/strategies/implementations/moving_average/ma_crossover.py
from src.strategies.core.base_strategy import StrategySignals, Signal

class MACrossoverSignals(StrategySignals):
    """Pure MA crossover signal generation - no infrastructure dependencies."""

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """Generate signals based purely on price data."""
        signals = []

        for symbol, df in market_data.items():
            # Calculate MAs from DataFrame directly (no portfolio dependency)
            df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
            df['slow_ma'] = df['close'].rolling(self.slow_period).mean()

            # Get latest values
            current = df.iloc[-1]
            previous = df.iloc[-2]

            # Golden cross (fast crosses above slow)
            if current['fast_ma'] > current['slow_ma'] and \
               previous['fast_ma'] <= previous['slow_ma']:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='BUY',
                    confidence=0.8,
                    metadata={
                        'fast_ma': current['fast_ma'],
                        'slow_ma': current['slow_ma'],
                        'strategy': 'MA_Crossover'
                    }
                ))

            # Death cross (fast crosses below slow)
            elif current['fast_ma'] < current['slow_ma'] and \
                 previous['fast_ma'] >= previous['slow_ma']:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='SELL',
                    confidence=0.8,
                    metadata={
                        'fast_ma': current['fast_ma'],
                        'slow_ma': current['slow_ma'],
                        'strategy': 'MA_Crossover'
                    }
                ))

        return signals

    def get_required_data(self) -> Dict[str, Dict]:
        """This strategy needs daily data with sufficient lookback for MAs."""
        # This would be populated by the adapter based on symbols traded
        return {
            # Symbol: requirements dict
            # Will be filled by adapter
        }
```

### Step 2: Create Backtest Adapter

```python
# src/backtesting/adapters/ma_backtest.py
from src.engine.base_strategy import Strategy
from src.strategies.implementations.moving_average.ma_crossover import MACrossoverSignals

class MABacktestAdapter(Strategy):
    """Adapter that integrates pure MA signals with backtesting engine."""

    def __init__(self, params: Dict):
        super().__init__(params)

        # Create pure signal generator
        self.signal_generator = MACrossoverSignals(
            fast_period=params.get('fast_period', 50),
            slow_period=params.get('slow_period', 200)
        )

    def generate_signals(self, data, timestamp):
        """Backtest engine interface - adapt to pure signal interface."""

        # Extract market data from backtest engine format
        market_data = self._extract_market_data(data)

        # Get pure signals
        signals = self.signal_generator.generate_signals(market_data, timestamp)

        # Convert pure signals to backtest engine format
        return self._convert_to_backtest_signals(signals)

    def _extract_market_data(self, backtest_data) -> Dict[str, pd.DataFrame]:
        """Convert backtest engine data to standard DataFrame format."""
        # Implementation depends on backtest engine structure
        pass

    def _convert_to_backtest_signals(self, pure_signals: List[Signal]) -> Dict:
        """Convert pure signals to backtest engine signal format."""
        # Implementation depends on backtest engine expectations
        pass
```

### Step 3: Create Live Trading Adapter

```python
# src/trading/adapters/ma_live.py
from src.trading.strategies.base_strategy import TradingStrategy
from src.strategies.implementations.moving_average.ma_crossover import MACrossoverSignals

class MALiveAdapter(TradingStrategy):
    """Adapter that integrates pure MA signals with live trading infrastructure."""

    def __init__(self, config: Dict):
        self.symbols = config.get('symbols', [])

        # Create pure signal generator
        self.signal_generator = MACrossoverSignals(
            fast_period=config.get('fast_period', 50),
            slow_period=config.get('slow_period', 200)
        )

        self.is_trained = True  # MA doesn't need training

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """MA strategy doesn't require training."""
        pass

    def generate_entry_signals(
        self,
        current_data: Dict[str, pd.DataFrame],
        broker: BrokerInterface
    ) -> List[Dict]:
        """Live trading interface - adapt to pure signal interface."""

        # Get pure signals
        timestamp = datetime.now()
        signals = self.signal_generator.generate_signals(current_data, timestamp)

        # Convert pure signals to live trading format
        live_signals = []
        for signal in signals:
            if signal.direction == 'BUY':
                quote = broker.get_latest_quote(signal.symbol)
                live_signals.append({
                    'symbol': signal.symbol,
                    'side': OrderSide.BUY,
                    'entry_price': quote['ask'],
                    'confidence': signal.confidence,
                    'metadata': signal.metadata
                })

        return live_signals

    def generate_exit_signals(self, broker: BrokerInterface) -> List[Dict]:
        """Generate exit signals for MA strategy."""
        # Get current positions
        positions = broker.get_positions()

        # Fetch current data
        # (Similar to entry signal generation)

        # Get pure signals
        signals = self.signal_generator.generate_signals(current_data, datetime.now())

        # Convert SELL signals to exits
        # ...
```

## Example: Reusing OMR Strategy

The OMR strategy components are **already well-structured** for reuse:
- `market_regime_detector.py` - Pure regime classification logic ✅
- `bayesian_reversion_model.py` - Pure probability calculations ✅
- `overnight_signal_generator.py` - Mostly pure signal generation ✅

Just need to remove hardcoded ETF list:

### Current (hardcoded):
```python
# src/strategies/advanced/overnight_signal_generator.py
class OvernightReversionSignals:
    LEVERAGED_ETFS = ['TQQQ', 'SQQQ', ...]  # ❌ Hardcoded

    def generate_signals(self, market_data, timestamp):
        for symbol in self.LEVERAGED_ETFS:  # ❌ Hardcoded
            ...
```

### Refactored (injected):
```python
# src/strategies/implementations/overnight/omr_signals.py
class OMRSignals(StrategySignals):
    def __init__(
        self,
        symbols: List[str],  # ✅ Injected
        regime_detector: MarketRegimeDetector,
        bayesian_model: BayesianReversionModel,
        ...
    ):
        self.symbols = symbols  # ✅ Not hardcoded
        self.regime_detector = regime_detector
        self.bayesian_model = bayesian_model
        ...

    def generate_signals(self, market_data, timestamp):
        for symbol in self.symbols:  # ✅ Configurable
            ...
```

### Backtest Adapter:
```python
# src/backtesting/adapters/omr_backtest.py
from src.strategies.implementations.overnight.omr_signals import OMRSignals

class OMRBacktestAdapter(Strategy):
    def __init__(self, params: Dict):
        # Get symbols from backtest config
        symbols = params.get('symbols', ['TQQQ', 'SQQQ', ...])

        # Create pure signal generator
        self.signal_generator = OMRSignals(
            symbols=symbols,
            regime_detector=MarketRegimeDetector(),
            bayesian_model=BayesianReversionModel(),
            ...
        )
```

### Live Trading Adapter:
```python
# src/trading/adapters/omr_live.py
from src.strategies.implementations.overnight.omr_signals import OMRSignals

class OMRLiveAdapter(TradingStrategy):
    def __init__(self, config: Dict):
        # Get symbols from live config
        symbols = config.get('symbols', ['TQQQ', 'SQQQ', ...])

        # Create pure signal generator
        self.signal_generator = OMRSignals(
            symbols=symbols,
            regime_detector=MarketRegimeDetector(),
            bayesian_model=BayesianReversionModel(),
            ...
        )
```

## Universe Management

Instead of hardcoding symbol lists, use universe classes:

```python
# src/strategies/universe/etf_universe.py
class ETFUniverse:
    """ETF symbol lists for different categories."""

    # Leveraged 3x Bull/Bear ETFs
    LEVERAGED_3X = [
        'TQQQ', 'SQQQ',  # Nasdaq 3x
        'UPRO', 'SPXU',  # S&P 500 3x
        'TMF', 'TMV',    # Treasury 3x
        'TECL', 'TECS',  # Tech 3x
        'FAS', 'FAZ',    # Financial 3x
        'TNA', 'TZA',    # Russell 2000 3x
    ]

    # Leveraged 2x ETFs
    LEVERAGED_2X = [
        'QLD', 'QID',    # Nasdaq 2x
        'SSO', 'SDS',    # S&P 500 2x
    ]

    # Sector ETFs
    SECTOR = [
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
        'XLP', 'XLY', 'XLU', 'XLB', 'XLRE'
    ]

# src/strategies/universe/equity_universe.py
class EquityUniverse:
    """Stock symbol lists."""

    FAANG = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']

    MEGA_CAP = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'BRK.B', 'V', 'JNJ'
    ]

    @staticmethod
    def load_sp500():
        """Dynamically load S&P 500 constituents."""
        import pandas as pd
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        return table['Symbol'].tolist()
```

### Usage:
```python
# Config file: config/strategies/omr_etf.yaml
symbols: !universe ETFUniverse.LEVERAGED_3X

# Or explicit list
symbols: ['TQQQ', 'SQQQ', 'UPRO']

# Or custom screener
symbols: !screener equity.momentum_screener(min_volume=1000000)
```

## Migration Path

### Phase 1: Extract Pure Strategy Logic (No Breaking Changes)
1. Create `src/strategies/core/` with abstract interfaces
2. Create `src/strategies/implementations/` with pure strategies
3. Keep existing backtest strategies as-is (no breaking changes)

### Phase 2: Create Adapters
1. Create `src/backtesting/adapters/`
2. Create `src/trading/adapters/`
3. Both adapters use same pure strategies from `src/strategies/implementations/`

### Phase 3: Migrate Existing Strategies
1. Move logic from `src/strategies/base_strategies/moving_average.py` to:
   - Pure: `src/strategies/implementations/moving_average/ma_crossover.py`
   - Backtest adapter: `src/backtesting/adapters/ma_backtest.py`
   - Live adapter: `src/trading/adapters/ma_live.py`

2. Update imports in backtest scripts to use adapters

### Phase 4: Deprecate Old Strategies
1. Mark old strategies as deprecated
2. Eventually remove once all code uses new adapters

## Benefits

1. **Single Source of Truth**: Strategy logic defined once, used everywhere
2. **Testability**: Pure strategies easy to unit test (no mocking needed)
3. **Flexibility**: Same strategy for backtesting and live trading
4. **Maintainability**: Fix bugs in one place, benefits both backtest and live
5. **Extensibility**: Add new strategies without touching infrastructure

## Summary

The key insight is **three-layer architecture**:

```
┌─────────────────────────────────────────────────┐
│  Pure Strategy Logic (src/strategies/)          │
│  - Signal generation                            │
│  - No infrastructure dependencies               │
│  - Reusable across backtest and live trading    │
└─────────────────────────────────────────────────┘
                       ↑
                       │ (used by)
         ┌─────────────┴─────────────┐
         │                           │
┌────────▼─────────┐      ┌──────────▼──────────┐
│ Backtest Adapter │      │ Live Trading Adapter│
│ (adapts to       │      │ (adapts to          │
│  Portfolio)      │      │  BrokerInterface)   │
└──────────────────┘      └─────────────────────┘
```

**Current state**: Most infrastructure is already generic!
**Main task**: Decouple strategy logic and create adapter layer.
