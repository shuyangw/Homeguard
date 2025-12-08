# Strategy Framework

**A modular framework for implementing, testing, and deploying trading strategies with clean separation between signal logic and execution.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Defines abstract base classes for pure strategy implementations
- Provides a strategy registry for dynamic lookup by name
- Separates signal generation from execution concerns (backtest vs live)
- Contains production and research strategy implementations

### Key Features
- **Pure Signal Generation**: Strategies produce signals without execution dependencies
- **Adapter Pattern**: Adapters connect strategies to backtest engine or live trading
- **Strategy Registry**: Dynamic lookup with lazy imports and display name aliases
- **Regime Detection**: Market regime analysis for adaptive strategies
- **Bayesian Models**: Probability-based signal generation

### Use Cases
- Implement new trading strategies following the framework
- Run strategies in backtesting or live trading via adapters
- Register custom strategies at runtime
- Analyze strategy behavior across market regimes

---

## Architecture

```
src/strategies/
├── __init__.py                      # Module overview and strategy locations
├── registry.py                      # Dynamic strategy lookup by name
├── core/
│   ├── __init__.py                  # Core exports: StrategySignals, Signal
│   ├── base_strategy.py             # StrategySignals abstract base class
│   └── signal.py                    # Signal and SignalBatch data structures
├── advanced/                        # Production-ready strategies
│   ├── __init__.py
│   ├── overnight_mean_reversion.py  # OMR: Entry 3:50 PM, exit 9:31 AM
│   ├── momentum_protection_strategy.py # MP: 1m-1w momentum with crash protection
│   ├── bayesian_reversion_model.py  # Bayesian probability model for OMR
│   ├── market_regime_detector.py    # Market regime detection (5 regimes)
│   └── overnight_signal_generator.py # Signal generation for OMR
├── research/                        # Experimental strategies
│   ├── __init__.py
│   ├── moving_average.py            # MA crossover strategies
│   ├── mean_reversion.py            # Mean reversion strategies
│   ├── momentum.py                  # Basic momentum strategies
│   ├── pairs_trading.py             # Statistical arbitrage
│   ├── cross_sectional_momentum.py  # Relative momentum
│   ├── volatility_targeted_momentum.py # Vol-adjusted momentum
│   ├── breakout_strategies.py       # Breakout detection
│   └── high52_breakout_strategy.py  # 52-week high breakout
├── implementations/                 # Pure signal implementations
│   ├── __init__.py
│   ├── moving_average/
│   │   └── ma_crossover_signals.py  # MA crossover signal generator
│   └── momentum/
│       └── momentum_signals.py      # Momentum signal generator
├── universe/                        # Trading universe definitions
│   ├── __init__.py
│   ├── equity_universe.py           # S&P 500, Russell 1000
│   ├── etf_universe.py              # Leveraged ETFs, sector ETFs
│   └── momentum_universe.py         # Momentum strategy universe
├── base_strategies/                 # Legacy (deprecated)
│   └── __init__.py
├── advanced_strategies/             # Legacy (deprecated)
│   └── __init__.py
└── custom/                          # User custom strategies
    ├── __init__.py
    └── template.py                  # Template for custom strategies
```

### Design Philosophy

1. **Pure Strategies**: Strategy logic is isolated from execution (no broker/backtest deps)
2. **Adapter Pattern**: Adapters in `src/backtesting/adapters/` and `src/trading/adapters/` connect strategies
3. **Lazy Imports**: Registry uses lazy loading to avoid circular import issues
4. **Production vs Research**: Clear separation between deployed and experimental code
5. **Data Validation**: Built-in validation for market data structure

---

## Key Components

### StrategySignals (`core/base_strategy.py`)

**Purpose**: Abstract base class for pure signal generation strategies.

**Key Methods**:
- `generate_signals(market_data, timestamp)`: Generate List[Signal] from market data
- `get_required_lookback()`: Return number of periods needed
- `validate_data(df, symbol)`: Validate DataFrame structure
- `get_parameters()`: Return strategy parameters

**Usage**:
```python
from src.strategies.core import StrategySignals, Signal

class MyStrategy(StrategySignals):
    def __init__(self, fast_period=10, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, market_data, timestamp):
        signals = []
        for symbol, df in market_data.items():
            fast_ma = df['close'].rolling(self.fast_period).mean()
            slow_ma = df['close'].rolling(self.slow_period).mean()

            if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='BUY',
                    confidence=0.7,
                    price=df['close'].iloc[-1]
                ))
        return signals

    def get_required_lookback(self):
        return self.slow_period
```

### Signal (`core/signal.py`)

**Purpose**: Pure data structure representing a trading signal.

**Attributes**:
- `timestamp`: When signal was generated
- `symbol`: Trading symbol (e.g., 'AAPL')
- `direction`: 'BUY', 'SELL', or 'HOLD'
- `confidence`: Signal confidence (0.0 to 1.0)
- `price`: Signal price (close when generated)
- `metadata`: Strategy-specific data

**Validation**:
- Direction must be 'BUY', 'SELL', or 'HOLD'
- Confidence must be between 0.0 and 1.0
- Price must be positive

**Usage**:
```python
from src.strategies.core import Signal

signal = Signal(
    timestamp=datetime.now(),
    symbol='TQQQ',
    direction='BUY',
    confidence=0.85,
    price=65.50,
    metadata={'regime': 'bull', 'win_prob': 0.62}
)

# Serialize
data = signal.to_dict()

# Deserialize
signal = Signal.from_dict(data)
```

### Strategy Registry (`registry.py`)

**Purpose**: Dynamic strategy lookup by class name or display name.

**Key Functions**:
- `get_strategy_class(name)`: Get strategy class by name
- `list_strategies()`: List all available strategies
- `get_strategy_info(name)`: Get strategy info with parameters
- `register_strategy(name, cls)`: Register custom strategy

**Supported Strategies**:
| Name | Class | Description |
|------|-------|-------------|
| `MovingAverageCrossover` | `MovingAverageCrossover` | MA crossover |
| `MeanReversion` | `MeanReversion` | Bollinger band reversion |
| `MomentumStrategy` | `MomentumStrategy` | Basic momentum |
| `OvernightMeanReversion` | `OvernightMeanReversionStrategy` | OMR (production) |
| `MomentumProtection` | `MomentumProtectionStrategy` | MP (production) |

**Display Name Aliases**:
- "OMR" → `OvernightMeanReversion`
- "MP" → `MomentumProtection`
- "Moving Average Crossover" → `MovingAverageCrossover`

**Usage**:
```python
from src.strategies.registry import get_strategy_class, list_strategies

# Get by class name
cls = get_strategy_class('MovingAverageCrossover')
strategy = cls(fast_period=10, slow_period=50)

# Get by display name
cls = get_strategy_class('OMR')
strategy = cls()

# List all strategies
print(list_strategies())
# ['MeanReversion', 'MomentumProtection', 'MomentumStrategy', 'MovingAverageCrossover', 'OvernightMeanReversion', ...]
```

### OvernightMeanReversionStrategy (`advanced/overnight_mean_reversion.py`)

**Purpose**: Production strategy for overnight mean reversion in leveraged ETFs.

**Key Features**:
- Entry at 3:50 PM EST, exit at 9:31 AM EST
- Bayesian probability model trained on 10 years of data
- 5 market regimes for adaptive signal generation
- Regime detection using SPY and VIX

**Parameters**:
- `min_probability`: Minimum win rate threshold (default: 0.55)
- `min_expected_return`: Minimum expected return (default: 0.002)
- `max_positions`: Maximum concurrent positions (default: 5)
- `position_size`: Position size as fraction (default: 0.20)

**Training**:
```python
strategy = OvernightMeanReversionStrategy()
strategy.train_models(historical_data)  # Dict[symbol, DataFrame]
```

### MomentumProtectionStrategy (`advanced/momentum_protection_strategy.py`)

**Purpose**: Production strategy for momentum with crash protection.

**Key Features**:
- Universe: S&P 500 stocks
- Selection: Top N by 1m-1w momentum (21-5 day returns)
- Rebalance: Daily at 3:55 PM EST
- Protection: 50% exposure when VIX > 25 or SPY drawdown > 5%

**Parameters**:
- `top_n`: Number of top momentum stocks (default: 10)
- `reduced_exposure`: Exposure during protection (default: 0.5)
- `vix_threshold`: VIX level for protection (default: 25.0)
- `spy_dd_threshold`: SPY drawdown threshold (default: -0.05)

**Decision History**:
- 2025-12-03: Changed from 3m-1m to 1m-1w based on walk-forward validation
- Simplified risk profile for better returns AND drawdown protection

---

## Data Flow

```
Market Data (Dict[symbol, DataFrame])
        ↓
  StrategySignals.validate_market_data()
        ↓
  StrategySignals.generate_signals()
        ↓
  List[Signal]
        ↓
  ┌────────────────────────────────────┐
  │        Adapter Layer               │
  ├──────────────┬─────────────────────┤
  │ BacktestAdapter │ LiveTradingAdapter │
  └──────────────┴─────────────────────┘
        ↓                    ↓
  PortfolioSimulator    ExecutionEngine
        ↓                    ↓
  Backtest Results      Live Orders
```

---

## Public API

### Core Exports

```python
from src.strategies.core import (
    StrategySignals,  # Base class
    Signal,           # Signal data structure
    SignalBatch,      # Collection of signals
)
```

### Registry Exports

```python
from src.strategies.registry import (
    get_strategy_class,       # Get class by name
    list_strategies,          # List all strategies
    list_strategy_display_names,  # Display name mapping
    get_strategy_info,        # Strategy info with params
    register_strategy,        # Register custom strategy
)
```

### Production Strategies

```python
from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversionStrategy
from src.strategies.advanced.momentum_protection_strategy import MomentumProtectionStrategy
```

---

## Configuration

### YAML Config (via registry)

```yaml
strategy:
  name: "MovingAverageCrossover"  # Or display name
  params:
    fast_period: 10
    slow_period: 50
```

### Environment Variables

None required. Strategies use data passed to them.

---

## Dependencies

### Internal (src/ modules)
- `src.backtesting.base.strategy` - BaseStrategy for backtest integration
- `src.utils.logger` - Logging utilities

### External (pip packages)
- `pandas` - DataFrames for market data
- `numpy` - Numerical computations
- `scikit-learn` - Bayesian model (optional)

---

## Strategy Categories

### Production Strategies (EC2 deployed)

| Strategy | Location | Schedule |
|----------|----------|----------|
| OMR | `advanced/overnight_mean_reversion.py` | Entry 3:50 PM, Exit 9:31 AM |
| MP | `advanced/momentum_protection_strategy.py` | Rebalance 3:55 PM daily |

### Research Strategies (backtesting only)

| Strategy | Location | Description |
|----------|----------|-------------|
| Moving Average | `research/moving_average.py` | MA crossover, triple MA |
| Mean Reversion | `research/mean_reversion.py` | Bollinger, RSI-based |
| Momentum | `research/momentum.py` | Basic momentum, breakout |
| Pairs Trading | `research/pairs_trading.py` | Statistical arbitrage |
| Cross-Sectional | `research/cross_sectional_momentum.py` | Relative momentum |
| Vol-Targeted | `research/volatility_targeted_momentum.py` | Vol-adjusted |
| Breakout | `research/breakout_strategies.py` | Price breakout |
| High-52 | `research/high52_breakout_strategy.py` | 52-week high |

---

## Creating Custom Strategies

### Step 1: Create Strategy Class

```python
# src/strategies/custom/my_strategy.py
from src.strategies.core import StrategySignals, Signal

class MyCustomStrategy(StrategySignals):
    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def generate_signals(self, market_data, timestamp):
        signals = []
        for symbol, df in market_data.items():
            returns = df['close'].pct_change().iloc[-1]
            if returns < -self.threshold:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='BUY',
                    confidence=min(abs(returns) / self.threshold, 1.0),
                    price=df['close'].iloc[-1]
                ))
        return signals

    def get_required_lookback(self):
        return 2  # Need 2 days for pct_change
```

### Step 2: Register Strategy

```python
from src.strategies.registry import register_strategy
from src.strategies.custom.my_strategy import MyCustomStrategy

register_strategy(
    name="MyCustomStrategy",
    strategy_cls=MyCustomStrategy,
    display_name="My Custom Strategy"
)
```

### Step 3: Use in Config

```yaml
strategy:
  name: "My Custom Strategy"
  params:
    threshold: 0.02
```

---

## Testing

### Test Location
- `tests/strategies/` - Unit tests
- `tests/integration/` - Integration tests

### Running Tests
```bash
pytest tests/strategies/ -v
pytest tests/strategies/test_registry.py -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Module Reference](../../docs/architecture/MODULE_REFERENCE.md)
- [Backtesting Engine](../backtesting/BACKTESTING_ENGINE.md)
- [Live Trading System](../trading/LIVE_TRADING_SYSTEM.md)

---

## Changelog

- **2025-12-08**: Initial documentation created
- **2025-12-06**: Reorganized into production vs research
- **2025-12-03**: MP strategy changed to 1m-1w momentum
- **2025-11-XX**: Strategy registry with lazy loading
- **2025-10-XX**: Initial strategy framework
