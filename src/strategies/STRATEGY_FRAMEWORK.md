# Strategy Framework

**A modular framework for implementing, testing, and deploying trading strategies with clean separation between signal logic and execution.**

**Last Updated**: 2026-05-17

---

## Overview

### What It Does
- Defines abstract base classes for pure strategy implementations
- Provides a strategy registry for dynamic lookup by name (backtest-side)
- Separates signal generation from execution concerns (backtest vs live)
- Contains production and research strategy implementations

### Key Features
- **Pure Signal Generation**: Strategies produce signals without execution dependencies
- **Adapter Pattern**: Adapters connect strategies to backtest engine or live trading
- **Strategy Registry**: Dynamic lookup with lazy imports and display name aliases
- **Regime Detection**: Market regime analysis for adaptive strategies (RAMP, OMR)
- **Bayesian Models**: Probability-based signal generation (OMR)

### Use Cases
- Implement new trading strategies following the framework
- Run strategies in backtesting or live trading via adapters
- Register custom strategies at runtime
- Analyze strategy behavior across market regimes

---

## Architecture

```
src/strategies/
|-- __init__.py                       # Module overview (no eager imports)
|-- registry.py                       # Dynamic strategy lookup (backtest registry)
|-- core/
|   |-- __init__.py                   # Re-exports StrategySignals, Signal, SignalBatch, DataRequirements
|   |-- base_strategy.py              # StrategySignals abstract base class + DataRequirements
|   `-- signal.py                     # Signal and SignalBatch data structures
|-- advanced/                         # Production-grade / advanced strategies
|   |-- overnight_mean_reversion.py   # OMR: Entry 3:50 PM, exit 9:31 AM
|   |-- momentum_protection_strategy.py # MP: 1m-1w momentum with crash protection (legacy, superseded by RAMP)
|   |-- ramp_strategy.py              # RAMP: Regime-aware momentum protection (production)
|   |-- ramp_target_planner.py        # Helper: regime -> target weights for RAMP
|   |-- bayesian_reversion_model.py   # Bayesian probability model for OMR
|   |-- market_regime_detector.py     # Market regime detection (5 regimes)
|   |-- overnight_signal_generator.py # Signal generation for OMR
|   |-- orb_strategy.py / orb_indicators.py / orb_numba_core.py  # Opening range breakout
|   |-- hv_orb_strategy.py / hv_orb_indicators.py  # High-volatility ORB ("stocks in play")
|   |-- ict_strategy.py / ict_indicators.py  # ICT / Smart Money Concepts
|   |-- bmsb_strategy.py / bmsb_indicators.py  # Bull-Market Support-Band
|   |-- ml_crypto_mr_strategy.py / ml_crypto_mr_indicators.py  # ML crypto mean reversion
|   |-- hurst_mr_strategy.py          # Hurst-based mean reversion
|   |-- opex_pinning_strategy.py      # OpEx / gamma pinning
|   |-- cscm_strategy.py / cscm_signals.py / cscm_indicators.py  # Cross-Sectional Crypto Momentum
|   |-- dsts_strategy.py / dsts_signals.py / dsts_indicators.py  # Dual-Signal Trend Sentinel
|   |-- frs_strategy.py / frs_indicators.py  # Fractal Regime Switching
|   |-- evr_strategy.py / evr_indicators.py  # Effort vs Result (VSA)
|   |-- exit_checker.py               # Shared exit/stop checker helpers
|   `-- zscore_mr_base.py             # Shared base for z-score mean reversion
|-- research/                         # Experimental strategies
|   |-- moving_average.py             # MovingAverageCrossover, TripleMovingAverage
|   |-- mean_reversion.py             # MeanReversion, RSIMeanReversion
|   |-- mean_reversion_long_short.py  # MeanReversionLongShort
|   |-- momentum.py                   # MomentumStrategy, BreakoutStrategy
|   |-- pairs_trading.py              # PairsTrading
|   |-- cross_sectional_momentum.py   # CrossSectionalMomentum
|   |-- volatility_targeted_momentum.py # VolatilityTargetedMomentum
|   |-- breakout_strategies.py        # Breakout variants
|   `-- high52_breakout_strategy.py   # 52-week high breakout
|-- implementations/                  # Pure signal-only implementations
|   |-- moving_average/
|   |   `-- ma_crossover_signals.py
|   `-- momentum/
|       `-- momentum_signals.py
|-- universe/                         # Trading universe definitions
|   |-- equity_universe.py            # S&P 500, Russell 1000
|   |-- etf_universe.py               # Leveraged ETFs, sector ETFs
|   |-- momentum_universe.py
|   `-- orb_universe.py
|-- options/                          # Options-specific strategies / helpers
|-- opex/                             # OpEx pinning helpers (signal generator, calendar, GEX)
|-- qqq_signal/                       # QQQ data-loader helpers
|-- base_strategies/                  # (LEGACY SHELL -- directory currently has no Python files)
`-- # Note: there is no top-level `custom/` directory yet -- see "Creating Custom Strategies" below.
```

### Design Philosophy

1. **Pure Strategies**: Strategy logic is isolated from execution (no broker/backtest deps)
2. **Adapter Pattern**: Adapters in `src/backtesting/adapters/` and `src/trading/adapters/` connect strategies
3. **Lazy Imports**: Registry uses lazy loading to avoid circular import issues
4. **Production vs Research vs Live**: Production code lives under `advanced/`, research lives under `research/`, live runtime concerns live under `src/trading/adapters/`.
5. **Data Validation**: Built-in validation for market data structure

> Note on `base_strategies/`: this directory used to hold a re-export shell that pointed into older modules. It is currently empty (no `__init__.py`, only stale `__pycache__`), so importing `src.strategies.base_strategies` will fail. New code should import from `src.strategies.core` (for `StrategySignals` / `Signal`) and from `src.strategies.research` / `src.strategies.advanced` for concrete strategies.

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

**Purpose**: Dynamic strategy lookup by class name or display name -- used by the **config-driven backtest runner** (`src/backtest_runner.py`). The registry maps name -> `(module_path, class_name)` and lazily imports the module on first access. All registered classes are subclasses of `src.backtesting.base.strategy.BaseStrategy`.

**IMPORTANT**: `RAMP` (and the pure-signal `RAMPSignals` class) is NOT in this registry. The registry is for `BaseStrategy`-based backtest strategies; RAMP runs through the live trading adapter / dedicated backtest pipeline. Calling `get_strategy_class("RAMP")` raises `ValueError: Unknown strategy: 'RAMP'`. The same applies to `CSCM` configurations that route through `src.strategies.advanced.cscm_strategy` -- they ARE in the registry under `CSCMStrategy`, but custom multi-asset adapters may bypass it.

**Key Functions**:
- `get_strategy_class(name)`: Get strategy class by name (raises `ValueError` if unknown, `ImportError` if module fails to load)
- `list_strategies()`: List all registered canonical class names (sorted)
- `list_strategy_display_names()`: Dict mapping display name -> canonical class name
- `get_strategy_info(name)`: Info dict with `class_name`, `module`, `description`, `parameters`
- `register_strategy(name, strategy_cls, display_name=None)`: Register a custom strategy at runtime
- `clear_cache()`: Clear the class-import cache (useful for tests)

**Registered Strategies** (current registry, 21 canonical entries; some have aliased class-name entries that resolve to the same class):

| Canonical Name | Module | Class | Notes |
|----------------|--------|-------|-------|
| `MovingAverageCrossover` | `src.strategies.research.moving_average` | `MovingAverageCrossover` | |
| `TripleMovingAverage` | `src.strategies.research.moving_average` | `TripleMovingAverage` | |
| `MeanReversion` | `src.strategies.research.mean_reversion` | `MeanReversion` | Bollinger-band based |
| `RSIMeanReversion` | `src.strategies.research.mean_reversion` | `RSIMeanReversion` | |
| `MeanReversionLongShort` | `src.strategies.research.mean_reversion_long_short` | `MeanReversionLongShort` | |
| `MomentumStrategy` | `src.strategies.research.momentum` | `MomentumStrategy` | |
| `BreakoutStrategy` | `src.strategies.research.momentum` | `BreakoutStrategy` | |
| `VolatilityTargetedMomentum` | `src.strategies.research.volatility_targeted_momentum` | `VolatilityTargetedMomentum` | |
| `CrossSectionalMomentum` | `src.strategies.research.cross_sectional_momentum` | `CrossSectionalMomentum` | |
| `PairsTrading` | `src.strategies.research.pairs_trading` | `PairsTrading` | |
| `OvernightMeanReversion` / `OvernightMeanReversionStrategy` | `src.strategies.advanced.overnight_mean_reversion` | `OvernightMeanReversionStrategy` | OMR (production) |
| `MomentumProtection` / `MomentumProtectionStrategy` | `src.strategies.advanced.momentum_protection_strategy` | `MomentumProtectionStrategy` | Legacy MP, superseded by RAMP |
| `ORBStrategy` | `src.strategies.advanced.orb_strategy` | `ORBStrategy` | Opening Range Breakout |
| `ICTStrategy` | `src.strategies.advanced.ict_strategy` | `ICTStrategy` | Smart Money Concepts |
| `HVORBStrategy` | `src.strategies.advanced.hv_orb_strategy` | `HVORBStrategy` | Stocks-in-play HVORB |
| `BMSBStrategy` | `src.strategies.advanced.bmsb_strategy` | `BMSBStrategy` | Bull-Market Support-Band |
| `MLCryptoMRStrategy` | `src.strategies.advanced.ml_crypto_mr_strategy` | `MLCryptoMRStrategy` | |
| `HurstMRStrategy` | `src.strategies.advanced.hurst_mr_strategy` | `HurstMRStrategy` | |
| `OpExPinningStrategy` | `src.strategies.advanced.opex_pinning_strategy` | `OpExPinningStrategy` | Gamma pinning |
| `CSCMStrategy` | `src.strategies.advanced.cscm_strategy` | `CSCMStrategy` | Cross-sectional crypto momentum |
| `DSTSStrategy` | `src.strategies.advanced.dsts_strategy` | `DSTSStrategy` | Dual-Signal Trend Sentinel |
| `FRSStrategy` | `src.strategies.advanced.frs_strategy` | `FRSStrategy` | Fractal Regime Switching |
| `EVRStrategy` | `src.strategies.advanced.evr_strategy` | `EVRStrategy` | Effort vs Result / VSA |

**Display-Name Aliases** (selected -- see `_DISPLAY_NAME_MAP` in `registry.py` for the full set):

| Display Name | Resolves To |
|--------------|-------------|
| `OMR`, `Overnight Mean Reversion` | `OvernightMeanReversion` |
| `MP`, `Momentum Protection`, `Protected Momentum`, ... | `MomentumProtection` |
| `Moving Average Crossover` | `MovingAverageCrossover` |
| `Mean Reversion`, `RSI Mean Reversion`, `Mean Reversion Long Short` | resp. `MeanReversion` / `RSIMeanReversion` / `MeanReversionLongShort` |
| `Momentum`, `Momentum Strategy` | `MomentumStrategy` |
| `Breakout`, `Breakout Strategy` | `BreakoutStrategy` |
| `Pairs`, `Pairs Trading` | `PairsTrading` |
| `Cross-Sectional Momentum`, `Cross Sectional Momentum` | `CrossSectionalMomentum` |
| `ORB`, `Opening Range Breakout` | `ORBStrategy` |
| `HVORB`, `HV ORB`, `SIP`, `Stocks in Play`, `High Volatility ORB` | `HVORBStrategy` |
| `ICT`, `SMC`, `Smart Money Concepts`, `Liquidity Strategy` | `ICTStrategy` |
| `BMSB`, `Bull Market Support Band`, `Bull Market Band` | `BMSBStrategy` |
| `ML Crypto MR`, `MLMR`, `ML Mean Reversion`, `Crypto Mean Reversion` | `MLCryptoMRStrategy` |
| `Hurst MR`, `HurstMR`, `Hurst Strategy`, `Hurst Mean Reversion` | `HurstMRStrategy` |
| `OpEx`, `OpEx Pinning`, `Gamma Pinning`, `GEX Strategy` | `OpExPinningStrategy` |
| `CSCM`, `Cross-Sectional Crypto Momentum`, `Crypto Momentum` | `CSCMStrategy` |
| `DSTS`, `Trend Sentinel`, `Z-Score Trend`, `Dual Signal Trend Sentinel` | `DSTSStrategy` |
| `FRS`, `Fractal Regime Switching`, `Hurst Regime`, `Regime Switching` | `FRSStrategy` |
| `EVR`, `VSA`, `Volume Spread Analysis`, `Absorption`, `Effort vs Result` | `EVRStrategy` |

**There is no `RAMP` or `RAMPSignals` display-name alias** -- attempting `get_strategy_class("RAMP")` will raise.

**Usage**:
```python
from src.strategies.registry import get_strategy_class, list_strategies

# Get by class name
cls = get_strategy_class('MovingAverageCrossover')
strategy = cls(fast_period=10, slow_period=50)

# Get by display name
cls = get_strategy_class('OMR')
strategy = cls()

# List all canonical class names
print(list_strategies())
# ['BMSBStrategy', 'BreakoutStrategy', 'CSCMStrategy', 'CrossSectionalMomentum', ...]
```

### OvernightMeanReversionStrategy (`advanced/overnight_mean_reversion.py`)

**Purpose**: Production-grade strategy for overnight mean reversion in leveraged ETFs.

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

### MomentumProtectionStrategy (`advanced/momentum_protection_strategy.py`) -- LEGACY

**Status**: Legacy. Superseded by RAMP for production deployment. Still in the registry for backtest comparisons.

**Purpose**: Basic momentum with crash protection (no regime awareness).

**Parameters**:
- `top_n`: Number of top momentum stocks (default: 10)
- `reduced_exposure`: Exposure during protection (default: 0.5)

---

### RAMP (`advanced/ramp_strategy.py`) -- PRODUCTION

**Purpose**: Regime-aware momentum protection, currently deployed on `homeguard-multi.service`.

**NOT in the backtest strategy registry**: RAMP is wired in through a dedicated live adapter (`src/trading/adapters/ramp_live_adapter.py`) and a target planner (`src/strategies/advanced/ramp_target_planner.py`) rather than the `registry.py` lookup. Backtests of RAMP go through dedicated walk-forward configs / scripts, not the generic `get_strategy_class("RAMP")` path.

**Key Features**:
- Universe: S&P 500 stocks
- Regime Detection: 5 market regimes (STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR)
- Selection: Top N by regime-specific momentum formula
- Rebalance: Daily at 3:55 PM EST
- Protection: reduced exposure in stress regimes (VIX / SPY drawdown driven)

**Momentum Formula**:
```
momentum = (long_weight * return_long_period) - (penalty_weight * return_short_period)
```

**Regime-Specific Parameters** (walk-forward validated; see strategy doc for current values):

| Regime | Long Period | Short Period | Long Weight | Penalty Weight | Top N |
|--------|-------------|--------------|-------------|----------------|-------|
| STRONG_BULL    | 21 | 5  | 0.3 | 5.0 | 20 |
| WEAK_BULL      | 21 | 5  | 0.3 | 5.0 | 10 |
| SIDEWAYS       | 21 | 5  | 0.5 | 2.0 | 5  |
| UNPREDICTABLE  | 42 | 21 | 0.5 | 4.0 | 10 |
| BEAR           | 21 | 5  | 0.3 | 3.0 | 10 |

**Position Sizing**:
- Dynamic 1/N: Each position = `max_capital_allocation / top_n`
- Example: 100% allocation with `top_n=10` -> 10% per position

**Performance** (walk-forward validation 2022-2024): 0.846 Sharpe ratio out-of-sample. See `docs/strategies/20251212_RAMP_WALK_FORWARD_VALIDATION.md` for the audit trail.

See [RAMP Strategy Documentation](../../docs/strategies/RAMP_STRATEGY.md) for full details.

---

## Data Flow

```
Market Data (Dict[symbol, DataFrame])
        v
  StrategySignals.validate_data() (per-symbol)
        v
  StrategySignals.generate_signals()
        v
  List[Signal]
        v
  +------------------------------------+
  |        Adapter Layer               |
  +------------------+-----------------+
  | BacktestAdapter  | LiveAdapter     |
  +------------------+-----------------+
        v                    v
  PortfolioSimulator    ExecutionEngine
        v                    v
  Backtest Results      Live Orders (Alpaca / IBKR / Coinbase)
```

---

## Public API

### Core Exports

```python
from src.strategies.core import (
    StrategySignals,    # Base class
    Signal,             # Signal data structure
    SignalBatch,        # Collection of signals
    DataRequirements,   # Per-strategy data spec
)
```

### Registry Exports

```python
from src.strategies.registry import (
    get_strategy_class,           # Get class by name
    list_strategies,              # List canonical class names
    list_strategy_display_names,  # Display-name -> class-name dict
    get_strategy_info,            # Strategy info with params
    register_strategy,            # Register custom strategy
    clear_cache,                  # Test helper: clear import cache
)
```

### Production / Live Strategies

```python
from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversionStrategy
from src.strategies.advanced.ramp_strategy import RAMPSignals          # NOT in registry; used by live adapter
from src.strategies.advanced.momentum_protection_strategy import MomentumProtectionStrategy  # Legacy
from src.strategies.advanced.cscm_strategy import CSCMStrategy
```

---

## Configuration

### YAML Config (via registry)

```yaml
strategy:
  name: "MovingAverageCrossover"   # canonical or display name (case-insensitive)
  params:
    fast_period: 10
    slow_period: 50
```

### Environment Variables

None required. Strategies use data passed to them.

---

## Dependencies

### Internal (src/ modules)
- `src.backtesting.base.strategy` - BaseStrategy (registry-loaded strategies inherit from this)
- `src.utils.logger` - Logging utilities

### External (pip packages)
- `pandas` - DataFrames for market data
- `numpy` - Numerical computations
- `scikit-learn` - Bayesian model (optional)

---

## Strategy Categories

### Production-Deployed Strategies

| Strategy | Location | Schedule | Broker | Service | In Registry |
|----------|----------|----------|--------|---------|-------------|
| RAMP | `advanced/ramp_strategy.py` | Rebalance 3:55 PM daily | IBKR paper | `homeguard-multi` | NO (live-adapter only) |
| OMR  | `advanced/overnight_mean_reversion.py` | Entry 3:50 PM, Exit 9:31 AM | IBKR paper | (disabled in toggle) | YES |
| CSCM | `advanced/cscm_strategy.py` | Weekly (Sun 00:00 UTC) | Coinbase | `homeguard-cscm` | YES |

Legacy / not deployed: `MomentumProtection` (registered, superseded by RAMP).

### Research Strategies (backtesting only)

All registered for use with the config-driven backtest runner:

| Strategy | Location | Description |
|----------|----------|-------------|
| Moving Average | `research/moving_average.py` | MA crossover, triple MA |
| Mean Reversion | `research/mean_reversion.py` | Bollinger, RSI-based |
| Mean Reversion Long/Short | `research/mean_reversion_long_short.py` | Cross-sectional MR |
| Momentum | `research/momentum.py` | Basic momentum, breakout |
| Pairs Trading | `research/pairs_trading.py` | Statistical arbitrage |
| Cross-Sectional Momentum | `research/cross_sectional_momentum.py` | Relative momentum |
| Vol-Targeted Momentum | `research/volatility_targeted_momentum.py` | Vol-adjusted |
| Breakout | `research/breakout_strategies.py` | Price breakout |
| High-52 Breakout | `research/high52_breakout_strategy.py` | 52-week high |

### Advanced Strategies (research / in-development / specialized)

`ORB`, `HVORB`, `ICT`, `BMSB`, `MLCryptoMR`, `HurstMR`, `OpExPinning`, `DSTS`, `FRS`, `EVR` -- all registered, all backtest-only at the moment.

---

## Creating Custom Strategies

### Step 1: Create Strategy Class

Put your strategy under `src/strategies/research/` (or under a new top-level
package if you prefer -- there is no required `custom/` directory). Inherit from
`StrategySignals` (pure) or `BaseStrategy` (if you want to register it with the
backtest runner):

```python
# src/strategies/research/my_strategy.py
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
        return 2
```

### Step 2: Register Strategy

```python
from src.strategies.registry import register_strategy
from src.strategies.research.my_strategy import MyCustomStrategy

register_strategy(
    name="MyCustomStrategy",
    strategy_cls=MyCustomStrategy,
    display_name="My Custom Strategy"
)
```

For permanent registration, add an entry to `_STRATEGY_REGISTRY` in
`src/strategies/registry.py` (and an alias to `_DISPLAY_NAME_MAP` if desired).

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

- **2026-05-17**: Corrected registry inventory (21 canonical entries), noted RAMP is NOT in the registry, clarified `base_strategies/` is an empty legacy shell, added missing strategies (ORB, HVORB, ICT, BMSB, MLCryptoMR, HurstMR, OpExPinning, CSCM, DSTS, FRS, EVR).
- **2025-12-08**: Added RAMP strategy documentation, deprecated MP
- **2025-12-08**: Initial documentation created
- **2025-12-06**: Reorganized into production vs research
- **2025-12-03**: MP strategy changed to 1m-1w momentum
- **2025-11-XX**: Strategy registry with lazy loading
- **2025-10-XX**: Initial strategy framework
