# QQQ-Signal Leveraged Semiconductor ETF Strategy

## Theory and Architecture Documentation

**Date:** 2025-11-25
**Module:** `src/strategies/qqq_signal/`
**Status:** Implemented (Backtesting), Pending (Live Trading)

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Signal Generation Variants](#signal-generation-variants)
4. [Cross-Asset Execution](#cross-asset-execution)
5. [Signal Lag and Lookahead Bias](#signal-lag-and-lookahead-bias)
6. [Architecture Design](#architecture-design)
7. [Integration with Backtesting Engine](#integration-with-backtesting-engine)
8. [Future: Live Trading Integration](#future-live-trading-integration)
9. [Parameter Optimization](#parameter-optimization)
10. [Risk Considerations](#risk-considerations)

---

## Strategy Overview

The QQQ-Signal strategy is a **cross-asset momentum/trend-following system** that uses the Nasdaq-100 ETF (QQQ) as a signal generator to time entries and exits in leveraged semiconductor ETFs.

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| Strategy Type | Cross-asset momentum |
| Signal Source | QQQ (Nasdaq-100 ETF) |
| Trade Assets | USD (2x leveraged) or SOXL (3x leveraged) |
| Position Types | Long only |
| Signal Methods | EMA-based (3 variants) |

### Why Cross-Asset?

The strategy exploits the relationship between broad tech sector momentum (QQQ) and semiconductor sector performance (USD/SOXL). The hypothesis is:

1. **QQQ as leading indicator**: The Nasdaq-100 captures broad technology momentum
2. **Semiconductor leverage**: Semiconductors are a high-beta subset of tech
3. **Amplified returns**: Using leveraged ETFs magnifies gains during bullish periods
4. **Risk reduction**: Exiting during QQQ weakness avoids leveraged drawdowns

---

## Theoretical Foundation

### Momentum Persistence

The strategy is built on the empirical observation that momentum tends to persist in equity markets. When QQQ shows upward momentum (via EMA alignment), the semiconductor sector—being a high-beta component—is likely to outperform.

### EMA as Trend Filter

Exponential Moving Averages (EMAs) are used because they:
- Weight recent prices more heavily than Simple Moving Averages (SMAs)
- React faster to price changes
- Provide clear, objective entry/exit rules

The EMA formula:
```
EMA_today = Price_today × k + EMA_yesterday × (1 - k)
where k = 2 / (period + 1)
```

**Important:** This implementation uses `adjust=False` in pandas EMA calculation:
```python
qqq_close.ewm(span=period, adjust=False).mean()
```

This matches the standard recursive EMA formula and ensures consistency between backtesting and live trading.

### Leverage Timing

The core insight is that **leverage timing matters more than leverage amount**. A 3x leveraged ETF in a downtrend will destroy wealth through:
- Volatility decay (beta slippage)
- Compounding losses
- Rebalancing drag

By using QQQ as a regime filter, the strategy aims to hold leveraged positions only during favorable conditions.

---

## Signal Generation Variants

### Variant 1: Two-EMA Crossover

**Logic:**
- Entry: Fast EMA > Slow EMA
- Exit: Fast EMA < Slow EMA

**Best Reported Parameters:** Fast=3, Slow=18

**Characteristics:**
- Most responsive to trend changes
- Higher trade frequency
- Works well in trending markets
- May whipsaw in choppy conditions

```
Signal = 1 if EMA(3) > EMA(18) else 0
```

### Variant 2: Three-EMA Regime Filter (Stateful)

**Logic:**
- Entry: Fast > Medium > Slow (bullish alignment)
- Exit: Fast < Medium < Slow (bearish alignment)
- Neutral: Maintain current position

**Best Reported Parameters:** Fast=7, Medium=21, Slow=25

**Characteristics:**
- Most conservative variant
- Requires strong trend confirmation for entry
- Requires strong trend reversal for exit
- **Stateful behavior**: Position maintained during neutral states
- Fewer trades, potentially larger gains per trade

```python
if fast > medium > slow:      # Bullish alignment
    position = LONG
elif fast < medium < slow:    # Bearish alignment
    position = FLAT
else:                         # Neutral - maintain position
    position = position  # No change
```

**Why Stateful?**

The three-EMA variant uses stateful logic because:
1. **Reduces whipsaws**: Doesn't exit on every minor EMA crossing
2. **Trend continuation**: Stays in position during consolidations
3. **Clear regime change**: Only exits on confirmed bearish reversal

### Variant 3: Single-EMA Price Filter

**Logic:**
- Entry: Close > EMA
- Exit: Close < EMA

**Best Reported Parameters:** Period=53

**Characteristics:**
- Simplest variant
- Uses price vs EMA instead of EMA vs EMA
- More responsive than two-EMA crossover
- Single parameter to optimize

```
Signal = 1 if Close > EMA(53) else 0
```

---

## Cross-Asset Execution

### Signal Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                        │
│                                                             │
│   QQQ Close Prices  ──►  EMA Calculation  ──►  Signal (0/1)│
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRADE EXECUTION                          │
│                                                             │
│   Signal (0/1)  ──►  Entry/Exit Logic  ──►  USD/SOXL Order │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Not Trade QQQ Directly?

1. **Leverage amplification**: USD (2x) and SOXL (3x) provide amplified exposure
2. **Sector focus**: Semiconductors have historically outperformed during tech rallies
3. **Signal-trade separation**: QQQ provides cleaner signals (more liquid, less volatile)

### Data Requirements

The strategy requires synchronized data for both symbols:

```python
data_dict = {
    "QQQ": DataFrame with OHLCV,   # For signal generation
    "USD": DataFrame with OHLCV    # For trade execution
}
```

Both DataFrames must have aligned DatetimeIndex for proper signal application.

---

## Signal Lag and Lookahead Bias

### The Lookahead Problem

The original strategy description implies same-day execution:
> "If Fast EMA > Slow EMA at close, buy USD at close"

This is problematic for backtesting because:
1. You can't know the closing price until the market closes
2. You can't execute a trade at the close price after knowing it
3. This creates unrealistic backtest results (lookahead bias)

### Solution: Signal Lag

The implementation applies a configurable signal lag (default: 1 day):

```python
# Signal generated on day T
raw_signals = core.get_signal_series(qqq_close)

# Position taken on day T+1
lagged_signals = raw_signals.shift(signal_lag)
```

**Timeline with signal_lag=1:**

| Day | QQQ Close | Signal Generated | Position Taken |
|-----|-----------|------------------|----------------|
| T   | $450      | LONG             | -              |
| T+1 | $452      | LONG             | Enter LONG     |
| T+2 | $448      | FLAT             | Hold LONG      |
| T+3 | $445      | FLAT             | Exit LONG      |

### Implementation Details

```python
# In backtest_adapter.py
if self.signal_lag > 0:
    lagged_signals = raw_signals.shift(self.signal_lag)
else:
    lagged_signals = raw_signals

# Align to trade asset index
aligned_signals = lagged_signals.reindex(trade_data.index, method='ffill')
```

---

## Architecture Design

### Hexagonal Architecture (Ports & Adapters)

The strategy uses a clean hexagonal architecture separating:
- **Core Logic**: Pure strategy calculations (no I/O, no dependencies)
- **Adapters**: Integration with backtesting and live trading systems

```
                    ┌─────────────────────┐
                    │   QQQSignalCore     │
                    │   (Pure Logic)      │
                    │                     │
                    │ - calculate_emas()  │
                    │ - get_signal_state()│
                    │ - get_signal_series()│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                │                ▼
┌─────────────────────┐        │    ┌─────────────────────┐
│ QQQSignalBacktest   │        │    │ QQQSignalLive       │
│ (Backtest Adapter)  │        │    │ (Live Adapter)      │
│                     │        │    │ [Future]            │
│ implements:         │        │    │                     │
│ MultiSymbolStrategy │        │    │ implements:         │
└─────────────────────┘        │    │ StrategySignals     │
                               │    └─────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │    SignalState      │
                    │    (Enum)           │
                    │                     │
                    │    LONG = "long"    │
                    │    FLAT = "flat"    │
                    └─────────────────────┘
```

### Module Structure

```
src/strategies/qqq_signal/
├── __init__.py           # Public exports
├── signal_state.py       # SignalState enum
├── core.py               # QQQSignalCore (pure logic)
└── backtest_adapter.py   # QQQSignalBacktest (MultiSymbolStrategy)
```

### Why This Architecture?

1. **Testability**: Core logic can be unit tested without infrastructure
2. **Reusability**: Same core logic for backtesting and live trading
3. **Maintainability**: Changes to backtesting don't affect core logic
4. **Extensibility**: Easy to add new adapters (paper trading, different brokers)

### Core Class Design

```python
class QQQSignalCore:
    """Pure strategy logic - no infrastructure dependencies."""

    def __init__(self, variant, fast_period, slow_period, ...):
        self._validate_parameters()

    def calculate_emas(self, qqq_close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate all required EMAs."""

    def get_signal_state(self, qqq_close: pd.Series) -> SignalState:
        """Get current signal state (for live trading)."""

    def get_signal_series(self, qqq_close: pd.Series) -> pd.Series:
        """Get signal series (for backtesting)."""
```

### Adapter Class Design

```python
class QQQSignalBacktest(MultiSymbolStrategy):
    """Adapts QQQSignalCore for backtesting engine."""

    def __init__(self, ..., signal_lag: int = 1):
        self.core = QQQSignalCore(...)  # Composition over inheritance

    def get_required_symbols(self) -> List[str]:
        return ["QQQ", self.core.trade_asset]

    def generate_multi_signals(self, data_dict) -> Dict[str, Tuple[Series, Series]]:
        """Generate entry/exit signals for backtest engine."""
```

---

## Integration with Backtesting Engine

### MultiSymbolStrategy Interface

The `QQQSignalBacktest` adapter implements the `MultiSymbolStrategy` interface required by the backtesting engine:

```python
class MultiSymbolStrategy(ABC):
    @abstractmethod
    def get_required_symbols(self) -> List[str]:
        """Return list of symbols needed by this strategy."""

    @abstractmethod
    def generate_multi_signals(self, data_dict: Dict[str, DataFrame])
        -> Dict[str, Tuple[Series, Series]]:
        """Generate entry/exit signals for each tradeable symbol."""
```

### Signal Generation Flow

```python
def generate_multi_signals(self, data_dict):
    # 1. Extract QQQ data for signal generation
    qqq_close = data_dict["QQQ"]['close']

    # 2. Generate raw signals using core logic
    raw_signals = self.core.get_signal_series(qqq_close)

    # 3. Apply signal lag
    lagged_signals = raw_signals.shift(self.signal_lag)

    # 4. Align to trade asset index
    aligned_signals = lagged_signals.reindex(trade_data.index, method='ffill')

    # 5. Convert to entry/exit signals
    signal_change = aligned_signals.diff()
    entries = (signal_change == 1)   # 0 → 1 transition
    exits = (signal_change == -1)    # 1 → 0 transition

    # 6. Return signals for trade asset only
    return {self.core.trade_asset: (entries, exits)}
```

### Usage Example

```python
from backtesting.engine import BacktestEngine
from strategies.qqq_signal import QQQSignalBacktest

# Create strategy
strategy = QQQSignalBacktest(
    variant="two_ema",
    fast_period=3,
    slow_period=18,
    trade_asset="USD",
    signal_lag=1
)

# Create engine
engine = BacktestEngine(
    initial_capital=100000,
    fees=0.001,
    slippage=0.001
)

# Run backtest
portfolio = engine.run(
    strategy=strategy,
    symbols=["QQQ", "USD"],  # Both required
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Get results
stats = portfolio.stats()
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
```

---

## Future: Live Trading Integration

### StrategySignals Interface

For live trading, a `QQQSignalLive` adapter would implement the `StrategySignals` interface:

```python
class QQQSignalLive(StrategySignals):
    """Live trading adapter for QQQ-Signal strategy."""

    def __init__(self, ...):
        self.core = QQQSignalCore(...)

    def should_enter(self, market_data: Dict) -> bool:
        """Check if we should enter a position."""
        qqq_close = self._get_qqq_history(market_data)
        return self.core.get_signal_state(qqq_close) == SignalState.LONG

    def should_exit(self, market_data: Dict, position: Position) -> bool:
        """Check if we should exit current position."""
        qqq_close = self._get_qqq_history(market_data)
        return self.core.get_signal_state(qqq_close) == SignalState.FLAT
```

### State Management for Three-EMA

For the three-EMA variant in live trading, the stateful behavior is automatically handled because `get_signal_state()` internally calls `get_signal_series()`:

```python
# In core.py
def get_signal_state(self, qqq_close: pd.Series) -> SignalState:
    if self.variant == "three_ema":
        # Uses full history to determine current state
        signals = self.get_signal_series(qqq_close)
        return SignalState.LONG if signals.iloc[-1] == 1 else SignalState.FLAT
```

This ensures the live trading signal is consistent with what backtesting would have produced at the same point in time.

---

## Parameter Optimization

### Optimization Script

The strategy includes an optimization script at `backtest_scripts/optimize_qqq_signal_leveraged_etf.py`:

```python
# Two-EMA parameter grid
param_grid = {
    'variant': ['two_ema'],
    'fast_period': [2, 3, 5, 7, 10, 15, 20],
    'slow_period': [15, 18, 21, 25, 30, 40, 50],
}

# Three-EMA parameter grid
param_grid = {
    'variant': ['three_ema'],
    'fast_period': [5, 7, 10, 12],
    'medium_period': [15, 18, 21, 25],
    'slow_period': [25, 30, 35, 40, 50],
}

# Single-EMA parameter grid
param_grid = {
    'variant': ['single_ema'],
    'fast_period': [10, 20, 30, 40, 50, 53, 60, 75, 100],
}
```

### Walk-Forward Validation

To detect overfitting, the optimizer uses walk-forward validation:

```
Training Windows:
├── 2020-01-01 to 2021-06-30 → Test 2021-07-01 to 2021-12-31
├── 2020-07-01 to 2022-01-01 → Test 2022-01-01 to 2022-06-30
├── 2021-01-01 to 2022-06-30 → Test 2022-07-01 to 2022-12-31
├── 2021-07-01 to 2023-01-01 → Test 2023-01-01 to 2023-06-30
└── 2022-01-01 to 2023-06-30 → Test 2023-07-01 to 2023-12-31
```

**Degradation Assessment:**
- < 30%: Parameters appear robust
- 30-50%: Some overfitting possible
- > 50%: Significant overfitting likely

---

## Risk Considerations

### Leverage Risk

Leveraged ETFs carry significant risks:

1. **Volatility decay**: Daily rebalancing erodes returns in volatile markets
2. **Path dependency**: Same start/end prices can yield different leveraged returns
3. **Compounding losses**: 2x or 3x daily losses compound rapidly

### Strategy-Specific Risks

1. **Regime change**: Strategy may underperform during sector rotation
2. **Correlation breakdown**: QQQ-semiconductor correlation may weaken
3. **Whipsaw risk**: Choppy markets generate false signals
4. **Execution risk**: Slippage in leveraged ETFs during volatility

### Risk Mitigation

The implementation includes several risk controls:

1. **Signal lag**: Reduces lookahead bias and allows realistic execution
2. **Stateful three-EMA**: Reduces whipsaws by requiring strong reversals
3. **Position sizing**: Controlled via RiskConfig (default 10% per trade)
4. **Walk-forward validation**: Detects overfitting before deployment

### Recommended Risk Profile

```python
from backtesting.utils.risk_config import RiskConfig

# Conservative for leveraged ETF strategy
engine.risk_config = RiskConfig.moderate()  # 10% position sizing
```

---

## Test Coverage

The strategy has comprehensive test coverage:

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `test_core.py` | 35 | Core logic validation |
| `test_backtest_adapter.py` | 20 | Adapter functionality |
| `test_strategy_spec_compliance.py` | 20 | Spec conformance |

### Key Test Areas

- EMA calculation with `adjust=False`
- All three variant signal generation
- Three-EMA stateful behavior
- Signal lag application
- Cross-asset execution
- Entry/exit generation
- Parameter validation

---

## References

- Strategy module: `src/strategies/qqq_signal/`
- Tests: `tests/strategies/qqq_signal/`
- Optimization script: `backtest_scripts/optimize_qqq_signal_leveraged_etf.py`
- Implementation plan: `docs/todos/20251125_QQQ_SIGNAL_LEVERAGED_ETF_STRATEGY_PLAN.md`
