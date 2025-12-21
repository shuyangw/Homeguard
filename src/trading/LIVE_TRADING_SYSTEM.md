# Live Trading System

**A broker-agnostic paper trading framework with multi-strategy support, execution engine, and state management.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Executes automated paper/live trading strategies via Alpaca API
- Provides broker-agnostic interfaces for order execution and market data
- Manages multi-strategy coordination with atomic state persistence
- Handles position tracking, risk management, and execution analytics

### Key Features
- **Broker-Agnostic Design**: Core logic depends on interfaces, not implementations
- **Multi-Strategy Support**: Concurrent strategies with execution locks and position isolation
- **Adapter Pattern**: Connect pure strategy logic to live trading infrastructure
- **State Persistence**: Atomic JSON state with file locking and automatic backups
- **Execution Engine**: Order execution with retry logic, status tracking, and analytics
- **Toggle Configuration**: Enable/disable strategies via YAML without code changes

### Use Cases
- Run paper trading for overnight mean reversion (OMR) strategy
- Execute regime-aware momentum protection (RAMP) strategy alongside OMR
- Test new strategies with isolated position tracking
- Monitor execution metrics and portfolio health

---

## Architecture

```
src/trading/
├── __init__.py                    # Public API: BrokerFactory, BrokerInterface
├── brokers/
│   ├── __init__.py
│   ├── broker_interface.py        # Composite interface (backward compat)
│   ├── broker_factory.py          # Factory pattern for broker creation
│   ├── alpaca_broker.py           # Alpaca API implementation
│   └── interfaces/
│       ├── __init__.py            # All interface exports
│       ├── base.py                # Base enums and exceptions
│       ├── account.py             # AccountInterface
│       ├── market_data.py         # MarketDataInterface
│       ├── market_hours.py        # MarketHoursInterface
│       ├── order_management.py    # OrderManagementInterface
│       ├── stock_trading.py       # StockTradingInterface
│       └── options_trading.py     # OptionsTradingInterface (future)
├── core/
│   ├── __init__.py
│   ├── paper_trading_bot.py       # Main orchestrator (legacy)
│   ├── execution_engine.py        # Order execution with retry logic
│   └── position_manager.py        # Position and risk tracking
├── adapters/
│   ├── __init__.py
│   ├── strategy_adapter.py        # Base adapter for strategies
│   ├── omr_live_adapter.py        # OMR strategy adapter
│   ├── ramp_live_adapter.py       # RAMP strategy adapter (production)
│   ├── momentum_live_adapter.py   # Momentum strategy adapter (deprecated)
│   └── ma_live_adapter.py         # Moving average adapter
├── strategies/
│   ├── __init__.py
│   └── omr_live_strategy.py       # OMR live trading logic
├── state/
│   ├── __init__.py
│   └── strategy_state_manager.py  # Multi-strategy state persistence
├── config/
│   ├── __init__.py
│   └── omr_config_loader.py       # Strategy configuration loading
└── utils/
    └── portfolio_health_check.py  # Portfolio monitoring utilities
```

### Design Philosophy

1. **Dependency Inversion**: Core components depend on `BrokerInterface`, not Alpaca directly
2. **Adapter Pattern**: Pure strategy logic is isolated; adapters handle live trading concerns
3. **Factory Pattern**: `BrokerFactory.create_from_yaml()` creates broker instances from config
4. **Atomic State**: File locking and temp file writes ensure state consistency
5. **Execution Locks**: Only one strategy executes at a time to prevent race conditions

---

## Key Components

### BrokerInterface (`brokers/broker_interface.py`)

**Purpose**: Composite interface providing backward compatibility with the original API.

**Key Interfaces** (in `brokers/interfaces/`):
- `AccountInterface`: Account info, buying power, equity
- `MarketDataInterface`: Quotes, trades, historical bars
- `MarketHoursInterface`: Market open/close times
- `StockTradingInterface`: Place/cancel orders, positions
- `OrderManagementInterface`: Order status, history

**Usage**:
```python
from src.trading import BrokerFactory, BrokerInterface

broker = BrokerFactory.create_from_yaml('config/trading/broker_alpaca.yaml')
account = broker.get_account()  # Returns dict, not object
positions = broker.get_positions()
```

### AlpacaBroker (`brokers/alpaca_broker.py`)

**Purpose**: Translates `BrokerInterface` to Alpaca's specific API.

**Key Features**:
- Automatic feed selection (IEX for paper, SIP for live)
- Stale quote detection with trade price fallback
- Timezone conversion (all data returned in Eastern Time)
- Error translation to broker-agnostic exceptions

**Exceptions**:
| Exception | Cause |
|-----------|-------|
| `BrokerConnectionError` | API connection/network issues |
| `InvalidOrderError` | Invalid order parameters |
| `InsufficientFundsError` | Not enough buying power |
| `OrderNotFoundError` | Order ID doesn't exist |
| `NoPositionError` | Position not found |

### ExecutionEngine (`core/execution_engine.py`)

**Purpose**: Order execution with retry logic, status tracking, and analytics.

**Key Features**:
- Configurable retry attempts and delays
- Fill timeout monitoring
- Batch order execution
- Execution metrics tracking

**Usage**:
```python
from src.trading.core.execution_engine import ExecutionEngine

engine = ExecutionEngine(
    broker=broker,
    max_retries=3,
    retry_delay=1.0,
    fill_timeout=30.0
)

result = engine.execute_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    wait_for_fill=True
)

metrics = engine.get_execution_metrics()
# {'total_orders': 5, 'successful_orders': 4, 'success_rate': 0.8, ...}
```

### StrategyAdapter (`adapters/strategy_adapter.py`)

**Purpose**: Base adapter connecting pure strategies to live trading infrastructure.

**Responsibilities**:
- Fetch market data from broker
- Call pure strategy for signal generation
- Convert signals to orders via ExecutionEngine
- Manage positions and risk
- Handle scheduling and lifecycle

**Key Methods**:
- `preload_historical_data()`: Cache historical data at market open
- `prefetch_intraday_data()`: Fetch today's minute bars before execution
- `generate_signals()`: Call strategy and return signals
- `execute_signals()`: Execute filtered signals via ExecutionEngine
- `run_once()`: Single iteration of strategy workflow

**Usage**:
```python
from src.trading.adapters.omr_live_adapter import OMRLiveAdapter

adapter = OMRLiveAdapter(
    strategy=omr_strategy,
    broker=broker,
    symbols=['TQQQ', 'SOXL', 'UPRO'],
    position_size=0.1,  # 10% per trade
    max_positions=5
)

adapter.preload_historical_data()  # At 9:30 AM
adapter.run_once()  # At 3:55 PM
```

### StrategyStateManager (`state/strategy_state_manager.py`)

**Purpose**: Manages state for multiple trading strategies with atomic persistence.

**Key Features**:
- **Execution Locks**: Serialize strategy execution (4-minute timeout)
- **Position Tracking**: Per-strategy position state with broker sync
- **Toggle Config**: Enable/disable strategies via YAML
- **Backup Recovery**: Automatic backups and corruption recovery
- **Drift Detection**: Detects when broker qty differs from state

**Usage**:
```python
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()

# Check if strategy is enabled
if manager.is_enabled('omr'):
    # Acquire execution lock
    if manager.acquire_execution_lock('omr'):
        try:
            # Add position
            manager.add_position('omr', 'TQQQ', 100, 65.00)
            # ... execute orders ...
        finally:
            manager.release_execution_lock('omr')

# For top-ups (accumulates qty, doesn't overwrite)
manager.add_or_update_position('omr', 'TQQQ', 50, 66.00)

# Sync with broker positions
broker_positions = {'TQQQ': 150, 'SOXL': 100}
changes = manager.sync_with_broker(broker_positions)
```

**State Files**:
- `data/trading/strategy_positions.json` - Position state
- `config/trading/strategy_toggle.yaml` - Enable/disable config

---

## Data Flow

```
Strategy Config (YAML)
        v
  StrategyAdapter
        v
  preload_historical_data() -> AlpacaBroker.get_bars()
        v
  prefetch_intraday_data() -> AlpacaBroker.get_historical_bars()
        v
  generate_signals() -> Pure Strategy
        v
  filter_signals() -> PositionManager (risk checks)
        v
  execute_signals() -> ExecutionEngine
        v
  ExecutionEngine.execute_order() -> AlpacaBroker.place_order()
        v
  StrategyStateManager.add_position()
        v
  State Persisted (JSON)
```

---

## Public API

### Primary Exports

```python
from src.trading import BrokerFactory, BrokerInterface

# Create broker from YAML config
broker = BrokerFactory.create_from_yaml('config/trading/broker_alpaca.yaml')

# Or create directly
from src.trading.brokers.alpaca_broker import AlpacaBroker
broker = AlpacaBroker(api_key='KEY', secret_key='SECRET', paper=True)
```

### Order Types and Enums

```python
from src.trading.brokers.broker_interface import (
    OrderSide,      # BUY, SELL
    OrderType,      # MARKET, LIMIT, STOP, STOP_LIMIT
    OrderStatus,    # NEW, FILLED, CANCELLED, REJECTED, etc.
    TimeInForce,    # DAY, GTC, IOC, FOK
)
```

### State Management

```python
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()
manager.is_enabled('omr')  # Check if enabled
manager.get_positions('omr')  # Get positions
manager.print_status()  # Print all strategy status
```

---

## Configuration

### Broker Config (`config/trading/broker_alpaca.yaml`)

```yaml
broker:
  type: alpaca
  paper: true

credentials:
  api_key: ${ALPACA_API_KEY}
  secret_key: ${ALPACA_SECRET_KEY}
```

### Strategy Toggle (`config/trading/strategy_toggle.yaml`)

```yaml
strategies:
  omr:
    enabled: true
    shutdown_requested: false
  ramp:
    enabled: true
    shutdown_requested: false
  mp:
    enabled: false  # Deprecated, replaced by RAMP
    shutdown_requested: false
last_modified: '2025-12-08T18:20:00-05:00'
modified_by: manual
```

### Environment Variables

- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret key

---

## Dependencies

### Internal (src/ modules)
- `src.strategies.core` - StrategySignals, Signal classes
- `src.data.providers` - DataProviderInterface for fallback data
- `src.utils.logger` - Centralized logging
- `src.utils.timezone` - Timezone utilities (tz.now())

### External (pip packages)
- `alpaca-py` - Alpaca trading and data API
- `pandas` - DataFrames for market data
- `pytz` - Timezone handling
- `pyyaml` - Configuration loading

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `BrokerConnectionError` | Network/API issues | Retry with backoff |
| `InvalidOrderError` | Bad order params | Fix parameters |
| `InsufficientFundsError` | Not enough buying power | Reduce position size |
| `State file corrupted` | JSON parse error | Auto-recover from backup |
| `Execution lock timeout` | Other strategy running | Wait or check logs |

---

## Deployment

### Systemd Services (EC2)

The live trading system runs as separate systemd services:

| Service | Strategy | Status |
|---------|----------|--------|
| `homeguard-omr.service` | Overnight Mean Reversion | Active |
| `homeguard-ramp.service` | Regime-Aware Momentum Protection | Active |
| `homeguard-mp.service` | Momentum Protection | Deprecated |
| `homeguard-trading.target` | Target to manage all | Active |

**Commands**:
```bash
# Start all strategies
sudo systemctl start homeguard-trading.target

# Check status
sudo systemctl status homeguard-omr homeguard-ramp

# View logs
journalctl -u homeguard-omr -u homeguard-ramp -f

# View RAMP-specific logs
journalctl -u homeguard-ramp --since "1 hour ago" | grep -E "Regime:|Position"
```

### Process Names

Each strategy runs with a distinct process name for monitoring:
- `homeguard-omr` - OMR strategy process
- `homeguard-ramp` - RAMP strategy process
- `homeguard-mp` - MP strategy process (deprecated)

---

## Testing

### Test Location
- `tests/trading/` - Unit tests
- `tests/integration/` - Integration tests

### Running Tests
```bash
pytest tests/trading/ -v
pytest tests/trading/test_execution_engine.py -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Module Reference](../../docs/architecture/MODULE_REFERENCE.md)
- [Multi-Strategy Position Management](../../docs/architecture/MULTI_STRATEGY_POSITION_MANAGEMENT.md)
- [Infrastructure Overview](../../docs/INFRASTRUCTURE_OVERVIEW.md)

---

## Changelog

- **2025-12-08**: Added RAMP service documentation, deprecated MP
- **2025-12-08**: Initial documentation created
- **2025-12-06**: Multi-strategy systemd services added
- **2025-11-XX**: StrategyStateManager for multi-strategy support
- **2025-10-XX**: Initial broker-agnostic architecture
