# Live Trading System

**A broker-agnostic paper trading framework with multi-strategy support, execution engine, state management, and IBKR + Alpaca + Coinbase broker implementations routed by strategy.**

**Last Updated**: 2026-05-17

---

## Overview

### What It Does
- Executes automated paper/live trading strategies via IBKR (primary for stocks/options), Alpaca (fallback / legacy), and Coinbase (crypto)
- Provides broker-agnostic interfaces (ISP) for order execution and market data
- Manages multi-strategy coordination with atomic state persistence
- Handles position tracking, risk management, and execution analytics
- Routes each strategy to the correct broker via `config/trading/broker_routing.yaml`

### Key Features
- **Broker-Agnostic Design**: Core logic depends on focused interfaces (AccountInterface, MarketDataInterface, StockTradingInterface, OptionsTradingInterface, ...), not implementations
- **Multi-Broker**: IBKR (`ib_async`), Alpaca (`alpaca-py`), Coinbase (`coinbase-advanced-py`); a strategy-broker router picks the right backend per strategy
- **Multi-Strategy Support**: Concurrent strategies with execution locks and position isolation
- **Adapter Pattern**: Connect pure strategy logic (`StrategySignals`) to live trading infrastructure
- **State Persistence**: Atomic JSON state with file locking and automatic backups
- **Execution Engine**: Order execution with retry logic, status tracking, and analytics
- **Toggle Configuration**: Enable/disable strategies via YAML without code changes

### Use Cases
- Run paper trading for RAMP (Regime-Aware Momentum Protection) on IBKR
- Hold OMR (Overnight Mean Reversion) and MP code paths warm for re-enablement on IBKR
- Run CSCM (Cross-Sectional Crypto Momentum) on Coinbase in a separate service
- Test new strategies with isolated position tracking
- Monitor execution metrics and portfolio health

---

## Architecture

```
src/trading/
|-- __init__.py
|-- price_oracle.py
|-- brokers/
|   |-- __init__.py
|   |-- broker_interface.py        # Composite interface (backward compat)
|   |-- broker_factory.py          # Factory: creates broker instances from YAML / routing
|   |-- alpaca_broker.py           # Alpaca (stocks/options) implementation
|   |-- alpaca_crypto_broker.py    # Alpaca crypto implementation
|   |-- coinbase_broker.py         # Coinbase Advanced Trade implementation (crypto)
|   |-- crypto_router.py           # Crypto-broker routing helper
|   |-- ibkr/                      # IBKR integration (see "IBKR Broker Integration" below)
|   |   |-- __init__.py
|   |   |-- config.py              # IBKRConfig (pydantic)
|   |   |-- connection.py          # IBKRConnectionManager (background event loop)
|   |   |-- contracts.py           # ContractResolver (symbol -> ib_async.Contract)
|   |   |-- data_download.py       # IBKRDataProvider (historical bars)
|   |   |-- errors.py              # IBKR error-code -> Homeguard exception mapping
|   |   |-- ibkr_broker.py         # IBKRBroker (Account/Market/Stock/Options)
|   |   |-- ibkr_futures_broker.py # IBKRFuturesBroker
|   |   |-- pacing.py              # Historical data pacing limiter
|   |   |-- streaming.py           # IBKRStreamingProvider (StreamingProviderInterface)
|   |   `-- symbols.py             # Domain<->IBKR symbol normalization (BRK.B vs BRK B)
|   `-- interfaces/                # ISP-compliant interfaces
|       |-- base.py                # Enums and broker-agnostic exceptions
|       |-- account.py             # AccountInterface
|       |-- crypto_trading.py      # CryptoTradingInterface
|       |-- futures_trading.py     # FuturesTradingInterface
|       |-- market_data.py         # MarketDataInterface
|       |-- market_hours.py        # MarketHoursInterface
|       |-- options_trading.py     # OptionsTradingInterface
|       |-- order_management.py    # OrderManagementInterface
|       `-- stock_trading.py       # StockTradingInterface
|-- core/
|   |-- paper_trading_bot.py       # Main orchestrator (legacy)
|   |-- execution_engine.py        # Order execution with retry logic
|   `-- position_manager.py        # Position and risk tracking
|-- adapters/
|   |-- strategy_adapter.py        # Base adapter for strategies
|   |-- omr_live_adapter.py        # OMR strategy adapter
|   |-- ramp_live_adapter.py       # RAMP strategy adapter (production)
|   |-- momentum_live_adapter.py   # MP adapter (legacy, superseded by RAMP)
|   |-- ma_live_adapter.py         # Moving average adapter
|   |-- cscm_live_adapter.py       # CSCM live adapter (Coinbase)
|   |-- cscm_paper_adapter.py      # CSCM paper-mode variant
|   `-- cscm_demo_adapter.py       # CSCM demo / sandbox variant
|-- strategies/
|   `-- omr_live_strategy.py       # OMR live trading logic
|-- state/
|   `-- strategy_state_manager.py  # Multi-strategy state persistence
|-- config/
|   `-- omr_config_loader.py       # Strategy configuration loading
|-- decision_log/                  # Per-execution decision logging
|-- futures/                       # Futures broker safeguards (expiration / margin guards / audit)
|-- portfolio/                     # Portfolio analytics helpers
|-- logging/                       # Trading-specific logging helpers
|-- demo/                          # Demo / sandbox flows
`-- utils/
    `-- portfolio_health_check.py  # Portfolio monitoring utilities
```

### Design Philosophy

1. **Dependency Inversion**: Core components depend on focused interfaces (`AccountInterface`, `StockTradingInterface`, `OptionsTradingInterface`, ...), not on Alpaca or IBKR directly
2. **Adapter Pattern**: Pure strategy logic is isolated; adapters handle live trading concerns
3. **Factory + Routing**: `BrokerFactory` plus `broker_routing.yaml` constructs and assigns the correct broker per strategy
4. **Atomic State**: File locking and temp file writes ensure state consistency
5. **Execution Locks**: Only one strategy executes at a time to prevent race conditions
6. **Boundary Translation**: Symbol formats (e.g. `BRK.B` vs `BRK B`), timestamps (everything returned to strategies in ET), and broker-specific exceptions are translated at the broker boundary -- strategy code stays broker-agnostic

---

## Key Components

### Broker Interfaces (`brokers/interfaces/`)

**Purpose**: ISP-compliant interfaces -- each broker advertises which it implements.

| Interface | What it covers |
|-----------|----------------|
| `AccountInterface`         | Account info, buying power, equity, connection lifecycle |
| `MarketHoursInterface`     | Market open/close times |
| `MarketDataInterface`      | Quotes, trades, historical bars |
| `OrderManagementInterface` | Order status, history, cancellation |
| `StockTradingInterface`    | Place/cancel stock orders, stock positions |
| `OptionsTradingInterface`  | Options chains, options positions, multi-leg orders |
| `FuturesTradingInterface`  | Futures contracts, expiration / margin safeguards |
| `CryptoTradingInterface`   | Crypto orders/positions (Coinbase, Alpaca crypto) |

Each focused interface lives in its own module under `brokers/interfaces/`. A composite `BrokerInterface` (`brokers/broker_interface.py`) re-exposes the older "everything in one" surface for backward compatibility.

**Usage**:
```python
from src.trading import BrokerFactory, BrokerInterface

# YAML-driven (single-broker config)
broker = BrokerFactory.create_from_yaml('config/trading/broker_alpaca.yaml')
account = broker.get_account()
positions = broker.get_positions()
```

### AlpacaBroker (`brokers/alpaca_broker.py`)

**Purpose**: Implements `AccountInterface`, `MarketDataInterface`, `MarketHoursInterface`, `StockTradingInterface`, `OrderManagementInterface` against Alpaca.

**Status**: Fallback / legacy. RAMP, OMR, and MP now route to IBKR per `broker_routing.yaml`; Alpaca is the `default_broker` for any strategy not explicitly routed.

**Key Features**:
- Automatic feed selection (IEX for paper, SIP for live)
- Stale quote detection with trade price fallback
- Timezone conversion (all data returned in Eastern Time)
- Error translation to broker-agnostic exceptions

### IBKRBroker (`brokers/ibkr/ibkr_broker.py`)

**Purpose**: Implements `AccountInterface`, `MarketHoursInterface`, `MarketDataInterface`, `StockTradingInterface`, and -- uniquely among Homeguard brokers -- `OptionsTradingInterface`.

**Status**: Primary stocks/options broker for RAMP / OMR / MP. Speaks to an IB Gateway (paper port 4002, live port 4001) via `ib_async`. The current production deployment uses paper port 4002 with `client_id=10`.

See the dedicated [IBKR Broker Integration](#ibkr-broker-integration) section below for the file-by-file breakdown.

### CoinbaseBroker (`brokers/coinbase_broker.py`)

**Purpose**: Implements `CryptoTradingInterface` against Coinbase Advanced Trade. Used by CSCM.

### Broker Routing (`config/trading/broker_routing.yaml`)

**Purpose**: Single source of truth for which broker each strategy uses. The live runner (`scripts/trading/run_live_paper_trading.py`) reads this file and constructs the broker before instantiating the strategy adapter.

```yaml
brokers:
  alpaca:
    paper: true
    # credentials from env: ALPACA_PAPER_KEY_ID, ALPACA_PAPER_SECRET_KEY

  ibkr:
    port: 4002          # 4002 paper / 4001 live
    client_id: 10
    readonly: false

strategies:
  omr:
    broker: ibkr
  ramp:
    broker: ibkr
  mp:
    broker: ibkr
  cscm:
    broker: coinbase

default_broker: alpaca   # used if a strategy is not listed above
```

### ExecutionEngine (`core/execution_engine.py`)

**Purpose**: Order execution with retry logic, status tracking, and analytics. Broker-agnostic -- takes a `BrokerInterface` / specific interface implementation.

**Key Features**:
- Configurable retry attempts and delays
- Fill timeout monitoring
- Batch order execution
- Execution metrics tracking

**Usage**:
```python
from src.trading.core.execution_engine import ExecutionEngine
from src.trading.brokers.interfaces import OrderSide

engine = ExecutionEngine(
    broker=broker,           # AlpacaBroker, IBKRBroker, etc.
    max_retries=3,
    retry_delay=1.0,
    fill_timeout=30.0,
)

result = engine.execute_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    wait_for_fill=True,
)

metrics = engine.get_execution_metrics()
# {'total_orders': 5, 'successful_orders': 4, 'success_rate': 0.8, ...}
```

### StrategyAdapter (`adapters/strategy_adapter.py`)

**Purpose**: Base adapter connecting pure strategies to live trading infrastructure.

**Responsibilities**:
- Fetch market data from broker / data provider
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
from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

adapter = RAMPLiveAdapter(
    broker=ibkr_broker,
    initial_capital=100_000,
    # ...regime / sizing parameters loaded from RAMP config...
)

adapter.preload_historical_data()  # At market open
adapter.run_once()                  # At 3:55 PM rebalance
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
if manager.is_enabled('ramp'):
    # Acquire execution lock
    if manager.acquire_execution_lock('ramp'):
        try:
            manager.add_position('ramp', 'AAPL', 50, 188.20)
            # ... execute orders ...
        finally:
            manager.release_execution_lock('ramp')

# For top-ups (accumulates qty, doesn't overwrite)
manager.add_or_update_position('ramp', 'AAPL', 25, 189.40)

# Sync with broker positions
broker_positions = {'AAPL': 75, 'MSFT': 50}
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
  StrategyAdapter (e.g. RAMPLiveAdapter)
        v
  preload_historical_data() -> IBKRDataProvider.get_bars()   (or AlpacaBroker.get_bars())
        v
  prefetch_intraday_data()   -> broker.get_historical_bars()
        v
  generate_signals() -> Pure Strategy (RAMPSignals / OvernightMeanReversionStrategy / ...)
        v
  filter_signals()   -> PositionManager (risk checks)
        v
  execute_signals()  -> ExecutionEngine
        v
  ExecutionEngine.execute_order() -> IBKRBroker.place_stock_order()  (or AlpacaBroker.place_stock_order(), CoinbaseBroker.place_order())
        v
  StrategyStateManager.add_position()
        v
  State Persisted (JSON, atomic)
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

# IBKR
from src.trading.brokers.ibkr import IBKRBroker, IBKRConfig
ibkr_broker = IBKRBroker(IBKRConfig(port=4002, client_id=10))
ibkr_broker.start()
# ... use it ...
ibkr_broker.stop()
```

### Order Types and Enums

```python
from src.trading.brokers.interfaces import (
    OrderSide,      # BUY, SELL
    OrderType,      # MARKET, LIMIT, STOP, STOP_LIMIT
    OrderStatus,    # NEW, FILLED, CANCELLED, REJECTED, ...
    TimeInForce,    # DAY, GTC, IOC, FOK
)
```

### State Management

```python
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()
manager.is_enabled('ramp')        # Check if enabled
manager.get_positions('ramp')     # Get positions
manager.print_status()            # Print all strategy status
```

---

## Configuration

### Broker Routing (`config/trading/broker_routing.yaml`)

The authoritative file for strategy -> broker assignment. See "Broker Routing" above.

### Standalone Broker Configs

For tooling or scripts that need a single broker (no routing):

```yaml
# config/trading/broker_alpaca.yaml
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
  ramp:
    enabled: true
    shutdown_requested: false
  omr:
    enabled: false
    shutdown_requested: false
  mp:
    enabled: false   # Legacy, superseded by RAMP
    shutdown_requested: false
  cscm:
    enabled: false   # Owned by homeguard-cscm service
    shutdown_requested: false
last_modified: '2026-05-15T17:41:14-04:00'
modified_by: claude-code
```

### Environment Variables

- `ALPACA_PAPER_KEY_ID`, `ALPACA_PAPER_SECRET_KEY` - Alpaca paper credentials
- `ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET_KEY` - Alpaca live credentials (if used)
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` - IBKR overrides (defaults from `IBKRConfig` / YAML)
- `COINBASE_API_KEY`, `COINBASE_API_SECRET` - Coinbase credentials

---

## Dependencies

### Internal (src/ modules)
- `src.strategies.core` - `StrategySignals`, `Signal` classes
- `src.data.providers` - `DataProviderInterface` for fallback data
- `src.streaming.interface` - `StreamingProviderInterface` (IBKR streaming implements this)
- `src.utils.logger` - Centralized logging
- `src.utils.timezone` - Timezone utilities (`tz.now()`)

### External (pip packages)
- `ib_async` - IBKR client (replaces deprecated `ibapi`)
- `alpaca-py` - Alpaca trading and data API
- `coinbase-advanced-py` - Coinbase Advanced Trade
- `pandas` - DataFrames for market data
- `pytz` - Timezone handling
- `pyyaml` - Configuration loading
- `pydantic` - `IBKRConfig` validation

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `BrokerConnectionError` | Network / API / gateway issues | Retry with backoff; for IBKR also check gateway port 4002 is bound |
| `InvalidOrderError` | Bad order params | Fix parameters |
| `InsufficientFundsError` | Not enough buying power | Reduce position size |
| `OrderNotFoundError` | Order ID doesn't exist or expired | Re-query, treat as cancelled |
| `NoPositionError` | Position not found in account | Re-sync state |
| `SymbolNotFoundError` | Broker can't resolve symbol (e.g. delisted, or IBKR symbol-format mismatch) | Check `src/trading/brokers/ibkr/symbols.py` for class-share normalization |
| State file corrupted | JSON parse error | Auto-recover from backup |
| Execution lock timeout | Other strategy running | Wait or check logs |
| `PacingViolationError` (internal) | IBKR historical-data rate limit exceeded | Handled internally by `PacingManager`; never raised to strategy code |

---

## Deployment

### Systemd Services (EC2)

The live trading system runs as systemd services. The active production unit is `homeguard-multi.service`, which is pinned to RAMP today.

| Service | Strategy / Process | Status |
|---------|--------------------|--------|
| `homeguard-multi.service` | `run_live_paper_trading.py --strategy ramp` (currently pinned to RAMP) | Active |
| `homeguard-cscm.service`  | CSCM weekly runner (Coinbase) | Active |
| `homeguard-gateway.service` | IB Gateway container / process (Java + IBC login) | Active |
| `homeguard-omr.service` (legacy)  | OMR-only unit; disabled, superseded by `homeguard-multi` | Disabled |
| `homeguard-ramp.service` (legacy) | RAMP-only unit; disabled, superseded by `homeguard-multi` | Disabled |
| `homeguard-mp.service` (legacy)   | MP-only unit; disabled, superseded by RAMP | Disabled |

`homeguard-multi.service` runs:
```
ExecStartPre=  wait up to 120s for IBKR gateway to bind port 4002
ExecStart=     python scripts/trading/run_live_paper_trading.py \
                  --strategy ramp --initial-capital 100000
```
The runner then reads `config/trading/broker_routing.yaml` and creates an `IBKRBroker` for RAMP.

> Note: the runner's `--strategy multi` mode exists but does NOT currently run multiple strategies concurrently; it picks the first enabled strategy in priority order (OMR > MP > RAMP). Until true multi-strategy support lands, use an explicit `--strategy <name>` so the unit is unambiguous about what it runs.

**Commands**:
```bash
# Start RAMP via the multi unit
sudo systemctl start homeguard-multi.service

# Check status
sudo systemctl status homeguard-multi homeguard-cscm homeguard-gateway

# View logs
journalctl -u homeguard-multi -f

# View RAMP-specific log lines
journalctl -u homeguard-multi --since "1 hour ago" | grep -E "Regime:|Position|RAMP"
```

### Process Identification

- `homeguard-multi` -- main equity trading process (RAMP today)
- `homeguard-cscm`  -- crypto rebalancer
- `homeguard-gateway` -- IB Gateway login process

---

## IBKR Smoke Test

**`scripts/trading/smoke_test_ibkr_paper.py`** -- end-to-end validation of the live trading call chain against IBKR paper. Run it after any change to `IBKRBroker`, `AlpacaBroker`, `ExecutionEngine`, the broker interfaces, `broker_routing.yaml`, or any adapter's order-submission path. ~25s, idempotent, safe after-hours.

```bash
# On EC2:
ssh ec2 'cd ~/Homeguard && source venv/bin/activate && python scripts/trading/smoke_test_ibkr_paper.py'

# Modes: --mode direct (broker only) | engine (ExecutionEngine only) | full (default, both)
```

It uses `client_id=99` so it never collides with the running `homeguard-multi` service (which holds `client_id=10`).

---

## Testing

### Test Location
- `tests/trading/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/trading/brokers/test_broker_contract.py` - Parametrized contract test across `AlpacaBroker` and `IBKRBroker` (no IBKR connection needed)

### Running Tests
```bash
pytest tests/trading/ -v
pytest tests/trading/test_execution_engine.py -v
pytest tests/trading/brokers/test_broker_contract.py -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Module Reference](../../docs/architecture/MODULE_REFERENCE.md) -- see "IBKR Broker Integration" subsection
- [Multi-Strategy Position Management](../../docs/architecture/MULTI_STRATEGY_POSITION_MANAGEMENT.md)
- [IBKR Integration Design (2026-04-07)](../../docs/architecture/2026-04-07-ibkr-integration-design.md)
- [IBKR Integration Plan (2026-04-07)](../../docs/architecture/2026-04-07-ibkr-integration-plan.md)
- [Infrastructure Overview](../../docs/INFRASTRUCTURE_OVERVIEW.md)

---

## IBKR Broker Integration

The IBKR client lives in `src/trading/brokers/ibkr/` and is the primary
backend for stocks and options. It is built on `ib_async` (not `ibapi`).

### File-by-file

| File | Purpose |
|------|---------|
| `__init__.py`            | Public API: `IBKRBroker`, `IBKRFuturesBroker`, `IBKRConfig`, `IBKRConnectionManager`, `IBKRDataProvider`, `IBKRStreamingProvider` |
| `config.py`              | `IBKRConfig` (pydantic) -- host, port (4001 live / 4002 paper / 7496-97 TWS), client_id, readonly, account, reconnect policy; loads from defaults / `config/ibkr.yaml` / env (`IBKR_HOST`, `IBKR_PORT`, ...) |
| `connection.py`          | `IBKRConnectionManager` -- singleton that owns the `ib_async.IB` instance, runs the asyncio event loop on a background daemon thread, and bridges sync code via `asyncio.run_coroutine_threadsafe()`. Handles auto-reconnect with exponential backoff and health monitoring. |
| `contracts.py`           | `ContractResolver` -- resolves stock/option/future symbols to fully-qualified `ib_async.Contract` objects with caching. Translates OCC option symbology (used by Alpaca) to IBKR's `(symbol, lastTradeDateOrContractMonth, strike, right)` form. |
| `data_download.py`       | `IBKRDataProvider` -- implements `DataProviderInterface`. Slots into `CompositeDataProvider` as a historical data source alongside Alpaca and yfinance. Returns DataFrames with ET-tz `DatetimeIndex` and lowercase `open/high/low/close/volume`. |
| `errors.py`              | Maps IBKR error codes to the broker-agnostic exceptions in `interfaces/base.py` (`BrokerConnectionError`, `InvalidOrderError`, `InsufficientFundsError`, `OrderNotFoundError`, `SymbolNotFoundError`). Defines internal-only `PacingViolationError`. |
| `ibkr_broker.py`         | `IBKRBroker` -- implements `AccountInterface`, `MarketHoursInterface`, `MarketDataInterface`, `StockTradingInterface`, and `OptionsTradingInterface` (~22 abstract methods total). First Homeguard broker to support options. |
| `ibkr_futures_broker.py` | `IBKRFuturesBroker` -- implements `FuturesTradingInterface`. Wraps `_build_future_contract` (symbol_root + contract_month -> `ib_async.Future`) and runs the ExpirationGuard / MarginGuard / AuditLog safeguard chain before submission. |
| `pacing.py`              | `PacingManager` -- token-bucket rate limiter tuned to IBKR's historical-data limits (~58 req / 10 min, 6 req / 2 s, 15 s gap for identical requests). Makes pacing invisible to callers: `acquire()` blocks until safe. |
| `streaming.py`           | `IBKRStreamingProvider` -- implements `src.streaming.interface.StreamingProviderInterface` using `reqMktData()` for quotes/trades and `reqRealTimeBars()` for 5s bars (aggregated to 1m). Converts to Homeguard `Bar`/`Quote`/`Trade` dataclasses at the boundary. |
| `symbols.py`             | `to_ibkr_symbol` / `from_ibkr_symbol` -- normalize multi-class US tickers (`BRK.B` <-> `BRK B`) at the IBKR boundary only. Single-class tickers and unknown patterns pass through unchanged. |

### Connection topology

- IB Gateway runs on the EC2 host on TCP port 4002 (paper) or 4001 (live).
- `homeguard-gateway.service` brings the gateway up. `homeguard-multi.service` has an `ExecStartPre` that waits up to 120s for port 4002 to bind before launching the runner, to avoid a race where `BrokerFactory` falls back to Alpaca on `ConnectionRefusedError`.
- The running `homeguard-multi` process holds `client_id=10`. Smoke tests and ad-hoc scripts must use a different `client_id` (the smoke test defaults to 99).

### Routing

Per `config/trading/broker_routing.yaml`:
- `omr`, `ramp`, `mp` -> `ibkr`
- `cscm` -> `coinbase`
- everything else falls back to `default_broker: alpaca`

---

## Changelog

- **2026-05-17**: Updated for IBKR migration -- IBKR is now primary for stocks/options, added IBKR file-by-file breakdown, replaced legacy per-strategy systemd table with `homeguard-multi`, added broker-routing section.
- **2026-04-07**: IBKR integration design + plan landed (`docs/architecture/2026-04-07-ibkr-integration-*.md`).
- **2025-12-08**: Added RAMP service documentation, deprecated MP
- **2025-12-08**: Initial documentation created
- **2025-12-06**: Multi-strategy systemd services added
- **2025-11-XX**: `StrategyStateManager` for multi-strategy support
- **2025-10-XX**: Initial broker-agnostic architecture
