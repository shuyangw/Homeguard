# IBKR Integration Design Spec

**Date**: 2026-04-07
**Status**: Approved
**Approach**: Layered Integration (V2 reference code + validation gate per phase)

---

## 1. Goal

Add Interactive Brokers as a broker backend alongside Alpaca. IBKR serves as a
trade execution engine (stocks + options) and optionally as a data source. The
architecture supports mix-and-match: e.g., IBKR for execution, Alpaca for data.

## 2. Scope

- StreamingProviderInterface ABC extraction from LiveDataProvider
- IBKR module: config, connection, contracts, pacing, errors
- IBKRDataProvider (DataProviderInterface) + IBKRStreamingProvider (StreamingProviderInterface)
- IBKRBroker (stocks + options -- first OptionsTradingInterface implementor)
- Config-driven broker routing (strategy -> broker assignment)
- EC2 Gateway deployment (IB Gateway + IBC + Xvfb on t4g.small ARM64)

## 3. What NOT To Do

- No changes to existing strategy signal logic
- No hot-swapping or runtime failover between brokers
- No new exception types leaking out of the IBKR module
- No extending DataProviderInterface for options (ISP -- use concrete type)
- No 2FA automation for live accounts (defer to production cutover)

---

## 4. Phase 1: StreamingProviderInterface Extraction

### New file: `src/streaming/interface.py`

ABC extracted from LiveDataProvider's actual public API:

**Lifecycle:**
- `start(symbols: Optional[List[str]] = None) -> None`
- `stop() -> None`
- `is_connected() -> bool`

**On-demand data:**
- `get_price(symbol: str) -> Optional[float]`
- `get_quote(symbol: str) -> Optional[Quote]`
- `get_trade(symbol: str) -> Optional[Trade]`
- `get_bar(symbol: str) -> Optional[Bar]`
- `get_bars(symbol: str, n: Optional[int] = None) -> pd.DataFrame`
- `get_vwap(symbol: str) -> Optional[float]`
- `get_spread(symbol: str) -> Optional[float]`

**Callbacks:**
- `on_bar(symbols: List[str], handler: Callable[[Bar], None]) -> str`
- `on_quote(symbols: List[str], handler: Callable[[Quote], None]) -> str`
- `on_trade(symbols: List[str], handler: Callable[[Trade], None]) -> str`
- `unsubscribe(subscription_id: str) -> None`

**Utility:**
- `get_subscribed_symbols() -> set`

**Identity:**
- `name: str` (property)

### Changes to existing code

1. `src/streaming/live_data_provider.py` -- add `(StreamingProviderInterface)` to
   class declaration. Convert `self.name` attribute to `@property` returning
   `self._name`. Zero method signature changes.

2. `src/streaming/__init__.py` -- add `StreamingProviderInterface` to exports.

3. Strategy adapters -- replace `hasattr(self._data_provider, 'get_bars')`
   duck-typing with `isinstance(self._data_provider, StreamingProviderInterface)`.
   Affected files:
   - `src/trading/adapters/omr_live_adapter.py` (line 357)
   - `src/trading/adapters/ramp_live_adapter.py` (line 734)

**Risk**: Zero. LiveDataProvider already implements every method.

### V2 discrepancy fixed

V2 interface omitted `on_trade()`. The actual LiveDataProvider has it (line 273).
Must include it in the ABC.

---

## 5. Phase 2: IBKR Module Core

### New files in `src/trading/brokers/ibkr/`

**`__init__.py`** (~40 lines)
Re-exports: IBKRBroker, IBKRDataProvider, IBKRStreamingProvider,
IBKRConnectionManager, IBKRConfig.

**`config.py`** (~110 lines)
Pydantic model with 3-tier loading:
1. Defaults in class
2. `config/ibkr.yaml` (if present)
3. Environment variables: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, etc.

Ports: 4001=Gateway live, 4002=Gateway paper, 7496=TWS live, 7497=TWS paper.

**`connection.py`** (~300 lines)
IBKRConnectionManager -- singleton owning the `ib_async.IB` instance.

Key challenge: ib_async is asyncio-native, Homeguard strategies are synchronous.
Solution: dedicated daemon thread running asyncio event loop, bridged via
`asyncio.run_coroutine_threadsafe()`.

Features:
- Singleton pattern (one connection shared across all IBKR components)
- Exponential backoff reconnection (max 50 attempts)
- Thread-safe `run(coro) -> result` bridge method
- Connection health monitoring
- Graceful shutdown

**`contracts.py`** (~290 lines)
ContractResolver -- symbol strings to qualified IBKR Contract objects.

Features:
- conId cache to avoid repeated `qualifyContracts()` round-trips
- OCC symbology bridge: `AAPL  260417C00190000` -> IBKR Option contract
- Stock, Option, Index, Future contract types
- Handles contract ambiguity (multiple exchanges)

**`pacing.py`** (~190 lines)
PacingManager -- token-bucket rate limiter for IBKR historical data.

IBKR pacing rules:
- Max 60 requests per 10-minute window (use 58 for headroom)
- Max 6 requests per 2 seconds
- Identical requests must be 15 seconds apart

Callers call `acquire()` which blocks until safe to proceed. Uses threading
primitives (not asyncio) since the connection bridge is already threaded.

**`errors.py`** (~75 lines)
Pure mapper -- no custom exception hierarchy. Maps IBKR error codes to existing
Homeguard exceptions from `src/trading/brokers/interfaces/base.py`:

| IBKR Codes | Homeguard Exception |
|---|---|
| 504, 1100-1102 | BrokerConnectionError |
| 200, 321 | SymbolNotFoundError |
| 201, 203 | InvalidOrderError |
| 135 | OrderNotFoundError |

One internal exception: `PacingViolationError` -- used only within pacing.py,
never leaks outside the ibkr package.

### New file: `config/ibkr.yaml`

Connection settings, pacing tuning, timeframe mappings.

### V2 discrepancy fixed

`pacing.py` uses `import logging` instead of `from src.utils.logger import get_logger`.
Must fix to follow Homeguard convention.

---

## 6. Phase 3: IBKRDataProvider + IBKRStreamingProvider

### IBKRDataProvider (`data_download.py`, ~275 lines)

Implements `DataProviderInterface`. Slots into `CompositeDataProvider` as another
data source in the fallback chain.

**Interface methods:**
- `get_historical_bars(symbol, start, end, timeframe='1D', force_refresh=False) -> Optional[pd.DataFrame]`
- `get_historical_bars_batch(symbols, start, end, timeframe='1D', force_refresh=False) -> Dict[str, pd.DataFrame]`
- `is_available() -> bool`
- `supports_timeframe(timeframe: str) -> bool`

**DataFrame contract:** DatetimeIndex in America/New_York, lowercase columns
(open, high, low, close, volume). Returns None on failure (enables fallback).

**IBKR-specific complexity handled transparently:**
- Duration/barSize mapping: Homeguard `"1D"` -> IBKR `"1 day"`, `"1Min"` -> `"1 min"`
- Pacing: all requests go through PacingManager.acquire()
- Large date ranges: IBKR caps request size; provider chunks automatically

**Additional methods beyond the interface (options data):**
- `get_options_chain(underlying, expiration) -> List[Dict]`
- `get_historical_option_bars(...) -> Optional[pd.DataFrame]`

Not on DataProviderInterface (ISP). Strategy code needing options data accepts
the concrete IBKRDataProvider type.

**Factory wiring:** Add `ibkr` case to `src/data/providers/factory.py` (~5 lines).
`create_data_provider()` gets optional `ibkr_connection` parameter.

### IBKRStreamingProvider (`streaming.py`, ~250 lines)

Implements `StreamingProviderInterface` (from Phase 1).

**IBKR streaming mechanics:**
- Uses `ib_async.reqMktData()` for quotes/trades
- Uses `ib_async.reqRealTimeBars()` for 5-second bars (aggregated to 1-min in buffer)
- All data converted to Homeguard `Bar`/`Quote`/`Trade` dataclasses at the boundary

**Return types must match exactly:**
- `get_quote()` -> `Quote(symbol, timestamp, bid_price, bid_size, ask_price, ask_size, ...)`
- `get_bar()` -> `Bar(symbol, timestamp, open, high, low, close, volume, ...)`
- `get_trade()` -> `Trade(symbol, timestamp, price, size, ...)`

**Lifecycle:** `start(symbols)` / `stop()` -- matches LiveDataProvider.

### V2 discrepancy fixed

V2 interface omits `on_trade()` callback. Must implement it. For IBKR this maps
to ib_async's tick-by-tick `Last` data.

---

## 7. Phase 4: IBKRBroker (Stock + Options Execution)

### `ibkr_broker.py` (~710 lines)

Implements 6 interfaces, 22+ abstract methods:

```
IBKRBroker(
    AccountInterface,           # get_account(), test_connection()
    MarketHoursInterface,       # is_market_open(), get_market_hours()
    MarketDataInterface,        # get_latest_quote(), get_latest_trade(), get_bars()
    StockTradingInterface,      # get_stock_positions(), place_stock_order(), ...
    OptionsTradingInterface,    # get_options_chain(), place_options_order(), ...
)
```

Note: OrderManagementInterface (cancel_order, get_order, get_orders) is inherited
by both StockTradingInterface and OptionsTradingInterface. Implemented once, shared.

### Translation layer

Follows AlpacaBroker patterns exactly:

**Order translation:** `_translate_order(trade: ib_async.Trade) -> Dict`
- order_id, symbol, quantity, side, order_type, status, limit_price, stop_price,
  created_at, filled_qty, filled_avg_price

**Position translation:** `_translate_position(pos) -> Dict`
- symbol, quantity, avg_entry_price, current_price, market_value, unrealized_pnl,
  unrealized_pnl_pct, side

**Enum mapping:**
- `OrderSide.BUY` ("buy") -> IBKR "BUY"
- `TimeInForce.DAY` ("day") -> IBKR "DAY"

**Status mapping:**
- IBKR "PendingSubmit"/"PreSubmitted"/"Submitted" -> "pending"
- IBKR "Filled" -> "filled"
- IBKR "Cancelled"/"ApiCancelled" -> "cancelled"
- IBKR "Inactive" -> "rejected"

### Options trading (first OptionsTradingInterface implementor)

- `get_options_chain(underlying, expiration)` -- reqSecDefOptParams() + reqMktData()
- `place_options_order(underlying, expiration, strike, option_type, quantity, side, ...)`
  -- resolves to IBKR Option contract via ContractResolver
- `place_multi_leg_order(legs: List[OptionLeg], ...)` -- IBKR combo/bag orders
- `get_greeks(contract_id)` -- delta, gamma, theta, vega, rho, IV
- `get_options_positions()` / `get_options_position(contract_id)`
- `close_options_position(contract_id, quantity)` / `close_all_options_positions()`

### get_bars MultiIndex

MarketDataInterface.get_bars() returns MultiIndex (symbol, timestamp). Fetches
per-symbol, concatenates, sets multi-index -- matching AlpacaBroker pattern.

### Shared connection

IBKRBroker, IBKRDataProvider, and IBKRStreamingProvider all share the same
IBKRConnectionManager singleton.

### V2 discrepancy fixed

`close_stock_position` signature: interface has
`(self, symbol: str, quantity: Optional[int] = None)` but V2 code omits Optional.
Must fix.

---

## 8. Phase 5: Broker Routing + Factory Wiring

### New file: `config/trading/broker_routing.yaml`

```yaml
brokers:
  alpaca:
    paper: true

  ibkr:
    port: 4002
    client_id: 10
    readonly: false

strategies:
  omr:
    broker: alpaca
  ramp:
    broker: alpaca
  cscm:
    broker: coinbase

default_broker: alpaca
```

### Changes to `src/trading/brokers/broker_factory.py`

Replace NotImplementedError placeholder (lines 69-75) with actual IBKR creation:

```python
elif broker_type in ['ib', 'interactive_brokers', 'interactivebrokers']:
    from .ibkr import IBKRBroker, IBKRConfig
    return IBKRBroker(IBKRConfig(**config))
```

Add `create_from_env()` support for IBKR (reads IBKR_HOST, IBKR_PORT,
IBKR_CLIENT_ID from env).

### New file: `src/trading/config/broker_routing.py` (~50 lines)

Loader that:
1. Reads broker_routing.yaml
2. Creates broker instances via BrokerFactory (shared -- two strategies assigned
   to `alpaca` get the same instance)
3. Returns `Dict[str, BrokerInterface]` mapping strategy name -> broker
4. Bot entry points call this instead of hardcoding AlpacaBroker

### Design decisions

- Config-driven, not runtime-switchable. Broker assignment is a deployment decision.
- Paper vs live is config. Alpaca: `paper: true/false`. IBKR: `port: 4002` vs `4001`.
- Broker instances shared across strategies using the same broker.
- No hot-swapping or automatic failover between brokers.
- Execution and data independently configurable.

---

## 9. Phase 6: EC2 Gateway Deployment

### Architecture

```
EC2 t4g.small (ARM64, Amazon Linux 2023, 2GB RAM)
+-- Xvfb :1 (virtual display)
|   +-- IB Gateway (renders login UI into virtual framebuffer)
|       +-- Listening on localhost:4002 (paper) or :4001 (live)
+-- IBC (automated login controller)
+-- Homeguard strategies (connect to Gateway via ib_async)
```

### Component stack

| Component | Purpose |
|---|---|
| Bellsoft Liberica JDK 17 Full (aarch64) | Java runtime (Full variant for JavaFX) |
| IB Gateway (stable) | IBKR API server |
| IBC (IB Controller) | Automated login, dialog dismissal |
| Xvfb | Virtual X11 framebuffer |

### Memory budget

```
Component              RAM (est.)
---------------------------------
Current usage          ~750 MB
IB Gateway             ~300-500 MB
Xvfb                   ~10 MB
---------------------------------
Total                  ~1,060-1,260 MB / 2,048 MB
```

Monitor with htop. Upgrade to t4g.medium (~$14/mo) if memory pressure appears.

### Systemd integration

New `homeguard-gateway.service` with `Before=homeguard-trading.target`.
Follows existing service patterns.

Boot sequence:
```
Lambda starts EC2 at 9:00 AM ET
  -> homeguard-gateway.service starts first
    -> Xvfb :1 + IBC launches Gateway + authenticates
  -> homeguard-omr.service (Alpaca)
  -> homeguard-ramp.service (Alpaca)
  -> future IBKR strategies connect to localhost:4002
Lambda stops EC2 at 4:30 PM ET
```

### Credentials

Added to `.env` (already gitignored):
- IBKR_USERNAME
- IBKR_PASSWORD
- IBKR_TRADING_MODE ("paper" or "live")
- IBKR_GATEWAY_PORT (4002 for paper, 4001 for live)

Matching placeholders in `.env.example`.

### 2FA

Paper accounts: not required (IBC handles username + password).
Live accounts: apply for IBKR "Automated Trading Session" (API login without
2FA for pre-approved IPs). Defer to production cutover.

### New files

- `infra/ec2/services/homeguard-gateway.service`
- `infra/ec2/install_ibkr_gateway.sh` (idempotent installer)
- `config/ibkr/ibc-config.ini.template` (templated from .env values)

### Open questions (resolve during implementation)

1. x64 Gateway installer on ARM64 -- community-validated but may need manual
   JAR extraction if official installer fails
2. Xvfb lifecycle -- ExecStartPre vs dedicated xvfb.service
3. IBC version compatibility with installed Gateway version
4. Actual Gateway RAM under load on t4g.small

---

## 10. V2 Reference Code Discrepancies

Issues found during validation against current codebase. All must be fixed
during implementation.

| File | Issue | Fix |
|---|---|---|
| `streaming/interface.py` | Missing `on_trade()` method | Add to ABC |
| `pacing.py` | Uses `import logging` | Change to `from src.utils.logger import get_logger` |
| `ibkr_broker.py` | `close_stock_position(symbol, quantity)` missing Optional | Change to `quantity: Optional[int] = None` |
| `streaming/interface.py` | `name` as @property but LiveDataProvider uses attribute | Convert LiveDataProvider.name to @property |

---

## 11. Complete File Inventory

### New files (IBKR module)

| File | Est. Lines | Purpose |
|---|---|---|
| `src/trading/brokers/ibkr/__init__.py` | 40 | Package exports |
| `src/trading/brokers/ibkr/config.py` | 110 | Pydantic config |
| `src/trading/brokers/ibkr/connection.py` | 300 | Event loop thread, reconnection |
| `src/trading/brokers/ibkr/contracts.py` | 290 | Contract resolution, OCC bridge |
| `src/trading/brokers/ibkr/pacing.py` | 190 | Rate limiter |
| `src/trading/brokers/ibkr/errors.py` | 75 | Error code mapping |
| `src/trading/brokers/ibkr/data_download.py` | 275 | DataProviderInterface impl |
| `src/trading/brokers/ibkr/streaming.py` | 250 | StreamingProviderInterface impl |
| `src/trading/brokers/ibkr/ibkr_broker.py` | 710 | BrokerInterface + OptionsTradingInterface |
| `config/ibkr.yaml` | 40 | Connection settings |
| `config/ibkr/ibc-config.ini.template` | 25 | IBC config template |

### New files (interface + routing)

| File | Est. Lines | Purpose |
|---|---|---|
| `src/streaming/interface.py` | 120 | StreamingProviderInterface ABC |
| `src/trading/config/broker_routing.py` | 50 | Strategy -> broker routing loader |
| `config/trading/broker_routing.yaml` | 25 | Routing config |

### New files (EC2 deployment)

| File | Est. Lines | Purpose |
|---|---|---|
| `infra/ec2/services/homeguard-gateway.service` | 30 | Systemd service |
| `infra/ec2/install_ibkr_gateway.sh` | 50 | Installer script |

### New files (tests)

| File | Est. Lines | Purpose |
|---|---|---|
| `tests/trading/brokers/ibkr/test_config_and_errors.py` | 106 | Config + error mapping tests |
| `tests/trading/brokers/ibkr/test_contracts.py` | 139 | Contract resolution tests |
| `tests/trading/brokers/ibkr/test_pacing.py` | 146 | Pacing manager tests |
| `tests/trading/brokers/ibkr/__init__.py` | 0 | Package marker |

### Modified files (existing code)

| File | Change | Risk |
|---|---|---|
| `src/streaming/live_data_provider.py` | Add (StreamingProviderInterface), `self.name` attr -> `self._name` + `@property name` | None |
| `src/streaming/__init__.py` | Add StreamingProviderInterface to exports | None |
| `src/trading/adapters/omr_live_adapter.py` | hasattr -> isinstance check | None |
| `src/trading/adapters/ramp_live_adapter.py` | hasattr -> isinstance check | None |
| `src/trading/brokers/broker_factory.py` | Replace NotImplementedError with IBKR creation | None |
| `src/data/providers/factory.py` | Add ibkr case (~5 lines) | None |
| `.env.example` | Add IBKR placeholder entries | None |

Total: ~3 files modified with meaningful changes, ~8 lines changed in existing code.

---

## 12. Dependency

One new Python package: `ib_async` (pip install ib_async). No ibapi needed.
ib_async is a modern asyncio wrapper around the IBKR API.

Pydantic is already in the project dependencies.

---

## 13. Validation Gates

Each phase has a validation step before proceeding:

| Phase | Validation |
|---|---|
| 1. StreamingProviderInterface | LiveDataProvider still passes all existing tests. isinstance checks work. |
| 2. IBKR Core | Unit tests for config, pacing, contracts, error mapping pass. Connection can reach paper Gateway. |
| 3. Data + Streaming | IBKRDataProvider.get_historical_bars('SPY', ...) returns ET-timezone DataFrame matching schema. IBKRStreamingProvider returns correct dataclass types. |
| 4. IBKRBroker | All 22 interface methods implemented. Paper order placed and confirmed. Options chain fetched. |
| 5. Broker Routing | Strategy wired to IBKR via config. Existing Alpaca strategies unaffected. |
| 6. EC2 Gateway | Gateway running on EC2, strategies connect via localhost:4002, surviving instance restart cycle. |
