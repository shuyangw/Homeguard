# Multi-Strategy Position Management

This document describes the architecture for running multiple trading strategies simultaneously while maintaining position isolation and state consistency.

## Overview

The system supports running **N strategies** concurrently with:
- Independent position tracking per strategy
- Toggle mechanism to enable/disable strategies
- Atomic state persistence with file locking
- Conflict prevention between strategies
- Graceful shutdown coordination
- Execution lock serialization

### Active Strategies

| ID | Name | Description | Broker | Service | Status |
|----|------|-------------|--------|---------|--------|
| `ramp` | Regime-Aware Momentum Protection | S&P 500 daily rebalance, 5 regimes | IBKR paper | `homeguard-multi` | Active (production) |
| `omr` | Overnight Mean Reversion | Leveraged ETF overnight holds | IBKR paper | (disabled in toggle) | Deployed, disabled |
| `cscm` | Cross-Sectional Crypto Momentum | Weekly crypto rebalance | Coinbase | `homeguard-cscm` | Active |
| `mp` | Momentum Protection | Static 1m-1w momentum on S&P 500 | IBKR paper (would-be) | (not deployed) | Legacy (superseded by RAMP) |

Notes:
- `homeguard-multi.service` runs `scripts/trading/run_live_paper_trading.py --strategy ramp`. The `--strategy multi` mode in the runner is not yet a true concurrent multi-strategy runtime -- only the highest-priority enabled strategy runs.
- OMR and MP code paths still exist (adapters, configs, registry entries) but are not currently dispatched in production. To bring OMR back you flip `omr.enabled: true` in `config/trading/strategy_toggle.yaml` AND change the service's `--strategy` argument or wait for true multi-mode.
- CSCM runs in its own service (`homeguard-cscm`) against Coinbase; it does not share state files with the equity strategies.

## Strategy Configuration

### Capital Allocation

| Strategy | Position Size | Max Positions | Max Exposure | Execution Time |
|----------|---------------|---------------|--------------|----------------|
| RAMP     | 1/N dynamic (typically 5-20% per name based on regime) | 5-20 (regime-dependent) | up to 100% | 3:55 PM (rebalance) |
| OMR      | ~15% per signal | 3-5             | ~45-75%       | 9:31 AM (exit), 3:50 PM (entry) |
| MP       | ~6.5% (legacy)  | 10              | ~65%          | (not deployed)  |
| CSCM     | per crypto config | crypto-universe top-K | ~100% | Weekly (Sun 0:00 UTC) |

**Note:** If multiple equity strategies are enabled simultaneously the combined exposure can exceed 100%. Orders are serialized by the execution lock and skipped if buying power is insufficient.

### Universe Isolation

Strategies MUST trade non-overlapping universes to prevent conflicts:

| Strategy | Universe | Example Symbols | Universe Source |
|----------|----------|-----------------|-----------------|
| RAMP     | S&P 500 (filtered) | AAPL, MSFT, NVDA, etc. | `config/universes/sp500-2025.csv` |
| OMR      | Leveraged ETFs | TQQQ, SOXL, UPRO, SPXL, TECL, FNGU | `ETFUniverse.LEVERAGED_3X` |
| MP       | S&P 500 (legacy) | overlaps with RAMP -- not deployable alongside RAMP | `config/universes/sp500-2025.csv` |
| CSCM     | Crypto top-K by mcap | BTC-USD, ETH-USD, ... | `config/strategies/cscm/universe.yaml` (separate service / Coinbase, no overlap) |

Universe isolation is validated on startup for the equity strategies that share `strategy_positions.json`. If overlap is detected the system logs an error and refuses to start. RAMP and MP overlap by design, which is why MP must not be enabled while RAMP is active.

### Adding a New Strategy

To add a new strategy to the system:

1. **Create Strategy Adapter** in `src/trading/adapters/`:
   ```python
   class NewStrategyAdapter(StrategyAdapter):
       STRATEGY_NAME = 'new_strat'  # Unique identifier

       def get_schedule(self) -> Dict:
           return {
               'execution_times': [
                   {'time': 'HH:MM', 'action': 'entry/exit/rebalance'}
               ],
               'market_hours_only': True
           }
   ```

2. **Define Universe** - Must not overlap with existing enabled strategies

3. **Register in Toggle Config** (`config/trading/strategy_toggle.yaml`):
   ```yaml
   strategies:
     new_strat:
       enabled: false
       shutdown_requested: false
   ```

4. **Route to a Broker** (`config/trading/broker_routing.yaml`):
   ```yaml
   strategies:
     new_strat:
       broker: ibkr   # or alpaca / coinbase
   ```

5. **Add / Reuse a Systemd Service** (for EC2 deployment) -- either extend `homeguard-multi.service` (once true multi-strategy is supported) or add a dedicated unit file.

6. **Update Documentation**:
   - Add to this file's strategy tables
   - Add a dedicated architecture doc under `docs/architecture/`

## State Files

### Toggle Configuration

**File:** `config/trading/strategy_toggle.yaml`

```yaml
strategies:
  ramp:
    enabled: true
    shutdown_requested: false
  omr:
    enabled: false
    shutdown_requested: false
  mp:
    enabled: false        # legacy, superseded by RAMP
    shutdown_requested: false
  cscm:
    enabled: false        # owned by homeguard-cscm.service
    shutdown_requested: false

last_modified: "2026-05-15T17:41:14-04:00"
modified_by: "claude-code"
```

- Read on each trading cycle
- Modified by `infra/ec2/toggle_strategy.sh`
- `shutdown_requested` enables graceful shutdown coordination

### Position State

**File:** `data/trading/strategy_positions.json`

```json
{
  "version": 1,
  "last_updated": "2026-05-15T15:55:00-04:00",
  "execution_lock": null,
  "strategies": {
    "ramp": {
      "positions": {
        "AAPL": {
          "qty": 50,
          "entry_price": 188.20,
          "entry_time": "2026-05-15T15:55:00-04:00",
          "order_id": "ibkr-12345"
        }
      },
      "last_execution": "2026-05-15T15:55:30-04:00"
    },
    "omr": {
      "positions": {},
      "last_execution": null
    }
  }
}
```

### State File Safety

**Atomic Writes:**
1. Write to `strategy_positions.json.tmp`
2. Use `os.replace()` (works on Windows and Linux)
3. Temp file atomically replaces original

**File Locking:**
- Acquire exclusive lock before read-modify-write
- Linux: `fcntl.flock()`
- Windows: `msvcrt.locking()`
- Prevents corruption if multiple processes access file

**Backup Strategy:**
- On startup, copy current state to `strategy_positions.json.bak`
- Keep last 3 backups with timestamps
- Validate JSON schema on load, reject malformed files

## Execution Lock

Only one strategy can execute orders at a time. This prevents interleaved order submission and buying power race conditions.

```json
{
  "execution_lock": {
    "holder": "ramp",
    "acquired": "2026-05-15T15:55:00-04:00",
    "expires": "2026-05-15T15:59:00-04:00"
  }
}
```

**Lock Lifecycle:**
1. Strategy attempts to acquire lock before execution
2. If lock held by another strategy, wait up to 30 seconds
3. If lock expired, force-acquire (previous holder crashed)
4. Execute orders while holding lock
5. Release lock when execution complete

**Timeout:** Each strategy has maximum 4 minutes to complete execution. If exceeded, lock expires automatically.

## Position Lifecycle Rules

### Rule 1: Position Entry

When a strategy opens a position:

1. **Acquire execution lock**
2. **Check pending orders** - abort if pending orders exist for symbol
3. **Check buying power** - verify sufficient funds
4. **Verify no conflict** - symbol not owned by another strategy
5. Submit buy order to broker (IBKR for RAMP/OMR/MP)
6. Wait for fill confirmation
7. **Handle partial fills** - track actual filled quantity
8. Write to state file (atomic with lock)
9. **Release execution lock**

```
[RAMP] Acquiring execution lock...
[RAMP] Lock acquired
[RAMP] Checking pending orders for AAPL: none
[RAMP] Buying power check: $95,000 available, $9,400 needed [+]
[RAMP] Symbol conflict check: AAPL not owned by other strategies [+]
[RAMP] Submitting order: BUY 50 AAPL @ MARKET (IBKR)
[RAMP] Order filled: 50 shares @ $188.20
[RAMP] State updated: AAPL added to RAMP positions
[RAMP] Releasing execution lock
```

### Rule 2: Position Exit (Normal)

When a strategy closes its own position:

1. **Acquire execution lock**
2. **Check pending orders** - abort if pending orders exist for symbol
3. Read owned quantity from state
4. Submit sell order for full quantity
5. Wait for fill confirmation
6. **Handle partial fills:**
   - Query actual filled quantity
   - If partial: update state with remaining shares
   - If complete: remove from state
7. Write to state file (atomic with lock)
8. **Release execution lock**

```
[RAMP] Acquiring execution lock...
[RAMP] Lock acquired
[RAMP] Closing position: AAPL (50 shares)
[RAMP] Submitting order: SELL 50 AAPL @ MARKET
[RAMP] Order filled: 50 shares @ $190.10 (complete fill)
[RAMP] State updated: AAPL removed from RAMP positions
[RAMP] Releasing execution lock
```

**Partial Fill Example:**
```
[RAMP] Order filled: 30 shares @ $190.10 (partial fill)
[RAMP] State updated: AAPL qty reduced from 50 to 20
[!] [RAMP] Partial close: 20 shares remaining for AAPL
```

### Rule 3: State Sync with Broker

On each trading cycle, before generating signals:

```python
def sync_state_with_broker():
    broker_positions = broker.get_positions()  # {symbol: qty}

    for strategy in ["ramp", "omr"]:
        for symbol, state_data in state[strategy]["positions"].items():

            if symbol not in broker_positions:
                # Position was closed externally
                log.warning(f"[{strategy}] Position {symbol} closed externally")
                del state[strategy]["positions"][symbol]

            elif broker_positions[symbol] < state_data["qty"]:
                # Partially closed externally
                new_qty = broker_positions[symbol]
                log.warning(f"[{strategy}] Position {symbol}: {state_data['qty']} -> {new_qty}")
                state[strategy]["positions"][symbol]["qty"] = new_qty
```

**Sync Timing:**
- Sync runs at start of each trading cycle
- Maximum age of sync data: 30 seconds before order submission
- If sync is stale, re-sync before submitting orders

### Rule 4: Pre-Order Pending Check

Before submitting ANY order:

```python
def can_submit_order(symbol: str) -> bool:
    pending = broker.get_orders(symbol=symbol, status='open')
    if pending:
        log.warning(f"Pending orders exist for {symbol}, skipping")
        return False
    return True
```

This prevents:
- Double-buying same symbol
- Selling shares that are already being sold
- Order collisions during slow fills

### Rule 5: Shutdown Coordination

**Problem:** Disable command must not run while strategy is submitting orders.

**Solution:** Shutdown flag pattern

```yaml
# strategy_toggle.yaml
strategies:
  ramp:
    enabled: true
    shutdown_requested: true  # <- Set by disable command
```

**Disable Command Flow:**
1. Set `shutdown_requested: true` in config
2. Wait for strategy's current execution to complete (poll `last_execution` timestamp)
3. Close positions if `--close-positions` specified
4. Set `enabled: false`
5. Set `shutdown_requested: false`

**Strategy Execution Flow:**
1. Check `shutdown_requested` before each order
2. If true, abort remaining orders gracefully
3. Update `last_execution` timestamp
4. Exit execution loop

```
[RAMP] Checking shutdown flag...
[!] [RAMP] Shutdown requested - aborting remaining 3 orders
[RAMP] Execution aborted gracefully
```

### Rule 6: Disable with Close Positions

When disabling a strategy with `--close-positions`:

```python
def disable_with_close(strategy: str):
    # Step 1: Request shutdown
    set_shutdown_requested(strategy, True)

    # Step 2: Wait for current execution to complete
    wait_for_execution_complete(strategy, timeout=60)

    # Step 3: Acquire execution lock
    acquire_lock(strategy)

    # Step 4: Close each position, updating state after each
    for symbol, data in list(positions.items()):
        try:
            order = broker.sell(symbol, qty=data["qty"])
            fill = wait_for_fill(order, timeout=30)

            filled_qty = fill.filled_quantity
            remaining = data["qty"] - filled_qty

            if remaining > 0:
                # Partial fill - update state with remaining
                positions[symbol]["qty"] = remaining
                log.warning(f"Partial close: {remaining} shares remaining")
            else:
                # Complete fill - remove from state
                del positions[symbol]

            save_state()  # Atomic write after EACH close

        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")
            # Position remains in state as orphaned

    # Step 5: Release lock and set disabled
    release_lock()
    set_enabled(strategy, False)
    set_shutdown_requested(strategy, False)
```

### Rule 7: Orphaned Position Handling

**Definition:** Position exists in state but strategy is disabled.

**Detection:** On startup and each cycle:
```python
def check_orphaned_positions():
    for strategy in ["ramp", "omr"]:
        if not is_enabled(strategy):
            positions = get_positions(strategy)
            if positions:
                log.warning(f"[{strategy}] DISABLED with orphaned positions:")
                for symbol, data in positions.items():
                    log.warning(f"  {symbol}: {data['qty']} shares")
```

**Sync While Disabled:**
Even disabled strategies sync their positions with broker on each cycle. This handles:
- Stop-loss triggered while strategy disabled
- Manual position close via broker UI

**Resolution Command:**
```bash
./toggle_strategy.sh ramp close-orphaned
```

### Rule 8: Symbol Conflict Prevention

Before opening any position:

```python
def can_open_position(strategy: str, symbol: str) -> bool:
    for other_strategy in all_strategies:
        if other_strategy != strategy:
            if symbol in other_strategy.positions:
                log.error(f"Cannot open {symbol}: owned by {other_strategy}")
                return False
    return True
```

**Startup Validation:**
```python
def validate_universes():
    ramp_symbols = set(ramp_universe)
    omr_symbols = set(omr_universe)
    overlap = ramp_symbols & omr_symbols
    if overlap:
        log.error(f"Universe overlap detected: {overlap}")
        raise ConfigurationError("Strategies must have non-overlapping universes")
```

### Rule 9: Buying Power Management

Before submitting orders, check available buying power:

```python
def execute_orders(orders: List[Order]):
    buying_power = broker.get_buying_power()

    # Sort orders by priority (highest momentum/probability first)
    orders.sort(key=lambda o: o.priority, reverse=True)

    for order in orders:
        if order.value > buying_power:
            log.warning(f"Insufficient buying power for {order.symbol}")
            log.warning(f"  Needed: ${order.value:,.2f}, Available: ${buying_power:,.2f}")
            continue  # Skip this order, try next

        if submit_order(order):
            buying_power -= order.value
```

### Rule 10: Execution Timeout

Each strategy has maximum 4 minutes to complete execution:

```python
EXECUTION_TIMEOUT = 240  # seconds

def execute_strategy(strategy):
    start_time = time.time()

    for order in orders_to_submit:
        elapsed = time.time() - start_time
        if elapsed > EXECUTION_TIMEOUT:
            remaining = len(orders_to_submit) - orders_submitted
            log.error(f"[{strategy}] Execution timeout after {elapsed:.0f}s")
            log.error(f"[{strategy}] Aborted {remaining} remaining orders")
            break

        submit_order(order)
```

## Toggle Command

### Usage

```bash
# Enable a strategy
./toggle_strategy.sh ramp enable

# Disable a strategy (keep positions open)
./toggle_strategy.sh ramp disable

# Disable and close all positions
./toggle_strategy.sh ramp disable --close-positions

# Close orphaned positions without enabling
./toggle_strategy.sh ramp close-orphaned

# Show current status
./toggle_strategy.sh status

# Emergency: force disable without waiting
./toggle_strategy.sh ramp disable --force
```

### Safe Restart Window

**Recommended restart times:** 9:35 AM - 3:40 PM EST

Avoid restarting:
- 9:30-9:35 AM (OMR exit window if OMR is enabled)
- 3:45-4:00 PM (OMR entry and RAMP rebalance window)

### Status Output

```
Strategy Toggle Status
======================
Time: 2026-05-15 15:30:00 EST

RAMP: ENABLED  (broker=ibkr, service=homeguard-multi)
  Last execution: 2026-05-15 15:55:30
  Positions: AAPL (50), MSFT (25), NVDA (40), ...
  Regime: WEAK_BULL
  Status: Waiting for next 15:55 rebalance

OMR: DISABLED  (broker=ibkr; not currently dispatched by homeguard-multi)
  Positions: (none)

CSCM: separate service (homeguard-cscm)

Execution Lock: None
Orphaned Positions: None
```

### Disable with Orphaned Warning

```
$ ./toggle_strategy.sh ramp disable --close-positions

Disabling RAMP strategy...
Setting shutdown_requested flag...
Waiting for current execution to complete... done

Closing positions:
  AAPL: Selling 50 shares... FILLED
  MSFT: Selling 25 shares... FILLED
  NVDA: Selling 40 shares... PARTIAL (30/40)

WARNING: Partial close for NVDA
  10 shares remaining as orphaned position

To close orphaned positions:
  ./toggle_strategy.sh ramp close-orphaned

RAMP strategy disabled.
```

## Error Handling

### Broker API Failure During Close

| Scenario | Behavior |
|----------|----------|
| Network timeout | Retry 3x with exponential backoff, then fail |
| Order rejected | Log error, position remains in state |
| Partial fill | Update state with remaining quantity |
| API rate limit | Back off 30s, retry once |

### State File Corruption Prevention

1. **Atomic writes:** Write to temp file, then `os.replace()`
2. **File locking:** Exclusive lock during read-modify-write
3. **Backup on startup:** Copy to timestamped backup file
4. **Validation on load:** JSON schema check, reject malformed
5. **Recovery:** If current file corrupt, try loading from backup

### Process Crash Recovery

On restart:
1. Load state from disk (validate JSON)
2. Sync all strategies with broker (regardless of enabled state)
3. Detect orphaned positions
4. Check for stale execution locks (force-release if expired)
5. Resume normal operation

## Trading Schedule

### Daily Timeline (All Times EST)

```
 9:30 AM ─── [OMR]  (if enabled) Pre-load historical data (VIX, SPY, leveraged ETFs)

 9:31 AM ─── [OMR]  (if enabled) EXIT: Sell all overnight positions
                    └─ Execution lock held for ~1-2 min

 3:55 PM ─┬─ [RAMP] REBALANCE: Buy/sell based on regime-aware momentum rankings
          │        ├─ Pre-load historical data (S&P 500, VIX)
          │        ├─ Detect regime (STRONG_BULL / WEAK_BULL / SIDEWAYS / UNPREDICTABLE / BEAR)
          │        ├─ Compute top-N targets per regime
          │        ├─ Sell stocks that dropped out of top-N
          │        ├─ Buy stocks that entered top-N
          │        └─ Execution lock held for ~2-3 min
          │
          └─ [OMR]  (if enabled) ENTRY: Open new overnight positions
                    ├─ Generate signals (Bayesian + regime filter)
                    ├─ Buy selected leveraged ETFs (TQQQ, SOXL, UPRO, ...)
                    └─ Execution lock held for ~2-3 min (after RAMP releases)

 4:00 PM ─── EOD reporting per enabled strategy
```

### Execution Lock Sequence

At **3:55 PM**, if both RAMP and OMR were enabled in a true multi-mode runtime, they would be serialized by execution lock:

```
3:55:00 -> [RAMP] Acquires lock, starts rebalancing
3:57:30 -> [RAMP] Releases lock after rebalance complete
3:57:31 -> [OMR]  Acquires lock, starts buying overnight positions
3:59:30 -> [OMR]  Releases lock after entry complete
```

Today (single-strategy `homeguard-multi`), only RAMP runs at 3:55 PM, so the lock is uncontended.

At **9:31 AM**, only OMR can run (and only if enabled):
```
9:31:00 -> [OMR] Acquires lock, starts selling overnight positions
9:31:45 -> [OMR] Releases lock after selling complete
```

### Strategy Execution Details

| Strategy | Signal Source | Universe | Trades/Day | Execution Time |
|----------|--------------|----------|------------|----------------|
| **RAMP** | Regime detector + regime-specific momentum formula | 503 S&P 500 stocks | 0-N (turnover-dependent) | ~2-4 min |
| **OMR**  | Bayesian model + VIX regime | 6 leveraged ETFs | 3-5 entry, 3-5 exit | ~1-2 min |
| **CSCM** | Cross-sectional crypto momentum + BTC regime filter | Crypto top-K | 0-K weekly | weekly cycle |

### What Each Strategy Trades

**RAMP (Regime-Aware Momentum Protection)** -- primary equity strategy:
- Rebalances at 3:55 PM based on today's close prices and the current market regime
- Trades: Any of ~503 S&P 500 stocks
- Holds until stock drops out of top-N for the current regime (days to weeks)

**OMR (Overnight Mean Reversion)** -- deployed but disabled:
- Would buy at 3:50 PM, sell at 9:31 AM next day
- Trades: TQQQ, SOXL, UPRO, SPXL, TECL, FNGU

**CSCM (Cross-Sectional Crypto Momentum)** -- separate service:
- Weekly rebalance on Sunday 00:00 UTC, runs against Coinbase, not in `strategy_positions.json`

## Monitoring

### Log Prefixes

All logs are prefixed with strategy identifier:
- `[RAMP]` - Regime-Aware Momentum Protection logs
- `[OMR]`  - Overnight Mean Reversion logs
- `[CSCM]` - Cross-Sectional Crypto Momentum logs

### Key Log Messages to Monitor

```
[!] [RAMP] Shutdown requested - aborting remaining orders
[!] [RAMP] Partial close: 20 shares remaining
[!] [RAMP] Insufficient buying power for NVDA
[-] [RAMP] Failed to close AAPL: IBKR connection timeout
[-] [RAMP] Execution timeout after 240s
[!] [RAMP] DISABLED with orphaned positions
```

### Health Checks

```bash
# View strategy status
./toggle_strategy.sh status

# View recent logs
./view_logs.sh   # or: journalctl -u homeguard-multi -f

# Daily 6-point health check
./daily_health_check.sh
```

## Implementation Files

| Component | File |
|-----------|------|
| Toggle config | `config/trading/strategy_toggle.yaml` |
| Broker routing | `config/trading/broker_routing.yaml` |
| Position state | `data/trading/strategy_positions.json` |
| State manager | `src/trading/state/strategy_state_manager.py` |
| Toggle command | `infra/ec2/toggle_strategy.sh` |
| Live runner | `scripts/trading/run_live_paper_trading.py` |
| RAMP adapter | `src/trading/adapters/ramp_live_adapter.py` |
| OMR adapter | `src/trading/adapters/omr_live_adapter.py` |
| MP adapter (legacy) | `src/trading/adapters/momentum_live_adapter.py` |
| CSCM adapter | `src/trading/adapters/cscm_live_adapter.py` |
| Multi-strategy systemd unit | `infra/ec2/homeguard-multi.service` |

## Safety Checklist

Before deploying multi-strategy:

- [ ] Verify universe isolation (no symbol overlap between enabled strategies)
- [ ] Test atomic write with simulated crash
- [ ] Test execution lock acquisition/release
- [ ] Test shutdown coordination (disable during execution)
- [ ] Test partial fill handling
- [ ] Test orphaned position detection
- [ ] Verify buying power checks work
- [ ] Test execution timeout
- [ ] Verify safe restart window documented
- [ ] Test toggle command with all options
- [ ] Verify each strategy is routed to the correct broker in `broker_routing.yaml`
