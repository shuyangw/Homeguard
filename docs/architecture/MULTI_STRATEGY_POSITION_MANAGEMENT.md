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

| ID | Name | Description | Status |
|----|------|-------------|--------|
| `omr` | Overnight Mean Reversion | Leveraged ETF overnight holds | Active |
| `mp` | Momentum Protection | S&P 500 momentum with crash protection | Active |
| `pairs` | Pairs Trading | Statistical arbitrage (future) | Planned |
| `vol` | Volatility Harvesting | VIX term structure (future) | Planned |

## Strategy Configuration

### Capital Allocation

| Strategy | Position Size | Max Positions | Max Exposure | Execution Time |
|----------|---------------|---------------|--------------|----------------|
| OMR      | 15%           | 3             | 45%          | 9:31 AM (exit), 3:55 PM (entry) |
| MP       | 6.5%          | 10            | 65%          | 3:55 PM (rebalance) |
| *Future* | TBD           | TBD           | TBD          | TBD |
| **Combined** | -         | 13+           | **110%+**    | - |

**Note:** Combined exposure can exceed 100%. In practice, strategies rarely reach maximum allocation simultaneously. If buying power is insufficient, orders are prioritized by signal strength.

### Universe Isolation

Strategies MUST trade non-overlapping universes to prevent conflicts:

| Strategy | Universe | Example Symbols | Universe Source |
|----------|----------|-----------------|-----------------|
| OMR      | Leveraged ETFs | TQQQ, SOXL, UPRO | `ETFUniverse.LEVERAGED_3X` |
| MP       | S&P 500 (filtered) | AAPL, MSFT, NVDA | `backtest_lists/sp500-2025.csv` |
| *pairs*  | Cointegrated pairs | XOM/CVX, KO/PEP | TBD |
| *vol*    | VIX products | VXX, UVXY, SVXY | TBD |

Universe isolation is validated on startup. If overlap is detected, the system logs an error and refuses to start.

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

2. **Define Universe** - Must not overlap with existing strategies

3. **Register in Toggle Config** (`config/trading/strategy_toggle.yaml`):
   ```yaml
   strategies:
     new_strat:
       enabled: false
       shutdown_requested: false
   ```

4. **Add Systemd Service** (for EC2 deployment):
   ```bash
   # /etc/systemd/system/homeguard-newstrat.service
   ```

5. **Update Documentation**:
   - Add to this file's strategy tables
   - Create `docs/architecture/NEW_STRATEGY_ARCHITECTURE.md`

## State Files

### Toggle Configuration

**File:** `config/trading/strategy_toggle.yaml`

```yaml
strategies:
  omr:
    enabled: true
    shutdown_requested: false
  mp:
    enabled: true
    shutdown_requested: false

# Metadata
last_modified: "2025-12-02T15:55:00-05:00"
modified_by: "toggle_strategy.sh"
```

- Read on each trading cycle
- Modified by toggle command
- `shutdown_requested` enables graceful shutdown coordination

### Position State

**File:** `data/trading/strategy_positions.json`

```json
{
  "version": 1,
  "last_updated": "2025-12-02T15:55:00-05:00",
  "execution_lock": null,
  "strategies": {
    "omr": {
      "positions": {
        "TQQQ": {
          "qty": 50,
          "entry_price": 45.20,
          "entry_time": "2025-12-02T15:55:00-05:00",
          "order_id": "abc123"
        }
      },
      "last_execution": "2025-12-02T15:55:30-05:00"
    },
    "mp": {
      "positions": {
        "PLTR": {
          "qty": 100,
          "entry_price": 65.00,
          "entry_time": "2025-12-02T15:55:00-05:00",
          "order_id": "def456"
        }
      },
      "last_execution": "2025-12-02T15:55:30-05:00"
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
    "holder": "mp",
    "acquired": "2025-12-02T15:55:00-05:00",
    "expires": "2025-12-02T15:59:00-05:00"
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
5. Submit buy order to broker
6. Wait for fill confirmation
7. **Handle partial fills** - track actual filled quantity
8. Write to state file (atomic with lock)
9. **Release execution lock**

```
[MP] Acquiring execution lock...
[MP] Lock acquired
[MP] Checking pending orders for PLTR: none
[MP] Buying power check: $65,000 available, $6,500 needed ✓
[MP] Symbol conflict check: PLTR not owned by other strategies ✓
[MP] Submitting order: BUY 100 PLTR @ MARKET
[MP] Order filled: 100 shares @ $65.00
[MP] State updated: PLTR added to MP positions
[MP] Releasing execution lock
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
[MP] Acquiring execution lock...
[MP] Lock acquired
[MP] Closing position: PLTR (100 shares)
[MP] Submitting order: SELL 100 PLTR @ MARKET
[MP] Order filled: 100 shares @ $68.50 (complete fill)
[MP] State updated: PLTR removed from MP positions
[MP] Releasing execution lock
```

**Partial Fill Example:**
```
[MP] Order filled: 60 shares @ $68.50 (partial fill)
[MP] State updated: PLTR qty reduced from 100 to 40
[!] [MP] Partial close: 40 shares remaining for PLTR
```

### Rule 3: State Sync with Broker

On each trading cycle, before generating signals:

```python
def sync_state_with_broker():
    broker_positions = broker.get_positions()  # {symbol: qty}

    for strategy in ["omr", "mp"]:
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
  mp:
    enabled: true
    shutdown_requested: true  # ← Set by disable command
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
[MP] Checking shutdown flag...
[!] [MP] Shutdown requested - aborting remaining 3 orders
[MP] Execution aborted gracefully
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
    for strategy in ["omr", "mp"]:
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
./toggle_strategy.sh mp close-orphaned
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
    omr_symbols = set(omr_universe)
    mp_symbols = set(mp_universe)
    overlap = omr_symbols & mp_symbols
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
./toggle_strategy.sh omr enable

# Disable a strategy (keep positions open)
./toggle_strategy.sh mp disable

# Disable and close all positions
./toggle_strategy.sh mp disable --close-positions

# Close orphaned positions without enabling
./toggle_strategy.sh mp close-orphaned

# Show current status
./toggle_strategy.sh status

# Emergency: force disable without waiting
./toggle_strategy.sh mp disable --force
```

### Safe Restart Window

**Recommended restart times:** 9:35 AM - 3:40 PM EST

Avoid restarting:
- 9:30-9:35 AM (OMR exit window)
- 3:45-4:00 PM (OMR entry and MP rebalance window)

### Status Output

```
Strategy Toggle Status
======================
Time: 2025-12-02 15:30:00 EST

OMR: ENABLED
  Last execution: 2025-12-02 09:31:15
  Positions: TQQQ (50), SOXL (30)
  Status: Waiting for 15:55 entry window

MP: ENABLED
  Last execution: 2025-12-02 15:55:30
  Positions: PLTR (100), AVGO (25), NVDA (50)
  Status: Waiting for next trading day

Execution Lock: None

Orphaned Positions: None
```

### Disable with Orphaned Warning

```
$ ./toggle_strategy.sh mp disable --close-positions

Disabling MP strategy...
Setting shutdown_requested flag...
Waiting for current execution to complete... done

Closing positions:
  PLTR: Selling 100 shares... FILLED
  AVGO: Selling 25 shares... FILLED
  NVDA: Selling 50 shares... PARTIAL (30/50)

WARNING: Partial close for NVDA
  20 shares remaining as orphaned position

To close orphaned positions:
  ./toggle_strategy.sh mp close-orphaned

MP strategy disabled.
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
 9:30 AM ─── [OMR] Pre-load historical data (VIX, SPY, leveraged ETFs)

 9:31 AM ─── [OMR] EXIT: Sell all overnight positions (TQQQ, SOXL, etc.)
                   └─ Execution lock held for ~1-2 min

 3:55 PM ─┬─ [OMR] ENTRY: Open new overnight positions
          │        ├─ Generate signals (Bayesian + regime filter)
          │        ├─ Buy selected leveraged ETFs (TQQQ, SOXL, UPRO, etc.)
          │        └─ Execution lock held for ~2-3 min
          │
          └─ [MP]  REBALANCE: Buy/sell based on today's momentum rankings
                   ├─ Pre-load historical data (S&P 500, VIX via yfinance)
                   ├─ Sell stocks that dropped out of top 10
                   ├─ Buy stocks that entered top 10
                   └─ Execution lock held for ~2-3 min (after OMR releases)

 4:00 PM ─┬─ [OMR] Generate EOD report
          └─ [MP]  Generate EOD report
```

### Execution Lock Sequence

At **3:55 PM**, both strategies trigger and are serialized by execution lock:

```
3:55:00 → [OMR] Acquires lock, starts buying overnight positions
3:57:30 → [OMR] Releases lock after entry complete
3:57:31 → [MP]  Acquires lock, starts rebalancing
3:59:30 → [MP]  Releases lock after rebalancing complete
```

At **9:31 AM**, only OMR runs:
```
9:31:00 → [OMR] Acquires lock, starts selling overnight positions
9:31:45 → [OMR] Releases lock after selling complete
```

### Strategy Execution Details

| Strategy | Signal Source | Universe | Trades/Day | Execution Time |
|----------|--------------|----------|------------|----------------|
| **OMR** | Bayesian model + VIX regime | 6 leveraged ETFs | 3-5 entry, 3-5 exit | ~1-2 min |
| **MP** | 1m-1w momentum | 503 S&P 500 stocks | 0-4 (avg 1.3) | ~2-4 min |

### What Each Strategy Trades

**OMR (Overnight Mean Reversion)**:
- Buys at 3:55 PM, sells at 9:31 AM next day
- Trades: TQQQ, SOXL, UPRO, SPXL, TECL, FNGU
- Holds ~16 hours overnight

**MP (Momentum Protection)**:
- Rebalances at 3:55 PM based on today's close prices
- Trades: Any of 503 S&P 500 stocks
- Holds until stock drops out of top 10 (days to weeks)

## Monitoring

### Log Prefixes

All logs are prefixed with strategy identifier:
- `[OMR]` - Overnight Mean Reversion logs
- `[MP]` - Momentum Protection logs

### Key Log Messages to Monitor

```
[!] [MP] Shutdown requested - aborting remaining orders
[!] [OMR] Partial close: 20 shares remaining
[!] [MP] Insufficient buying power for NVDA
[-] [OMR] Failed to close TQQQ: Connection timeout
[-] [MP] Execution timeout after 240s
[!] [MP] DISABLED with orphaned positions
```

### Health Checks

```bash
# View strategy status
./toggle_strategy.sh status

# View recent logs
./view_logs.sh

# Check for issues
./daily_health_check.sh
```

## Implementation Files

| Component | File |
|-----------|------|
| Toggle config | `config/trading/strategy_toggle.yaml` |
| Position state | `data/trading/strategy_positions.json` |
| State manager | `src/trading/state/strategy_state_manager.py` |
| Toggle command | `scripts/ec2/toggle_strategy.sh` |
| Live runner | `scripts/trading/run_live_paper_trading.py` |
| OMR adapter | `src/trading/adapters/omr_live_adapter.py` |
| MP adapter | `src/trading/adapters/momentum_live_adapter.py` |

## Safety Checklist

Before deploying multi-strategy:

- [ ] Verify universe isolation (no symbol overlap)
- [ ] Test atomic write with simulated crash
- [ ] Test execution lock acquisition/release
- [ ] Test shutdown coordination (disable during execution)
- [ ] Test partial fill handling
- [ ] Test orphaned position detection
- [ ] Verify buying power checks work
- [ ] Test execution timeout
- [ ] Verify safe restart window documented
- [ ] Test toggle command with all options
