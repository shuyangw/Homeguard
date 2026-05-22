"""
Strategy State Manager for Multi-Strategy Trading.

Provides:
- Atomic state persistence with file locking
- Execution lock management (one strategy at a time)
- Position tracking per strategy
- Toggle config reading
- Shutdown coordination
- Broker sync for external position changes

See docs/architecture/MULTI_STRATEGY_POSITION_MANAGEMENT.md for full documentation.
"""

import copy
import json
import os
import sys
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import yaml

from src.utils.logger import logger
from src.utils.timezone import tz

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt

    def lock_file(f):
        """Acquire exclusive lock on file (Windows)."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock_file(f):
        """Release lock on file (Windows)."""
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception as e:
            logger.warning(f"Failed to unlock file: {e}")
else:
    import fcntl

    def lock_file(f):
        """Acquire exclusive lock on file (Unix)."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock_file(f):
        """Release lock on file (Unix)."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclass
class PositionInfo:
    """Information about a tracked position."""
    qty: int
    entry_price: float
    entry_time: str
    order_id: Optional[str] = None


@dataclass
class ExecutionLock:
    """Execution lock state."""
    holder: str
    acquired: str
    expires: str


@dataclass
class StrategyState:
    """State for a single strategy."""
    positions: Dict[str, PositionInfo] = field(default_factory=dict)
    last_execution: Optional[str] = None


@dataclass
class ToggleConfig:
    """Toggle configuration for a strategy."""
    enabled: bool = True
    shutdown_requested: bool = False


class StrategyStateManager:
    """
    Manages state for multiple trading strategies.

    Provides thread-safe, atomic state persistence with:
    - File locking to prevent corruption
    - Execution locks to serialize strategy execution
    - Position tracking per strategy
    - Broker synchronization

    Usage:
        manager = StrategyStateManager()

        # Check if strategy is enabled
        if manager.is_enabled('mp'):
            # Acquire execution lock
            if manager.acquire_execution_lock('mp'):
                try:
                    # Add position
                    manager.add_position('mp', 'PLTR', 100, 65.00)
                    # ... execute orders ...
                finally:
                    manager.release_execution_lock('mp')
    """

    # Execution timeout in seconds (4 minutes)
    EXECUTION_TIMEOUT = 240

    # Lock wait timeout in seconds
    LOCK_WAIT_TIMEOUT = 30

    def __init__(
        self,
        state_file: Optional[Path] = None,
        toggle_file: Optional[Path] = None,
        backup_count: int = 3
    ):
        """
        Initialize state manager.

        Args:
            state_file: Path to position state file
            toggle_file: Path to toggle config file
            backup_count: Number of backup files to keep
        """
        # Resolve file paths
        project_root = Path(__file__).resolve().parent.parent.parent.parent

        self.state_file = state_file or project_root / 'data' / 'trading' / 'strategy_positions.json'
        self.toggle_file = toggle_file or project_root / 'config' / 'trading' / 'strategy_toggle.yaml'
        self.backup_count = backup_count

        # Ensure directories exist
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self._state: Dict[str, Any] = {}
        self._toggle: Dict[str, Any] = {}
        self._file_handle = None

        # Load initial state
        self._load_state()
        self._load_toggle()

        # Create backup on startup
        self._create_backup()

        logger.info(f"StrategyStateManager initialized")
        logger.info(f"  State file: {self.state_file}")
        logger.info(f"  Toggle file: {self.toggle_file}")

    # =========================================================================
    # Toggle Configuration
    # =========================================================================

    def _load_toggle(self) -> None:
        """Load toggle configuration from YAML file."""
        try:
            if self.toggle_file.exists():
                with open(self.toggle_file, 'r') as f:
                    self._toggle = yaml.safe_load(f) or {}
            else:
                # Create default toggle config
                self._toggle = {
                    'strategies': {
                        'omr': {'enabled': True, 'shutdown_requested': False},
                        'mp': {'enabled': False, 'shutdown_requested': False},
                        'ramp': {'enabled': True, 'shutdown_requested': False},
                        'cscm': {'enabled': True, 'shutdown_requested': False},
                    },
                    'last_modified': tz.iso_timestamp(),
                    'modified_by': 'auto'
                }
                self._save_toggle()
        except Exception as e:
            logger.error(f"Failed to load toggle config: {e}")
            self._toggle = {'strategies': {}}

    def _save_toggle(self) -> None:
        """Save toggle configuration to YAML file."""
        try:
            self._toggle['last_modified'] = tz.iso_timestamp()

            # Write to temp file first
            temp_file = self.toggle_file.with_suffix('.yaml.tmp')
            with open(temp_file, 'w') as f:
                yaml.dump(self._toggle, f, default_flow_style=False)

            # Atomic replace
            os.replace(temp_file, self.toggle_file)

        except Exception as e:
            logger.error(f"Failed to save toggle config: {e}")

    def reload_toggle(self) -> None:
        """Reload toggle configuration from disk."""
        self._load_toggle()

    def is_enabled(self, strategy: str) -> bool:
        """Check if a strategy is enabled."""
        self.reload_toggle()  # Always read fresh
        strategies = self._toggle.get('strategies', {})
        config = strategies.get(strategy, {})
        return config.get('enabled', False)

    def is_shutdown_requested(self, strategy: str) -> bool:
        """Check if shutdown is requested for a strategy."""
        self.reload_toggle()
        strategies = self._toggle.get('strategies', {})
        config = strategies.get(strategy, {})
        return config.get('shutdown_requested', False)

    def set_enabled(self, strategy: str, enabled: bool, modified_by: str = 'api') -> None:
        """Set strategy enabled state."""
        if 'strategies' not in self._toggle:
            self._toggle['strategies'] = {}
        if strategy not in self._toggle['strategies']:
            self._toggle['strategies'][strategy] = {}

        self._toggle['strategies'][strategy]['enabled'] = enabled
        self._toggle['modified_by'] = modified_by
        self._save_toggle()

        logger.info(f"Strategy '{strategy}' {'enabled' if enabled else 'disabled'}")

    def set_shutdown_requested(self, strategy: str, requested: bool) -> None:
        """Set shutdown requested flag for a strategy."""
        if 'strategies' not in self._toggle:
            self._toggle['strategies'] = {}
        if strategy not in self._toggle['strategies']:
            self._toggle['strategies'][strategy] = {}

        self._toggle['strategies'][strategy]['shutdown_requested'] = requested
        self._save_toggle()

        if requested:
            logger.warning(f"[{strategy.upper()}] Shutdown requested")

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names."""
        self.reload_toggle()
        strategies = self._toggle.get('strategies', {})
        return [name for name, config in strategies.items() if config.get('enabled', False)]

    # =========================================================================
    # State File Management
    # =========================================================================

    def _load_state(self) -> None:
        """Load state from JSON file with validation and v1->v2 migration."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    self._state = json.load(f)

                # Validate structure
                if 'strategies' not in self._state:
                    self._state['strategies'] = {}
                if 'version' not in self._state:
                    self._state['version'] = 1

                # v1 -> v2 migration: tag every untagged position as 'alpaca'.
                # Alpaca is the only broker in production as of commit b42b09e,
                # so this default is correct. Post-migration, untagged positions
                # are a bug and will raise in sync/write paths.
                if self._state.get('version', 1) < 2:
                    self._migrate_v1_to_v2()
            else:
                # Initialize empty state
                self._state = {
                    'version': 2,
                    'last_updated': tz.iso_timestamp(),
                    'execution_lock': None,
                    'strategies': {
                        'omr': {'positions': {}, 'last_execution': None},
                        'mp': {'positions': {}, 'last_execution': None},
                        'ramp': {'positions': {}, 'last_execution': None},
                        'cscm': {'positions': {}, 'last_execution': None},
                    }
                }
                self._save_state()

        except json.JSONDecodeError as e:
            logger.error(f"State file corrupted: {e}")
            self._recover_from_backup()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self._state = {'strategies': {}}

    def _migrate_v1_to_v2(self) -> None:
        """One-shot migration: stamp broker='alpaca' on every untagged position."""
        tagged = 0
        strategies = self._state.get('strategies', {})
        if not isinstance(strategies, dict):
            self._state['version'] = 2
            self._save_state()
            logger.warning("Migrated state file v1 -> v2: 'strategies' was not a dict, no positions tagged")
            return
        for strategy, data in strategies.items():
            if not isinstance(data, dict):
                continue
            positions = data.get('positions', {})
            if not isinstance(positions, dict):
                continue
            for symbol, pos in positions.items():
                if isinstance(pos, dict) and 'broker' not in pos:
                    pos['broker'] = 'alpaca'
                    tagged += 1
        self._state['version'] = 2
        self._save_state()
        logger.info(f"Migrated state file v1 -> v2: tagged {tagged} positions as 'alpaca'")

    def _save_state(self) -> None:
        """Save state to JSON file atomically with file locking.

        On Windows + Dropbox-watched dirs, the atomic os.replace() can
        spuriously fail with PermissionError when the file watcher has
        the destination open. Retry briefly to absorb the race.
        """
        try:
            self._state['last_updated'] = tz.iso_timestamp()

            # Write to temp file
            temp_file = self.state_file.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self._state, f, indent=2)

            # Atomic replace with retry-on-PermissionError (Windows/Dropbox race)
            import time
            for attempt in range(3):
                try:
                    os.replace(temp_file, self.state_file)
                    break
                except PermissionError as e:
                    if attempt == 2:
                        raise
                    logger.warning(
                        f"State write race on {self.state_file}, "
                        f"retrying ({attempt + 1}/3): {e}"
                    )
                    time.sleep(0.05 * (attempt + 1))

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise

    def _create_backup(self) -> None:
        """Create backup of current state file."""
        try:
            if not self.state_file.exists():
                return

            timestamp = tz.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.state_file.with_suffix(f'.{timestamp}.bak')
            shutil.copy2(self.state_file, backup_file)

            # Clean old backups
            backup_pattern = self.state_file.stem + '.*.bak'
            backups = sorted(self.state_file.parent.glob(backup_pattern), reverse=True)
            for old_backup in backups[self.backup_count:]:
                old_backup.unlink()

            logger.info(f"Created state backup: {backup_file.name}")

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def _recover_from_backup(self) -> None:
        """Attempt to recover state from backup file."""
        try:
            backup_pattern = self.state_file.stem + '.*.bak'
            backups = sorted(self.state_file.parent.glob(backup_pattern), reverse=True)

            for backup in backups:
                try:
                    with open(backup, 'r') as f:
                        self._state = json.load(f)
                    logger.warning(f"Recovered state from backup: {backup.name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load backup {backup.name}: {e}")
                    continue

            # No valid backup found
            logger.error("No valid backup found, initializing empty state")
            self._state = {'strategies': {}}

        except Exception as e:
            logger.error(f"Failed to recover from backup: {e}")
            self._state = {'strategies': {}}

    # =========================================================================
    # Execution Lock
    # =========================================================================

    def acquire_execution_lock(self, strategy: str, timeout: int = None) -> bool:
        """
        Acquire execution lock for a strategy.

        Args:
            strategy: Strategy name
            timeout: Wait timeout in seconds (default: LOCK_WAIT_TIMEOUT)

        Returns:
            True if lock acquired, False if timeout
        """
        timeout = timeout or self.LOCK_WAIT_TIMEOUT
        start_time = time.time()

        while True:
            self._load_state()  # Refresh

            lock = self._state.get('execution_lock')

            if lock is None:
                # Lock available, acquire it
                expires = tz.now() + timedelta(seconds=self.EXECUTION_TIMEOUT)
                self._state['execution_lock'] = {
                    'holder': strategy,
                    'acquired': tz.iso_timestamp(),
                    'expires': expires.isoformat()
                }
                self._save_state()
                logger.info(f"[{strategy.upper()}] Acquired execution lock")
                return True

            # Check if lock is expired
            try:
                expires = datetime.fromisoformat(lock['expires'])
                if tz.now() > expires:
                    logger.warning(f"Force-acquiring expired lock from {lock['holder']}")
                    expires = tz.now() + timedelta(seconds=self.EXECUTION_TIMEOUT)
                    self._state['execution_lock'] = {
                        'holder': strategy,
                        'acquired': tz.iso_timestamp(),
                        'expires': expires.isoformat()
                    }
                    self._save_state()
                    return True
            except Exception as e:
                logger.warning(f"Failed to check lock expiration: {e}")

            # Check if we already hold the lock
            if lock.get('holder') == strategy:
                logger.info(f"[{strategy.upper()}] Already holds execution lock")
                return True

            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"[{strategy.upper()}] Timeout waiting for execution lock (held by {lock['holder']})")
                return False

            # Wait and retry
            time.sleep(1)

    def release_execution_lock(self, strategy: str) -> None:
        """Release execution lock if held by this strategy."""
        self._load_state()

        lock = self._state.get('execution_lock')
        if lock and lock.get('holder') == strategy:
            self._state['execution_lock'] = None
            self._save_state()
            logger.info(f"[{strategy.upper()}] Released execution lock")
        else:
            logger.warning(f"[{strategy.upper()}] Cannot release lock not held by this strategy")

    def get_execution_lock_holder(self) -> Optional[str]:
        """Get the strategy currently holding the execution lock."""
        self._load_state()
        lock = self._state.get('execution_lock')
        if lock:
            return lock.get('holder')
        return None

    # =========================================================================
    # Position Management
    # =========================================================================

    def get_positions(self, strategy: str) -> Dict[str, Dict]:
        """Get all positions for a strategy."""
        self._load_state()
        strategies = self._state.get('strategies', {})
        strategy_data = strategies.get(strategy, {})
        return strategy_data.get('positions', {}).copy()

    def add_position(
        self,
        strategy: str,
        symbol: str,
        qty: int,
        entry_price: float,
        order_id: Optional[str] = None,
        broker: Optional[str] = None,
    ) -> None:
        """
        Add a position to a strategy's tracked positions.

        Args:
            strategy: Strategy name
            symbol: Stock symbol
            qty: Quantity
            entry_price: Entry price
            order_id: Optional order ID
            broker: Name of the broker that opened the position (required in v2)
        """
        if not broker:
            raise ValueError("add_position requires broker name (v2 schema)")

        self._load_state()

        if 'strategies' not in self._state:
            self._state['strategies'] = {}
        if strategy not in self._state['strategies']:
            self._state['strategies'][strategy] = {'positions': {}, 'last_execution': None}

        self._state['strategies'][strategy]['positions'][symbol] = {
            'qty': qty,
            'entry_price': entry_price,
            'entry_time': tz.iso_timestamp(),
            'order_id': order_id,
            'broker': broker,
        }

        self._save_state()
        logger.info(
            f"[{strategy.upper()}] Added position: {symbol} "
            f"({qty} shares @ ${entry_price:.2f}) on {broker}"
        )

    def update_position_qty(self, strategy: str, symbol: str, new_qty: int) -> None:
        """Update the quantity of an existing position."""
        self._load_state()

        positions = self._state.get('strategies', {}).get(strategy, {}).get('positions', {})
        if symbol in positions:
            old_qty = positions[symbol]['qty']
            positions[symbol]['qty'] = new_qty
            self._save_state()
            logger.info(f"[{strategy.upper()}] Updated {symbol}: {old_qty} -> {new_qty} shares")

    def add_or_update_position(
        self,
        strategy: str,
        symbol: str,
        qty_delta: int,
        price: float,
        order_id: Optional[str] = None,
        broker: Optional[str] = None,
    ) -> int:
        """
        Add to an existing position or create a new one.

        - New positions: broker is REQUIRED.
        - Top-ups: broker is optional. If passed and differs from the existing
          tag, raises ValueError. If omitted, the existing tag is preserved.

        Returns new total quantity after update.
        """
        self._load_state()

        if 'strategies' not in self._state:
            self._state['strategies'] = {}
        if strategy not in self._state['strategies']:
            self._state['strategies'][strategy] = {'positions': {}, 'last_execution': None}

        positions = self._state['strategies'][strategy]['positions']

        if symbol in positions:
            existing_broker = positions[symbol].get('broker')
            if broker and existing_broker and broker != existing_broker:
                raise ValueError(
                    f"Cannot top up {strategy}:{symbol}: state tagged "
                    f"{existing_broker!r}, caller passed {broker!r}"
                )
            old_qty = positions[symbol]['qty']
            new_qty = old_qty + qty_delta
            positions[symbol]['qty'] = new_qty
            self._save_state()
            logger.info(
                f"[{strategy.upper()}] Topped up {symbol}: "
                f"{old_qty} + {qty_delta} = {new_qty} shares"
            )
            return new_qty

        # New position -- broker is required.
        if not broker:
            raise ValueError(
                "add_or_update_position requires broker for new positions (v2 schema)"
            )
        positions[symbol] = {
            'qty': qty_delta,
            'entry_price': price,
            'entry_time': tz.iso_timestamp(),
            'order_id': order_id,
            'broker': broker,
        }
        self._save_state()
        logger.info(
            f"[{strategy.upper()}] Added position: {symbol} "
            f"({qty_delta} shares @ ${price:.2f}) on {broker}"
        )
        return qty_delta

    def remove_position(self, strategy: str, symbol: str) -> None:
        """Remove a position from a strategy's tracked positions."""
        self._load_state()

        positions = self._state.get('strategies', {}).get(strategy, {}).get('positions', {})
        if symbol in positions:
            del positions[symbol]
            self._save_state()
            logger.info(f"[{strategy.upper()}] Removed position: {symbol}")

    def has_position(self, strategy: str, symbol: str) -> bool:
        """Check if a strategy has a position in a symbol."""
        positions = self.get_positions(strategy)
        return symbol in positions

    def get_position_qty(self, strategy: str, symbol: str) -> int:
        """Get quantity of a position."""
        positions = self.get_positions(strategy)
        pos = positions.get(symbol, {})
        return pos.get('qty', 0)

    def symbol_owned_by_other(self, strategy: str, symbol: str) -> Optional[str]:
        """
        Check if symbol is owned by another strategy.

        Returns:
            Name of owning strategy, or None if not owned
        """
        self._load_state()

        for other_strategy, data in self._state.get('strategies', {}).items():
            if other_strategy != strategy:
                if symbol in data.get('positions', {}):
                    return other_strategy

        return None

    def update_last_execution(self, strategy: str) -> None:
        """Update the last execution timestamp for a strategy."""
        self._load_state()

        if strategy in self._state.get('strategies', {}):
            self._state['strategies'][strategy]['last_execution'] = tz.iso_timestamp()
            self._save_state()

    def get_last_execution(self, strategy: str) -> Optional[str]:
        """Get the last execution timestamp for a strategy."""
        self._load_state()
        return self._state.get('strategies', {}).get(strategy, {}).get('last_execution')

    # =========================================================================
    # Runner Session State (peak equity, day-open equity)
    # =========================================================================
    # These persist `LiveTradingRunner._peak_equity` and `_session_open_equity`
    # across process restarts so the drawdown and day-PnL gauges keep their
    # baselines instead of resetting to 0 every time the bot is restarted.

    def get_runner_session_state(self, strategy: str) -> Dict[str, Any]:
        """Return persisted runner session state for `strategy`.

        Keys: peak_equity_usd (float), session_open_equity_usd (float),
        session_open_date (str YYYY-MM-DD ET). Missing keys return None values.
        """
        self._load_state()
        strat = self._state.get('strategies', {}).get(strategy, {}) or {}
        return {
            'peak_equity_usd': strat.get('peak_equity_usd'),
            'session_open_equity_usd': strat.get('session_open_equity_usd'),
            'session_open_date': strat.get('session_open_date'),
        }

    def update_runner_session_state(
        self,
        strategy: str,
        peak_equity_usd: Optional[float] = None,
        session_open_equity_usd: Optional[float] = None,
        session_open_date: Optional[str] = None,
    ) -> None:
        """Persist runner session state for `strategy`. Only non-None values are written."""
        self._load_state()
        if strategy not in self._state.get('strategies', {}):
            self._state.setdefault('strategies', {})[strategy] = {'positions': {}, 'last_execution': None}
        strat = self._state['strategies'][strategy]
        if peak_equity_usd is not None:
            strat['peak_equity_usd'] = float(peak_equity_usd)
        if session_open_equity_usd is not None:
            strat['session_open_equity_usd'] = float(session_open_equity_usd)
        if session_open_date is not None:
            strat['session_open_date'] = str(session_open_date)
        self._save_state()

    # =========================================================================
    # Broker Synchronization
    # =========================================================================

    def sync_with_broker(
        self,
        broker_name: str,
        broker_positions: Dict[str, int],
    ) -> Dict[str, List[str]]:
        """
        Synchronize state with actual broker positions for a specific broker.

        Only reconciles positions tagged with `broker_name`. Positions tagged
        with a different broker are left untouched and reported in 'skipped'.

        Untagged positions indicate a post-migration bug and raise ValueError.

        Args:
            broker_name: Name of the broker being synced ('alpaca', 'ibkr', ...)
            broker_positions: Dict of symbol -> quantity from that broker

        Returns:
            Dict with 'removed', 'updated', 'drift_detected', 'skipped' lists.
        """
        self._load_state()

        changes: Dict[str, List[str]] = {
            'removed': [],
            'updated': [],
            'drift_detected': [],
            'skipped': [],
        }

        for strategy, data in self._state.get('strategies', {}).items():
            positions = data.get('positions', {})

            for symbol in list(positions.keys()):
                pos_broker = positions[symbol].get('broker')
                if pos_broker is None:
                    logger.error(
                        f"[{strategy.upper()}] {symbol} has no broker tag "
                        f"(post-migration bug)"
                    )
                    raise ValueError(f"Untagged position: {strategy}:{symbol}")

                if pos_broker != broker_name:
                    changes['skipped'].append(f"{strategy}:{symbol} (on {pos_broker})")
                    continue

                state_qty = positions[symbol]['qty']
                broker_qty = broker_positions.get(symbol, 0)

                if broker_qty == 0:
                    logger.warning(
                        f"[{strategy.upper()}] {symbol} closed on {broker_name}"
                    )
                    del positions[symbol]
                    changes['removed'].append(f"{strategy}:{symbol}")

                elif broker_qty < state_qty:
                    logger.warning(
                        f"[{strategy.upper()}] {symbol} reduced on {broker_name}: "
                        f"{state_qty} -> {broker_qty}"
                    )
                    positions[symbol]['qty'] = broker_qty
                    changes['updated'].append(f"{strategy}:{symbol}")

                elif broker_qty > state_qty:
                    logger.error(
                        f"[{strategy.upper()}] DRIFT on {broker_name}: {symbol} "
                        f"state={state_qty} broker={broker_qty} "
                        f"(+{broker_qty - state_qty} untracked)"
                    )
                    positions[symbol]['qty'] = broker_qty
                    changes['drift_detected'].append(f"{strategy}:{symbol}")

        if changes['removed'] or changes['updated'] or changes['drift_detected']:
            self._save_state()

        return changes

    def get_positions_by_broker(self, strategy: str) -> Dict[str, Dict]:
        """Get positions for a strategy, grouped by the broker that opened them."""
        positions = self.get_positions(strategy)
        by_broker: Dict[str, Dict] = {}
        for symbol, info in positions.items():
            broker = info.get('broker')
            if broker is None:
                raise ValueError(f"Untagged position: {strategy}:{symbol}")
            by_broker.setdefault(broker, {})[symbol] = info
        return by_broker

    def check_broker_switch_safety(
        self,
        strategy: str,
        current_broker_name: str,
        new_broker_name: str,
        current_broker,
        new_broker,
    ) -> Dict[str, Any]:
        """Query both brokers and report whether it's safe to switch. Does NOT mutate state."""
        state_positions = self.get_positions(strategy)
        blocking_reasons: List[str] = []

        try:
            current_positions = {
                p['symbol']: abs(int(p['quantity'])) for p in current_broker.get_stock_positions()
            }
        except Exception as e:
            logger.error(f"Failed to fetch positions from {current_broker_name}: {e}")
            current_positions = {}
            blocking_reasons.append(f"Cannot reach current broker ({current_broker_name}): {e}")

        try:
            new_positions = {
                p['symbol']: abs(int(p['quantity'])) for p in new_broker.get_stock_positions()
            }
        except Exception as e:
            logger.error(f"Failed to fetch positions from {new_broker_name}: {e}")
            new_positions = {}
            blocking_reasons.append(f"Cannot reach new broker ({new_broker_name}): {e}")

        for symbol, pos_info in state_positions.items():
            pos_broker = pos_info.get('broker')
            if pos_broker is None:
                raise ValueError(f"Untagged position: {strategy}:{symbol}")
            state_qty = pos_info.get('qty', 0)
            if pos_broker == current_broker_name and state_qty > 0:
                broker_qty = current_positions.get(symbol, 0)
                if broker_qty != 0:
                    blocking_reasons.append(
                        f"{symbol}: {broker_qty} shares still open on {current_broker_name}"
                    )

        safe = len(blocking_reasons) == 0
        if blocking_reasons:
            action = (
                f"Close {len(blocking_reasons)} position(s) on {current_broker_name} "
                f"before switching to {new_broker_name}:\n"
                + '\n'.join(f"  - {r}" for r in blocking_reasons)
            )
        else:
            action = f"Safe to switch {strategy} from {current_broker_name} to {new_broker_name}"

        return {
            'safe': safe,
            'positions_on_current': current_positions,
            'positions_on_new': new_positions,
            'state_positions': copy.deepcopy(state_positions),
            'blocking_reasons': blocking_reasons,
            'action_required': action,
        }

    # =========================================================================
    # Orphaned Positions
    # =========================================================================

    def get_orphaned_positions(self) -> Dict[str, Dict]:
        """
        Get positions for disabled strategies (orphaned).

        Returns:
            Dict of strategy -> positions for disabled strategies with positions
        """
        self._load_state()
        self.reload_toggle()

        orphaned = {}

        for strategy, data in self._state.get('strategies', {}).items():
            if not self.is_enabled(strategy):
                positions = data.get('positions', {})
                if positions:
                    orphaned[strategy] = positions.copy()

        return orphaned

    def check_and_warn_orphaned(self) -> None:
        """Check for orphaned positions and log warnings."""
        orphaned = self.get_orphaned_positions()

        for strategy, positions in orphaned.items():
            logger.warning(f"[{strategy.upper()}] DISABLED with orphaned positions:")
            for symbol, data in positions.items():
                logger.warning(f"  {symbol}: {data['qty']} shares (entry: ${data['entry_price']:.2f})")
            logger.warning(f"Run: ./toggle_strategy.sh {strategy} close-orphaned")

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all strategies."""
        self._load_state()
        self.reload_toggle()

        status = {
            'timestamp': tz.iso_timestamp(),
            'execution_lock': self._state.get('execution_lock'),
            'strategies': {}
        }

        for strategy in ['omr', 'mp', 'ramp', 'cscm']:
            toggle = self._toggle.get('strategies', {}).get(strategy, {})
            state = self._state.get('strategies', {}).get(strategy, {})

            positions = state.get('positions', {})

            status['strategies'][strategy] = {
                'enabled': toggle.get('enabled', False),
                'shutdown_requested': toggle.get('shutdown_requested', False),
                'last_execution': state.get('last_execution'),
                'position_count': len(positions),
                'positions': list(positions.keys())
            }

        return status

    def print_status(self) -> None:
        """Print formatted status to logger."""
        status = self.get_status()

        logger.info("=" * 50)
        logger.info("STRATEGY STATUS")
        logger.info("=" * 50)
        logger.info(f"Time: {status['timestamp']}")
        logger.info("")

        lock = status['execution_lock']
        if lock:
            logger.info(f"Execution Lock: {lock['holder']} (expires: {lock['expires']})")
        else:
            logger.info("Execution Lock: None")
        logger.info("")

        for strategy, data in status['strategies'].items():
            enabled_str = "ENABLED" if data['enabled'] else "DISABLED"
            logger.info(f"{strategy.upper()}: {enabled_str}")

            if data['shutdown_requested']:
                logger.warning(f"  Shutdown requested!")

            if data['last_execution']:
                logger.info(f"  Last execution: {data['last_execution']}")

            if data['positions']:
                logger.info(f"  Positions: {', '.join(data['positions'])}")
            else:
                logger.info("  Positions: None")

            logger.info("")

        # Check for orphaned
        orphaned = self.get_orphaned_positions()
        if orphaned:
            logger.warning("ORPHANED POSITIONS DETECTED:")
            for strategy, positions in orphaned.items():
                for symbol, data in positions.items():
                    logger.warning(f"  [{strategy}] {symbol}: {data['qty']} shares")
