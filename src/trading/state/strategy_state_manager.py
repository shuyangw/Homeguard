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
        except Exception:
            pass
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
                        'mp': {'enabled': False, 'shutdown_requested': False}
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
        """Load state from JSON file with validation."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    self._state = json.load(f)

                # Validate structure
                if 'strategies' not in self._state:
                    self._state['strategies'] = {}
                if 'version' not in self._state:
                    self._state['version'] = 1

            else:
                # Initialize empty state
                self._state = {
                    'version': 1,
                    'last_updated': tz.iso_timestamp(),
                    'execution_lock': None,
                    'strategies': {
                        'omr': {'positions': {}, 'last_execution': None},
                        'mp': {'positions': {}, 'last_execution': None}
                    }
                }
                self._save_state()

        except json.JSONDecodeError as e:
            logger.error(f"State file corrupted: {e}")
            self._recover_from_backup()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self._state = {'strategies': {}}

    def _save_state(self) -> None:
        """Save state to JSON file atomically with file locking."""
        try:
            self._state['last_updated'] = tz.iso_timestamp()

            # Write to temp file
            temp_file = self.state_file.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self._state, f, indent=2)

            # Atomic replace
            os.replace(temp_file, self.state_file)

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
                except Exception:
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
            except Exception:
                pass

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
        order_id: Optional[str] = None
    ) -> None:
        """
        Add a position to a strategy's tracked positions.

        Args:
            strategy: Strategy name
            symbol: Stock symbol
            qty: Quantity
            entry_price: Entry price
            order_id: Optional order ID
        """
        self._load_state()

        if 'strategies' not in self._state:
            self._state['strategies'] = {}
        if strategy not in self._state['strategies']:
            self._state['strategies'][strategy] = {'positions': {}, 'last_execution': None}

        self._state['strategies'][strategy]['positions'][symbol] = {
            'qty': qty,
            'entry_price': entry_price,
            'entry_time': tz.iso_timestamp(),
            'order_id': order_id
        }

        self._save_state()
        logger.info(f"[{strategy.upper()}] Added position: {symbol} ({qty} shares @ ${entry_price:.2f})")

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
        order_id: Optional[str] = None
    ) -> int:
        """
        Add to existing position or create new one.

        CRITICAL: Use this for top-ups to avoid state drift!
        - If position exists: adds qty_delta to existing qty
        - If position doesn't exist: creates new position with qty_delta

        Args:
            strategy: Strategy name
            symbol: Stock symbol
            qty_delta: Quantity to ADD (not total)
            price: Current price (for new positions or logging)
            order_id: Optional order ID

        Returns:
            New total quantity after update
        """
        self._load_state()

        if 'strategies' not in self._state:
            self._state['strategies'] = {}
        if strategy not in self._state['strategies']:
            self._state['strategies'][strategy] = {'positions': {}, 'last_execution': None}

        positions = self._state['strategies'][strategy]['positions']

        if symbol in positions:
            # Existing position - ADD to qty (don't overwrite!)
            old_qty = positions[symbol]['qty']
            new_qty = old_qty + qty_delta
            positions[symbol]['qty'] = new_qty
            # Keep original entry_price and entry_time
            self._save_state()
            logger.info(
                f"[{strategy.upper()}] Topped up {symbol}: {old_qty} + {qty_delta} = {new_qty} shares"
            )
            return new_qty
        else:
            # New position
            positions[symbol] = {
                'qty': qty_delta,
                'entry_price': price,
                'entry_time': tz.iso_timestamp(),
                'order_id': order_id
            }
            self._save_state()
            logger.info(
                f"[{strategy.upper()}] Added position: {symbol} ({qty_delta} shares @ ${price:.2f})"
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
    # Broker Synchronization
    # =========================================================================

    def sync_with_broker(self, broker_positions: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Synchronize state with actual broker positions.

        Handles:
        - Positions closed externally (stop-loss, manual)
        - Positions partially closed
        - Positions increased externally (logs warning but updates to match)
        - STATE DRIFT DETECTION: When broker qty > state qty unexpectedly

        Args:
            broker_positions: Dict of symbol -> quantity from broker

        Returns:
            Dict with 'removed', 'updated', and 'drift_detected' lists of symbols
        """
        self._load_state()

        changes = {'removed': [], 'updated': [], 'drift_detected': []}

        for strategy, data in self._state.get('strategies', {}).items():
            positions = data.get('positions', {})

            for symbol in list(positions.keys()):
                state_qty = positions[symbol]['qty']
                broker_qty = broker_positions.get(symbol, 0)

                if broker_qty == 0:
                    # Position no longer exists at broker
                    logger.warning(f"[{strategy.upper()}] Position {symbol} closed externally")
                    del positions[symbol]
                    changes['removed'].append(f"{strategy}:{symbol}")

                elif broker_qty < state_qty:
                    # Position partially closed
                    logger.warning(f"[{strategy.upper()}] Position {symbol} reduced: {state_qty} -> {broker_qty}")
                    positions[symbol]['qty'] = broker_qty
                    changes['updated'].append(f"{strategy}:{symbol}")

                elif broker_qty > state_qty:
                    # STATE DRIFT DETECTED: Broker has MORE than state expected
                    # This indicates a bug in position tracking (e.g., top-up not recorded)
                    logger.error(
                        f"[{strategy.upper()}] STATE DRIFT DETECTED: {symbol} "
                        f"state={state_qty} but broker={broker_qty} (+{broker_qty - state_qty} untracked)"
                    )
                    # Update state to match broker (self-heal)
                    positions[symbol]['qty'] = broker_qty
                    changes['drift_detected'].append(f"{strategy}:{symbol}")

        if changes['removed'] or changes['updated'] or changes['drift_detected']:
            self._save_state()

        return changes

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

        for strategy in ['omr', 'mp']:
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
