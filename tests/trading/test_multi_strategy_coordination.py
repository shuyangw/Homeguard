"""
Tests for Multi-Strategy Coordination.

Validates that OMR and MP strategies can run simultaneously without conflicts.
Tests execution locks, position ownership, and universe isolation.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Set
import tempfile
import pytest

from src.trading.state import StrategyStateManager
from src.strategies.universe import ETFUniverse


class TestExecutionLock:
    """Tests for execution lock mechanism."""

    @pytest.fixture
    def temp_state_dir(self, tmp_path):
        """Create temporary directory for state files."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"
        return state_file, toggle_file

    @pytest.fixture
    def manager(self, temp_state_dir):
        """Create a fresh StrategyStateManager with temp files."""
        state_file, toggle_file = temp_state_dir
        return StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    def test_acquire_lock_when_available(self, manager):
        """Test acquiring lock when no one holds it."""
        result = manager.acquire_execution_lock('omr', timeout=2)
        assert result is True
        assert manager.get_execution_lock_holder() == 'omr'
        manager.release_execution_lock('omr')

    def test_mutual_exclusion(self, manager):
        """Only one strategy can hold the lock at a time."""
        # OMR acquires lock
        assert manager.acquire_execution_lock('omr', timeout=2) is True

        # MP should fail to acquire (short timeout)
        assert manager.acquire_execution_lock('mp', timeout=2) is False

        # OMR still holds it
        assert manager.get_execution_lock_holder() == 'omr'

        # Release and MP should succeed
        manager.release_execution_lock('omr')
        assert manager.acquire_execution_lock('mp', timeout=2) is True
        manager.release_execution_lock('mp')

    def test_reentrant_lock(self, manager):
        """Same strategy can acquire lock multiple times."""
        assert manager.acquire_execution_lock('omr', timeout=2) is True
        assert manager.acquire_execution_lock('omr', timeout=2) is True
        assert manager.get_execution_lock_holder() == 'omr'
        manager.release_execution_lock('omr')

    def test_release_only_by_holder(self, manager):
        """Strategy can only release lock it holds."""
        manager.acquire_execution_lock('omr', timeout=2)

        # MP tries to release OMR's lock - should not work
        manager.release_execution_lock('mp')

        # OMR should still hold it
        assert manager.get_execution_lock_holder() == 'omr'

        # OMR releases
        manager.release_execution_lock('omr')
        assert manager.get_execution_lock_holder() is None

    def test_lock_released_after_execution(self, manager):
        """Lock should be properly released after use."""
        manager.acquire_execution_lock('omr', timeout=2)
        manager.release_execution_lock('omr')
        assert manager.get_execution_lock_holder() is None

    def test_second_strategy_waits_and_acquires(self, temp_state_dir):
        """Test that second strategy can acquire after first releases."""
        state_file, toggle_file = temp_state_dir
        results = {'omr_acquired': False, 'mp_acquired': False, 'order': []}

        def omr_execution():
            mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
            if mgr.acquire_execution_lock('omr', timeout=10):
                results['omr_acquired'] = True
                results['order'].append('omr')
                time.sleep(1)  # Simulate execution
                mgr.release_execution_lock('omr')

        def mp_execution():
            time.sleep(0.2)  # Start slightly after OMR
            mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
            if mgr.acquire_execution_lock('mp', timeout=10):
                results['mp_acquired'] = True
                results['order'].append('mp')
                mgr.release_execution_lock('mp')

        t1 = threading.Thread(target=omr_execution)
        t2 = threading.Thread(target=mp_execution)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results['omr_acquired'] is True
        assert results['mp_acquired'] is True
        assert results['order'] == ['omr', 'mp']  # OMR first, then MP


class TestPositionOwnership:
    """Tests for position tracking and ownership."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a fresh StrategyStateManager with temp files."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"
        return StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    def test_add_position(self, manager):
        """Test adding a position."""
        manager.add_position('omr', 'TQQQ', 100, 45.00, 'order123')

        positions = manager.get_positions('omr')
        assert 'TQQQ' in positions
        assert positions['TQQQ']['qty'] == 100
        assert positions['TQQQ']['entry_price'] == 45.00
        assert positions['TQQQ']['order_id'] == 'order123'

    def test_remove_position(self, manager):
        """Test removing a position."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)
        manager.remove_position('omr', 'TQQQ')

        positions = manager.get_positions('omr')
        assert 'TQQQ' not in positions

    def test_symbol_owned_by_other_detection(self, manager):
        """Test detecting when symbol is owned by another strategy."""
        # OMR takes position in TQQQ
        manager.add_position('omr', 'TQQQ', 100, 45.00)

        # MP should see TQQQ is owned by OMR
        owner = manager.symbol_owned_by_other('mp', 'TQQQ')
        assert owner == 'omr'

        # OMR should not see its own position as "other"
        owner = manager.symbol_owned_by_other('omr', 'TQQQ')
        assert owner is None

    def test_separate_positions_per_strategy(self, manager):
        """Each strategy has separate position tracking."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)
        manager.add_position('mp', 'AAPL', 50, 150.00)

        omr_positions = manager.get_positions('omr')
        mp_positions = manager.get_positions('mp')

        assert 'TQQQ' in omr_positions
        assert 'AAPL' not in omr_positions
        assert 'AAPL' in mp_positions
        assert 'TQQQ' not in mp_positions

    def test_update_position_qty(self, manager):
        """Test updating position quantity (partial close)."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)
        manager.update_position_qty('omr', 'TQQQ', 60)

        positions = manager.get_positions('omr')
        assert positions['TQQQ']['qty'] == 60

    def test_has_position(self, manager):
        """Test checking if strategy has position."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)

        assert manager.has_position('omr', 'TQQQ') is True
        assert manager.has_position('omr', 'SOXL') is False
        assert manager.has_position('mp', 'TQQQ') is False


class TestBrokerSync:
    """Tests for broker position synchronization."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a fresh StrategyStateManager with temp files."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"
        return StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    def test_sync_detects_closed_position(self, manager):
        """Sync should detect externally closed positions."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)

        # Broker says no TQQQ position
        broker_positions = {}
        changes = manager.sync_with_broker(broker_positions)

        assert 'omr:TQQQ' in changes['removed']
        assert manager.has_position('omr', 'TQQQ') is False

    def test_sync_detects_partial_close(self, manager):
        """Sync should detect partially closed positions."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)

        # Broker says only 60 shares
        broker_positions = {'TQQQ': 60}
        changes = manager.sync_with_broker(broker_positions)

        assert 'omr:TQQQ' in changes['updated']
        assert manager.get_position_qty('omr', 'TQQQ') == 60

    def test_sync_no_changes(self, manager):
        """Sync should return empty when positions match."""
        manager.add_position('omr', 'TQQQ', 100, 45.00)

        broker_positions = {'TQQQ': 100}
        changes = manager.sync_with_broker(broker_positions)

        assert len(changes['removed']) == 0
        assert len(changes['updated']) == 0


class TestUniverseIsolation:
    """Tests for universe isolation between strategies."""

    def test_omr_universe_is_leveraged_etfs(self):
        """OMR should trade leveraged 3x ETFs."""
        omr_symbols = set(ETFUniverse.LEVERAGED_3X)

        # Verify these are leveraged ETFs
        leveraged_indicators = ['3X', 'TQQ', 'SOX', 'UPR', 'SPX', 'TMF', 'TMV']
        for symbol in omr_symbols:
            is_leveraged = any(ind in symbol for ind in leveraged_indicators) or ETFUniverse.is_leveraged(symbol)
            assert is_leveraged, f"{symbol} should be a leveraged ETF"

    def test_mp_filters_out_leveraged(self):
        """MP should filter out leveraged ETFs from its universe."""
        # Simulate MP's filtering logic
        sample_sp500 = ['AAPL', 'MSFT', 'GOOGL', 'TQQQ', 'SOXL', 'NVDA']
        filtered = [s for s in sample_sp500 if not ETFUniverse.is_leveraged(s)]

        assert 'AAPL' in filtered
        assert 'MSFT' in filtered
        assert 'GOOGL' in filtered
        assert 'NVDA' in filtered
        assert 'TQQQ' not in filtered
        assert 'SOXL' not in filtered

    def test_no_universe_overlap(self):
        """OMR and MP universes should not overlap."""
        omr_symbols = set(ETFUniverse.LEVERAGED_3X)

        # Sample S&P 500 symbols (filtered)
        sample_sp500 = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'BRK.B', 'JPM', 'JNJ', 'V', 'PG', 'XOM', 'UNH', 'MA'
        ]
        mp_symbols = set(s for s in sample_sp500 if not ETFUniverse.is_leveraged(s))

        overlap = omr_symbols & mp_symbols
        assert len(overlap) == 0, f"Universe overlap detected: {overlap}"


class TestToggleConfiguration:
    """Tests for strategy toggle configuration."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a fresh StrategyStateManager with temp files."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"
        return StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    def test_enable_disable_strategy(self, manager):
        """Test enabling and disabling strategies."""
        manager.set_enabled('mp', True)
        assert manager.is_enabled('mp') is True

        manager.set_enabled('mp', False)
        assert manager.is_enabled('mp') is False

    def test_shutdown_requested_flag(self, manager):
        """Test shutdown requested flag."""
        manager.set_shutdown_requested('mp', True)
        assert manager.is_shutdown_requested('mp') is True

        manager.set_shutdown_requested('mp', False)
        assert manager.is_shutdown_requested('mp') is False

    def test_get_enabled_strategies(self, manager):
        """Test getting list of enabled strategies."""
        manager.set_enabled('omr', True)
        manager.set_enabled('mp', False)

        enabled = manager.get_enabled_strategies()
        assert 'omr' in enabled
        assert 'mp' not in enabled


class TestConcurrentExecution:
    """Tests for concurrent strategy execution scenarios."""

    def test_concurrent_lock_acquisition(self, tmp_path):
        """Test that concurrent lock requests are properly serialized."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"

        results = {'first': None, 'second': None}
        lock = threading.Lock()

        def try_acquire(name, delay):
            time.sleep(delay)
            mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
            acquired = mgr.acquire_execution_lock(name, timeout=5)
            with lock:
                if results['first'] is None:
                    results['first'] = name
                else:
                    results['second'] = name
            if acquired:
                time.sleep(0.5)
                mgr.release_execution_lock(name)

        # Start both nearly simultaneously
        t1 = threading.Thread(target=try_acquire, args=('omr', 0))
        t2 = threading.Thread(target=try_acquire, args=('mp', 0.01))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should have gotten a chance
        assert results['first'] is not None
        assert results['second'] is not None
        # They should be different
        assert results['first'] != results['second']

    def test_position_ownership_prevents_conflict(self, tmp_path):
        """Test that position ownership check prevents conflicts."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"

        mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

        # OMR takes position in a symbol
        mgr.add_position('omr', 'TEST', 100, 50.00)

        # Simulate MP checking before trading
        owner = mgr.symbol_owned_by_other('mp', 'TEST')
        assert owner == 'omr'

        # MP should skip this symbol
        should_trade = owner is None
        assert should_trade is False


class TestStateFileIntegrity:
    """Tests for state file integrity and recovery."""

    def test_atomic_write(self, tmp_path):
        """Test that state writes are atomic."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"

        mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

        # Add position
        mgr.add_position('omr', 'TQQQ', 100, 45.00)

        # Verify file exists and is valid JSON
        assert state_file.exists()
        with open(state_file) as f:
            data = json.load(f)
        assert 'strategies' in data
        assert 'omr' in data['strategies']

    def test_backup_creation(self, tmp_path):
        """Test that backups are created on startup."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"

        # First manager creates initial state
        mgr1 = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
        mgr1.add_position('omr', 'TQQQ', 100, 45.00)

        # Second manager should create backup
        mgr2 = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

        # Check for backup files
        backup_files = list(tmp_path.glob("*.bak"))
        assert len(backup_files) >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a fresh StrategyStateManager with temp files."""
        state_file = tmp_path / "strategy_positions.json"
        toggle_file = tmp_path / "strategy_toggle.yaml"
        return StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    def test_remove_nonexistent_position(self, manager):
        """Removing nonexistent position should not error."""
        manager.remove_position('omr', 'NONEXISTENT')
        # Should complete without error

    def test_release_unheld_lock(self, manager):
        """Releasing lock not held should not error."""
        manager.release_execution_lock('omr')
        # Should complete without error

    def test_get_positions_empty_strategy(self, manager):
        """Getting positions for strategy with none should return empty dict."""
        positions = manager.get_positions('omr')
        assert positions == {}

    def test_symbol_ownership_no_positions(self, manager):
        """Symbol ownership check with no positions should return None."""
        owner = manager.symbol_owned_by_other('mp', 'AAPL')
        assert owner is None
