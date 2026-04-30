"""
Unit tests for strategy state tracking.

Tests for:
- Position tracking (add, update, remove)
- Top-up behavior (add_or_update_position)
- Broker synchronization and drift detection
- Cross-strategy isolation
- End-to-end scenarios
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime

from src.trading.state.strategy_state_manager import StrategyStateManager


@pytest.fixture
def temp_state_dir():
    """Create temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def state_manager(temp_state_dir):
    """Create state manager with temporary files."""
    state_file = Path(temp_state_dir) / "strategy_positions.json"
    toggle_file = Path(temp_state_dir) / "strategy_toggle.yaml"

    # Create initial toggle file
    with open(toggle_file, 'w') as f:
        f.write("strategies:\n  mp:\n    enabled: true\n  omr:\n    enabled: true\n")

    # Create state manager with temp file paths
    manager = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
    # Clear any existing state
    manager._state = {'strategies': {}}
    manager._save_state()
    yield manager


class TestAddPosition:
    """Tests for basic add_position functionality."""

    def test_add_new_position(self, state_manager):
        """Test adding a brand new position."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, 'order_123', broker='alpaca')

        positions = state_manager.get_positions('mp')
        assert 'AAPL' in positions
        assert positions['AAPL']['qty'] == 100
        assert positions['AAPL']['entry_price'] == 150.0
        assert positions['AAPL']['order_id'] == 'order_123'

    def test_add_position_overwrites_existing(self, state_manager):
        """Test that add_position OVERWRITES existing position (the bug we fixed)."""
        # Add initial position
        state_manager.add_position('mp', 'AAPL', 100, 150.0, 'order_1', broker='alpaca')

        # Add again with different qty - this OVERWRITES
        state_manager.add_position('mp', 'AAPL', 50, 160.0, 'order_2', broker='alpaca')

        positions = state_manager.get_positions('mp')
        # BUG BEHAVIOR: qty is 50, not 150!
        assert positions['AAPL']['qty'] == 50
        assert positions['AAPL']['entry_price'] == 160.0

    def test_add_position_different_strategies(self, state_manager):
        """Test positions are isolated between strategies."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        state_manager.add_position('omr', 'TQQQ', 50, 45.0, broker='alpaca')

        mp_positions = state_manager.get_positions('mp')
        omr_positions = state_manager.get_positions('omr')

        assert 'AAPL' in mp_positions
        assert 'TQQQ' not in mp_positions
        assert 'TQQQ' in omr_positions
        assert 'AAPL' not in omr_positions


class TestAddOrUpdatePosition:
    """Tests for the safe add_or_update_position method."""

    def test_add_or_update_new_position(self, state_manager):
        """Test creating a new position."""
        total = state_manager.add_or_update_position('mp', 'AAPL', 100, 150.0, 'order_1', broker='alpaca')

        assert total == 100
        positions = state_manager.get_positions('mp')
        assert positions['AAPL']['qty'] == 100
        assert positions['AAPL']['entry_price'] == 150.0

    def test_add_or_update_topup_existing(self, state_manager):
        """Test topping up an existing position adds to qty."""
        # Create initial position
        state_manager.add_or_update_position('mp', 'AAPL', 100, 150.0, 'order_1', broker='alpaca')

        # Top up with additional shares
        total = state_manager.add_or_update_position('mp', 'AAPL', 25, 145.0, 'order_2')

        # Should have 125 total (100 + 25)
        assert total == 125
        positions = state_manager.get_positions('mp')
        assert positions['AAPL']['qty'] == 125
        # Entry price should remain original
        assert positions['AAPL']['entry_price'] == 150.0

    def test_add_or_update_multiple_topups(self, state_manager):
        """Test multiple top-ups accumulate correctly."""
        state_manager.add_or_update_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        state_manager.add_or_update_position('mp', 'AAPL', 20, 145.0)
        state_manager.add_or_update_position('mp', 'AAPL', 30, 155.0)

        positions = state_manager.get_positions('mp')
        assert positions['AAPL']['qty'] == 150  # 100 + 20 + 30

    def test_add_or_update_preserves_entry_time(self, state_manager):
        """Test that original entry_time is preserved on top-up."""
        state_manager.add_or_update_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        original_time = state_manager.get_positions('mp')['AAPL']['entry_time']

        # Wait a moment and top up
        import time
        time.sleep(0.1)
        state_manager.add_or_update_position('mp', 'AAPL', 25, 145.0)

        new_time = state_manager.get_positions('mp')['AAPL']['entry_time']
        assert new_time == original_time


class TestSyncWithBroker:
    """Tests for broker synchronization and drift detection."""

    def test_sync_removes_closed_positions(self, state_manager):
        """Test that positions closed at broker are removed from state."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        state_manager.add_position('mp', 'MSFT', 50, 300.0, broker='alpaca')

        # Broker only has MSFT
        broker_positions = {'MSFT': 50}

        changes = state_manager.sync_with_broker('alpaca', broker_positions)

        assert 'mp:AAPL' in changes['removed']
        positions = state_manager.get_positions('mp')
        assert 'AAPL' not in positions
        assert 'MSFT' in positions

    def test_sync_updates_partial_closes(self, state_manager):
        """Test that partial closes update state qty."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        # Broker shows only 60 shares (partial close)
        broker_positions = {'AAPL': 60}

        changes = state_manager.sync_with_broker('alpaca', broker_positions)

        assert 'mp:AAPL' in changes['updated']
        positions = state_manager.get_positions('mp')
        assert positions['AAPL']['qty'] == 60

    def test_sync_detects_drift_broker_higher(self, state_manager):
        """Test that state drift is detected when broker has MORE than state."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        # Broker shows 150 shares - MORE than state expected
        # This indicates a tracking bug (e.g., top-up not recorded)
        broker_positions = {'AAPL': 150}

        changes = state_manager.sync_with_broker('alpaca', broker_positions)

        assert 'mp:AAPL' in changes['drift_detected']
        # State should be healed to match broker
        positions = state_manager.get_positions('mp')
        assert positions['AAPL']['qty'] == 150

    def test_sync_no_changes_when_matched(self, state_manager):
        """Test that no changes when state matches broker."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        broker_positions = {'AAPL': 100}

        changes = state_manager.sync_with_broker('alpaca', broker_positions)

        assert len(changes['removed']) == 0
        assert len(changes['updated']) == 0
        assert len(changes['drift_detected']) == 0

    def test_sync_multiple_strategies(self, state_manager):
        """Test sync handles multiple strategies correctly."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        state_manager.add_position('omr', 'TQQQ', 50, 45.0, broker='alpaca')

        # Broker has both
        broker_positions = {'AAPL': 100, 'TQQQ': 50}

        changes = state_manager.sync_with_broker('alpaca', broker_positions)

        # No changes expected
        assert len(changes['removed']) == 0
        assert len(changes['updated']) == 0


class TestCrossStrategyIsolation:
    """Tests for multi-strategy isolation."""

    def test_symbol_owned_by_other_returns_owner(self, state_manager):
        """Test detecting when another strategy owns a symbol."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        owner = state_manager.symbol_owned_by_other('omr', 'AAPL')

        assert owner == 'mp'

    def test_symbol_owned_by_other_returns_none_if_not_owned(self, state_manager):
        """Test None returned when symbol not owned by another."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        owner = state_manager.symbol_owned_by_other('omr', 'MSFT')

        assert owner is None

    def test_symbol_owned_by_self_returns_none(self, state_manager):
        """Test that checking own positions returns None."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        owner = state_manager.symbol_owned_by_other('mp', 'AAPL')

        assert owner is None

    def test_remove_position_only_affects_own_strategy(self, state_manager):
        """Test removing position doesn't affect other strategies."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        state_manager.add_position('omr', 'AAPL', 50, 160.0, broker='alpaca')  # Same symbol, different strategy

        state_manager.remove_position('mp', 'AAPL')

        mp_positions = state_manager.get_positions('mp')
        omr_positions = state_manager.get_positions('omr')

        assert 'AAPL' not in mp_positions
        assert 'AAPL' in omr_positions
        assert omr_positions['AAPL']['qty'] == 50


class TestMPTopUpScenario:
    """End-to-end test for MP top-up scenario that was causing drift."""

    def test_mp_topup_maintains_correct_qty(self, state_manager):
        """
        Simulate MP rebalance top-up scenario:
        1. Day 1: Buy 100 shares of AAPL at $150
        2. Day 2: Price drops, need to buy 25 more to reach target
        3. State should show 125 total shares
        """
        # Day 1: Initial buy
        state_manager.add_or_update_position('mp', 'AAPL', 100, 150.0, 'order_1', broker='alpaca')

        positions = state_manager.get_positions('mp')
        assert positions['AAPL']['qty'] == 100

        # Day 2: Top-up (using add_or_update_position, not add_position)
        state_manager.add_or_update_position('mp', 'AAPL', 25, 140.0, 'order_2')

        positions = state_manager.get_positions('mp')
        # CRITICAL: Should be 125, not 25!
        assert positions['AAPL']['qty'] == 125
        # Entry price should remain original $150
        assert positions['AAPL']['entry_price'] == 150.0

    def test_mp_topup_with_old_add_position_causes_drift(self, state_manager):
        """
        Demonstrate the bug: using add_position for top-ups causes drift.
        This test documents the bug behavior before it was fixed.
        """
        # Day 1: Initial buy
        state_manager.add_position('mp', 'AAPL', 100, 150.0, 'order_1', broker='alpaca')

        # Day 2: Top-up using OLD buggy method (add_position)
        state_manager.add_position('mp', 'AAPL', 25, 140.0, 'order_2', broker='alpaca')

        positions = state_manager.get_positions('mp')
        # BUG: Shows 25 instead of 125!
        assert positions['AAPL']['qty'] == 25  # This is the bug!

        # Sync with broker would detect drift
        broker_positions = {'AAPL': 125}  # Broker has correct qty
        changes = state_manager.sync_with_broker('alpaca', broker_positions)

        # Drift should be detected and healed
        assert 'mp:AAPL' in changes['drift_detected']
        assert state_manager.get_positions('mp')['AAPL']['qty'] == 125


class TestOMRPositionTracking:
    """Tests for OMR-specific position tracking."""

    def test_omr_open_and_close_cycle(self, state_manager):
        """Test OMR daily cycle: open at 3:50 PM, close at 9:31 AM."""
        # 3:50 PM: Open positions
        state_manager.add_or_update_position('omr', 'TQQQ', 100, 45.0, broker='alpaca')
        state_manager.add_or_update_position('omr', 'SQQQ', 50, 12.0, broker='alpaca')

        positions = state_manager.get_positions('omr')
        assert len(positions) == 2

        # 9:31 AM next day: Close all positions
        state_manager.remove_position('omr', 'TQQQ')
        state_manager.remove_position('omr', 'SQQQ')

        positions = state_manager.get_positions('omr')
        assert len(positions) == 0

    def test_omr_positions_dont_affect_mp(self, state_manager):
        """Test OMR positions don't interfere with MP."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')
        state_manager.add_position('omr', 'TQQQ', 50, 45.0, broker='alpaca')

        # Remove all OMR positions
        state_manager.remove_position('omr', 'TQQQ')

        # MP should be unaffected
        mp_positions = state_manager.get_positions('mp')
        assert 'AAPL' in mp_positions
        assert mp_positions['AAPL']['qty'] == 100


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_positions_empty_strategy(self, state_manager):
        """Test getting positions for strategy with no positions."""
        positions = state_manager.get_positions('mp')
        assert positions == {}

    def test_get_positions_nonexistent_strategy(self, state_manager):
        """Test getting positions for strategy that doesn't exist."""
        positions = state_manager.get_positions('nonexistent')
        assert positions == {}

    def test_remove_nonexistent_position(self, state_manager):
        """Test removing position that doesn't exist (should not error)."""
        # Should not raise
        state_manager.remove_position('mp', 'NONEXISTENT')

    def test_has_position(self, state_manager):
        """Test has_position helper."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        assert state_manager.has_position('mp', 'AAPL') is True
        assert state_manager.has_position('mp', 'MSFT') is False
        assert state_manager.has_position('omr', 'AAPL') is False

    def test_get_position_qty(self, state_manager):
        """Test get_position_qty helper."""
        state_manager.add_position('mp', 'AAPL', 100, 150.0, broker='alpaca')

        assert state_manager.get_position_qty('mp', 'AAPL') == 100
        assert state_manager.get_position_qty('mp', 'MSFT') == 0
        assert state_manager.get_position_qty('omr', 'AAPL') == 0


class TestRunnerSessionState:
    """Tests for persisted runner session state (peak equity, day-open equity)."""

    def test_get_returns_none_when_unset(self, state_manager):
        out = state_manager.get_runner_session_state('ramp')
        assert out['peak_equity_usd'] is None
        assert out['session_open_equity_usd'] is None
        assert out['session_open_date'] is None

    def test_round_trip(self, state_manager):
        state_manager.update_runner_session_state(
            'ramp',
            peak_equity_usd=1023456.78,
            session_open_equity_usd=1014351.11,
            session_open_date='2026-04-28',
        )
        out = state_manager.get_runner_session_state('ramp')
        assert out['peak_equity_usd'] == 1023456.78
        assert out['session_open_equity_usd'] == 1014351.11
        assert out['session_open_date'] == '2026-04-28'

    def test_partial_update_preserves_other_fields(self, state_manager):
        state_manager.update_runner_session_state(
            'ramp',
            peak_equity_usd=1000.0,
            session_open_equity_usd=900.0,
            session_open_date='2026-04-28',
        )
        # Update only the peak; other fields should stay
        state_manager.update_runner_session_state('ramp', peak_equity_usd=1100.0)
        out = state_manager.get_runner_session_state('ramp')
        assert out['peak_equity_usd'] == 1100.0
        assert out['session_open_equity_usd'] == 900.0  # unchanged
        assert out['session_open_date'] == '2026-04-28'  # unchanged

    def test_strategies_isolated(self, state_manager):
        state_manager.update_runner_session_state('ramp', peak_equity_usd=1000.0)
        state_manager.update_runner_session_state('cscm', peak_equity_usd=2000.0)
        assert state_manager.get_runner_session_state('ramp')['peak_equity_usd'] == 1000.0
        assert state_manager.get_runner_session_state('cscm')['peak_equity_usd'] == 2000.0

    def test_persists_across_manager_reload(self, state_manager, temp_state_dir):
        """Write via one manager, read via a fresh manager pointed at the same files."""
        state_manager.update_runner_session_state(
            'ramp',
            peak_equity_usd=1023456.78,
            session_open_equity_usd=1014351.11,
            session_open_date='2026-04-28',
        )
        # Spin up a fresh manager on the same files (simulates process restart)
        from pathlib import Path
        state_file = Path(temp_state_dir) / "strategy_positions.json"
        toggle_file = Path(temp_state_dir) / "strategy_toggle.yaml"
        fresh = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
        out = fresh.get_runner_session_state('ramp')
        assert out['peak_equity_usd'] == 1023456.78
        assert out['session_open_equity_usd'] == 1014351.11
        assert out['session_open_date'] == '2026-04-28'

    def test_unknown_strategy_creates_entry(self, state_manager):
        """Strategies not pre-seeded should still accept session state."""
        state_manager.update_runner_session_state('newstrat', peak_equity_usd=500.0)
        out = state_manager.get_runner_session_state('newstrat')
        assert out['peak_equity_usd'] == 500.0
