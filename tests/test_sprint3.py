"""
Tests for Sprint 3: Advanced UX Features

Sprint 3 Features:
- D3: Keyboard shortcuts
- D4: Toast notifications
- C4: Run history tracking
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.utils.run_history import RunHistory


class TestRunHistory:
    """Test run history tracking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temp directory for history
        self.temp_dir = Path(tempfile.mkdtemp())
        self.history_file = self.temp_dir / "test_history.json"
        self.run_history = RunHistory(history_file=self.history_file)

    def teardown_method(self):
        """Clean up test files."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

    def test_add_run(self):
        """Test adding a backtest run to history."""
        self.run_history.add_run(
            strategy_name="MovingAverageCrossover",
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2024-01-01",
            config={"initial_capital": 100000, "fees": 0.001},
            results={"completed": 2, "failed": 0, "total": 2}
        )

        # Verify history file was created
        assert self.history_file.exists()

        # Load and check content
        runs = self.run_history.get_all_runs()
        assert len(runs) == 1
        assert runs[0]['strategy'] == "MovingAverageCrossover"
        assert runs[0]['symbols'] == ["AAPL", "MSFT"]
        assert runs[0]['num_symbols'] == 2

    def test_add_multiple_runs(self):
        """Test adding multiple runs."""
        for i in range(5):
            self.run_history.add_run(
                strategy_name=f"Strategy{i}",
                symbols=[f"SYM{i}"],
                start_date="2023-01-01",
                end_date="2024-01-01",
                config={},
                results={}
            )

        runs = self.run_history.get_all_runs()
        assert len(runs) == 5

        # Most recent should be first
        assert runs[0]['strategy'] == "Strategy4"
        assert runs[4]['strategy'] == "Strategy0"

    def test_run_has_timestamp(self):
        """Test that runs have timestamps."""
        self.run_history.add_run(
            strategy_name="TestStrategy",
            symbols=["TEST"],
            start_date="2023-01-01",
            end_date="2024-01-01",
            config={},
            results={}
        )

        runs = self.run_history.get_all_runs()
        assert 'timestamp' in runs[0]

        # Verify timestamp is valid ISO format
        timestamp = runs[0]['timestamp']
        datetime.fromisoformat(timestamp)  # Should not raise exception

    def test_get_recent_runs(self):
        """Test retrieving recent runs with limit."""
        # Add 15 runs
        for i in range(15):
            self.run_history.add_run(
                strategy_name=f"Strategy{i}",
                symbols=["TEST"],
                start_date="2023-01-01",
                end_date="2024-01-01",
                config={},
                results={}
            )

        # Get only 10 most recent
        recent = self.run_history.get_recent_runs(limit=10)
        assert len(recent) == 10
        assert recent[0]['strategy'] == "Strategy14"  # Most recent
        assert recent[9]['strategy'] == "Strategy5"

    def test_history_limit_50_runs(self):
        """Test that history is limited to 50 runs."""
        # Add 60 runs
        for i in range(60):
            self.run_history.add_run(
                strategy_name=f"Strategy{i}",
                symbols=["TEST"],
                start_date="2023-01-01",
                end_date="2024-01-01",
                config={},
                results={}
            )

        # Should only keep last 50
        runs = self.run_history.get_all_runs()
        assert len(runs) == 50

        # Should have most recent 50 (10-59)
        assert runs[0]['strategy'] == "Strategy59"
        assert runs[49]['strategy'] == "Strategy10"

    def test_clear_history(self):
        """Test clearing run history."""
        # Add some runs
        for i in range(5):
            self.run_history.add_run(
                strategy_name=f"Strategy{i}",
                symbols=["TEST"],
                start_date="2023-01-01",
                end_date="2024-01-01",
                config={},
                results={}
            )

        assert len(self.run_history.get_all_runs()) == 5

        # Clear history
        self.run_history.clear_history()
        assert len(self.run_history.get_all_runs()) == 0

    def test_run_stores_all_fields(self):
        """Test that all fields are stored correctly."""
        self.run_history.add_run(
            strategy_name="TestStrategy",
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date="2023-01-01",
            end_date="2024-01-01",
            config={
                "initial_capital": 50000,
                "fees": 0.002,
                "workers": 4
            },
            results={
                "completed": 2,
                "failed": 1,
                "total": 3
            }
        )

        runs = self.run_history.get_all_runs()
        run = runs[0]

        assert run['strategy'] == "TestStrategy"
        assert run['symbols'] == ["AAPL", "MSFT", "GOOGL"]
        assert run['num_symbols'] == 3
        assert run['start_date'] == "2023-01-01"
        assert run['end_date'] == "2024-01-01"
        assert run['initial_capital'] == 50000
        assert run['fees'] == 0.002
        assert run['workers'] == 4
        assert run['results']['completed'] == 2
        assert run['results']['failed'] == 1
        assert run['results']['total'] == 3

    def test_empty_history(self):
        """Test getting runs from empty history."""
        runs = self.run_history.get_all_runs()
        assert runs == []

        recent = self.run_history.get_recent_runs()
        assert recent == []

    def test_history_file_not_exists(self):
        """Test that missing history file is handled gracefully."""
        # Create new history with non-existent file
        non_existent = self.temp_dir / "does_not_exist.json"
        new_history = RunHistory(history_file=non_existent)

        # Should return empty list, not crash
        runs = new_history.get_all_runs()
        assert runs == []


class TestKeyboardShortcuts:
    """
    Test keyboard shortcut implementation.

    Note: Full keyboard testing requires Flet integration test,
    these tests verify the handler exists and is configured.
    """

    def test_keyboard_handler_exists(self):
        """Test that keyboard handler function exists in app."""
        from gui.app import BacktestApp

        # Verify the handler method exists
        assert hasattr(BacktestApp, '_on_keyboard')
        assert callable(getattr(BacktestApp, '_on_keyboard'))


class TestNotifications:
    """
    Test notification implementation.

    Note: Full notification testing requires Flet integration test,
    these tests verify the notification method exists.
    """

    def test_notification_method_exists(self):
        """Test that notification method exists in app."""
        from gui.app import BacktestApp

        # Verify the notification method exists
        assert hasattr(BacktestApp, '_show_notification')
        assert callable(getattr(BacktestApp, '_show_notification'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
