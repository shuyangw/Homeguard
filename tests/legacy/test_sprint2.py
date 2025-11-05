"""
Tests for Sprint 2: Tooltips and Progress Estimates

Sprint 2 Features:
- D1: Tooltips on all main UI controls
- D2: Progress time tracking and estimates
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import flet as ft

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.views.setup_view import SetupView
from gui.views.run_view import RunView
from gui.utils.config_manager import ConfigManager
import shutil


class TestTooltips:
    """Test that all main UI controls have tooltips."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create SetupView (no page needed for property testing)
        self.setup_view = SetupView(on_run_clicked=lambda config: None)

    def test_strategy_dropdown_has_tooltip(self):
        """Test that strategy dropdown has a tooltip."""
        assert self.setup_view.strategy_dropdown.tooltip is not None
        assert len(self.setup_view.strategy_dropdown.tooltip) > 0
        assert "strategy" in self.setup_view.strategy_dropdown.tooltip.lower()

    def test_symbols_input_has_tooltip(self):
        """Test that symbols input has a tooltip."""
        assert self.setup_view.symbols_input.tooltip is not None
        assert len(self.setup_view.symbols_input.tooltip) > 0
        assert "symbol" in self.setup_view.symbols_input.tooltip.lower()

    def test_date_pickers_have_tooltips(self):
        """Test that date pickers have tooltips."""
        assert self.setup_view.start_date_picker.tooltip is not None
        assert len(self.setup_view.start_date_picker.tooltip) > 0

        assert self.setup_view.end_date_picker.tooltip is not None
        assert len(self.setup_view.end_date_picker.tooltip) > 0

    def test_capital_input_has_tooltip(self):
        """Test that initial capital input has a tooltip."""
        assert self.setup_view.initial_capital_input.tooltip is not None
        assert len(self.setup_view.initial_capital_input.tooltip) > 0
        assert "portfolio" in self.setup_view.initial_capital_input.tooltip.lower() or "capital" in self.setup_view.initial_capital_input.tooltip.lower()

    def test_fees_input_has_tooltip(self):
        """Test that fees input has a tooltip."""
        assert self.setup_view.fees_input.tooltip is not None
        assert len(self.setup_view.fees_input.tooltip) > 0
        assert "fee" in self.setup_view.fees_input.tooltip.lower()

    def test_workers_slider_has_tooltip(self):
        """Test that workers slider has a tooltip."""
        assert self.setup_view.workers_slider.tooltip is not None
        assert len(self.setup_view.workers_slider.tooltip) > 0
        assert "worker" in self.setup_view.workers_slider.tooltip.lower()

    def test_parallel_checkbox_has_tooltip(self):
        """Test that parallel execution checkbox has a tooltip."""
        assert self.setup_view.parallel_checkbox.tooltip is not None
        assert len(self.setup_view.parallel_checkbox.tooltip) > 0

    def test_run_button_has_tooltip(self):
        """Test that run button has a tooltip."""
        assert self.setup_view.run_button.tooltip is not None
        assert len(self.setup_view.run_button.tooltip) > 0

    def test_quick_rerun_button_has_tooltip(self):
        """Test that quick re-run button has a tooltip."""
        assert self.setup_view.quick_rerun_button.tooltip is not None
        assert len(self.setup_view.quick_rerun_button.tooltip) > 0
        assert "last" in self.setup_view.quick_rerun_button.tooltip.lower()

    def test_preset_controls_have_tooltips(self):
        """Test that preset controls have tooltips."""
        assert self.setup_view.preset_dropdown.tooltip is not None
        assert len(self.setup_view.preset_dropdown.tooltip) > 0

        assert self.setup_view.load_preset_button.tooltip is not None
        assert len(self.setup_view.load_preset_button.tooltip) > 0

        assert self.setup_view.save_preset_button.tooltip is not None
        assert len(self.setup_view.save_preset_button.tooltip) > 0

    def test_symbol_list_controls_have_tooltips(self):
        """Test that symbol list controls have tooltips."""
        assert self.setup_view.symbol_list_dropdown.tooltip is not None
        assert len(self.setup_view.symbol_list_dropdown.tooltip) > 0

        assert self.setup_view.load_symbol_list_button.tooltip is not None
        assert len(self.setup_view.load_symbol_list_button.tooltip) > 0

        assert self.setup_view.save_symbol_list_button.tooltip is not None
        assert len(self.setup_view.save_symbol_list_button.tooltip) > 0


class TestProgressEstimates:
    """Test progress time tracking and estimates."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create RunView (no page needed for property testing)
        self.run_view = RunView()

    def test_time_tracking_components_exist(self):
        """Test that time tracking UI components exist."""
        assert hasattr(self.run_view, 'time_elapsed_text')
        assert hasattr(self.run_view, 'time_remaining_text')
        assert hasattr(self.run_view, 'eta_text')
        assert isinstance(self.run_view.time_elapsed_text, ft.Text)
        assert isinstance(self.run_view.time_remaining_text, ft.Text)
        assert isinstance(self.run_view.eta_text, ft.Text)

    def test_time_tracking_state_variables_exist(self):
        """Test that time tracking state variables are initialized."""
        assert hasattr(self.run_view, 'start_time')
        assert hasattr(self.run_view, 'completion_times')
        assert hasattr(self.run_view, 'total_symbols')
        assert self.run_view.start_time is None
        assert isinstance(self.run_view.completion_times, list)
        assert self.run_view.total_symbols == 0

    def test_initialize_symbols_sets_total(self):
        """Test that initialize_symbols sets total_symbols correctly."""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        self.run_view.initialize_symbols(symbols)
        assert self.run_view.total_symbols == 4

    def test_mark_running_starts_time_tracking(self):
        """Test that mark_running initializes time tracking."""
        self.run_view.mark_running()
        assert self.run_view.start_time is not None
        assert isinstance(self.run_view.start_time, datetime)
        assert len(self.run_view.completion_times) == 0

    def test_update_time_estimates_with_no_start_time(self):
        """Test that update_time_estimates handles missing start_time gracefully."""
        self.run_view.start_time = None
        self.run_view.total_symbols = 5
        # Should not raise an error
        self.run_view.update_time_estimates(2)

    def test_update_time_estimates_with_zero_total(self):
        """Test that update_time_estimates handles zero total_symbols gracefully."""
        self.run_view.start_time = datetime.now()
        self.run_view.total_symbols = 0
        # Should not raise an error
        self.run_view.update_time_estimates(0)

    def test_update_time_estimates_calculates_elapsed(self):
        """Test that elapsed time is calculated and displayed."""
        # Set start time to 1 minute ago
        self.run_view.start_time = datetime.now() - timedelta(minutes=1)
        self.run_view.total_symbols = 5

        self.run_view.update_time_estimates(2)

        # Check that elapsed time is displayed
        assert self.run_view.time_elapsed_text.value is not None
        assert "Elapsed" in self.run_view.time_elapsed_text.value
        assert "0:01" in self.run_view.time_elapsed_text.value  # Should show ~1 minute

    def test_update_time_estimates_calculates_remaining(self):
        """Test that remaining time is calculated when progress > 0."""
        # Set start time to 2 minutes ago
        self.run_view.start_time = datetime.now() - timedelta(minutes=2)
        self.run_view.total_symbols = 4

        # 2 symbols completed in 2 minutes = 1 min/symbol
        # 2 symbols remaining = 2 minutes remaining
        self.run_view.update_time_estimates(2)

        # Check that remaining time is displayed
        assert self.run_view.time_remaining_text.value is not None
        assert "Remaining" in self.run_view.time_remaining_text.value

    def test_update_time_estimates_calculates_eta(self):
        """Test that ETA is calculated when progress > 0."""
        self.run_view.start_time = datetime.now() - timedelta(minutes=2)
        self.run_view.total_symbols = 4

        self.run_view.update_time_estimates(2)

        # Check that ETA is displayed
        assert self.run_view.eta_text.value is not None
        assert "ETA" in self.run_view.eta_text.value

    def test_update_time_estimates_with_zero_completed(self):
        """Test that estimates are cleared when no symbols completed."""
        self.run_view.start_time = datetime.now()
        self.run_view.total_symbols = 5

        self.run_view.update_time_estimates(0)

        # Elapsed should be shown, but remaining/ETA should be empty
        assert "Elapsed" in self.run_view.time_elapsed_text.value
        assert self.run_view.time_remaining_text.value == ""
        assert self.run_view.eta_text.value == ""

    def test_update_overall_progress_calls_time_estimates(self):
        """Test that update_overall_progress triggers time tracking."""
        self.run_view.start_time = datetime.now()
        self.run_view.total_symbols = 5

        # This should call update_time_estimates internally
        self.run_view.update_overall_progress(completed=2, total=5, running=1, failed=0)

        # Verify time tracking was updated
        assert self.run_view.time_elapsed_text.value is not None
        assert "Elapsed" in self.run_view.time_elapsed_text.value

    def test_mark_complete_shows_final_time(self):
        """Test that mark_complete displays total execution time."""
        # Set start time to 5 minutes ago
        self.run_view.start_time = datetime.now() - timedelta(minutes=5)

        self.run_view.mark_complete()

        # Check that final time is displayed
        assert self.run_view.time_elapsed_text.value is not None
        assert "Total Time" in self.run_view.time_elapsed_text.value
        assert self.run_view.time_remaining_text.value == ""
        assert "Complete" in self.run_view.eta_text.value

    def test_mark_complete_without_start_time(self):
        """Test that mark_complete handles missing start_time gracefully."""
        self.run_view.start_time = None
        # Should not raise an error
        self.run_view.mark_complete()
        # Status should still be updated
        assert "Complete" in self.run_view.progress_text.value


class TestTimeEstimateAccuracy:
    """Test accuracy of time estimate calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_view = RunView()

    def test_estimate_accuracy_half_complete(self):
        """Test time estimates when half of symbols are completed."""
        # Start time: 10 minutes ago
        # Completed: 5 out of 10 symbols
        # Expected: 10 more minutes remaining (2 min/symbol × 5 remaining)

        self.run_view.start_time = datetime.now() - timedelta(minutes=10)
        self.run_view.total_symbols = 10

        self.run_view.update_time_estimates(5)

        # Verify elapsed is ~10 minutes
        assert "0:10" in self.run_view.time_elapsed_text.value

        # Verify remaining is calculated (should be ~10 minutes)
        assert "Remaining" in self.run_view.time_remaining_text.value

    def test_estimate_accuracy_nearly_complete(self):
        """Test time estimates when nearly complete."""
        # Start time: 9 minutes ago
        # Completed: 9 out of 10 symbols
        # Expected: 1 more minute remaining (1 min/symbol × 1 remaining)

        self.run_view.start_time = datetime.now() - timedelta(minutes=9)
        self.run_view.total_symbols = 10

        self.run_view.update_time_estimates(9)

        # Verify estimates are reasonable
        assert "Elapsed" in self.run_view.time_elapsed_text.value
        assert "Remaining" in self.run_view.time_remaining_text.value
        assert "ETA" in self.run_view.eta_text.value

    def test_estimate_updates_as_progress_increases(self):
        """Test that estimates update correctly as more symbols complete."""
        self.run_view.start_time = datetime.now() - timedelta(minutes=6)
        self.run_view.total_symbols = 6

        # After 1 symbol (1 minute elapsed per symbol)
        self.run_view.update_time_estimates(1)
        first_remaining = self.run_view.time_remaining_text.value

        # After 3 symbols (should show less remaining)
        self.run_view.update_time_estimates(3)
        second_remaining = self.run_view.time_remaining_text.value

        # Both should have remaining time estimates
        assert "Remaining" in first_remaining
        assert "Remaining" in second_remaining
        # Estimates should be different
        assert first_remaining != second_remaining


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
