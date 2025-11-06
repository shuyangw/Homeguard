"""
Unit tests for OptimizationDialog GUI component.

Tests parameter grid collection, combination estimation, and dialog behavior.
Note: These tests focus on the data logic, not the Flet UI rendering.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from gui.views.optimization_dialog import OptimizationDialog


class MockTextField:
    """Mock Flet TextField for testing."""
    def __init__(self, value: str = ""):
        self.value = value
        self.label = ""
        self.hint_text = ""
        self.keyboard_type = None
        self.width = None

    def strip(self):
        """Mock string strip() for compatibility."""
        return self.value.strip()


class MockCheckbox:
    """Mock Flet Checkbox for testing."""
    def __init__(self, value: bool = False):
        self.value = value
        self.label = ""


class MockPage:
    """Mock Flet Page for testing."""
    def __init__(self):
        self.dialogs = []

    def open(self, dialog):
        self.dialogs.append(dialog)

    def close(self, dialog):
        if dialog in self.dialogs:
            self.dialogs.remove(dialog)


class TestParameterGridCollection:
    """Test parameter grid collection from UI controls."""

    def test_collect_numeric_parameter_range(self):
        """Test collecting numeric parameter range (min, max, step)."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'fast_window': 10},
            param_types={'fast_window': int},
            on_optimize=lambda grid, metric: None
        )

        # Simulate user input for numeric range
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField('5'),
            'max': MockTextField('15'),
            'step': MockTextField('5'),
            'type': 'numeric'
        }

        param_grid = dialog._collect_param_grid()

        assert 'fast_window' in param_grid
        assert param_grid['fast_window'] == [5, 10, 15]

    def test_collect_boolean_parameter(self):
        """Test collecting boolean parameter (test both True/False)."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'use_stops': True},
            param_types={'use_stops': bool},
            on_optimize=lambda grid, metric: None
        )

        # Simulate checkbox checked
        dialog.grid_controls['use_stops'] = {
            'checkbox': MockCheckbox(True),
            'type': 'bool'
        }

        param_grid = dialog._collect_param_grid()

        assert 'use_stops' in param_grid
        assert param_grid['use_stops'] == [True, False]

    def test_collect_boolean_parameter_unchecked(self):
        """Test boolean parameter when checkbox unchecked (use current only)."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'use_stops': True},
            param_types={'use_stops': bool},
            on_optimize=lambda grid, metric: None
        )

        # Simulate checkbox unchecked
        dialog.grid_controls['use_stops'] = {
            'checkbox': MockCheckbox(False),
            'type': 'bool'
        }

        param_grid = dialog._collect_param_grid()

        # Should not include parameter if checkbox unchecked
        assert 'use_stops' not in param_grid

    def test_collect_value_list_parameter(self):
        """Test collecting comma-separated value list."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'ma_type': 'sma'},
            param_types={'ma_type': str},
            on_optimize=lambda grid, metric: None
        )

        # Simulate comma-separated values
        dialog.grid_controls['ma_type'] = {
            'values': MockTextField('sma, ema, wma'),
            'type': 'values'
        }

        param_grid = dialog._collect_param_grid()

        assert 'ma_type' in param_grid
        assert param_grid['ma_type'] == ['sma', 'ema', 'wma']

    def test_collect_mixed_parameters(self):
        """Test collecting multiple different parameter types together."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={
                'fast_window': 10,
                'slow_window': 20,
                'use_stops': False,
                'ma_type': 'sma'
            },
            param_types={
                'fast_window': int,
                'slow_window': int,
                'use_stops': bool,
                'ma_type': str
            },
            on_optimize=lambda grid, metric: None
        )

        # Set up mixed controls
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField('10'),
            'max': MockTextField('20'),
            'step': MockTextField('5'),
            'type': 'numeric'
        }
        dialog.grid_controls['slow_window'] = {
            'min': MockTextField('40'),
            'max': MockTextField('60'),
            'step': MockTextField('10'),
            'type': 'numeric'
        }
        dialog.grid_controls['use_stops'] = {
            'checkbox': MockCheckbox(True),
            'type': 'bool'
        }
        dialog.grid_controls['ma_type'] = {
            'values': MockTextField('sma, ema'),
            'type': 'values'
        }

        param_grid = dialog._collect_param_grid()

        assert len(param_grid) == 4
        assert param_grid['fast_window'] == [10, 15, 20]
        assert param_grid['slow_window'] == [40, 50, 60]
        assert param_grid['use_stops'] == [True, False]
        assert param_grid['ma_type'] == ['sma', 'ema']


class TestParameterGridEdgeCases:
    """Test edge cases in parameter grid collection."""

    def test_empty_numeric_fields(self):
        """Test that empty numeric fields are ignored."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'fast_window': 10},
            param_types={'fast_window': int},
            on_optimize=lambda grid, metric: None
        )

        # Empty fields - should use current value only (not in grid)
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField(''),
            'max': MockTextField(''),
            'step': MockTextField(''),
            'type': 'numeric'
        }

        param_grid = dialog._collect_param_grid()

        # No parameters defined
        assert 'fast_window' not in param_grid

    def test_partial_numeric_fields(self):
        """Test that partially filled numeric fields are ignored."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'fast_window': 10},
            param_types={'fast_window': int},
            on_optimize=lambda grid, metric: None
        )

        # Only min filled - should be ignored
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField('5'),
            'max': MockTextField(''),
            'step': MockTextField(''),
            'type': 'numeric'
        }

        param_grid = dialog._collect_param_grid()

        assert 'fast_window' not in param_grid

    def test_empty_value_list(self):
        """Test that empty value list is ignored."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'ma_type': 'sma'},
            param_types={'ma_type': str},
            on_optimize=lambda grid, metric: None
        )

        dialog.grid_controls['ma_type'] = {
            'values': MockTextField(''),
            'type': 'values'
        }

        param_grid = dialog._collect_param_grid()

        assert 'ma_type' not in param_grid

    def test_value_list_with_spaces(self):
        """Test value list parsing with extra spaces."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'ma_type': 'sma'},
            param_types={'ma_type': str},
            on_optimize=lambda grid, metric: None
        )

        dialog.grid_controls['ma_type'] = {
            'values': MockTextField('  sma  ,  ema  ,  wma  '),
            'type': 'values'
        }

        param_grid = dialog._collect_param_grid()

        # Should strip whitespace
        assert param_grid['ma_type'] == ['sma', 'ema', 'wma']

    def test_float_parameter_range(self):
        """Test collecting float parameter range."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'threshold': 1.5},
            param_types={'threshold': float},
            on_optimize=lambda grid, metric: None
        )

        dialog.grid_controls['threshold'] = {
            'min': MockTextField('1.0'),
            'max': MockTextField('2.0'),
            'step': MockTextField('0.5'),
            'type': 'numeric'
        }

        param_grid = dialog._collect_param_grid()

        assert 'threshold' in param_grid
        # Should be [1.0, 1.5, 2.0]
        assert len(param_grid['threshold']) == 3
        assert param_grid['threshold'][0] == pytest.approx(1.0)
        assert param_grid['threshold'][1] == pytest.approx(1.5)
        assert param_grid['threshold'][2] == pytest.approx(2.0)

    def test_single_value_numeric_range(self):
        """Test numeric range with min == max (single value)."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'window': 10},
            param_types={'window': int},
            on_optimize=lambda grid, metric: None
        )

        dialog.grid_controls['window'] = {
            'min': MockTextField('10'),
            'max': MockTextField('10'),
            'step': MockTextField('1'),
            'type': 'numeric'
        }

        param_grid = dialog._collect_param_grid()

        assert 'window' in param_grid
        assert param_grid['window'] == [10]


class TestCombinationEstimation:
    """Test parameter combination count estimation."""

    def test_estimate_single_parameter(self):
        """Test combination count for single parameter."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'fast_window': 10},
            param_types={'fast_window': int},
            on_optimize=lambda grid, metric: None
        )

        # 3 values for fast_window
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField('5'),
            'max': MockTextField('15'),
            'step': MockTextField('5'),
            'type': 'numeric'
        }

        # Manually calculate like _on_estimate_combinations
        param_grid = dialog._collect_param_grid()
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)

        assert total_combinations == 3  # [5, 10, 15]

    def test_estimate_two_parameters(self):
        """Test combination count for two parameters (cartesian product)."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={'fast_window': 10, 'slow_window': 20},
            param_types={'fast_window': int, 'slow_window': int},
            on_optimize=lambda grid, metric: None
        )

        # 3 values x 2 values = 6 combinations
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField('5'),
            'max': MockTextField('15'),
            'step': MockTextField('5'),
            'type': 'numeric'
        }
        dialog.grid_controls['slow_window'] = {
            'min': MockTextField('20'),
            'max': MockTextField('30'),
            'step': MockTextField('10'),
            'type': 'numeric'
        }

        param_grid = dialog._collect_param_grid()
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)

        assert total_combinations == 6  # 3 * 2

    def test_estimate_mixed_types(self):
        """Test combination count with mixed parameter types."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={
                'fast_window': 10,
                'use_stops': False,
                'ma_type': 'sma'
            },
            param_types={
                'fast_window': int,
                'use_stops': bool,
                'ma_type': str
            },
            on_optimize=lambda grid, metric: None
        )

        # 3 numeric x 2 boolean x 2 values = 12 combinations
        dialog.grid_controls['fast_window'] = {
            'min': MockTextField('10'),
            'max': MockTextField('20'),
            'step': MockTextField('5'),
            'type': 'numeric'
        }
        dialog.grid_controls['use_stops'] = {
            'checkbox': MockCheckbox(True),
            'type': 'bool'
        }
        dialog.grid_controls['ma_type'] = {
            'values': MockTextField('sma, ema'),
            'type': 'values'
        }

        param_grid = dialog._collect_param_grid()
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)

        assert total_combinations == 12  # 3 * 2 * 2


class TestDialogInitialization:
    """Test OptimizationDialog initialization."""

    def test_dialog_initializes_with_params(self):
        """Test dialog initializes with provided parameters."""
        page = MockPage()
        current_params = {'fast_window': 10, 'slow_window': 20}
        param_types = {'fast_window': int, 'slow_window': int}

        dialog = OptimizationDialog(
            page=page,
            strategy_name="MA Crossover",
            current_params=current_params,
            param_types=param_types,
            on_optimize=lambda grid, metric: None
        )

        assert dialog.strategy_name == "MA Crossover"
        assert dialog.current_params == current_params
        assert dialog.param_types == param_types
        assert dialog.page == page

    def test_dialog_creates_controls_for_all_params(self):
        """Test dialog creates grid controls for all parameters."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={
                'fast_window': 10,
                'slow_window': 20,
                'use_stops': False
            },
            param_types={
                'fast_window': int,
                'slow_window': int,
                'use_stops': bool
            },
            on_optimize=lambda grid, metric: None
        )

        # Should have controls for all 3 parameters
        assert len(dialog.grid_controls) == 3
        assert 'fast_window' in dialog.grid_controls
        assert 'slow_window' in dialog.grid_controls
        assert 'use_stops' in dialog.grid_controls

    def test_dialog_control_types_match_param_types(self):
        """Test that control types match parameter types."""
        dialog = OptimizationDialog(
            page=MockPage(),
            strategy_name="Test Strategy",
            current_params={
                'window': 10,
                'use_stops': False,
                'ma_type': 'sma'
            },
            param_types={
                'window': int,
                'use_stops': bool,
                'ma_type': str
            },
            on_optimize=lambda grid, metric: None
        )

        # Check control types
        assert dialog.grid_controls['window']['type'] == 'numeric'
        assert dialog.grid_controls['use_stops']['type'] == 'bool'
        assert dialog.grid_controls['ma_type']['type'] == 'values'
