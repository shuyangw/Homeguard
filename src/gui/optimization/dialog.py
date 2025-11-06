"""
Optimization dialog for parameter grid search.
"""

import flet as ft
from typing import Dict, Any, Callable, List, Optional
from gui.utils.error_logger import log_info, log_error, log_exception


class OptimizationDialog:
    """
    Dialog for defining parameter grids and running optimization.

    Allows users to:
    - Define ranges for each strategy parameter
    - Select optimization metric
    - Run grid search optimization
    - View results and apply best parameters
    """

    def __init__(
        self,
        page: ft.Page,
        strategy_name: str,
        current_params: Dict[str, Any],
        param_types: Dict[str, type],
        on_optimize: Callable[[Dict[str, List[Any]], str], None]
    ):
        """
        Initialize optimization dialog.

        Args:
            page: Flet page
            strategy_name: Name of strategy being optimized
            current_params: Current parameter values
            param_types: Parameter types for each parameter
            on_optimize: Callback when optimization is started (param_grid, metric)
        """
        self.page = page
        self.strategy_name = strategy_name
        self.current_params = current_params
        self.param_types = param_types
        self.on_optimize = on_optimize

        # Parameter grid controls
        self.grid_controls: Dict[str, Dict[str, ft.Control]] = {}

        # Metric selector
        self.metric_dropdown = None

        # Dialog
        self.dialog = None

        log_info(f"Initializing optimization dialog for strategy: {strategy_name}")
        log_info(f"Parameters to optimize: {list(current_params.keys())}")

        self._build_dialog()

    def _build_dialog(self):
        """Build the optimization dialog."""
        # Title
        title = ft.Text(
            f"Optimize: {self.strategy_name}",
            size=20,
            weight=ft.FontWeight.BOLD
        )

        # Instructions
        instructions = ft.Text(
            "Define parameter ranges to test. Leave fields empty to use current value only.",
            size=12,
            color=ft.Colors.GREY_400,
            italic=True
        )

        # Parameter grid inputs
        param_grid_column = ft.Column(spacing=15, scroll=ft.ScrollMode.AUTO)

        for param_name, current_value in self.current_params.items():
            param_type = self.param_types.get(param_name, str)

            # Parameter label
            param_label = ft.Text(
                param_name.replace('_', ' ').title(),
                size=14,
                weight=ft.FontWeight.W_500
            )

            # Current value display
            current_display = ft.Text(
                f"Current: {current_value}",
                size=11,
                color=ft.Colors.CYAN_400
            )

            if param_type in [int, float]:
                # Numeric parameter: min, max, step
                min_input = ft.TextField(
                    label="Min",
                    hint_text=str(current_value),
                    keyboard_type=ft.KeyboardType.NUMBER,
                    width=100
                )
                max_input = ft.TextField(
                    label="Max",
                    hint_text=str(current_value),
                    keyboard_type=ft.KeyboardType.NUMBER,
                    width=100
                )
                step_input = ft.TextField(
                    label="Step",
                    hint_text="1" if param_type == int else "0.1",
                    keyboard_type=ft.KeyboardType.NUMBER,
                    width=100
                )

                grid_row = ft.Row([
                    min_input,
                    max_input,
                    step_input
                ], spacing=10)

                self.grid_controls[param_name] = {
                    'min': min_input,
                    'max': max_input,
                    'step': step_input,
                    'type': 'numeric'
                }

            elif param_type == bool:
                # Boolean: test both values
                checkbox = ft.Checkbox(
                    label="Test both True and False",
                    value=False
                )

                grid_row = checkbox

                self.grid_controls[param_name] = {
                    'checkbox': checkbox,
                    'type': 'bool'
                }

            else:
                # String/other: comma-separated values
                values_input = ft.TextField(
                    label="Values (comma-separated)",
                    hint_text=f"{current_value}",
                    width=300
                )

                grid_row = values_input

                self.grid_controls[param_name] = {
                    'values': values_input,
                    'type': 'values'
                }

            # Container for parameter
            param_container = ft.Container(
                content=ft.Column([
                    ft.Row([param_label, current_display], spacing=10),
                    grid_row
                ], spacing=5),
                border=ft.border.all(2, ft.Colors.BLUE_700),
                border_radius=8,
                padding=10,
                bgcolor=ft.Colors.GREY_900
            )

            param_grid_column.controls.append(param_container)

        # Optimization metric selector
        self.metric_dropdown = ft.Dropdown(
            label="Optimization Metric",
            value="sharpe_ratio",
            options=[
                ft.dropdown.Option("sharpe_ratio", "Sharpe Ratio (risk-adjusted returns)"),
                ft.dropdown.Option("total_return", "Total Return (maximize profit)"),
                ft.dropdown.Option("max_drawdown", "Max Drawdown (minimize risk)")
            ],
            width=400,
            tooltip="Metric to optimize. Sharpe Ratio is recommended for balanced risk/return."
        )

        # Combination count estimate
        self.combination_text = ft.Text(
            "Define parameter ranges to see total combinations",
            size=12,
            color=ft.Colors.AMBER_400
        )

        # Update combination count button
        estimate_button = ft.TextButton(
            "Estimate Combinations",
            icon=ft.Icons.CALCULATE,
            on_click=self._on_estimate_combinations
        )

        # Actions
        cancel_button = ft.TextButton(
            "Cancel",
            on_click=lambda e: self.page.close(self.dialog)
        )

        optimize_button = ft.ElevatedButton(
            "Run Optimization",
            icon=ft.Icons.ROCKET_LAUNCH,
            on_click=self._on_run_optimization,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN_700
            )
        )

        # Build dialog
        self.dialog = ft.AlertDialog(
            title=title,
            content=ft.Container(
                content=ft.Column([
                    instructions,
                    ft.Divider(),
                    ft.Container(
                        content=param_grid_column,
                        height=400,
                        border=ft.border.all(2, ft.Colors.PURPLE_700),
                        border_radius=8,
                        padding=10
                    ),
                    ft.Divider(),
                    self.metric_dropdown,
                    ft.Row([
                        self.combination_text,
                        estimate_button
                    ], spacing=10)
                ], spacing=10, scroll=ft.ScrollMode.AUTO),
                width=600
            ),
            actions=[
                cancel_button,
                optimize_button
            ],
            actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

    def _on_estimate_combinations(self, e):
        """Estimate total parameter combinations."""
        try:
            total_combinations = 1
            param_grid = self._collect_param_grid()

            if param_grid:
                for param_values in param_grid.values():
                    total_combinations *= len(param_values)

                log_info(f"Estimated {total_combinations} parameter combinations from grid: {param_grid}")
                self.combination_text.value = f"Total combinations: {total_combinations}"
                self.combination_text.color = ft.Colors.GREEN_400
            else:
                log_error("No parameter ranges defined for optimization")
                self.combination_text.value = "No parameter ranges defined"
                self.combination_text.color = ft.Colors.ORANGE_400

            self.combination_text.update()

        except Exception as ex:
            log_exception(ex, "Error estimating parameter combinations")
            self.combination_text.value = f"Error: {ex}"
            self.combination_text.color = ft.Colors.RED_400
            self.combination_text.update()

    def _collect_param_grid(self) -> Dict[str, List[Any]]:
        """
        Collect parameter grid from UI controls.

        Returns:
            Dictionary mapping parameter name to list of values to test
        """
        param_grid = {}

        for param_name, controls in self.grid_controls.items():
            control_type = controls['type']

            if control_type == 'numeric':
                min_val = controls['min'].value.strip()
                max_val = controls['max'].value.strip()
                step_val = controls['step'].value.strip()

                if min_val and max_val and step_val:
                    param_type = self.param_types.get(param_name, float)
                    min_num = param_type(min_val)
                    max_num = param_type(max_val)
                    step_num = param_type(step_val)

                    # Generate range
                    values = []
                    current = min_num
                    while current <= max_num:
                        values.append(current)
                        current += step_num

                    if values:
                        param_grid[param_name] = values

            elif control_type == 'bool':
                if controls['checkbox'].value:
                    param_grid[param_name] = [True, False]

            elif control_type == 'values':
                values_str = controls['values'].value.strip()
                if values_str:
                    # Parse comma-separated values
                    param_type = self.param_types.get(param_name, str)
                    values = [param_type(v.strip()) for v in values_str.split(',') if v.strip()]
                    if values:
                        param_grid[param_name] = values

        return param_grid

    def _on_run_optimization(self, e):
        """Handle Run Optimization button click."""
        try:
            log_info("User clicked 'Run Optimization' button")
            param_grid = self._collect_param_grid()

            if not param_grid:
                # Show error
                log_error("Cannot start optimization: No parameter ranges defined")
                error_dlg = ft.AlertDialog(
                    title=ft.Text("No Parameters"),
                    content=ft.Text("Please define at least one parameter range to optimize."),
                    actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(error_dlg))]
                )
                self.page.open(error_dlg)
                return

            metric = self.metric_dropdown.value or 'sharpe_ratio'
            log_info(f"Optimization metric selected: {metric}")
            log_info(f"Parameter grid collected: {param_grid}")

            # Show staging/confirmation dialog
            self._show_confirmation_dialog(param_grid, metric)

        except Exception as ex:
            log_exception(ex, "Error starting optimization")
            error_dlg = ft.AlertDialog(
                title=ft.Text("Error"),
                content=ft.Text(f"Failed to start optimization: {ex}"),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(error_dlg))]
            )
            self.page.open(error_dlg)

    def _show_confirmation_dialog(self, param_grid: Dict[str, List[Any]], metric: str):
        """
        Show confirmation dialog with optimization preview.

        Args:
            param_grid: Parameter grid to optimize
            metric: Optimization metric
        """
        from itertools import product

        log_info("Showing optimization confirmation dialog")

        # Calculate total combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)

        # Generate preview of combinations (first 10)
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(product(*param_values))

        log_info(f"Generated {len(combinations)} total parameter combinations")
        log_info(f"Preview (first 10): {combinations[:10]}")

        preview_text = "Parameter Combinations Preview:\n\n"
        for i, combo in enumerate(combinations[:10], 1):
            params_str = ", ".join(f"{name}={val}" for name, val in zip(param_names, combo))
            preview_text += f"{i}. {params_str}\n"

        if len(combinations) > 10:
            preview_text += f"\n... and {len(combinations) - 10} more combinations"

        # Estimate time (assume ~2 seconds per backtest)
        estimated_seconds = total_combinations * 2
        if estimated_seconds < 60:
            time_estimate = f"{estimated_seconds} seconds"
        elif estimated_seconds < 3600:
            time_estimate = f"{estimated_seconds / 60:.1f} minutes"
        else:
            time_estimate = f"{estimated_seconds / 3600:.1f} hours"

        log_info(f"Estimated optimization time: {time_estimate} ({estimated_seconds} seconds)")

        # Build confirmation dialog
        def confirm_and_run(e):
            log_info("User confirmed optimization - starting optimization run")
            log_info(f"Final parameter grid: {param_grid}")
            log_info(f"Optimization metric: {metric}")
            self.page.close(confirmation_dialog)
            self.page.close(self.dialog)
            self.on_optimize(param_grid, metric)

        confirmation_dialog = ft.AlertDialog(
            title=ft.Text("Confirm Optimization", size=20, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                content=ft.Column([
                    ft.Text(f"Strategy: {self.strategy_name}", weight=ft.FontWeight.W_500),
                    ft.Text(f"Optimization Metric: {metric}", weight=ft.FontWeight.W_500),
                    ft.Divider(),
                    ft.Row([
                        ft.Icon(ft.Icons.GRID_ON, color=ft.Colors.CYAN_400),
                        ft.Text(
                            f"Total Combinations: {total_combinations}",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.CYAN_400
                        )
                    ], spacing=5),
                    ft.Row([
                        ft.Icon(ft.Icons.SCHEDULE, color=ft.Colors.AMBER_400),
                        ft.Text(
                            f"Estimated Time: ~{time_estimate}",
                            size=14,
                            color=ft.Colors.AMBER_400
                        )
                    ], spacing=5),
                    ft.Divider(),
                    ft.Text("Preview:", weight=ft.FontWeight.W_500),
                    ft.Container(
                        content=ft.Text(
                            preview_text,
                            size=11,
                            selectable=True,
                            color=ft.Colors.GREY_300
                        ),
                        bgcolor=ft.Colors.GREY_900,
                        border=ft.border.all(1, ft.Colors.GREY_700),
                        border_radius=5,
                        padding=10,
                        height=200,
                        scroll=ft.ScrollMode.AUTO
                    ),
                    ft.Divider(),
                    ft.Text(
                        "Results will be exported to CSV for detailed analysis.",
                        size=12,
                        italic=True,
                        color=ft.Colors.GREEN_400
                    )
                ], spacing=10),
                width=600,
                height=500
            ),
            actions=[
                ft.TextButton(
                    "Cancel",
                    on_click=lambda _: self.page.close(confirmation_dialog)
                ),
                ft.ElevatedButton(
                    "Confirm & Run Optimization",
                    icon=ft.Icons.ROCKET_LAUNCH,
                    on_click=confirm_and_run,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.GREEN_700
                    )
                )
            ],
            actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        self.page.open(confirmation_dialog)

    def show(self):
        """Show the optimization dialog."""
        self.page.open(self.dialog)
