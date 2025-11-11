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
        on_optimize: Callable[[Dict[str, Any]], None]
    ):
        """
        Initialize optimization dialog.

        Args:
            page: Flet page
            strategy_name: Name of strategy being optimized
            current_params: Current parameter values
            param_types: Parameter types for each parameter
            on_optimize: Callback when optimization is started (receives opt_config dict)
                opt_config contains:
                - 'param_space': Parameter grid (Grid Search) or ranges (Random Search)
                - 'metric': Optimization metric
                - 'method': 'grid_search' or 'random_search'
                - 'n_iterations': Number of iterations (Random Search only)
        """
        self.page = page
        self.strategy_name = strategy_name
        self.current_params = current_params
        self.param_types = param_types
        self.on_optimize = on_optimize

        # Parameter grid controls
        self.grid_controls: Dict[str, Dict[str, ft.Control]] = {}

        # Method selector
        self.method_dropdown = None

        # Metric selector
        self.metric_dropdown = None

        # Random search settings
        self.n_iterations_input = None

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

        # Check if Bayesian optimization is available
        try:
            from backtesting.optimization import BAYESIAN_AVAILABLE
        except ImportError:
            BAYESIAN_AVAILABLE = False

        # Optimization method selector
        method_options = [
            ft.dropdown.Option("grid_search", "Grid Search (Exhaustive)"),
            ft.dropdown.Option("random_search", "Random Search (Fast Sampling)"),
            ft.dropdown.Option("genetic", "Genetic Algorithm (Evolutionary)")
        ]

        # Add Bayesian if available
        if BAYESIAN_AVAILABLE:
            method_options.insert(
                2,  # Insert before Genetic
                ft.dropdown.Option("bayesian", "Bayesian Optimization (Smart, requires scikit-optimize)")
            )

        self.method_dropdown = ft.Dropdown(
            label="Optimization Method",
            value="grid_search",
            options=method_options,
            width=400,
            on_change=self._on_method_changed,
            tooltip="Grid Search tests all combinations. Random Search samples randomly. Bayesian uses Gaussian Processes. Genetic uses evolutionary algorithms."
        )

        # Random Search settings panel
        self.n_iterations_input = ft.TextField(
            label="Number of Iterations",
            value="100",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Number of random parameter combinations to test (typically 50-500)",
            visible=False  # Hidden by default (Grid Search is default)
        )

        random_search_info = ft.Text(
            "Random Search samples parameter combinations randomly. Faster for large parameter spaces.",
            size=11,
            color=ft.Colors.GREY_400,
            italic=True,
            visible=False
        )

        self.random_search_panel = ft.Container(
            content=ft.Column([
                self.n_iterations_input,
                random_search_info
            ], spacing=5),
            visible=False,
            padding=10,
            bgcolor=ft.Colors.GREY_900,
            border=ft.border.all(2, ft.Colors.ORANGE_700),
            border_radius=8
        )

        # Bayesian Optimization settings panel
        self.bayesian_iterations_input = ft.TextField(
            label="Total Iterations",
            value="50",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Total number of iterations (typically 30-100). Fewer needed than Random Search.",
            visible=False
        )

        self.bayesian_initial_points_input = ft.TextField(
            label="Initial Random Points",
            value="10",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Number of random points before Bayesian selection starts (typically 5-20)",
            visible=False
        )

        self.acquisition_function_dropdown = ft.Dropdown(
            label="Acquisition Function",
            value="EI",
            options=[
                ft.dropdown.Option("EI", "Expected Improvement (Recommended)"),
                ft.dropdown.Option("LCB", "Lower Confidence Bound (Conservative)"),
                ft.dropdown.Option("PI", "Probability of Improvement (Greedy)")
            ],
            width=400,
            tooltip="Strategy for selecting next point. EI balances exploration vs exploitation.",
            visible=False
        )

        bayesian_info = ft.Text(
            "Bayesian Optimization uses Gaussian Processes to intelligently select parameters. "
            "5-20x more efficient than Random Search.",
            size=11,
            color=ft.Colors.GREY_400,
            italic=True,
            visible=False
        )

        self.bayesian_panel = ft.Container(
            content=ft.Column([
                self.bayesian_iterations_input,
                self.bayesian_initial_points_input,
                self.acquisition_function_dropdown,
                bayesian_info
            ], spacing=5),
            visible=False,
            padding=10,
            bgcolor=ft.Colors.GREY_900,
            border=ft.border.all(2, ft.Colors.CYAN_700),
            border_radius=8
        )

        # Genetic Algorithm settings panel
        self.genetic_population_input = ft.TextField(
            label="Population Size",
            value="50",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Number of individuals in population (typically 30-100)",
            visible=False
        )

        self.genetic_generations_input = ft.TextField(
            label="Generations",
            value="20",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Number of generations to evolve (typically 10-50)",
            visible=False
        )

        self.genetic_mutation_rate_input = ft.TextField(
            label="Mutation Rate",
            value="0.1",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Probability of mutation (0.01-0.3, recommended: 0.1)",
            visible=False
        )

        self.genetic_crossover_rate_input = ft.TextField(
            label="Crossover Rate",
            value="0.7",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=200,
            tooltip="Probability of crossover (0.5-1.0, recommended: 0.7)",
            visible=False
        )

        genetic_info = ft.Text(
            "Genetic Algorithm evolves populations using selection, crossover, and mutation. "
            "Good for multi-modal optimization.",
            size=11,
            color=ft.Colors.GREY_400,
            italic=True,
            visible=False
        )

        self.genetic_panel = ft.Container(
            content=ft.Column([
                self.genetic_population_input,
                self.genetic_generations_input,
                self.genetic_mutation_rate_input,
                self.genetic_crossover_rate_input,
                genetic_info
            ], spacing=5),
            visible=False,
            padding=10,
            bgcolor=ft.Colors.GREY_900,
            border=ft.border.all(2, ft.Colors.GREEN_700),
            border_radius=8
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
                    self.method_dropdown,
                    self.random_search_panel,
                    self.bayesian_panel,
                    self.genetic_panel,
                    ft.Divider(),
                    ft.Container(
                        content=param_grid_column,
                        height=300,
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

    def _on_method_changed(self, e):
        """Handle optimization method selection change."""
        method = self.method_dropdown.value

        # Show/hide method-specific settings panels
        self.random_search_panel.visible = (method == "random_search")
        self.bayesian_panel.visible = (method == "bayesian")
        self.genetic_panel.visible = (method == "genetic")

        # Update instructions text based on method
        if method == "random_search":
            self.combination_text.value = "Random Search: Specify n_iterations above"
            self.combination_text.color = ft.Colors.ORANGE_400
        elif method == "bayesian":
            self.combination_text.value = "Bayesian Optimization: Specify iterations above"
            self.combination_text.color = ft.Colors.CYAN_400
        elif method == "genetic":
            self.combination_text.value = "Genetic Algorithm: Specify population and generations above"
            self.combination_text.color = ft.Colors.GREEN_400
        else:  # grid_search
            self.combination_text.value = "Define parameter ranges to see total combinations"
            self.combination_text.color = ft.Colors.AMBER_400

        # Update UI
        self.random_search_panel.update()
        self.bayesian_panel.update()
        self.genetic_panel.update()
        self.combination_text.update()

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
        Collect parameter grid from UI controls (Grid Search format).

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

    def _collect_range_params(self) -> Dict[str, Any]:
        """
        Collect parameter ranges from UI controls (Random Search format).

        Returns:
            Dictionary mapping parameter name to range tuple or list:
            - Numeric: (min, max) for uniform sampling
            - Boolean: [True, False] for discrete choice
            - String: [val1, val2, ...] for discrete choice
        """
        param_ranges = {}

        for param_name, controls in self.grid_controls.items():
            control_type = controls['type']

            if control_type == 'numeric':
                min_val = controls['min'].value.strip()
                max_val = controls['max'].value.strip()

                if min_val and max_val:
                    param_type = self.param_types.get(param_name, float)
                    min_num = param_type(min_val)
                    max_num = param_type(max_val)

                    # Return as tuple (min, max) for Random Search
                    param_ranges[param_name] = (min_num, max_num)

            elif control_type == 'bool':
                if controls['checkbox'].value:
                    param_ranges[param_name] = [True, False]

            elif control_type == 'values':
                values_str = controls['values'].value.strip()
                if values_str:
                    # Parse comma-separated values
                    param_type = self.param_types.get(param_name, str)
                    values = [param_type(v.strip()) for v in values_str.split(',') if v.strip()]
                    if values:
                        param_ranges[param_name] = values

        return param_ranges

    def _on_run_optimization(self, e):
        """Handle Run Optimization button click."""
        try:
            log_info("User clicked 'Run Optimization' button")

            # Collect method and parameters
            method = self.method_dropdown.value or 'grid_search'
            log_info(f"Optimization method selected: {method}")

            # Collect parameters based on method
            genetic_params = {}
            if method == 'random_search':
                param_space = self._collect_range_params()
                n_iterations = int(self.n_iterations_input.value or "100")
                n_initial_points = None
                acquisition_func = None
                log_info(f"Random Search iterations: {n_iterations}")
            elif method == 'bayesian':
                param_space = self._collect_range_params()  # Same format as Random Search
                n_iterations = int(self.bayesian_iterations_input.value or "50")
                n_initial_points = int(self.bayesian_initial_points_input.value or "10")
                acquisition_func = self.acquisition_function_dropdown.value or "EI"
                log_info(f"Bayesian Optimization iterations: {n_iterations}")
                log_info(f"Initial random points: {n_initial_points}")
                log_info(f"Acquisition function: {acquisition_func}")
            elif method == 'genetic':
                param_space = self._collect_range_params()  # Same format as Random Search
                genetic_params['population_size'] = int(self.genetic_population_input.value or "50")
                genetic_params['n_generations'] = int(self.genetic_generations_input.value or "20")
                genetic_params['mutation_rate'] = float(self.genetic_mutation_rate_input.value or "0.1")
                genetic_params['crossover_rate'] = float(self.genetic_crossover_rate_input.value or "0.7")
                n_iterations = None
                n_initial_points = None
                acquisition_func = None
                log_info(f"Genetic Algorithm population: {genetic_params['population_size']}")
                log_info(f"Generations: {genetic_params['n_generations']}")
                log_info(f"Mutation rate: {genetic_params['mutation_rate']}")
                log_info(f"Crossover rate: {genetic_params['crossover_rate']}")
            else:  # grid_search
                param_space = self._collect_param_grid()
                n_iterations = None
                n_initial_points = None
                acquisition_func = None

            if not param_space:
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
            log_info(f"Parameter space collected: {param_space}")

            # Show staging/confirmation dialog
            self._show_confirmation_dialog(
                param_space,
                metric,
                method,
                n_iterations,
                n_initial_points,
                acquisition_func,
                genetic_params
            )

        except Exception as ex:
            log_exception(ex, "Error starting optimization")
            error_dlg = ft.AlertDialog(
                title=ft.Text("Error"),
                content=ft.Text(f"Failed to start optimization: {ex}"),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(error_dlg))]
            )
            self.page.open(error_dlg)

    def _show_confirmation_dialog(
        self,
        param_space: Dict[str, Any],
        metric: str,
        method: str = 'grid_search',
        n_iterations: Optional[int] = None,
        n_initial_points: Optional[int] = None,
        acquisition_func: Optional[str] = None,
        genetic_params: Optional[Dict] = None
    ):
        """
        Show confirmation dialog with optimization preview.

        Args:
            param_space: Parameter space (grid or ranges depending on method)
            metric: Optimization metric
            method: Optimization method ('grid_search' or 'random_search')
            n_iterations: Number of iterations for Random Search (ignored for Grid Search)
        """
        from itertools import product

        log_info("Showing optimization confirmation dialog")
        log_info(f"Method: {method}")

        # Calculate total combinations and generate preview based on method
        if method == 'random_search':
            # Random Search: Show iteration count and sample ranges
            total_combinations = n_iterations or 100
            log_info(f"Random Search with {total_combinations} iterations")

            preview_text = "Parameter Ranges (Random Sampling):\n\n"
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple):
                    preview_text += f"{param_name}: {param_range[0]} to {param_range[1]}\n"
                elif isinstance(param_range, list):
                    preview_text += f"{param_name}: {param_range}\n"

            preview_text += f"\nWill test {total_combinations} random combinations from these ranges."

        elif method == 'bayesian':
            # Bayesian: Show iteration count, initial points, and acquisition function
            total_combinations = n_iterations or 50
            log_info(f"Bayesian Optimization with {total_combinations} iterations")

            preview_text = "Parameter Ranges (Bayesian Optimization):\n\n"
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple):
                    preview_text += f"{param_name}: {param_range[0]} to {param_range[1]}\n"
                elif isinstance(param_range, list):
                    preview_text += f"{param_name}: {param_range}\n"

            preview_text += f"\nTotal iterations: {total_combinations}\n"
            preview_text += f"Initial random points: {n_initial_points or 10}\n"
            preview_text += f"Acquisition function: {acquisition_func or 'EI'}\n"
            preview_text += f"\nBayesian optimization will intelligently select {total_combinations} parameter combinations."

        elif method == 'genetic':
            # Genetic: Show population, generations, and settings
            genetic_params = genetic_params or {}
            population_size = genetic_params.get('population_size', 50)
            n_generations = genetic_params.get('n_generations', 20)
            total_combinations = population_size * n_generations
            log_info(f"Genetic Algorithm with {population_size} individuals × {n_generations} generations")

            preview_text = "Parameter Ranges (Genetic Algorithm):\n\n"
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple):
                    preview_text += f"{param_name}: {param_range[0]} to {param_range[1]}\n"
                elif isinstance(param_range, list):
                    preview_text += f"{param_name}: {param_range}\n"

            preview_text += f"\nPopulation size: {population_size}\n"
            preview_text += f"Generations: {n_generations}\n"
            preview_text += f"Mutation rate: {genetic_params.get('mutation_rate', 0.1):.2f}\n"
            preview_text += f"Crossover rate: {genetic_params.get('crossover_rate', 0.7):.2f}\n"
            preview_text += f"\nGenetic algorithm will evolve {population_size} individuals over {n_generations} generations."

        else:
            # Grid Search: Calculate exact combinations
            total_combinations = 1
            for param_values in param_space.values():
                total_combinations *= len(param_values)

            # Generate preview of combinations (first 10)
            param_names = list(param_space.keys())
            param_values = [param_space[name] for name in param_names]
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
            log_info(f"Final parameter space: {param_space}")
            log_info(f"Optimization metric: {metric}")
            log_info(f"Method: {method}")
            if n_iterations:
                log_info(f"Random Search iterations: {n_iterations}")

            self.page.close(confirmation_dialog)
            self.page.close(self.dialog)

            # Build optimization config dict
            opt_config = {
                'param_space': param_space,
                'metric': metric,
                'method': method
            }
            if n_iterations is not None:
                opt_config['n_iterations'] = n_iterations
            if n_initial_points is not None:
                opt_config['n_initial_points'] = n_initial_points
            if acquisition_func is not None:
                opt_config['acquisition_func'] = acquisition_func
            if genetic_params:
                opt_config.update(genetic_params)

            self.on_optimize(opt_config)

        # Format method name for display
        if method == 'grid_search':
            method_display = "Grid Search (Exhaustive)"
        elif method == 'random_search':
            method_display = "Random Search (Fast Sampling)"
        elif method == 'bayesian':
            method_display = "Bayesian Optimization (Intelligent Sampling)"
        elif method == 'genetic':
            method_display = "Genetic Algorithm (Evolutionary)"
        else:
            method_display = method

        confirmation_dialog = ft.AlertDialog(
            title=ft.Text("Confirm Optimization", size=20, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                content=ft.Column([
                    ft.Text(f"Strategy: {self.strategy_name}", weight=ft.FontWeight.W_500),
                    ft.Text(f"Method: {method_display}", weight=ft.FontWeight.W_500),
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
