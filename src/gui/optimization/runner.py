"""
GUI optimization runner for parameter optimization workflows.

Handles the complete optimization execution flow including:
- Progress dialog display
- Result tracking and CSV export
- Results dialog with parameter application
"""

import sys
import flet as ft
import pandas as pd
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer, RandomSearchOptimizer, GeneticOptimizer, BAYESIAN_AVAILABLE
if BAYESIAN_AVAILABLE:
    from backtesting.optimization import BayesianOptimizer
from config import get_log_output_dir
from gui.utils.error_logger import log_info, log_error, log_exception


class OptimizationRunner:
    """
    Handles optimization execution and results display in the GUI.
    """

    def __init__(
        self,
        page: ft.Page,
        setup_view: Any,
        show_notification: Callable[[str, str], None],
        show_error_dialog: Callable[[str, str], None],
        show_setup_view: Callable[[], None]
    ):
        """
        Initialize optimization runner.

        Args:
            page: Flet Page object
            setup_view: Setup view instance (for applying best params)
            show_notification: Notification callback
            show_error_dialog: Error dialog callback
            show_setup_view: Setup view navigation callback
        """
        self.page = page
        self.setup_view = setup_view
        self.show_notification = show_notification
        self.show_error_dialog = show_error_dialog
        self.show_setup_view = show_setup_view

    def run_optimization(self, config: Dict[str, Any]):
        """
        Run parameter optimization.

        Args:
            config: Configuration with optimization parameters:
                - strategy_class: Strategy class to optimize
                - param_space: Parameter space (grid or ranges)
                - optimization_metric: Metric to optimize
                - optimization_method: 'grid_search' or 'random_search'
                - n_iterations: Number of iterations (Random Search only)
                - symbols: List of symbols
                - start_date: Start date
                - end_date: End date
                - initial_capital: Initial capital
                - fees: Trading fees
        """
        progress_dialog = None
        try:
            strategy_class = config['strategy_class']
            param_space = config['param_space']
            metric = config['optimization_metric']
            method = config.get('optimization_method', 'grid_search')
            symbols = config['symbols']
            start_date = config['start_date']
            end_date = config['end_date']

            log_info(f"Starting optimization: {strategy_class.__name__}")
            log_info(f"Method: {method}")
            log_info(f"Parameter space: {param_space}")
            log_info(f"Optimization metric: {metric}")

            # Format method display name
            method_display = "Grid Search" if method == 'grid_search' else "Random Search"

            # Show progress dialog
            progress_dialog = ft.AlertDialog(
                title=ft.Text("Running Optimization..."),
                content=ft.Column([
                    ft.Text(f"Strategy: {strategy_class.__name__}"),
                    ft.Text(f"Method: {method_display}"),
                    ft.Text(f"Metric: {metric}"),
                    ft.ProgressRing(),
                    ft.Text("This may take several minutes...")
                ], spacing=10, height=180),
                modal=True
            )
            self.page.open(progress_dialog)
            self.page.update()

            # Run optimization
            result = self._execute_optimization(config)

            # Close progress dialog
            self.page.close(progress_dialog)

            # Show results dialog
            self._show_results_dialog(result, config)

            log_info(f"Optimization complete. Best {metric}: {result['best_value']:.4f}")

        except Exception as e:
            log_exception(e, "Error running optimization")
            # Close progress dialog if it exists
            if progress_dialog:
                try:
                    self.page.close(progress_dialog)
                except:
                    pass
            self.show_notification(f"Optimization failed: {str(e)}", "error")
            self.show_error_dialog("Optimization Error", str(e))

    def _execute_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the optimization and return results.

        Args:
            config: Configuration dictionary

        Returns:
            Results dictionary with best_params, best_value, csv_path, total_tested
        """
        strategy_class = config['strategy_class']
        param_space = config['param_space']
        metric = config['optimization_metric']
        method = config.get('optimization_method', 'grid_search')
        symbols = config['symbols']
        start_date = config['start_date']
        end_date = config['end_date']

        # Create engine
        engine = BacktestEngine(
            initial_capital=config['initial_capital'],
            fees=config['fees'],
            allow_shorts=config.get('allow_shorts', True)  # Default True for optimization
        )

        # Select optimizer based on method
        if method == 'random_search':
            optimizer = RandomSearchOptimizer(engine)
            n_iterations = config.get('n_iterations', 100)
            log_info(f"Using Random Search with {n_iterations} iterations")

            # Run Random Search optimization
            result = optimizer.optimize(
                strategy_class=strategy_class,
                param_ranges=param_space,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                metric=metric,
                n_iterations=n_iterations,
                random_seed=None,  # Different results each time
                max_workers=config.get('workers', 1)
            )

        elif method == 'bayesian':
            if not BAYESIAN_AVAILABLE:
                raise ImportError(
                    "Bayesian optimization requires scikit-optimize. "
                    "Install it with: pip install scikit-optimize"
                )

            # Convert param_ranges to skopt.space dimensions
            from skopt.space import Integer, Real, Categorical

            param_space_skopt = []
            for param_name, param_spec in param_space.items():
                if isinstance(param_spec, tuple) and len(param_spec) >= 2:
                    # Numeric range (min, max) or (min, max, 'log')
                    min_val, max_val = param_spec[0], param_spec[1]
                    is_log = (len(param_spec) == 3 and param_spec[2] == 'log')

                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        param_space_skopt.append(Integer(min_val, max_val, name=param_name))
                    else:
                        # Real parameter
                        prior = 'log-uniform' if is_log else 'uniform'
                        param_space_skopt.append(Real(min_val, max_val, prior=prior, name=param_name))

                elif isinstance(param_spec, list):
                    # Categorical parameter
                    param_space_skopt.append(Categorical(param_spec, name=param_name))

            optimizer = BayesianOptimizer(engine)
            n_iterations = config.get('n_iterations', 50)
            n_initial_points = config.get('n_initial_points', 10)
            acquisition_func = config.get('acquisition_func', 'EI')

            log_info(f"Using Bayesian Optimization with {n_iterations} iterations")
            log_info(f"Initial random points: {n_initial_points}")
            log_info(f"Acquisition function: {acquisition_func}")

            # Run Bayesian optimization
            result = optimizer.optimize(
                strategy_class=strategy_class,
                param_space=param_space_skopt,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                metric=metric,
                n_iterations=n_iterations,
                n_initial_points=n_initial_points,
                acquisition_func=acquisition_func,
                random_seed=None,
                max_workers=config.get('workers', 1)
            )

        elif method == 'genetic':
            optimizer = GeneticOptimizer(engine)
            population_size = config.get('population_size', 50)
            n_generations = config.get('n_generations', 20)
            mutation_rate = config.get('mutation_rate', 0.1)
            crossover_rate = config.get('crossover_rate', 0.7)

            log_info(f"Using Genetic Algorithm")
            log_info(f"Population: {population_size}, Generations: {n_generations}")
            log_info(f"Mutation rate: {mutation_rate}, Crossover rate: {crossover_rate}")

            # Run Genetic Algorithm optimization
            result = optimizer.optimize(
                strategy_class=strategy_class,
                param_ranges=param_space,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                metric=metric,
                population_size=population_size,
                n_generations=n_generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                max_workers=config.get('workers', 1)
            )

        else:  # grid_search
            optimizer = GridSearchOptimizer(engine)
            log_info(f"Using Grid Search")

            # Run Grid Search optimization
            result = optimizer.optimize(
                strategy_class=strategy_class,
                param_grid=param_space,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                metric=metric,
                max_workers=config.get('workers', 1)
            )

        # Extract results from optimizer result
        best_params = result['best_params']
        best_value = result['best_value']
        csv_path = result.get('csv_path')  # CSV path from optimizer

        # Get total tested from result
        if method == 'random_search' or method == 'bayesian':
            total_tested = result.get('n_iterations', result.get('total_tested', 0))
        elif method == 'genetic':
            total_tested = result.get('total_evaluations', result.get('total_tested', 0))
        else:
            total_tested = result.get('total_tested', 0)

        log_info(f"Optimization complete. Best {metric}: {best_value:.4f}")
        log_info(f"Total combinations tested: {total_tested}")

        return {
            'best_params': best_params,
            'best_value': best_value,
            'metric': metric,
            'csv_path': csv_path,
            'total_tested': total_tested,
            'method': method
        }

    def _export_results_to_csv(
        self,
        all_results: list,
        strategy_class: type,
        metric: str
    ) -> Optional[Path]:
        """
        Export optimization results to CSV.

        Args:
            all_results: List of result dictionaries
            strategy_class: Strategy class being optimized
            metric: Optimization metric

        Returns:
            Path to CSV file, or None if no results
        """
        if not all_results:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}_{strategy_class.__name__}_optimization.csv"
        log_dir = get_log_output_dir()
        csv_path = Path(log_dir) / csv_filename
        csv_path.parent.mkdir(exist_ok=True, parents=True)

        df = pd.DataFrame(all_results)
        df = df.sort_values('metric_value', ascending=(metric == 'max_drawdown'))
        df.to_csv(csv_path, index=False)
        log_info(f"Optimization results exported to: {csv_path}")

        return csv_path

    def _show_results_dialog(self, result: Dict[str, Any], config: Dict[str, Any]):
        """
        Show optimization results dialog.

        Args:
            result: Results dictionary
            config: Original configuration
        """
        best_params = result['best_params']
        best_value = result['best_value']
        csv_path = result['csv_path']
        total_tested = result['total_tested']
        metric = result['metric']

        results_text = f"Best {metric}: {best_value:.4f}\n"
        results_text += f"Tested {total_tested} valid combinations\n\n"
        results_text += "Best Parameters:\n"
        for param_name, value in best_params.items():
            results_text += f"  • {param_name}: {value}\n"

        def apply_best_params(_):
            # Load best parameters into setup view
            for param_name, value in best_params.items():
                if param_name in self.setup_view.param_controls:
                    control = self.setup_view.param_controls[param_name]
                    if isinstance(control, ft.TextField):
                        control.value = str(value)
                    elif isinstance(control, ft.Checkbox):
                        control.value = bool(value)
                    elif isinstance(control, ft.Dropdown):
                        control.value = value
                    control.update()

            self.page.close(results_dialog)
            self.show_notification("Best parameters applied!", "success")
            self.show_setup_view()

        def open_csv(_):
            import os
            import subprocess
            if csv_path and csv_path.exists():
                if os.name == 'nt':  # Windows
                    os.startfile(str(csv_path))
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(csv_path)])

        actions: list = [
            ft.TextButton("Close", on_click=lambda _: self.page.close(results_dialog))
        ]

        if csv_path:
            actions.append(
                ft.ElevatedButton(
                    "Open CSV",
                    icon=ft.Icons.TABLE_CHART,
                    on_click=open_csv,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.ORANGE_700
                    )
                )
            )

        actions.append(
            ft.ElevatedButton(
                "Apply Best Parameters",
                icon=ft.Icons.CHECK_CIRCLE,
                on_click=apply_best_params,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREEN_700
                )
            )
        )

        content_items = [
            ft.Text(results_text, selectable=True),
            ft.Divider()
        ]

        if csv_path:
            content_items.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.TABLE_CHART, color=ft.Colors.GREEN_400),
                        ft.Text(
                            f"Results exported to:\n{csv_path.name}",
                            size=12,
                            color=ft.Colors.GREEN_400,
                            selectable=True
                        )
                    ], spacing=5),
                    bgcolor=ft.Colors.GREY_900,
                    border=ft.border.all(1, ft.Colors.GREEN_700),
                    border_radius=5,
                    padding=10
                )
            )
            content_items.append(ft.Divider())

        content_items.append(
            ft.Text("Apply these parameters to your strategy?", weight=ft.FontWeight.BOLD)
        )

        results_dialog = ft.AlertDialog(
            title=ft.Text("Optimization Complete!", color=ft.Colors.GREEN_400),
            content=ft.Column(
                content_items,
                spacing=10,
                height=350
            ),
            actions=actions,
            actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )
        self.page.open(results_dialog)
        self.page.update()
