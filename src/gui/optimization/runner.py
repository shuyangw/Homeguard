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
                - param_grid: Parameter grid dictionary
                - optimization_metric: Metric to optimize
                - symbols: List of symbols
                - start_date: Start date
                - end_date: End date
                - initial_capital: Initial capital
                - fees: Trading fees
        """
        progress_dialog = None
        try:
            strategy_class = config['strategy_class']
            param_grid = config['param_grid']
            metric = config['optimization_metric']
            symbols = config['symbols']
            start_date = config['start_date']
            end_date = config['end_date']

            log_info(f"Starting optimization: {strategy_class.__name__}")
            log_info(f"Parameter grid: {param_grid}")
            log_info(f"Optimization metric: {metric}")

            # Show progress dialog
            progress_dialog = ft.AlertDialog(
                title=ft.Text("Running Optimization..."),
                content=ft.Column([
                    ft.Text(f"Strategy: {strategy_class.__name__}"),
                    ft.Text(f"Metric: {metric}"),
                    ft.ProgressRing(),
                    ft.Text("This may take several minutes...")
                ], spacing=10, height=150),
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
        param_grid = config['param_grid']
        metric = config['optimization_metric']
        symbols = config['symbols']
        start_date = config['start_date']
        end_date = config['end_date']

        # Create engine
        engine = BacktestEngine(
            initial_capital=config['initial_capital'],
            fees=config['fees']
        )

        # Track all results for CSV export
        all_results = []
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(product(*param_values))

        log_info(f"Testing {len(combinations)} parameter combinations...")

        best_value = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = None
        valid_combinations = 0

        for param_combo in combinations:
            params = dict(zip(param_names, param_combo))

            try:
                # Test if strategy can be instantiated with these params
                strategy = strategy_class(**params)

                # Run backtest
                if len(symbols) == 1:
                    data = engine.data_loader.load_symbols(symbols, start_date, end_date)
                    portfolio = engine._run_single_symbol(strategy, data, symbols[0], 'close')
                else:
                    data = engine.data_loader.load_symbols(symbols, start_date, end_date)
                    portfolio = engine._run_multiple_symbols(strategy, data, symbols, 'close')

                stats = portfolio.stats()

                if stats is not None:
                    # Extract metric value
                    if metric == 'sharpe_ratio':
                        value = float(stats.get('Sharpe Ratio', float('-inf')))
                    elif metric == 'total_return':
                        value = float(stats.get('Total Return [%]', float('-inf')))
                    elif metric == 'max_drawdown':
                        value = float(stats.get('Max Drawdown [%]', float('inf')))
                    else:
                        value = float('-inf')

                    # Track result
                    result_row = params.copy()
                    result_row['metric_value'] = value
                    result_row['sharpe_ratio'] = float(stats.get('Sharpe Ratio', 0))
                    result_row['total_return'] = float(stats.get('Total Return [%]', 0))
                    result_row['max_drawdown'] = float(stats.get('Max Drawdown [%]', 0))
                    result_row['win_rate'] = float(stats.get('Win Rate [%]', 0))
                    result_row['total_trades'] = int(stats.get('# Trades', 0))
                    all_results.append(result_row)

                    valid_combinations += 1

                    # Check if this is the best
                    is_better = (
                        (metric != 'max_drawdown' and value > best_value) or
                        (metric == 'max_drawdown' and value < best_value)
                    )

                    if is_better:
                        best_value = value
                        best_params = params
                        log_info(f"New best {metric}: {best_value:.4f} with params {best_params}")

            except ValueError as ve:
                # Invalid parameter combination (e.g., fast_window >= slow_window)
                log_info(f"Skipping invalid combination {params}: {ve}")
                continue
            except Exception as e:
                log_error(f"Error testing params {params}: {e}")
                continue

        log_info(f"Optimization complete. Tested {valid_combinations}/{len(combinations)} valid combinations")

        # Export results to CSV
        csv_path = self._export_results_to_csv(all_results, strategy_class, metric)

        return {
            'best_params': best_params,
            'best_value': best_value,
            'metric': metric,
            'csv_path': csv_path,
            'total_tested': valid_combinations
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
