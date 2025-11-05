"""
Main Flet application for backtesting GUI.
"""

import flet as ft
import asyncio
from typing import Dict, Any

from gui.views.setup_view import SetupView
from gui.views.run_view import RunView
from gui.views.results_view import ResultsView
from gui.workers.gui_controller import GUIBacktestController
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.engine.results_aggregator import ResultsAggregator
from gui.utils.error_logger import log_error, log_info, log_exception
from gui.utils.run_history import RunHistory
from utils.cache_manager import CacheManager


class BacktestApp:
    """
    Main backtesting GUI application.

    Manages navigation between setup, execution, and results views.
    Integrates with GUIBacktestController for running backtests.
    """

    def __init__(self, page: ft.Page):
        """
        Initialize application.

        Args:
            page: Flet Page object
        """
        try:
            log_info("Initializing BacktestApp")
            self.page = page
            self.page.title = "Backtest Runner"
            self.page.theme_mode = ft.ThemeMode.DARK
            self.page.padding = 0

            # Set window size to accommodate all UI elements
            self.page.window.width = 1400
            self.page.window.height = 900
            self.page.window.min_width = 1200
            self.page.window.min_height = 700

            # Views
            self.setup_view = None
            self.run_view = None
            self.results_view = None
            self.current_view = None  # Track active view

            # Controller
            self.controller: GUIBacktestController = None

            # State
            self.current_config: Dict[str, Any] = None
            self.poll_task = None

            # Run history tracker
            self.run_history = RunHistory()

            # Cache manager
            self.cache_manager = CacheManager()

            # Setup keyboard shortcuts
            self.page.on_keyboard_event = self._on_keyboard

            # Build UI
            self._build_ui()
            log_info("BacktestApp initialized successfully")

        except Exception as e:
            log_exception(e, "Error initializing BacktestApp")
            self._show_error_dialog("Initialization Error", str(e))
            raise

    def _build_ui(self):
        """Build the application UI."""
        # Create views
        self.setup_view = SetupView(on_run_clicked=self._on_run_backtests)
        self.run_view = RunView()
        self.run_view.cancel_button.on_click = lambda e: self._on_cancel_backtests()
        self.run_view.view_results_button.on_click = lambda e: self._show_results_view()
        self.run_view.return_to_menu_button.on_click = lambda e: self._show_setup_view()
        self.results_view = ResultsView(on_back_clicked=self._show_setup_view)

        # Start with setup view
        self._show_setup_view()

    def _show_setup_view(self):
        """Show the setup view."""
        self.current_view = 'setup'
        self.page.controls.clear()
        self.page.controls.append(self.setup_view)
        self.page.update()

    def _show_run_view(self):
        """Show the combined run view."""
        self.current_view = 'run'
        self.page.controls.clear()
        self.page.controls.append(self.run_view)
        self.page.update()

    def _show_results_view(self):
        """Show the results view."""
        try:
            self.current_view = 'results'

            # Stop polling if still running
            if self.poll_task:
                self.poll_task = None

            # Get results
            if self.controller:
                results = self.controller.get_results()
                df = ResultsAggregator.aggregate_results(results)
                self.results_view.load_results(df)
                log_info(f"Loaded {len(df)} results")

            self.page.controls.clear()
            self.page.controls.append(self.results_view)
            self.page.update()

        except Exception as e:
            log_exception(e, "Error showing results view")
            self._show_error_dialog("Results Error", f"Failed to load results: {str(e)}")

    def _on_run_backtests(self, config: Dict[str, Any]):
        """
        Handle Run Backtests button click.

        Args:
            config: Configuration dictionary from SetupView
        """
        try:
            log_info(f"Starting backtests for {len(config['symbols'])} symbols")
            log_info(f"Strategy: {config['strategy_class'].__name__}")
            log_info(f"Date range: {config['start_date']} to {config['end_date']}")

            self.current_config = config

            # Check if results are cached
            if self.cache_manager.is_cached(config):
                self._show_cache_found_dialog(config)
                return

            # No cache found, run new backtest
            self._run_new_backtest(config)

        except Exception as e:
            log_exception(e, "Error starting backtests")
            self._show_notification(f"Failed to start backtests: {str(e)}", "error")
            self._show_error_dialog("Backtest Start Error", f"Failed to start backtests: {str(e)}")

    def _start_polling(self):
        """Start polling for backtest updates."""
        if self.poll_task is None:
            self.poll_task = True
            self.page.run_task(self._poll_updates)

    async def _poll_updates(self):
        """Async task to poll for backtest updates."""
        try:
            while self.poll_task and self.controller:
                try:
                    # Get updates from controller
                    updates = self.controller.get_updates()

                    for symbol, update_data in updates.items():
                        # Update status
                        status = self.controller.get_status(symbol)
                        self.run_view.update_symbol_status(symbol, status)

                        # Update progress
                        if 'progress' in update_data and update_data['progress']:
                            latest_progress = update_data['progress'][-1]
                            self.run_view.update_symbol_progress(
                                symbol,
                                latest_progress.progress,
                                latest_progress.message
                            )

                    # Update overall progress
                    summary = self.controller.get_progress_summary()
                    self.run_view.update_overall_progress(
                        completed=summary['completed'],
                        total=summary['total'],
                        running=summary['running'],
                        failed=summary['failed']
                    )

                    # Phase 3: Get worker updates
                    worker_updates = self.controller.get_worker_updates()

                    # Update worker status
                    for status_update in worker_updates['status_updates']:
                        self.run_view.update_worker_status(
                            status_update.worker_id,
                            status_update.symbol,
                            status_update.status
                        )

                    # Update worker logs
                    for worker_id, log_messages in worker_updates['logs'].items():
                        for log_msg in log_messages:
                            self.run_view.add_worker_log(
                                worker_id,
                                log_msg.message,
                                log_msg.level
                            )

                    # Check if all done
                    if summary['completed'] + summary['failed'] >= summary['total']:
                        # Detect breaking errors: all symbols failed OR no results available
                        has_breaking_errors = (summary['completed'] == 0 and summary['failed'] > 0)

                        self.run_view.mark_complete(has_breaking_errors=has_breaking_errors)
                        self.poll_task = None
                        log_info("All backtests completed")

                        # Add to run history
                        if self.current_config:
                            strategy_name = self.current_config.get('strategy_class').__name__
                            self.run_history.add_run(
                                strategy_name=strategy_name,
                                symbols=self.current_config.get('symbols', []),
                                start_date=self.current_config.get('start_date', ''),
                                end_date=self.current_config.get('end_date', ''),
                                config=self.current_config,
                                results={
                                    'completed': summary['completed'],
                                    'failed': summary['failed'],
                                    'total': summary['total']
                                }
                            )

                        # Cache results if successful
                        if summary['completed'] > 0 and not has_breaking_errors:
                            try:
                                results = self.controller.get_results()
                                df = ResultsAggregator.aggregate_results(results)
                                portfolios = self.controller.get_portfolios() if hasattr(self.controller, 'get_portfolios') else None

                                strategy_name = self.current_config.get('strategy_class').__name__
                                description = f"{strategy_name} - {len(self.current_config.get('symbols', []))} symbols"

                                self.cache_manager.cache_results(
                                    self.current_config,
                                    df,
                                    portfolios,
                                    description
                                )
                                log_info("Results cached successfully")
                            except Exception as e:
                                log_error(f"Failed to cache results: {e}")

                        # Show completion notification
                        if summary['failed'] > 0:
                            self._show_notification(
                                f"Backtests complete! {summary['completed']} succeeded, {summary['failed']} failed",
                                "warning"
                            )
                        else:
                            self._show_notification(
                                f"All {summary['completed']} backtests completed successfully!",
                                "success"
                            )
                        break

                except Exception as e:
                    # Log but continue polling
                    log_error(f"Error during polling update: {e}")

                # Poll every 200ms
                await asyncio.sleep(0.2)

        except Exception as e:
            log_exception(e, "Fatal error in polling loop")
            self.poll_task = None
            self._show_error_dialog("Polling Error", f"Lost connection to backtest runner: {str(e)}")

    def _on_cancel_backtests(self):
        """Handle Cancel button click."""
        try:
            if self.controller:
                self.controller.cancel()

            # Stop polling
            if self.poll_task:
                self.poll_task = None

            # Show notification
            self._show_notification("Backtests cancelled - stopping after current symbols", "warning")

            # Show dialog
            dlg = ft.AlertDialog(
                title=ft.Text("Cancellation Requested"),
                content=ft.Text("Backtests will stop after current symbols complete."),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(dlg))]
            )
            self.page.open(dlg)

        except Exception as e:
            log_exception(e, "Error cancelling backtests")
            self._show_notification(f"Cancellation failed: {str(e)}", "error")
            self._show_error_dialog("Cancel Error", str(e))

    def _on_keyboard(self, e: ft.KeyboardEvent):
        """
        Handle keyboard shortcuts.

        Shortcuts:
        - Ctrl+R: Run backtests (setup view)
        - Esc: Cancel backtests (run view)
        - Ctrl+Q: Quick re-run (setup view)
        - F5: Refresh/view results
        """
        try:
            # Check if key event is key down
            if e.key_type != ft.KeyType.KEY_DOWN:
                return

            # Ctrl+R: Run backtests
            if e.key == "R" and e.ctrl:
                if self.current_view == 'setup':
                    self.setup_view.run_button.on_click(None)
                    log_info("Keyboard shortcut: Ctrl+R - Run backtests")

            # Esc: Cancel backtests
            elif e.key == "Escape":
                if self.current_view == 'run':
                    self._on_cancel_backtests()
                    log_info("Keyboard shortcut: Esc - Cancel backtests")

            # Ctrl+Q: Quick re-run
            elif e.key == "Q" and e.ctrl:
                if self.current_view == 'setup' and not self.setup_view.quick_rerun_button.disabled:
                    self.setup_view.quick_rerun_button.on_click(None)
                    log_info("Keyboard shortcut: Ctrl+Q - Quick re-run")

            # F5: View results
            elif e.key == "F5":
                if self.current_view == 'run' and not self.run_view.view_results_button.disabled:
                    self._show_results_view()
                    log_info("Keyboard shortcut: F5 - View results")

        except Exception as ex:
            log_error(f"Error handling keyboard shortcut: {ex}")

    def _show_notification(self, message: str, notification_type: str = "info"):
        """
        Show a toast-style notification to the user.

        Args:
            message: Notification message
            notification_type: Type of notification ("success", "error", "warning", "info")
        """
        try:
            # Color based on type
            colors = {
                "success": ft.Colors.GREEN_700,
                "error": ft.Colors.RED_700,
                "warning": ft.Colors.AMBER_700,
                "info": ft.Colors.BLUE_700
            }

            snackbar = ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE),
                bgcolor=colors.get(notification_type, ft.Colors.BLUE_700),
                duration=3000  # 3 seconds
            )

            self.page.overlay.append(snackbar)
            snackbar.open = True
            self.page.update()

        except Exception as e:
            log_error(f"Error showing notification: {e}")

    def _show_error_dialog(self, title: str, message: str):
        """
        Show error dialog to user.

        Args:
            title: Error dialog title
            message: Error message
        """
        try:
            dlg = ft.AlertDialog(
                title=ft.Text(title),
                content=ft.Text(message),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(dlg))]
            )
            self.page.open(dlg)
        except Exception as e:
            # Last resort - log but don't throw
            log_error(f"Failed to show error dialog: {e}")

    def _show_cache_found_dialog(self, config: Dict[str, Any]):
        """
        Show dialog when cached results are found.

        Args:
            config: Configuration dictionary
        """
        try:
            strategy_name = config['strategy_class'].__name__
            num_symbols = len(config['symbols'])

            content = ft.Column([
                ft.Text(
                    "Cached results found for this configuration!",
                    size=14,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Divider(),
                ft.Text(f"Strategy: {strategy_name}", size=12),
                ft.Text(f"Symbols: {num_symbols}", size=12),
                ft.Text(f"Date Range: {config['start_date']} to {config['end_date']}", size=12),
                ft.Divider(),
                ft.Text(
                    "Would you like to load cached results or run a new backtest?",
                    size=13,
                    color=ft.Colors.GREY_600
                )
            ], spacing=8, tight=True)

            def load_cached(e):
                self.page.close(dlg)
                self._load_cached_results(config)

            def run_new(e):
                self.page.close(dlg)
                self._run_new_backtest(config)

            dlg = ft.AlertDialog(
                title=ft.Text("Cached Results Available"),
                content=content,
                actions=[
                    ft.TextButton("Run New", on_click=run_new),
                    ft.ElevatedButton(
                        "Load Cached",
                        icon=ft.Icons.CACHED,
                        on_click=load_cached,
                        style=ft.ButtonStyle(
                            color=ft.Colors.WHITE,
                            bgcolor=ft.Colors.GREEN_700
                        )
                    )
                ],
                actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            )

            self.page.open(dlg)

        except Exception as e:
            log_exception(e, "Error showing cache dialog")
            self._run_new_backtest(config)

    def _load_cached_results(self, config: Dict[str, Any]):
        """
        Load cached results and display them.

        Args:
            config: Configuration dictionary
        """
        try:
            cached = self.cache_manager.get_cached_results(config)

            if cached is None:
                self._show_notification("Cache not found, running new backtest", "warning")
                self._run_new_backtest(config)
                return

            log_info("Loading cached results")

            # Load results directly into results view
            self.results_view.load_results(cached['results_df'])
            self._show_results_view()

            self._show_notification("Loaded cached results", "success")

        except Exception as e:
            log_exception(e, "Error loading cached results")
            self._show_notification("Failed to load cache, running new backtest", "error")
            self._run_new_backtest(config)

    def _run_new_backtest(self, config: Dict[str, Any]):
        """
        Run a new backtest (skip cache).

        Args:
            config: Configuration dictionary
        """
        try:
            # Create strategy instance
            strategy_class = config['strategy_class']
            strategy_params = config['strategy_params']
            strategy = strategy_class(**strategy_params)

            # Create controller
            self.controller = GUIBacktestController(max_workers=config['workers'])

            # Show combined run view with configuration
            self.run_view.set_configuration(config)

            # Initialize symbols and workers based on portfolio mode
            portfolio_mode = config.get('portfolio_mode', 'Single-Symbol')
            if portfolio_mode == "Multi-Symbol Portfolio":
                # Single portfolio entry
                symbols_display = f"Portfolio ({', '.join(config['symbols'])})"
                display_symbols = [symbols_display]
                # Portfolio mode uses only 1 worker
                num_workers = 1
            else:
                # Individual symbol entries
                display_symbols = config['symbols']
                # Single-symbol mode uses configured workers
                num_workers = config['workers']

            self.run_view.initialize_symbols(display_symbols)
            self.run_view.initialize_workers(num_workers)
            self.run_view.mark_running()
            self._show_run_view()

            # Show notification
            if portfolio_mode == "Multi-Symbol Portfolio":
                self._show_notification(
                    f"Running portfolio backtest with {len(config['symbols'])} symbols...",
                    "info"
                )
            else:
                self._show_notification(
                    f"Running {len(config['symbols'])} backtests...",
                    "info"
                )

            # Start backtests in background
            self.controller.start_backtests(
                strategy=strategy,
                symbols=config['symbols'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                initial_capital=config.get('initial_capital', 100000.0),
                fees=config.get('fees', 0.001),
                risk_profile=config.get('risk_profile', 'Moderate'),
                generate_full_output=config.get('generate_full_output', True),
                portfolio_mode=config.get('portfolio_mode', 'Single-Symbol'),
                position_sizing_method=config.get('position_sizing_method', 'equal_weight'),
                rebalancing_frequency=config.get('rebalancing_frequency', 'never'),
                rebalancing_threshold_pct=config.get('rebalancing_threshold_pct', 0.05)
            )

            # Start polling for updates
            self._start_polling()

        except Exception as e:
            log_exception(e, "Error running backtest")
            self._show_notification(f"Failed to run backtest: {str(e)}", "error")


def main(page: ft.Page):
    """
    Main entry point for Flet application.

    Args:
        page: Flet Page object
    """
    try:
        log_info("Starting Backtest Runner GUI")
        app = BacktestApp(page)
    except Exception as e:
        log_exception(e, "Fatal error starting GUI")
        # Show error in page
        error_text = ft.Text(
            f"Fatal Error: {str(e)}\n\nCheck logs at: C:\\Users\\qwqw1\\Dropbox\\cs\\stonk\\homeguard_gui_logs",
            color=ft.Colors.RED,
            size=14
        )
        page.add(error_text)
        page.update()


if __name__ == "__main__":
    ft.app(target=main)
