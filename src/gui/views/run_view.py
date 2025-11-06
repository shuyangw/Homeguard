"""
Combined run view - shows configuration and execution monitoring in one responsive layout.
"""

import flet as ft
from typing import Dict, Any, List
from datetime import datetime, timedelta
from gui.views.worker_log_viewer import WorkerLogViewer


class RunView(ft.Container):
    """
    Combined view showing backtest configuration and live execution monitoring.

    Layout:
    - Left panel (30%): Configuration summary (read-only during execution)
    - Right panel (70%): Execution monitoring with symbol progress and worker logs
    """

    def __init__(self):
        super().__init__()

        # Configuration display
        self.config_text = ft.Text("", size=12, selectable=True)

        # Execution state
        self.symbol_cards: Dict[str, Dict[str, Any]] = {}
        self.worker_viewers: List[WorkerLogViewer] = []

        # Time tracking for progress estimates
        self.start_time = None
        self.completion_times: List[float] = []  # Track time for each completed symbol
        self.total_symbols = 0

        # Progress components
        self.overall_progress = ft.ProgressBar(value=0, height=20, expand=True)
        self.progress_text = ft.Text("Ready to run", size=14, weight=ft.FontWeight.BOLD)
        self.time_elapsed_text = ft.Text("", size=12, color=ft.Colors.CYAN_300)
        self.time_remaining_text = ft.Text("", size=12, color=ft.Colors.AMBER_300)
        self.eta_text = ft.Text("", size=12, color=ft.Colors.GREEN_300)
        self.cancel_button = ft.ElevatedButton(
            "Cancel",
            icon=ft.Icons.CANCEL,
            on_click=lambda e: None,  # Will be set by app
            disabled=True,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.RED_700
        )
        self.view_results_button = ft.ElevatedButton(
            "View Results",
            icon=ft.Icons.ASSESSMENT,
            on_click=lambda e: None,  # Will be set by app
            disabled=True,
            visible=False,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE_700
        )
        self.return_to_menu_button = ft.ElevatedButton(
            "Return to Main Menu",
            icon=ft.Icons.HOME,
            on_click=lambda e: None,  # Will be set by app
            disabled=False,
            visible=False,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.ORANGE_700
        )

        # Symbol progress container
        self.symbols_column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            spacing=10,
            expand=True
        )

        # Worker logs container
        self.workers_column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            spacing=10,
            expand=True
        )

        self._build_ui()

    def _build_ui(self):
        """Build the combined responsive layout."""

        # Left panel: Configuration summary (30%)
        left_panel = ft.Container(
            content=ft.Column([
                ft.Text("Configuration", size=18, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Container(
                    content=self.config_text,
                    bgcolor=ft.Colors.GREY_900,
                    border=ft.border.all(2, ft.Colors.CYAN_700),
                    border_radius=8,
                    padding=15,
                    expand=True
                )
            ], spacing=10, expand=True),
            expand=3,  # 30% width
            padding=10
        )

        # Right panel: Execution monitoring (70%)
        right_panel = ft.Container(
            content=ft.Column([
                # Overall progress header
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            self.progress_text,
                            self.cancel_button,
                            self.view_results_button,
                            self.return_to_menu_button
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        self.overall_progress,
                        # Time tracking row
                        ft.Row([
                            self.time_elapsed_text,
                            self.time_remaining_text,
                            self.eta_text
                        ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
                    ], spacing=10),
                    bgcolor=ft.Colors.GREY_900,
                    border=ft.border.all(2, ft.Colors.PURPLE_700),
                    border_radius=8,
                    padding=15
                ),

                # Split: Symbols (left) and Workers (right)
                ft.Container(
                    content=ft.Row([
                        # Symbol progress (40%)
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Symbol Progress", size=16, weight=ft.FontWeight.BOLD),
                                ft.Divider(),
                                self.symbols_column
                            ], spacing=5, expand=True),
                            bgcolor=ft.Colors.GREY_900,
                            border=ft.border.all(2, ft.Colors.BLUE_700),
                            border_radius=8,
                            padding=10,
                            expand=4  # 40% of right panel
                        ),

                        # Worker logs (60%)
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Worker Threads", size=16, weight=ft.FontWeight.BOLD),
                                ft.Divider(),
                                self.workers_column
                            ], spacing=5, expand=True),
                            bgcolor=ft.Colors.GREY_900,
                            border=ft.border.all(2, ft.Colors.GREEN_700),
                            border_radius=8,
                            padding=10,
                            expand=6  # 60% of right panel
                        )
                    ], spacing=10, expand=True),
                    expand=True
                )
            ], spacing=10, expand=True),
            expand=7,  # 70% width
            padding=10
        )

        # Main responsive row
        self.content = ft.Row([
            left_panel,
            right_panel
        ], spacing=0, expand=True)

        self.expand = True

    def set_configuration(self, config: Dict[str, Any]):
        """
        Display configuration summary.

        Args:
            config: Configuration dictionary from setup
        """
        strategy_name = config['strategy_class'].__name__
        symbols = ', '.join(config['symbols'])
        params_str = '\n'.join([f"  {k}: {v}" for k, v in config['strategy_params'].items()])

        # Get risk profile with description
        risk_profile = config.get('risk_profile', 'Moderate')
        risk_descriptions = {
            'Conservative': 'Conservative (5% per trade, 1% stop loss)',
            'Moderate': 'Moderate (10% per trade, 2% stop loss)',
            'Aggressive': 'Aggressive (20% per trade, 3% stop loss)',
            'Disabled': '⚠️ Disabled (99% per trade - unrealistic)'
        }
        risk_desc = risk_descriptions.get(risk_profile, risk_profile)

        config_summary = f"""Strategy: {strategy_name}

Symbols: {symbols}

Date Range:
  Start: {config['start_date']}
  End: {config['end_date']}

Parameters:
{params_str}

Risk Management:
  Profile: {risk_desc}

Execution:
  Workers: {config['workers']}
  Parallel: {config['parallel']}
  Full Output: {config.get('generate_full_output', True)}
"""

        self.config_text.value = config_summary
        if self.page:
            self.update()

    def initialize_symbols(self, symbols: List[str]):
        """Initialize symbol progress cards."""
        self.total_symbols = len(symbols)
        self.symbol_cards.clear()
        self.symbols_column.controls.clear()

        for symbol in symbols:
            # Progress bar
            progress_bar = ft.ProgressBar(value=0, width=200, height=10)

            # Status text
            status_text = ft.Text("Pending", size=12, color=ft.Colors.GREY_400)

            # Message text
            message_text = ft.Text("", size=11, color=ft.Colors.GREY_500)

            # Card
            card = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text(symbol, size=14, weight=ft.FontWeight.BOLD),
                        status_text
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    progress_bar,
                    message_text
                ], spacing=5),
                bgcolor=ft.Colors.GREY_800,
                border=ft.border.all(1, ft.Colors.GREY_600),
                border_radius=8,
                padding=10
            )

            self.symbol_cards[symbol] = {
                'card': card,
                'progress_bar': progress_bar,
                'status_text': status_text,
                'message_text': message_text
            }

            self.symbols_column.controls.append(card)

        if self.page:
            self.update()

    def initialize_workers(self, num_workers: int):
        """Initialize worker log viewers."""
        self.worker_viewers.clear()
        self.workers_column.controls.clear()

        for i in range(num_workers):
            viewer = WorkerLogViewer(worker_id=i)
            self.worker_viewers.append(viewer)
            self.workers_column.controls.append(viewer)

        if self.page:
            self.update()

    def update_symbol_status(self, symbol: str, status: str):
        """Update symbol status."""
        if symbol not in self.symbol_cards:
            return

        status_text = self.symbol_cards[symbol]['status_text']
        status_text.value = status.title()

        # Color code status
        if status == "running":
            status_text.color = ft.Colors.BLUE_400
        elif status == "completed":
            status_text.color = ft.Colors.GREEN_400
        elif status == "failed":
            status_text.color = ft.Colors.RED_400
        else:
            status_text.color = ft.Colors.GREY_400

        if self.page:
            status_text.update()

    def update_symbol_progress(self, symbol: str, progress: float, message: str):
        """Update symbol progress."""
        if symbol not in self.symbol_cards:
            return

        self.symbol_cards[symbol]['progress_bar'].value = progress
        self.symbol_cards[symbol]['message_text'].value = message

        if self.page:
            self.symbol_cards[symbol]['progress_bar'].update()
            self.symbol_cards[symbol]['message_text'].update()

    def update_overall_progress(self, completed: int, total: int, running: int, failed: int):
        """Update overall progress."""
        if total > 0:
            progress = completed / total
            self.overall_progress.value = progress

            status = f"Progress: {completed}/{total} completed"
            if running > 0:
                status += f" | {running} running"
            if failed > 0:
                status += f" | {failed} failed"

            self.progress_text.value = status

            # Update time estimates (pass total to avoid mismatch)
            self.update_time_estimates(completed, total)

            if self.page:
                self.overall_progress.update()
                self.progress_text.update()

    def update_time_estimates(self, completed: int, total: int):
        """
        Calculate and display time estimates based on completed symbols.

        Args:
            completed: Number of completed symbols
            total: Total number of symbols (passed to avoid using stale self.total_symbols)
        """
        if not self.start_time or total == 0:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.time_elapsed_text.value = f"⏱️ Elapsed: {timedelta(seconds=int(elapsed))}"

        if completed > 0:
            avg_time = elapsed / completed
            remaining_symbols = total - completed
            remaining_seconds = avg_time * remaining_symbols

            # Guard against negative remaining time
            if remaining_seconds < 0:
                remaining_seconds = 0

            self.time_remaining_text.value = f"⏳ Remaining: ~{timedelta(seconds=int(remaining_seconds))}"

            eta = datetime.now() + timedelta(seconds=remaining_seconds)
            self.eta_text.value = f"🎯 ETA: {eta.strftime('%H:%M:%S')}"
        else:
            self.time_remaining_text.value = ""
            self.eta_text.value = ""

        if self.page:
            self.time_elapsed_text.update()
            self.time_remaining_text.update()
            self.eta_text.update()

    def mark_complete(self, has_breaking_errors: bool = False):
        """
        Mark execution as complete.

        Args:
            has_breaking_errors: If True, show "Return to Main Menu" instead of "View Results"
        """
        # Show final time
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.time_elapsed_text.value = f"⏱️ Total Time: {timedelta(seconds=int(elapsed))}"
            self.time_remaining_text.value = ""
            self.eta_text.value = "✅ Complete!" if not has_breaking_errors else "❌ Failed!"
            if self.page:
                self.time_elapsed_text.update()
                self.time_remaining_text.update()
                self.eta_text.update()

        self.progress_text.value = "Backtests Complete!" if not has_breaking_errors else "Backtests Failed!"
        self.progress_text.color = ft.Colors.GREEN_400 if not has_breaking_errors else ft.Colors.RED_400
        self.cancel_button.visible = False  # Hide cancel button when complete

        if has_breaking_errors:
            # Show return to menu button instead of view results
            self.view_results_button.visible = False
            self.view_results_button.disabled = True
            self.return_to_menu_button.visible = True
            self.return_to_menu_button.disabled = False
        else:
            # Show both view results and return to menu buttons
            self.view_results_button.disabled = False
            self.view_results_button.visible = True
            self.return_to_menu_button.visible = True
            self.return_to_menu_button.disabled = False

        if self.page:
            self.update()

    def mark_running(self):
        """Mark execution as running."""
        # Start time tracking
        self.start_time = datetime.now()
        self.completion_times.clear()

        self.cancel_button.disabled = False
        self.view_results_button.disabled = True
        self.view_results_button.visible = False
        self.return_to_menu_button.visible = False

        if self.page:
            self.update()

    def add_worker_log(self, worker_id: int, message: str, level: str = "info"):
        """Add log message to worker viewer."""
        if 0 <= worker_id < len(self.worker_viewers):
            self.worker_viewers[worker_id].add_log(message, level)

    def set_worker_symbol(self, worker_id: int, symbol: str):
        """Set current symbol for worker."""
        if 0 <= worker_id < len(self.worker_viewers):
            self.worker_viewers[worker_id].set_symbol(symbol)

    def set_worker_idle(self, worker_id: int):
        """Mark worker as idle."""
        if 0 <= worker_id < len(self.worker_viewers):
            self.worker_viewers[worker_id].set_idle()

    def update_worker_status(self, worker_id: int, symbol: str | None, status: str):
        """Update worker status."""
        if status == "idle" or symbol is None:
            self.set_worker_idle(worker_id)
        else:
            self.set_worker_symbol(worker_id, symbol)
