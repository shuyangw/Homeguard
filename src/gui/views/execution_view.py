"""
Execution view for monitoring backtest progress in real-time.
"""

import flet as ft
from typing import Dict, Callable, List
from datetime import datetime

from gui.views.worker_log_viewer import WorkerLogViewer


class SymbolCard(ft.Container):
    """Card displaying progress for a single symbol."""

    def __init__(self, symbol: str):
        """
        Initialize symbol card.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        """
        super().__init__()

        self.symbol = symbol

        # Status indicators
        self.status_icon = ft.Icon(ft.Icons.PENDING, color=ft.Colors.GREY_400, size=20)
        self.symbol_text = ft.Text(symbol, size=18, weight=ft.FontWeight.BOLD)
        self.status_text = ft.Text("Pending", size=12, color=ft.Colors.GREY_600)

        # Progress bar
        self.progress_bar = ft.ProgressBar(value=0, width=300, color=ft.Colors.BLUE)

        # Message text
        self.message_text = ft.Text("", size=12, italic=True)

        # Build card
        self._build_card()

    def _build_card(self):
        """Build the card layout."""
        self.content = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            self.status_icon,
                            self.symbol_text,
                            ft.Container(expand=True),  # Spacer
                            self.status_text
                        ],
                        alignment=ft.MainAxisAlignment.START
                    ),
                    self.progress_bar,
                    self.message_text
                ],
                spacing=8
            ),
            padding=15,
            width=400,
            border=ft.border.all(2, ft.Colors.BLUE_700),
            border_radius=10,
            bgcolor=ft.Colors.GREY_900
        )

    def update_status(self, status: str):
        """
        Update symbol status.

        Args:
            status: One of "pending", "running", "completed", "failed"
        """
        status_config = {
            "pending": (ft.Icons.PENDING, ft.Colors.GREY_400, "Pending"),
            "running": (ft.Icons.HOURGLASS_EMPTY, ft.Colors.BLUE, "Running"),
            "completed": (ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN, "Completed"),
            "failed": (ft.Icons.ERROR, ft.Colors.RED, "Failed")
        }

        if status in status_config:
            icon, color, text = status_config[status]
            self.status_icon.name = icon
            self.status_icon.color = color
            self.status_text.value = text
            self.status_text.color = color

    def update_progress(self, progress: float, message: str = ""):
        """
        Update progress bar and message.

        Args:
            progress: Progress value 0.0 - 1.0
            message: Progress message (e.g., "Loading data...")
        """
        self.progress_bar.value = progress
        self.message_text.value = message


class ExecutionView(ft.Container):
    """
    View for monitoring backtest execution in real-time.

    Shows:
    - Overall progress summary
    - Individual progress cards for each symbol
    - Cancel button
    """

    def __init__(self, on_cancel_clicked: Callable[[], None], on_view_results_clicked: Callable[[], None]):
        """
        Initialize execution view.

        Args:
            on_cancel_clicked: Callback when Cancel button is clicked
            on_view_results_clicked: Callback when View Results button is clicked
        """
        super().__init__()

        self.on_cancel_clicked = on_cancel_clicked
        self.on_view_results_clicked = on_view_results_clicked

        # Symbol cards
        self.symbol_cards: Dict[str, SymbolCard] = {}
        self.cards_container = None

        # Worker log viewers
        self.worker_viewers: List[WorkerLogViewer] = []
        self.workers_container = None

        # Summary components
        self.summary_text = None
        self.overall_progress = None
        self.cancel_button = None
        self.view_results_button = None

        # State
        self.is_running = False
        self.num_workers = 0

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the execution view UI."""
        # Summary section
        self.summary_text = ft.Text(
            "Waiting to start...",
            size=16,
            weight=ft.FontWeight.W_500
        )

        self.overall_progress = ft.ProgressBar(
            value=0,
            width=500,
            color=ft.Colors.GREEN_700
        )

        # Buttons
        self.cancel_button = ft.ElevatedButton(
            "Cancel",
            icon=ft.Icons.CANCEL,
            on_click=lambda e: self.on_cancel_clicked(),
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.RED_700
            ),
            visible=False
        )

        self.view_results_button = ft.ElevatedButton(
            "View Results",
            icon=ft.Icons.ASSESSMENT,
            on_click=lambda e: self.on_view_results_clicked(),
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700
            ),
            visible=False
        )

        # Symbol cards container (left side)
        self.cards_container = ft.Column(
            spacing=15,
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )

        # Worker log viewers container (right side)
        self.workers_container = ft.Column(
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )

        # Split layout: Left (symbols) | Right (worker logs)
        split_content = ft.Row(
            [
                # Left column: Symbol progress
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Symbol Progress", size=18, weight=ft.FontWeight.W_500),
                            self.cards_container
                        ],
                        spacing=10,
                        expand=True
                    ),
                    expand=2,  # 2/5 of width
                    padding=15,
                    border=ft.border.all(2, ft.Colors.CYAN_700),
                    border_radius=10,
                    bgcolor=ft.Colors.BLACK
                ),

                # Vertical divider
                ft.VerticalDivider(width=3, color=ft.Colors.GREY_600),

                # Right column: Worker logs
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Worker Threads", size=18, weight=ft.FontWeight.W_500),
                            self.workers_container
                        ],
                        spacing=10,
                        expand=True
                    ),
                    expand=3,  # 3/5 of width
                    padding=15,
                    border=ft.border.all(2, ft.Colors.GREEN_700),
                    border_radius=10,
                    bgcolor=ft.Colors.BLACK
                )
            ],
            expand=True,
            spacing=10
        )

        # Main layout
        self.content = ft.Column(
            [
                ft.Text("Backtest Execution", size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),

                # Summary section
                self.summary_text,
                self.overall_progress,
                ft.Row(
                    [self.cancel_button, self.view_results_button],
                    spacing=10
                ),
                ft.Divider(),

                # Split content (symbols + worker logs)
                split_content
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=15,
            expand=True
        )

        self.padding = 20
        self.expand = True

    def initialize_symbols(self, symbols: list):
        """
        Initialize cards for all symbols.

        Args:
            symbols: List of symbol strings
        """
        self.symbol_cards.clear()
        self.cards_container.controls.clear()

        for symbol in symbols:
            card = SymbolCard(symbol)
            self.symbol_cards[symbol] = card
            self.cards_container.controls.append(card)

        self.summary_text.value = f"Initialized {len(symbols)} symbols"
        self.is_running = True
        self.cancel_button.visible = True
        self.view_results_button.visible = False

        if self.page:
            self.update()

    def update_symbol_status(self, symbol: str, status: str):
        """Update status for a specific symbol."""
        if symbol in self.symbol_cards:
            self.symbol_cards[symbol].update_status(status)
            if self.page:
                self.symbol_cards[symbol].update()

    def update_symbol_progress(self, symbol: str, progress: float, message: str = ""):
        """Update progress for a specific symbol."""
        if symbol in self.symbol_cards:
            self.symbol_cards[symbol].update_progress(progress, message)
            if self.page:
                self.symbol_cards[symbol].update()

    def update_overall_progress(self, completed: int, total: int, running: int, failed: int):
        """
        Update overall progress summary.

        Args:
            completed: Number of completed symbols
            total: Total number of symbols
            running: Number of currently running symbols
            failed: Number of failed symbols
        """
        progress = completed / total if total > 0 else 0
        self.overall_progress.value = progress

        self.summary_text.value = (
            f"Progress: {completed}/{total} completed | "
            f"{running} running | {failed} failed"
        )

        # Show View Results button when all done
        if completed + failed >= total:
            self.is_running = False
            self.cancel_button.visible = False
            self.view_results_button.visible = True

        if self.page:
            self.update()

    def mark_complete(self):
        """Mark execution as complete."""
        self.is_running = False
        self.cancel_button.visible = False
        self.view_results_button.visible = True

        if self.page:
            self.update()

    def initialize_workers(self, num_workers: int):
        """
        Initialize worker log viewer panels.

        Args:
            num_workers: Number of worker threads
        """
        self.num_workers = num_workers
        self.worker_viewers.clear()
        self.workers_container.controls.clear()

        for i in range(num_workers):
            viewer = WorkerLogViewer(worker_id=i)
            self.worker_viewers.append(viewer)
            self.workers_container.controls.append(viewer)

        if self.page:
            self.update()

    def add_worker_log(self, worker_id: int, message: str, level: str = "info"):
        """
        Add a log message to a specific worker's log viewer.

        Args:
            worker_id: Worker thread ID (0-indexed)
            message: Log message
            level: Log level ("info", "success", "warning", "error")
        """
        if 0 <= worker_id < len(self.worker_viewers):
            self.worker_viewers[worker_id].add_log(message, level)

    def set_worker_symbol(self, worker_id: int, symbol: str):
        """
        Set the current symbol being processed by a worker.

        Args:
            worker_id: Worker thread ID (0-indexed)
            symbol: Symbol being processed
        """
        if 0 <= worker_id < len(self.worker_viewers):
            self.worker_viewers[worker_id].set_symbol(symbol)

    def set_worker_idle(self, worker_id: int):
        """
        Mark a worker as idle.

        Args:
            worker_id: Worker thread ID (0-indexed)
        """
        if 0 <= worker_id < len(self.worker_viewers):
            self.worker_viewers[worker_id].set_idle()

    def update_worker_status(self, worker_id: int, symbol: str | None, status: str):
        """
        Update worker status based on status update from controller.

        Args:
            worker_id: Worker thread ID (0-indexed)
            symbol: Symbol being processed (None if idle)
            status: Status string ("started" or "idle")
        """
        if status == "idle" or symbol is None:
            self.set_worker_idle(worker_id)
        else:
            self.set_worker_symbol(worker_id, symbol)
