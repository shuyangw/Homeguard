"""
Worker thread log viewer component.

Displays real-time logs from a specific worker thread.
"""

import flet as ft
from typing import List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WorkerLogEntry:
    """Single log entry from a worker."""
    timestamp: datetime
    message: str
    level: str  # "info", "success", "warning", "error"


class WorkerLogViewer(ft.Container):
    """
    Component that displays real-time logs from a single worker thread.

    Shows:
    - Worker ID
    - Current symbol being processed
    - Scrollable log output with timestamps
    """

    def __init__(self, worker_id: int):
        """
        Initialize worker log viewer.

        Args:
            worker_id: ID of the worker thread (0-indexed)
        """
        super().__init__()

        self.worker_id = worker_id
        self.current_symbol = "Idle"
        self.log_entries: List[WorkerLogEntry] = []

        # UI Components
        self.worker_header = None
        self.symbol_text = None
        self.log_container = None
        self.log_column = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the worker log viewer UI."""
        # Header
        self.worker_header = ft.Text(
            f"Worker {self.worker_id + 1}",
            size=14,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.CYAN_300
        )

        # Current symbol
        self.symbol_text = ft.Text(
            "Idle",
            size=12,
            italic=True,
            color=ft.Colors.GREY_400
        )

        # Log column (scrollable) - must expand to fill container
        self.log_column = ft.Column(
            spacing=2,
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,  # Auto-scroll to latest
            expand=True  # Expand to fill available vertical space
        )

        # Log container with border (removed fixed height, let it expand)
        self.log_container = ft.Container(
            content=self.log_column,
            bgcolor=ft.Colors.BLACK,
            border=ft.border.all(2, ft.Colors.TEAL_800),
            border_radius=8,
            padding=10,
            expand=True  # Expand to fill available space
        )

        # Main layout (Column must expand vertically AND stretch children horizontally)
        self.content = ft.Column(
            [
                ft.Row(
                    [
                        self.worker_header,
                        ft.Container(expand=True),  # Spacer
                        self.symbol_text
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                ft.Divider(height=2, color=ft.Colors.CYAN_800),
                self.log_container
            ],
            spacing=8,
            expand=True,  # Expand vertically to fill available height
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH  # Stretch children horizontally to fill width
        )

        self.border = ft.border.all(3, ft.Colors.ORANGE_700)
        self.border_radius = 10
        self.padding = 12
        self.bgcolor=ft.Colors.GREY_900
        self.expand = True

    def set_symbol(self, symbol: str):
        """
        Set the current symbol being processed by this worker.

        Args:
            symbol: Symbol name (e.g., "AAPL")
        """
        self.current_symbol = symbol
        self.symbol_text.value = f"Processing: {symbol}"
        self.symbol_text.color = ft.Colors.GREEN_300

        if self.page:
            self.symbol_text.update()

    def set_idle(self):
        """Mark this worker as idle."""
        self.current_symbol = "Idle"
        self.symbol_text.value = "Idle"
        self.symbol_text.color = ft.Colors.GREY_400

        if self.page:
            self.symbol_text.update()

    def add_log(self, message: str, level: str = "info"):
        """
        Add a log entry.

        Args:
            message: Log message
            level: Log level ("info", "success", "warning", "error")
        """
        timestamp = datetime.now()
        entry = WorkerLogEntry(timestamp, message, level)
        self.log_entries.append(entry)

        # Color based on level
        color = ft.Colors.WHITE
        if level == "success":
            color = ft.Colors.GREEN_300
        elif level == "warning":
            color = ft.Colors.YELLOW_300
        elif level == "error":
            color = ft.Colors.RED_300
        elif level == "info":
            color = ft.Colors.BLUE_200

        # Format timestamp
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

        # Create log line (monospace font for readability)
        log_line = ft.Text(
            f"[{time_str}] {message}",
            size=10,
            color=color,
            selectable=True,
            font_family="Consolas",  # Monospace font for logs
            no_wrap=False,  # Allow wrapping for long lines
            expand=True  # Expand to use full width
        )

        self.log_column.controls.append(log_line)

        # Limit to last 200 log entries to prevent memory bloat
        if len(self.log_column.controls) > 200:
            self.log_column.controls.pop(0)

        if self.page:
            self.log_column.update()

    def clear_logs(self):
        """Clear all log entries."""
        self.log_entries.clear()
        self.log_column.controls.clear()

        if self.page:
            self.log_column.update()
