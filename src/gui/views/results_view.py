"""
Results view for displaying backtest results.
"""

import flet as ft
import pandas as pd
from typing import Callable, Optional, Dict
from pathlib import Path
from gui.views.regime_analysis_tab import RegimeAnalysisTab


class ResultsView(ft.Container):
    """
    View for displaying backtest results.

    Shows:
    - Summary statistics
    - Results table (sortable)
    - Export buttons
    """

    def __init__(self, on_back_clicked: Callable[[], None], on_view_logs_clicked: Optional[Callable[[], None]] = None):
        """
        Initialize results view.

        Args:
            on_back_clicked: Callback when Back button is clicked
            on_view_logs_clicked: Optional callback when View Execution Logs button is clicked
        """
        super().__init__()

        self.on_back_clicked = on_back_clicked
        self.on_view_logs_clicked = on_view_logs_clicked

        # Components
        self.summary_cards = None
        self.results_table = None
        self.export_csv_button = None
        self.export_html_button = None
        self.back_button = None
        self.view_logs_button = None
        self.regime_tab = None  # Level 4: Regime analysis tab
        self.main_tabs = None  # Level 4: Main tab container

        # State
        self.results_df: Optional[pd.DataFrame] = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the results view UI."""
        # Summary cards
        self.summary_cards = ft.Row(spacing=15, wrap=True)

        # Results table
        self.results_table = ft.DataTable(
            columns=[],
            rows=[],
            border=ft.border.all(2, ft.Colors.CYAN_700),
            border_radius=10,
            vertical_lines=ft.border.BorderSide(2, ft.Colors.GREY_600),
            horizontal_lines=ft.border.BorderSide(2, ft.Colors.GREY_600),
            heading_row_color=ft.Colors.GREY_800,
            heading_row_height=50,
            data_row_max_height=60
        )

        # Open log directory button
        self.open_logs_button = ft.ElevatedButton(
            "Open Log Directory",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_open_logs_clicked,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.ORANGE_700
            )
        )

        # View execution logs button
        self.view_logs_button = ft.ElevatedButton(
            "View Execution Logs",
            icon=ft.Icons.ARTICLE,
            on_click=lambda e: self.on_view_logs_clicked() if self.on_view_logs_clicked else None,
            visible=self.on_view_logs_clicked is not None,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.PURPLE_700
            ),
            tooltip="Go back to worker threads and execution logs"
        )

        self.back_button = ft.ElevatedButton(
            "Back to Setup",
            icon=ft.Icons.ARROW_BACK,
            on_click=lambda e: self.on_back_clicked(),
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700
            )
        )

        # Level 4: Create regime analysis tab
        self.regime_tab = RegimeAnalysisTab()

        # Level 4: Create tabbed interface
        results_table_content = ft.Column(
            [
                ft.Text("Detailed Results", size=18, weight=ft.FontWeight.W_500),
                ft.Container(
                    content=self.results_table,
                    border=ft.border.all(3, ft.Colors.BLUE_700),
                    border_radius=12,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )

        self.main_tabs = ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(
                    text="Results Table",
                    icon=ft.Icons.TABLE_CHART,
                    content=results_table_content
                ),
                ft.Tab(
                    text="Regime Analysis",
                    icon=ft.Icons.ANALYTICS,
                    content=self.regime_tab
                )
            ],
            expand=True
        )

        # Layout
        self.content = ft.Column(
            [
                ft.Text("Backtest Results", size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),

                # Summary
                ft.Text("Summary Statistics", size=18, weight=ft.FontWeight.W_500),
                self.summary_cards,
                ft.Divider(),

                # Tabs (Results Table + Regime Analysis)
                self.main_tabs,
                ft.Divider(),

                # Actions
                ft.Row(
                    [
                        self.back_button,
                        self.view_logs_button,
                        self.open_logs_button
                    ],
                    spacing=15
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=15,
            expand=True
        )

        self.padding = 20
        self.expand = True

    def load_results(self, results_df: pd.DataFrame):
        """
        Load and display results.

        Args:
            results_df: DataFrame with backtest results
        """
        self.results_df = results_df

        if results_df.empty:
            self._show_no_results()
            return

        # Build summary cards
        self._build_summary_cards(results_df)

        # Build results table
        self._build_results_table(results_df)

        if self.page:
            self.update()

    def load_regime_results(self, regime_results: Dict):
        """
        Load regime analysis results into the regime tab (Level 4).

        Args:
            regime_results: Dictionary mapping symbol -> RegimeAnalysisResults
        """
        if regime_results and self.regime_tab:
            self.regime_tab.load_results(regime_results)
            if self.page:
                self.update()

    def _build_summary_cards(self, df: pd.DataFrame):
        """Build summary statistic cards."""
        self.summary_cards.controls.clear()

        # Calculate summary stats
        total_symbols = len(df)
        avg_return = df['Total Return [%]'].mean() if 'Total Return [%]' in df.columns else 0
        avg_sharpe = df['Sharpe Ratio'].mean() if 'Sharpe Ratio' in df.columns else 0
        win_rate = (df['Total Return [%]'] > 0).sum() / total_symbols * 100 if 'Total Return [%]' in df.columns else 0

        # Create cards
        cards_data = [
            ("Total Symbols", str(total_symbols), ft.Icons.SHOW_CHART, ft.Colors.BLUE),
            ("Avg Return", f"{avg_return:.2f}%", ft.Icons.TRENDING_UP,
             ft.Colors.GREEN if avg_return >= 0 else ft.Colors.RED),
            ("Avg Sharpe", f"{avg_sharpe:.2f}", ft.Icons.SPEED,
             ft.Colors.GREEN if avg_sharpe >= 1.0 else ft.Colors.ORANGE),
            ("Win Rate", f"{win_rate:.1f}%", ft.Icons.PIE_CHART,
             ft.Colors.GREEN if win_rate >= 50 else ft.Colors.ORANGE)
        ]

        for title, value, icon, color in cards_data:
            card = ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(icon, size=30, color=color),
                        ft.Text(value, size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        ft.Text(title, size=12, color=ft.Colors.GREY_400)
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=5
                ),
                padding=20,
                width=150,
                height=120,
                border=ft.border.all(2, color),
                border_radius=10,
                bgcolor=ft.Colors.GREY_900
            )
            self.summary_cards.controls.append(card)

    def _build_results_table(self, df: pd.DataFrame):
        """Build results table from DataFrame."""
        # Select key columns to display
        display_columns = []
        for col in ['Symbol', 'Total Return [%]', 'Annual Return [%]', 'Sharpe Ratio',
                    'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades']:
            if col in df.columns:
                display_columns.append(col)

        if not display_columns:
            display_columns = df.columns[:7].tolist()  # Show first 7 columns

        # Build column headers
        self.results_table.columns = [
            ft.DataColumn(ft.Text(col, weight=ft.FontWeight.BOLD, size=12))
            for col in display_columns
        ]

        # Build rows
        self.results_table.rows = []
        for _, row in df.iterrows():
            cells = []
            for col in display_columns:
                value = row[col]

                # Format value
                if isinstance(value, float):
                    formatted = f"{value:.2f}"
                else:
                    formatted = str(value)

                # Color code returns
                color = ft.Colors.WHITE  # Default to white for readability on dark background
                if col in ['Total Return [%]', 'Annual Return [%]']:
                    color = ft.Colors.GREEN_400 if value >= 0 else ft.Colors.RED_400
                elif col == 'Sharpe Ratio':
                    color = ft.Colors.GREEN_400 if value >= 1.0 else ft.Colors.ORANGE_400
                elif col == 'Max Drawdown [%]':
                    color = ft.Colors.RED_400 if value < -15 else ft.Colors.ORANGE_400

                cells.append(ft.DataCell(ft.Text(formatted, size=11, color=color)))

            self.results_table.rows.append(ft.DataRow(cells=cells))

    def _show_no_results(self):
        """Show message when no results available."""
        self.summary_cards.controls.clear()
        self.summary_cards.controls.append(
            ft.Text("No results to display", size=16, italic=True, color=ft.Colors.GREY_400)
        )

        self.results_table.columns = []
        self.results_table.rows = []

    def _on_open_logs_clicked(self, e):
        """Open the log output directory in file explorer."""
        try:
            from config import get_log_output_dir
            import subprocess
            import sys

            output_dir = get_log_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)

            # Open directory in file explorer (cross-platform)
            if sys.platform == 'win32':
                subprocess.run(['explorer', str(output_dir)])
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(output_dir)])
            else:  # Linux
                subprocess.run(['xdg-open', str(output_dir)])

        except Exception as ex:
            self._show_error(f"Failed to open directory: {ex}")

    def _show_success(self, message: str):
        """Show success dialog."""
        if self.page:
            dlg = ft.AlertDialog(
                title=ft.Text("Export Successful"),
                content=ft.Text(message),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(dlg))]
            )
            self.page.open(dlg)

    def _show_error(self, message: str):
        """Show error dialog."""
        if self.page:
            dlg = ft.AlertDialog(
                title=ft.Text("Export Error"),
                content=ft.Text(message),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.close(dlg))]
            )
            self.page.open(dlg)
