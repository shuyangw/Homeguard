"""
Regime Analysis Tab for ResultsView.

Displays regime-based performance analysis in the GUI with:
- Summary card (robustness score, best/worst regime)
- Performance tables for each regime type
- Symbol selector for multi-symbol backtests
"""

import flet as ft
from typing import Dict, Optional, List
from backtesting.regimes.analyzer import RegimeAnalysisResults, RegimePerformance, RegimeLabel


class RegimeAnalysisTab(ft.Container):
    """Tab for displaying regime analysis results."""

    def __init__(self):
        """Initialize regime analysis tab."""
        super().__init__()

        # State
        self.regime_results: Dict[str, RegimeAnalysisResults] = {}
        self.current_symbol: Optional[str] = None

        # UI Components
        self.symbol_selector: Optional[ft.Dropdown] = None
        self.summary_card: Optional[ft.Container] = None
        self.regime_tabs: Optional[ft.Tabs] = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the tab UI structure."""
        # Symbol selector (shown only for multi-symbol)
        self.symbol_selector = ft.Dropdown(
            label="Select Symbol",
            options=[],
            on_change=self._on_symbol_changed,
            visible=False,
            width=300
        )

        # Summary card placeholder
        self.summary_card = ft.Container(
            content=ft.Text("No regime analysis data available", color=ft.Colors.GREY_400),
            padding=20,
            border=ft.border.all(2, ft.Colors.GREY_700),
            border_radius=10,
            bgcolor=ft.Colors.GREY_900
        )

        # Regime type tabs placeholder
        self.regime_tabs = ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(text="Trend Regimes", content=ft.Container(padding=10)),
                ft.Tab(text="Volatility Regimes", content=ft.Container(padding=10)),
                ft.Tab(text="Drawdown Regimes", content=ft.Container(padding=10))
            ],
            visible=False
        )

        # Main layout
        self.content = ft.Column(
            controls=[
                ft.Text(
                    "Regime-Based Performance Analysis",
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.PURPLE_300
                ),
                ft.Divider(height=20, color=ft.Colors.GREY_700),
                self.symbol_selector,
                self.summary_card,
                ft.Container(height=20),  # Spacer
                self.regime_tabs
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )

        self.padding = 20
        self.expand = True

    def load_results(self, results: Dict[str, RegimeAnalysisResults]):
        """
        Load and display regime analysis results.

        Args:
            results: Dictionary mapping symbol -> RegimeAnalysisResults
        """
        self.regime_results = results

        if not results:
            self._show_no_data()
            return

        # Set up symbol selector if multiple symbols
        symbols = list(results.keys())
        if len(symbols) > 1:
            self.symbol_selector.options = [
                ft.dropdown.Option(symbol) for symbol in symbols
            ]
            self.symbol_selector.value = symbols[0]
            self.symbol_selector.visible = True
        else:
            self.symbol_selector.visible = False

        # Display first symbol's results
        self.current_symbol = symbols[0]
        self._display_symbol_results(self.current_symbol)

    def _on_symbol_changed(self, e):
        """Handle symbol selection change."""
        if e.control.value:
            self.current_symbol = e.control.value
            self._display_symbol_results(self.current_symbol)
            self.update()

    def _show_no_data(self):
        """Show message when no data is available."""
        self.summary_card.content = ft.Column([
            ft.Icon(ft.Icons.INFO_OUTLINE, size=48, color=ft.Colors.GREY_600),
            ft.Text(
                "No regime analysis data available",
                size=16,
                color=ft.Colors.GREY_400
            ),
            ft.Text(
                "Enable 'Regime Analysis' in setup to generate this data",
                size=12,
                color=ft.Colors.GREY_600
            )
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)
        self.regime_tabs.visible = False
        self.update()

    def _display_symbol_results(self, symbol: str):
        """Display results for a specific symbol."""
        if symbol not in self.regime_results:
            self._show_no_data()
            return

        results = self.regime_results[symbol]

        # Update summary card
        self.summary_card.content = self._create_summary_card_content(results)

        # Update regime tabs
        self.regime_tabs.tabs[0].content = self._create_regime_table(
            results.trend_performance, "Trend"
        )
        self.regime_tabs.tabs[1].content = self._create_regime_table(
            results.volatility_performance, "Volatility"
        )
        self.regime_tabs.tabs[2].content = self._create_regime_table(
            results.drawdown_performance, "Drawdown"
        )
        self.regime_tabs.visible = True

        self.update()

    def _create_summary_card_content(self, results: RegimeAnalysisResults) -> ft.Column:
        """Create summary card content."""
        # Determine robustness color and label
        if results.robustness_score >= 70:
            robustness_color = ft.Colors.GREEN_400
            robustness_label = "Excellent"
        elif results.robustness_score >= 50:
            robustness_color = ft.Colors.BLUE_400
            robustness_label = "Good"
        else:
            robustness_color = ft.Colors.RED_400
            robustness_label = "Needs Improvement"

        # Sharpe ratio color
        sharpe_color = ft.Colors.GREEN_400 if results.overall_sharpe > 0 else ft.Colors.RED_400

        # Return color
        return_color = ft.Colors.GREEN_400 if results.overall_return > 0 else ft.Colors.RED_400

        return ft.Column([
            ft.Row([
                # Robustness Score
                ft.Container(
                    content=ft.Column([
                        ft.Text("Robustness Score", size=12, color=ft.Colors.GREY_400),
                        ft.Text(
                            f"{results.robustness_score:.0f}/100",
                            size=32,
                            weight=ft.FontWeight.BOLD,
                            color=robustness_color
                        ),
                        ft.Text(robustness_label, size=14, color=robustness_color)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                    padding=15,
                    border=ft.border.all(2, robustness_color),
                    border_radius=10,
                    bgcolor=ft.Colors.GREY_900
                ),

                # Overall Sharpe
                ft.Container(
                    content=ft.Column([
                        ft.Text("Overall Sharpe", size=12, color=ft.Colors.GREY_400),
                        ft.Text(
                            f"{results.overall_sharpe:.2f}",
                            size=32,
                            weight=ft.FontWeight.BOLD,
                            color=sharpe_color
                        )
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                    padding=15,
                    border=ft.border.all(2, ft.Colors.GREY_700),
                    border_radius=10,
                    bgcolor=ft.Colors.GREY_900
                ),

                # Overall Return
                ft.Container(
                    content=ft.Column([
                        ft.Text("Overall Return", size=12, color=ft.Colors.GREY_400),
                        ft.Text(
                            f"{results.overall_return:.1f}%",
                            size=32,
                            weight=ft.FontWeight.BOLD,
                            color=return_color
                        )
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                    padding=15,
                    border=ft.border.all(2, ft.Colors.GREY_700),
                    border_radius=10,
                    bgcolor=ft.Colors.GREY_900
                )
            ], spacing=15, wrap=True),

            ft.Container(height=15),  # Spacer

            # Best/Worst Regimes
            ft.Row([
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.TRENDING_UP, color=ft.Colors.GREEN_400, size=20),
                        ft.Text(
                            f"Best: {results.best_regime}",
                            size=14,
                            color=ft.Colors.GREEN_400,
                            weight=ft.FontWeight.BOLD
                        )
                    ], spacing=8),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREEN_400),
                    border_radius=8,
                    bgcolor=ft.Colors.GREEN_900 + "20"  # Semi-transparent
                ),
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.TRENDING_DOWN, color=ft.Colors.RED_400, size=20),
                        ft.Text(
                            f"Worst: {results.worst_regime}",
                            size=14,
                            color=ft.Colors.RED_400,
                            weight=ft.FontWeight.BOLD
                        )
                    ], spacing=8),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.RED_400),
                    border_radius=8,
                    bgcolor=ft.Colors.RED_900 + "20"  # Semi-transparent
                )
            ], spacing=15)
        ], spacing=10)

    def _create_regime_table(
        self,
        performance_dict: Dict[RegimeLabel, RegimePerformance],
        regime_type: str
    ) -> ft.Container:
        """Create performance table for a regime type."""
        if not performance_dict:
            return ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.INFO_OUTLINE, size=40, color=ft.Colors.GREY_600),
                    ft.Text(
                        f"No {regime_type.lower()} regime data available",
                        size=14,
                        color=ft.Colors.GREY_400
                    )
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                padding=40,
                alignment=ft.alignment.center
            )

        # Create table rows
        rows = []
        for regime_label, perf in performance_dict.items():
            regime_name = regime_label.value if hasattr(regime_label, 'value') else str(regime_label)

            # Color code cells
            sharpe_color = ft.Colors.GREEN_400 if perf.sharpe_ratio > 0 else ft.Colors.RED_400
            return_color = ft.Colors.GREEN_400 if perf.total_return > 0 else ft.Colors.RED_400

            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(regime_name, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)),
                        ft.DataCell(ft.Text(f"{perf.sharpe_ratio:.2f}", color=sharpe_color)),
                        ft.DataCell(ft.Text(f"{perf.total_return:.1f}%", color=return_color)),
                        ft.DataCell(ft.Text(f"{perf.max_drawdown:.1f}%", color=ft.Colors.RED_300)),
                        ft.DataCell(ft.Text(f"{perf.win_rate:.1f}%", color=ft.Colors.CYAN_300)),
                        ft.DataCell(ft.Text(str(perf.num_trades), color=ft.Colors.GREY_300)),
                        ft.DataCell(ft.Text(str(perf.num_periods), color=ft.Colors.GREY_300))
                    ]
                )
            )

        # Create data table
        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Regime", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Sharpe", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Return %", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Drawdown %", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Win Rate %", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Trades", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Periods", weight=ft.FontWeight.BOLD))
            ],
            rows=rows,
            border=ft.border.all(2, ft.Colors.CYAN_700),
            border_radius=10,
            vertical_lines=ft.border.BorderSide(1, ft.Colors.GREY_700),
            horizontal_lines=ft.border.BorderSide(1, ft.Colors.GREY_700),
            heading_row_color=ft.Colors.GREY_800,
            heading_row_height=50,
            data_row_max_height=60
        )

        return ft.Container(
            content=ft.Column([
                ft.Text(
                    f"{regime_type} Regime Performance",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.CYAN_300
                ),
                ft.Container(height=10),
                table
            ], scroll=ft.ScrollMode.AUTO),
            padding=10
        )
