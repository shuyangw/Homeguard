"""
Setup view for configuring backtest parameters.
"""

import flet as ft
from datetime import datetime, timedelta
import multiprocessing
from typing import Callable, Dict, Any, Optional

from gui.utils.strategy_utils import (
    get_strategy_registry,
    get_strategy_parameters,
    get_strategy_param_types,
    get_strategy_description
)
from gui.utils.config_manager import ConfigManager


class SetupView(ft.Container):
    """
    View for setting up backtest configuration.

    Includes:
    - Strategy selector
    - Dynamic parameter editor
    - Symbol input
    - Date range picker
    - Worker count slider
    """

    def __init__(self, on_run_clicked: Callable[[Dict[str, Any]], None]):
        """
        Initialize setup view.

        Args:
            on_run_clicked: Callback when Run button is clicked,
                           receives config dict with all parameters
        """
        super().__init__()

        self.on_run_clicked = on_run_clicked
        self.strategy_registry = get_strategy_registry()
        self.config_manager = ConfigManager()

        # UI Components
        self.strategy_dropdown = None
        self.strategy_desc_text = None
        self.params_container = None
        self.param_controls: Dict[str, ft.Control] = {}
        self.symbols_input = None
        self.start_date_picker = None
        self.end_date_picker = None
        self.workers_slider = None
        self.workers_text = None
        self.run_button = None
        self.parallel_checkbox = None

        # Preset/Symbol List UI components
        self.quick_rerun_button = None
        self.preset_dropdown = None
        self.save_preset_button = None
        self.load_preset_button = None
        self.symbol_list_dropdown = None
        self.save_symbol_list_button = None
        self.load_symbol_list_button = None

        # Cache management buttons
        self.view_cache_button = None
        self.clear_cache_button = None

        # Portfolio mode components
        self.portfolio_mode_dropdown = None
        self.portfolio_settings_container = None
        self.position_sizing_dropdown = None
        self.rebalancing_frequency_dropdown = None
        self.rebalancing_threshold_input = None

        # Current state
        self.current_strategy_class = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the setup view UI."""
        # Get CPU count for worker recommendation
        cpu_count = multiprocessing.cpu_count()
        recommended_workers = min(cpu_count, 8)

        # Strategy selector
        strategy_names = sorted(self.strategy_registry.keys())
        self.strategy_dropdown = ft.Dropdown(
            label="Select Strategy",
            options=[ft.dropdown.Option(name) for name in strategy_names],
            value=strategy_names[0] if strategy_names else None,
            on_change=self._on_strategy_changed,
            width=400,
            tooltip="Choose a trading strategy to backtest. Each strategy has different parameters and logic."
        )

        # Strategy description
        self.strategy_desc_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
            italic=True
        )

        # Parameters container (dynamic)
        self.params_container = ft.Column(spacing=10)

        # Symbols input
        self.symbols_input = ft.TextField(
            label="Symbols (comma-separated)",
            hint_text="AAPL, MSFT, GOOGL",
            value="AAPL, MSFT",
            multiline=False,
            width=400,
            tooltip="Enter stock ticker symbols separated by commas. The strategy will be tested on each symbol independently."
        )

        # Date pickers
        default_start = datetime(2023, 1, 1)
        default_end = datetime(2024, 1, 1)

        # Date pickers with calendar support
        self.start_date_value = default_start
        self.end_date_value = default_end

        self.start_date_picker = ft.TextField(
            label="Start Date",
            hint_text="YYYY-MM-DD",
            value=default_start.strftime("%Y-%m-%d"),
            width=140,
            read_only=True,
            tooltip="Beginning of backtest period. Click calendar icon to select date."
        )

        self.start_date_button = ft.IconButton(
            icon=ft.Icons.CALENDAR_MONTH,
            tooltip="Select start date from calendar",
            on_click=self._on_start_date_click
        )

        self.end_date_picker = ft.TextField(
            label="End Date",
            hint_text="YYYY-MM-DD",
            value=default_end.strftime("%Y-%m-%d"),
            width=140,
            read_only=True,
            tooltip="End of backtest period. Click calendar icon to select date."
        )

        self.end_date_button = ft.IconButton(
            icon=ft.Icons.CALENDAR_MONTH,
            tooltip="Select end date from calendar",
            on_click=self._on_end_date_click
        )

        # Capital and fees inputs
        self.initial_capital_input = ft.TextField(
            label="Initial Capital ($)",
            value="100000",
            keyboard_type=ft.KeyboardType.NUMBER,
            width=190,
            tooltip="Starting portfolio value in dollars. Each backtest starts with this amount."
        )

        self.fees_input = ft.TextField(
            label="Transaction Fees (%)",
            value="0",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="0 = no fees",
            width=190,
            tooltip="Trading commission as a percentage. Example: 0.1% means $1 fee per $1000 traded. Set to 0 for no fees."
        )

        # Risk management dropdown
        self.risk_profile_dropdown = ft.Dropdown(
            label="Risk Profile",
            value="Moderate",
            options=[
                ft.dropdown.Option("Conservative", "Conservative (5% per trade, 1% stop loss)"),
                ft.dropdown.Option("Moderate", "Moderate (10% per trade, 2% stop loss)"),
                ft.dropdown.Option("Aggressive", "Aggressive (20% per trade, 3% stop loss)"),
                ft.dropdown.Option("Disabled", "⚠️ Disabled (99% per trade - unrealistic)")
            ],
            width=400,
            tooltip="Risk management profile controls position sizing and stop losses. Moderate is recommended for realistic results."
        )

        # Portfolio mode dropdown
        self.portfolio_mode_dropdown = ft.Dropdown(
            label="Backtest Mode",
            value="Single-Symbol",
            options=[
                ft.dropdown.Option("Single-Symbol", "Test each symbol independently (sweep mode)"),
                ft.dropdown.Option("Multi-Symbol Portfolio", "Hold multiple symbols in one portfolio")
            ],
            width=400,
            on_change=self._on_portfolio_mode_changed,
            tooltip="Single-Symbol mode tests each symbol separately. Multi-Symbol Portfolio mode holds multiple positions simultaneously with portfolio-level allocation and rebalancing."
        )

        # Portfolio settings (shown only in Multi-Symbol Portfolio mode)
        self.position_sizing_dropdown = ft.Dropdown(
            label="Position Sizing Method",
            value="equal_weight",
            options=[
                ft.dropdown.Option("equal_weight", "Equal Weight (e.g., 10% per position)"),
                ft.dropdown.Option("risk_parity", "Risk Parity (inverse volatility)"),
                ft.dropdown.Option("fixed_count", "Fixed Count (max N positions)"),
                ft.dropdown.Option("ranked", "Ranked (by signal strength)"),
                ft.dropdown.Option("adaptive", "Adaptive (multi-factor)")
            ],
            width=400,
            tooltip="How to allocate capital across multiple positions in the portfolio."
        )

        self.rebalancing_frequency_dropdown = ft.Dropdown(
            label="Rebalancing Frequency",
            value="never",
            options=[
                ft.dropdown.Option("never", "Never (no rebalancing)"),
                ft.dropdown.Option("monthly", "Monthly"),
                ft.dropdown.Option("quarterly", "Quarterly"),
                ft.dropdown.Option("on_signal", "On Signal Change"),
                ft.dropdown.Option("drift", "Threshold-Based (drift)")
            ],
            width=400,
            tooltip="When to rebalance the portfolio back to target weights."
        )

        self.rebalancing_threshold_input = ft.TextField(
            label="Rebalancing Threshold (%)",
            value="5.0",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Trigger rebalance if weight drifts by this %",
            width=400,
            tooltip="For drift-based rebalancing: if any position weight drifts beyond this threshold, trigger rebalance. Example: 5% means rebalance when 10% target becomes 15% or 5%."
        )

        # Container for portfolio settings (initially hidden)
        self.portfolio_settings_container = ft.Container(
            content=ft.Column([
                ft.Text("Portfolio Settings", size=16, weight=ft.FontWeight.W_400),
                self.position_sizing_dropdown,
                self.rebalancing_frequency_dropdown,
                self.rebalancing_threshold_input,
                ft.Text(
                    "ℹ️ These settings apply when holding multiple positions in a single portfolio.",
                    size=11,
                    color=ft.Colors.GREY_400,
                    italic=True
                )
            ], spacing=10),
            visible=False,  # Hidden by default
            padding=ft.padding.only(top=10)
        )

        # Worker count slider
        self.workers_slider = ft.Slider(
            min=1,
            max=16,
            divisions=15,
            value=recommended_workers,
            label="{value} workers",
            on_change=self._on_workers_changed,
            width=350,
            tooltip="Number of parallel worker threads. More workers = faster backtests but more CPU usage. Auto-capped to number of symbols."
        )

        self.workers_text = ft.Text(
            f"Parallel Workers: {recommended_workers} (Recommended: {recommended_workers} of {cpu_count} CPUs)",
            size=14
        )

        # Parallel mode checkbox
        self.parallel_checkbox = ft.Checkbox(
            label="Run in parallel mode",
            value=True,
            tooltip="Enable parallel execution to test multiple symbols simultaneously. Significantly faster for multiple symbols."
        )

        # Full output generation checkbox
        self.generate_full_output_checkbox = ft.Checkbox(
            label="Generate full output (tearsheets, charts, detailed logs)",
            value=True,
            tooltip="Generate comprehensive output: CSV/HTML reports, trade logs, equity curves, and portfolio states. Files saved to logs directory."
        )

        # Run button
        self.run_button = ft.ElevatedButton(
            "Run Backtests",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run_button_clicked,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN_700
            ),
            height=50,
            width=200,
            tooltip="Execute backtest with current configuration. Configuration will be saved for Quick Re-run."
        )

        # Quick Re-run button
        self.quick_rerun_button = ft.ElevatedButton(
            "Quick Re-run",
            icon=ft.Icons.REPLAY,
            on_click=self._on_quick_rerun_clicked,
            disabled=not self.config_manager.has_last_run(),
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.PURPLE_700
            ),
            height=50,
            width=200,
            tooltip="Load the last executed configuration with one click. Perfect for iterating on parameters."
        )

        # Cache management buttons
        self.view_cache_button = ft.ElevatedButton(
            "View Cache",
            icon=ft.Icons.STORAGE,
            on_click=self._on_view_cache_clicked,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700
            ),
            height=40,
            tooltip="View cached backtest results"
        )

        self.clear_cache_button = ft.ElevatedButton(
            "Clear Cache",
            icon=ft.Icons.DELETE_SWEEP,
            on_click=self._on_clear_cache_clicked,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.ORANGE_700
            ),
            height=40,
            tooltip="Clear all cached results"
        )

        # Symbol List controls
        self.symbol_list_dropdown = ft.Dropdown(
            label="Symbol Lists",
            hint_text="Select saved list",
            options=[],
            width=200,
            tooltip="Select a saved symbol list. Lists show symbol count in parentheses."
        )
        self._refresh_symbol_list_dropdown()

        self.load_symbol_list_button = ft.IconButton(
            icon=ft.Icons.DOWNLOAD,
            tooltip="Load the selected symbol list into the Symbols field",
            on_click=self._on_load_symbol_list_clicked,
            icon_color=ft.Colors.GREEN_400
        )

        self.save_symbol_list_button = ft.IconButton(
            icon=ft.Icons.SAVE,
            tooltip="Save current symbols as a reusable list (e.g., 'FAANG', 'Tech Stocks')",
            on_click=self._on_save_symbol_list_clicked,
            icon_color=ft.Colors.BLUE_400
        )

        # Config Preset controls
        self.preset_dropdown = ft.Dropdown(
            label="Config Presets",
            hint_text="Select saved preset",
            options=[],
            width=200,
            tooltip="Select a saved configuration preset. Presets include strategy, parameters, symbols, dates, and settings."
        )
        self._refresh_preset_dropdown()

        self.load_preset_button = ft.IconButton(
            icon=ft.Icons.DOWNLOAD,
            tooltip="Load the selected configuration preset. All settings will be restored.",
            on_click=self._on_load_preset_clicked,
            icon_color=ft.Colors.GREEN_400
        )

        self.save_preset_button = ft.IconButton(
            icon=ft.Icons.SAVE,
            tooltip="Save entire current configuration as a reusable preset (e.g., 'MA Cross FAANG Q1')",
            on_click=self._on_save_preset_clicked,
            icon_color=ft.Colors.BLUE_400
        )

        # Layout - Horizontal two-column design
        # Left column: Strategy and Strategy Parameters
        left_column = ft.Column(
            [
                # Strategy section
                ft.Container(
                    content=ft.Column([
                        ft.Text("Strategy", size=18, weight=ft.FontWeight.W_500),
                        self.strategy_dropdown,
                        self.strategy_desc_text,
                    ], spacing=10),
                    border=ft.border.all(2, ft.Colors.CYAN_700),
                    border_radius=8,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                ),

                # Parameters section
                ft.Container(
                    content=ft.Column([
                        ft.Text("Strategy Parameters", size=18, weight=ft.FontWeight.W_500),
                        self.params_container,
                    ], spacing=10),
                    border=ft.border.all(2, ft.Colors.BLUE_700),
                    border_radius=8,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                ),
            ],
            spacing=15,
            scroll=ft.ScrollMode.AUTO
        )

        # Right column: Symbols, Date Range, Backtest Parameters, Execution Settings
        right_column = ft.Column(
            [
                # Symbols section
                ft.Container(
                    content=ft.Column([
                        ft.Text("Symbols", size=18, weight=ft.FontWeight.W_500),
                        self.symbols_input,
                    ], spacing=10),
                    border=ft.border.all(2, ft.Colors.PURPLE_700),
                    border_radius=8,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                ),

                # Date range section
                ft.Container(
                    content=ft.Column([
                        ft.Text("Date Range", size=18, weight=ft.FontWeight.W_500),
                        ft.Row(
                            [
                                ft.Row([self.start_date_picker, self.start_date_button], spacing=5),
                                ft.Row([self.end_date_picker, self.end_date_button], spacing=5)
                            ],
                            spacing=20
                        ),
                    ], spacing=10),
                    border=ft.border.all(2, ft.Colors.ORANGE_700),
                    border_radius=8,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                ),

                # Backtest parameters section
                ft.Container(
                    content=ft.Column([
                        ft.Text("Backtest Parameters", size=18, weight=ft.FontWeight.W_500),
                        ft.Row(
                            [self.initial_capital_input, self.fees_input],
                            spacing=20
                        ),
                        self.risk_profile_dropdown,
                        ft.Divider(),
                        self.portfolio_mode_dropdown,
                        self.portfolio_settings_container,
                    ], spacing=10),
                    border=ft.border.all(2, ft.Colors.PINK_700),
                    border_radius=8,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                ),

                # Execution settings
                ft.Container(
                    content=ft.Column([
                        ft.Text("Execution Settings", size=18, weight=ft.FontWeight.W_500),
                        self.parallel_checkbox,
                        self.workers_text,
                        self.workers_slider,
                        ft.Divider(),
                        ft.Text("Output Settings", size=16, weight=ft.FontWeight.W_400),
                        self.generate_full_output_checkbox,
                    ], spacing=10),
                    border=ft.border.all(2, ft.Colors.TEAL_700),
                    border_radius=8,
                    padding=15,
                    bgcolor=ft.Colors.GREY_900
                ),
            ],
            spacing=15,
            scroll=ft.ScrollMode.AUTO
        )

        # Main layout
        self.content = ft.Column(
            [
                ft.Text("Backtest Configuration", size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),

                # Two-column row
                ft.Row(
                    [left_column, right_column],
                    spacing=15,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START
                ),

                # Bottom controls section
                ft.Container(
                    content=ft.Column([
                        # Preset and Symbol List Management
                        ft.Row([
                            # Config Presets
                            ft.Container(
                                content=ft.Row([
                                    self.preset_dropdown,
                                    self.load_preset_button,
                                    self.save_preset_button
                                ], spacing=5),
                                border=ft.border.all(2, ft.Colors.AMBER_700),
                                border_radius=8,
                                padding=10,
                                bgcolor=ft.Colors.GREY_900
                            ),
                            # Symbol Lists
                            ft.Container(
                                content=ft.Row([
                                    self.symbol_list_dropdown,
                                    self.load_symbol_list_button,
                                    self.save_symbol_list_button
                                ], spacing=5),
                                border=ft.border.all(2, ft.Colors.DEEP_PURPLE_700),
                                border_radius=8,
                                padding=10,
                                bgcolor=ft.Colors.GREY_900
                            )
                        ], spacing=15, alignment=ft.MainAxisAlignment.CENTER),

                        # Run Buttons
                        ft.Row([
                            self.run_button,
                            self.quick_rerun_button
                        ], spacing=15, alignment=ft.MainAxisAlignment.CENTER),

                        # Cache Management
                        ft.Row([
                            self.view_cache_button,
                            self.clear_cache_button
                        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER)
                    ], spacing=15),
                    padding=20
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=15,
            expand=True
        )

        self.padding = 20
        self.expand = True

        # Initialize with first strategy
        if strategy_names:
            self._on_strategy_changed(None)

    def _on_strategy_changed(self, e):
        """Handle strategy selection change."""
        strategy_name = self.strategy_dropdown.value
        if not strategy_name:
            return

        strategy_class = self.strategy_registry[strategy_name]
        self.current_strategy_class = strategy_class

        # Update description
        desc = get_strategy_description(strategy_class)
        self.strategy_desc_text.value = desc

        # Update parameters
        self._build_parameter_controls(strategy_class)

        if self.page:
            self.update()

    def _build_parameter_controls(self, strategy_class):
        """Build dynamic parameter input controls for the selected strategy."""
        params = get_strategy_parameters(strategy_class)
        param_types = get_strategy_param_types(strategy_class)

        self.param_controls.clear()
        self.params_container.controls.clear()

        for param_name, default_value in params.items():
            param_type = param_types.get(param_name, str)

            # Create appropriate control based on type
            if param_type == int:
                control = ft.TextField(
                    label=param_name.replace('_', ' ').title(),
                    value=str(default_value) if default_value is not None else "0",
                    keyboard_type=ft.KeyboardType.NUMBER,
                    width=300
                )
            elif param_type == float:
                control = ft.TextField(
                    label=param_name.replace('_', ' ').title(),
                    value=str(default_value) if default_value is not None else "0.0",
                    keyboard_type=ft.KeyboardType.NUMBER,
                    width=300
                )
            elif param_type == bool:
                control = ft.Checkbox(
                    label=param_name.replace('_', ' ').title(),
                    value=bool(default_value) if default_value is not None else False
                )
            elif param_type == str:
                # Check if it's a choice parameter (like 'sma' or 'ema')
                if isinstance(default_value, str) and default_value in ['sma', 'ema']:
                    control = ft.Dropdown(
                        label=param_name.replace('_', ' ').title(),
                        options=[
                            ft.dropdown.Option("sma"),
                            ft.dropdown.Option("ema")
                        ],
                        value=default_value,
                        width=300
                    )
                # Check for rebalance period choices
                elif param_name == 'rebalance_period' and isinstance(default_value, str):
                    control = ft.Dropdown(
                        label=param_name.replace('_', ' ').title(),
                        options=[
                            ft.dropdown.Option("daily"),
                            ft.dropdown.Option("weekly"),
                            ft.dropdown.Option("monthly"),
                            ft.dropdown.Option("quarterly")
                        ],
                        value=default_value,
                        width=300
                    )
                else:
                    control = ft.TextField(
                        label=param_name.replace('_', ' ').title(),
                        value=str(default_value) if default_value is not None else "",
                        width=300
                    )
            elif isinstance(default_value, list):
                # Handle list types - format as comma-separated values
                if default_value:
                    formatted_value = ', '.join(str(v) for v in default_value)
                else:
                    formatted_value = ""
                control = ft.TextField(
                    label=param_name.replace('_', ' ').title(),
                    value=formatted_value,
                    hint_text="Comma-separated values",
                    width=300
                )
            else:
                # Default to text field
                formatted_value = ""
                if default_value is not None:
                    if isinstance(default_value, list):
                        formatted_value = ', '.join(str(v) for v in default_value)
                    else:
                        formatted_value = str(default_value)

                control = ft.TextField(
                    label=param_name.replace('_', ' ').title(),
                    value=formatted_value,
                    width=300
                )

            self.param_controls[param_name] = control
            self.params_container.controls.append(control)

        if not params:
            self.params_container.controls.append(
                ft.Text("No configurable parameters", italic=True, color=ft.Colors.GREY_400)
            )

    def _on_workers_changed(self, e):
        """Handle worker count slider change."""
        cpu_count = multiprocessing.cpu_count()
        workers = int(self.workers_slider.value)
        self.workers_text.value = f"Parallel Workers: {workers} (Recommended: {min(cpu_count, 8)} of {cpu_count} CPUs)"
        if self.page:
            self.update()

    def _on_portfolio_mode_changed(self, e):
        """Handle portfolio mode dropdown change."""
        mode = self.portfolio_mode_dropdown.value

        # Show/hide portfolio settings based on mode
        if mode == "Multi-Symbol Portfolio":
            self.portfolio_settings_container.visible = True
        else:
            self.portfolio_settings_container.visible = False

        if self.page:
            self.portfolio_settings_container.update()

    def _on_run_button_clicked(self, e):
        """Handle Run button click with symbol validation."""
        config = self._collect_configuration()
        if config:
            # Validate symbols before running
            self._validate_and_run_backtest(config)

    def _validate_and_run_backtest(self, config: Dict[str, Any]):
        """
        Validate that symbols exist in database, download if missing, then run backtest.

        Args:
            config: Backtest configuration dictionary
        """
        from backtesting.engine.data_loader import DataLoader
        from gui.utils.symbol_downloader import SymbolDownloader

        symbols = config['symbols']
        start_date = config['start_date']
        end_date = config['end_date']

        # Check symbol availability
        loader = DataLoader()
        availability = loader.check_symbols_availability(symbols)

        missing = availability['missing']

        if not missing:
            # All symbols available, proceed with backtest
            self._execute_backtest(config)
            return

        # Some symbols are missing - show download dialog
        self._show_download_dialog(config, missing, start_date, end_date)

    def _execute_backtest(self, config: Dict[str, Any]):
        """Execute the backtest after validation."""
        # Save as last run for quick re-run feature
        self.config_manager.save_last_run(config)

        # Enable quick re-run button
        self.quick_rerun_button.disabled = False
        if self.page:
            self.quick_rerun_button.update()

        # Execute the backtest
        self.on_run_clicked(config)

    def _collect_configuration(self) -> Optional[Dict[str, Any]]:
        """
        Collect all configuration values from UI controls.

        Returns:
            Dictionary with all configuration, or None if validation fails
        """
        try:
            # Get strategy class
            strategy_name = self.strategy_dropdown.value
            strategy_class = self.strategy_registry[strategy_name]

            # Collect strategy parameters
            strategy_params = {}
            defaults = get_strategy_parameters(strategy_class)

            for param_name, control in self.param_controls.items():
                if isinstance(control, ft.TextField):
                    value_str = control.value.strip()

                    # Get the default value to infer type
                    default = defaults.get(param_name)

                    # Try to infer type from control's keyboard type
                    if control.keyboard_type == ft.KeyboardType.NUMBER:
                        # Check if it's int or float based on default
                        if isinstance(default, int):
                            strategy_params[param_name] = int(value_str)
                        else:
                            strategy_params[param_name] = float(value_str)
                    elif isinstance(default, list):
                        # Parse comma-separated list
                        if not value_str:
                            strategy_params[param_name] = []
                        else:
                            # Infer element type from first element of default list
                            parts = [p.strip() for p in value_str.split(',') if p.strip()]
                            if default and len(default) > 0:
                                elem_type = type(default[0])
                                strategy_params[param_name] = [elem_type(p) for p in parts]
                            else:
                                # Default to float for numeric lists
                                try:
                                    strategy_params[param_name] = [float(p) for p in parts]
                                except ValueError:
                                    strategy_params[param_name] = parts
                    else:
                        strategy_params[param_name] = value_str
                elif isinstance(control, ft.Checkbox):
                    strategy_params[param_name] = control.value
                elif isinstance(control, ft.Dropdown):
                    strategy_params[param_name] = control.value

            # Parse symbols
            symbols_str = self.symbols_input.value.strip()
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]

            if not symbols:
                self._show_error("Please enter at least one symbol")
                return None

            # Get dates
            start_date = self.start_date_picker.value.strip()
            end_date = self.end_date_picker.value.strip()

            # Validate date format (basic)
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                self._show_error("Invalid date format. Use YYYY-MM-DD")
                return None

            # Get backtest parameters
            initial_capital = float(self.initial_capital_input.value)
            fees_pct = float(self.fees_input.value)
            fees = fees_pct / 100.0  # Convert percentage to decimal
            risk_profile = self.risk_profile_dropdown.value

            # Get worker count and parallel mode
            workers = int(self.workers_slider.value)
            parallel = self.parallel_checkbox.value
            generate_full_output = self.generate_full_output_checkbox.value

            # Cap workers at number of symbols (no point having more workers than symbols)
            if workers > len(symbols):
                workers = len(symbols)

            # Get portfolio mode and settings
            portfolio_mode = self.portfolio_mode_dropdown.value

            # Portfolio settings (only relevant in Multi-Symbol Portfolio mode)
            position_sizing_method = self.position_sizing_dropdown.value
            rebalancing_frequency = self.rebalancing_frequency_dropdown.value
            rebalancing_threshold_pct = float(self.rebalancing_threshold_input.value) / 100.0

            return {
                'strategy_class': strategy_class,
                'strategy_params': strategy_params,
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'fees': fees,
                'risk_profile': risk_profile,
                'workers': workers,
                'parallel': parallel,
                'generate_full_output': generate_full_output,
                'portfolio_mode': portfolio_mode,
                'position_sizing_method': position_sizing_method,
                'rebalancing_frequency': rebalancing_frequency,
                'rebalancing_threshold_pct': rebalancing_threshold_pct
            }

        except ValueError as e:
            self._show_error(f"Invalid parameter value: {e}")
            return None
        except Exception as e:
            self._show_error(f"Configuration error: {e}")
            return None

    def _show_error(self, message: str):
        """Show error dialog."""
        if self.page:
            dlg = ft.AlertDialog(
                title=ft.Text("Configuration Error"),
                content=ft.Text(message),
                actions=[
                    ft.TextButton("OK", on_click=lambda e: self.page.close(dlg))
                ]
            )
            self.page.open(dlg)

    def _show_success(self, message: str):
        """Show success dialog."""
        if self.page:
            dlg = ft.AlertDialog(
                title=ft.Text("Success"),
                content=ft.Text(message),
                actions=[
                    ft.TextButton("OK", on_click=lambda e: self.page.close(dlg))
                ]
            )
            self.page.open(dlg)

    def _show_input_dialog(self, title: str, hint: str, on_submit: callable):
        """Show input dialog for preset/list names."""
        if not self.page:
            return

        input_field = ft.TextField(
            label="Name",
            hint_text=hint,
            autofocus=True,
            width=300
        )

        def on_ok(e):
            name = input_field.value.strip()
            if name:
                self.page.close(dlg)
                on_submit(name)
            else:
                input_field.error_text = "Name cannot be empty"
                input_field.update()

        dlg = ft.AlertDialog(
            title=ft.Text(title),
            content=input_field,
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self.page.close(dlg)),
                ft.TextButton("OK", on_click=on_ok)
            ]
        )
        self.page.open(dlg)

    # ========== Preset Management ==========

    def _refresh_preset_dropdown(self):
        """Refresh preset dropdown with saved presets."""
        preset_names = self.config_manager.get_preset_names()
        self.preset_dropdown.options = [
            ft.dropdown.Option(name) for name in preset_names
        ]
        if self.page:
            self.preset_dropdown.update()

    def _on_save_preset_clicked(self, e):
        """Handle save preset button click."""
        def save_preset(name):
            config = self._collect_configuration()
            if config:
                self.config_manager.save_preset(name, config)
                self._refresh_preset_dropdown()
                self._show_success(f"Preset '{name}' saved successfully!")

        self._show_input_dialog(
            "Save Configuration Preset",
            "Enter a name for this preset",
            save_preset
        )

    def _on_load_preset_clicked(self, e):
        """Handle load preset button click."""
        preset_name = self.preset_dropdown.value
        if not preset_name:
            self._show_error("Please select a preset to load")
            return

        config = self.config_manager.load_preset(preset_name)
        if config:
            self._load_configuration(config)
            self._show_success(f"Preset '{preset_name}' loaded successfully!")
        else:
            self._show_error(f"Failed to load preset '{preset_name}'")

    def _load_configuration(self, config: Dict[str, Any]):
        """Load configuration into UI controls."""
        # Set strategy
        strategy_class = config.get('strategy_class')
        if strategy_class:
            # Handle both string names (from JSON) and class objects (legacy)
            strategy_name = strategy_class if isinstance(strategy_class, str) else strategy_class.__name__
            self.strategy_dropdown.value = strategy_name
            self._on_strategy_changed(None)

            # Set strategy parameters
            strategy_params = config.get('strategy_params', {})
            for param_name, value in strategy_params.items():
                if param_name in self.param_controls:
                    control = self.param_controls[param_name]
                    if isinstance(control, ft.TextField):
                        control.value = str(value)
                    elif isinstance(control, ft.Checkbox):
                        control.value = bool(value)
                    elif isinstance(control, ft.Dropdown):
                        control.value = value

        # Set symbols
        if 'symbols' in config:
            self.symbols_input.value = ', '.join(config['symbols'])

        # Set dates
        if 'start_date' in config:
            start_str = config['start_date']
            self.start_date_picker.value = start_str
            try:
                self.start_date_value = datetime.strptime(start_str, "%Y-%m-%d")
            except:
                pass

        if 'end_date' in config:
            end_str = config['end_date']
            self.end_date_picker.value = end_str
            try:
                self.end_date_value = datetime.strptime(end_str, "%Y-%m-%d")
            except:
                pass

        # Set backtest parameters
        if 'initial_capital' in config:
            self.initial_capital_input.value = str(int(config['initial_capital']))
        if 'fees' in config:
            fees_pct = config['fees'] * 100.0
            self.fees_input.value = str(fees_pct)
        if 'risk_profile' in config:
            self.risk_profile_dropdown.value = config['risk_profile']

        # Set execution settings
        if 'workers' in config:
            self.workers_slider.value = config['workers']
        if 'parallel' in config:
            self.parallel_checkbox.value = config['parallel']
        if 'generate_full_output' in config:
            self.generate_full_output_checkbox.value = config['generate_full_output']

        # Set portfolio mode and settings
        if 'portfolio_mode' in config:
            self.portfolio_mode_dropdown.value = config['portfolio_mode']
            # Trigger mode change to show/hide portfolio settings
            self._on_portfolio_mode_changed(None)

        if 'position_sizing_method' in config:
            self.position_sizing_dropdown.value = config['position_sizing_method']
        if 'rebalancing_frequency' in config:
            self.rebalancing_frequency_dropdown.value = config['rebalancing_frequency']
        if 'rebalancing_threshold_pct' in config:
            threshold_pct = config['rebalancing_threshold_pct'] * 100.0
            self.rebalancing_threshold_input.value = str(threshold_pct)

        # Update UI
        if self.page:
            self.update()

    # ========== Symbol List Management ==========

    def _refresh_symbol_list_dropdown(self):
        """Refresh symbol list dropdown with saved lists."""
        list_info = self.config_manager.get_symbol_list_info()
        self.symbol_list_dropdown.options = [
            ft.dropdown.Option(f"{name} ({count})", key=name)
            for name, count in list_info.items()
        ]
        if self.page:
            self.symbol_list_dropdown.update()

    def _on_save_symbol_list_clicked(self, e):
        """Handle save symbol list button click."""
        symbols_str = self.symbols_input.value.strip()
        symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]

        if not symbols:
            self._show_error("Please enter symbols before saving")
            return

        def save_list(name):
            self.config_manager.save_symbol_list(name, symbols)
            self._refresh_symbol_list_dropdown()
            self._show_success(f"Symbol list '{name}' saved with {len(symbols)} symbols!")

        self._show_input_dialog(
            "Save Symbol List",
            "Enter a name for this symbol list",
            save_list
        )

    def _on_load_symbol_list_clicked(self, e):
        """Handle load symbol list button click."""
        if not self.symbol_list_dropdown.value:
            self._show_error("Please select a symbol list to load")
            return

        # Extract the list name from dropdown option key
        list_name = self.symbol_list_dropdown.value

        symbols = self.config_manager.load_symbol_list(list_name)
        if symbols:
            self.symbols_input.value = ', '.join(symbols)
            if self.page:
                self.symbols_input.update()
            self._show_success(f"Loaded {len(symbols)} symbols from '{list_name}'")
        else:
            self._show_error(f"Failed to load symbol list '{list_name}'")

    # ========== Quick Re-run ==========

    def _on_quick_rerun_clicked(self, e):
        """Handle quick re-run button click."""
        last_config = self.config_manager.load_last_run()
        if last_config:
            self._load_configuration(last_config)
            self._show_success("Last configuration loaded! Click 'Run Backtests' to execute.")
        else:
            self._show_error("No previous run configuration found")

    # ========== Symbol Download Dialog ==========

    def _show_download_dialog(self, config: Dict[str, Any], missing_symbols: list, start_date: str, end_date: str):
        """
        Show dialog asking user if they want to download missing symbols.

        Args:
            config: Backtest configuration
            missing_symbols: List of missing symbol strings
            start_date: Start date for download
            end_date: End date for download
        """
        if not self.page:
            return

        missing_text = ', '.join(missing_symbols)
        symbol_count = len(missing_symbols)

        content = ft.Column([
            ft.Text(
                f"The following {symbol_count} symbol(s) are not in your database:",
                size=14,
                weight=ft.FontWeight.BOLD
            ),
            ft.Text(missing_text, size=12, color=ft.Colors.AMBER_400),
            ft.Divider(),
            ft.Text(
                "Would you like to download them from Alpaca IEX feed?",
                size=13
            ),
            ft.Text(
                f"Date range: {start_date} to {end_date}",
                size=12,
                color=ft.Colors.GREY_400,
                italic=True
            ),
            ft.Text(
                "Note: Download may take a few minutes depending on date range.",
                size=11,
                color=ft.Colors.GREY_500,
                italic=True
            )
        ], spacing=10, tight=True)

        def on_download_confirmed(e):
            self.page.close(dlg)
            self._download_missing_symbols(config, missing_symbols, start_date, end_date)

        def on_cancel(e):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            title=ft.Text("Missing Symbols Detected"),
            content=content,
            actions=[
                ft.TextButton("Cancel", on_click=on_cancel),
                ft.ElevatedButton(
                    "Download & Run",
                    icon=ft.Icons.DOWNLOAD,
                    on_click=on_download_confirmed,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.GREEN_700
                    )
                )
            ],
            actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        self.page.open(dlg)

    def _download_missing_symbols(self, config: Dict[str, Any], missing_symbols: list, start_date: str, end_date: str):
        """
        Download missing symbols with progress dialog.

        Args:
            config: Backtest configuration
            missing_symbols: List of missing symbols
            start_date: Start date
            end_date: End date
        """
        from gui.utils.symbol_downloader import SymbolDownloader

        if not self.page:
            return

        # Create progress dialog
        progress_text = ft.Text("Preparing to download...", size=13, color=ft.Colors.WHITE)
        progress_bar = ft.ProgressBar(width=400, value=0)
        status_text = ft.Text("", size=11, color=ft.Colors.GREY_400)

        progress_dlg = ft.AlertDialog(
            title=ft.Text("Downloading Symbols"),
            content=ft.Container(
                content=ft.Column([
                    progress_text,
                    progress_bar,
                    status_text
                ], spacing=10, tight=True),
                width=450
            ),
            modal=True
        )

        self.page.open(progress_dlg)

        def update_progress(symbol: str, current: int, total: int):
            """Update progress UI."""
            progress_text.value = f"Downloading {symbol}... ({current}/{total})"
            progress_bar.value = current / total
            status_text.value = f"{current} of {total} completed"
            if self.page:
                progress_text.update()
                progress_bar.update()
                status_text.update()

        # Download in background
        def download_task():
            downloader = SymbolDownloader()
            result = downloader.download_symbols(
                missing_symbols,
                start_date,
                end_date,
                progress_callback=update_progress
            )

            # Close progress dialog
            self.page.close(progress_dlg)

            # Show results
            successful = result['successful']
            failed = result['failed']

            if failed:
                failed_symbols = [sym for sym, _ in failed]
                failed_text = ', '.join(failed_symbols)

                # Some downloads failed
                error_content = ft.Column([
                    ft.Text(
                        f"{len(successful)} symbol(s) downloaded successfully.",
                        size=13,
                        color=ft.Colors.GREEN_400
                    ),
                    ft.Text(
                        f"{len(failed)} symbol(s) failed to download:",
                        size=13,
                        color=ft.Colors.RED_400
                    ),
                    ft.Text(failed_text, size=11, color=ft.Colors.RED_300),
                    ft.Divider(),
                    ft.Text(
                        "These symbols may not exist on the IEX feed or the date range may be invalid.",
                        size=11,
                        color=ft.Colors.GREY_400,
                        italic=True
                    )
                ], spacing=8, tight=True)

                if successful:
                    # Ask if user wants to proceed with successful symbols only
                    def proceed_with_available(e):
                        self.page.close(error_dlg)
                        # Update config to only include available symbols
                        all_available = successful + [s for s in config['symbols'] if s not in missing_symbols]
                        config['symbols'] = all_available
                        self._execute_backtest(config)

                    def cancel_run(e):
                        self.page.close(error_dlg)

                    error_dlg = ft.AlertDialog(
                        title=ft.Text("Partial Download Success"),
                        content=error_content,
                        actions=[
                            ft.TextButton("Cancel", on_click=cancel_run),
                            ft.ElevatedButton(
                                "Proceed with Available Symbols",
                                on_click=proceed_with_available,
                                style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700)
                            )
                        ]
                    )
                else:
                    # All downloads failed
                    def close_error(e):
                        self.page.close(error_dlg)

                    error_dlg = ft.AlertDialog(
                        title=ft.Text("Download Failed"),
                        content=error_content,
                        actions=[ft.TextButton("OK", on_click=close_error)]
                    )

                self.page.open(error_dlg)

            else:
                # All downloads successful - proceed with backtest
                self._show_success(f"Successfully downloaded {len(successful)} symbol(s)!")
                self._execute_backtest(config)

        # Run download in thread to avoid blocking UI
        import threading
        download_thread = threading.Thread(target=download_task, daemon=True)
        download_thread.start()

    # ========== Cache Management ==========

    def _on_view_cache_clicked(self, e):
        """Handle View Cache button click."""
        from utils.cache_manager import CacheManager

        cache_manager = CacheManager()
        runs = cache_manager.list_cached_runs(limit=50)
        cache_size = cache_manager.get_cache_size()

        if not runs:
            self._show_error("No cached results found")
            return

        content = ft.Column([
            ft.Text(
                f"Cache Size: {cache_size['total_size_mb']:.2f} MB ({cache_size['file_count']} files)",
                size=12,
                color=ft.Colors.GREY_400
            ),
            ft.Divider(),
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("Timestamp", size=11, weight=ft.FontWeight.BOLD, expand=2),
                        ft.Text("Strategy", size=11, weight=ft.FontWeight.BOLD, expand=2),
                        ft.Text("Symbols", size=11, weight=ft.FontWeight.BOLD, expand=1)
                    ]),
                    ft.Divider(height=1),
                    *[
                        ft.Row([
                            ft.Text(run['timestamp'][:19], size=10, expand=2),
                            ft.Text(run['strategy'], size=10, expand=2),
                            ft.Text(str(run['num_symbols']), size=10, expand=1)
                        ])
                        for run in runs[:10]
                    ]
                ], spacing=5, scroll=ft.ScrollMode.AUTO),
                height=300
            )
        ], spacing=10, tight=True)

        dlg = ft.AlertDialog(
            title=ft.Text("Cached Results"),
            content=content,
            actions=[ft.TextButton("Close", on_click=lambda e: self.page.close(dlg))]
        )

        self.page.open(dlg)

    def _on_clear_cache_clicked(self, e):
        """Handle Clear Cache button click."""
        from utils.cache_manager import CacheManager

        cache_manager = CacheManager()
        cache_size = cache_manager.get_cache_size()

        content = ft.Column([
            ft.Text(
                "Are you sure you want to clear all cached results?",
                size=14,
                weight=ft.FontWeight.BOLD
            ),
            ft.Divider(),
            ft.Text(f"Cache Size: {cache_size['total_size_mb']:.2f} MB", size=12),
            ft.Text(f"Cached Runs: {cache_size['num_cached_runs']}", size=12),
            ft.Text(f"Files: {cache_size['file_count']}", size=12),
            ft.Divider(),
            ft.Text(
                "This action cannot be undone.",
                size=11,
                color=ft.Colors.RED_300,
                italic=True
            )
        ], spacing=8, tight=True)

        def confirm_clear(e):
            self.page.close(dlg)
            cleared = cache_manager.clear_cache()
            self._show_success(f"Cleared {cleared} cached result(s)")

        def cancel_clear(e):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            title=ft.Text("Clear Cache"),
            content=content,
            actions=[
                ft.TextButton("Cancel", on_click=cancel_clear),
                ft.ElevatedButton(
                    "Clear All",
                    on_click=confirm_clear,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.RED_700
                    )
                )
            ],
            actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        self.page.open(dlg)

    # ========== Date Picker Handlers ==========

    def _on_start_date_click(self, e):
        """Handle start date calendar button click."""
        if not self.page:
            return

        date_picker = ft.DatePicker(
            first_date=datetime(2000, 1, 1),
            last_date=datetime.now(),
            value=self.start_date_value,
            on_change=self._on_start_date_change,
            on_dismiss=lambda e: None
        )

        self.page.overlay.append(date_picker)
        date_picker.open = True
        self.page.update()

    def _on_start_date_change(self, e):
        """Handle start date selection from calendar."""
        if e.control.value:
            self.start_date_value = e.control.value
            self.start_date_picker.value = e.control.value.strftime("%Y-%m-%d")
            self.start_date_picker.update()

    def _on_end_date_click(self, e):
        """Handle end date calendar button click."""
        if not self.page:
            return

        date_picker = ft.DatePicker(
            first_date=datetime(2000, 1, 1),
            last_date=datetime.now(),
            value=self.end_date_value,
            on_change=self._on_end_date_change,
            on_dismiss=lambda e: None
        )

        self.page.overlay.append(date_picker)
        date_picker.open = True
        self.page.update()

    def _on_end_date_change(self, e):
        """Handle end date selection from calendar."""
        if e.control.value:
            self.end_date_value = e.control.value
            self.end_date_picker.value = e.control.value.strftime("%Y-%m-%d")
            self.end_date_picker.update()
