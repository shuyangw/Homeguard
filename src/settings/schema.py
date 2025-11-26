"""
Pydantic models for backtest configuration validation.

Provides type-safe configuration with automatic validation and
sensible defaults for all backtest parameters.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class BacktestMode(str, Enum):
    """Backtest execution modes."""
    SINGLE = "single"
    SWEEP = "sweep"
    OPTIMIZE = "optimize"
    WALK_FORWARD = "walk_forward"


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED_PERCENT = "fixed_percent"
    KELLY = "kelly"
    VOLATILITY_TARGET = "volatility_target"
    EQUAL_WEIGHT = "equal_weight"
    CUSTOM = "custom"


class StrategyConfig(BaseModel):
    """Strategy selection and parameters."""
    name: str = Field(..., description="Strategy class name from registry")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


class SymbolsConfig(BaseModel):
    """Symbol selection - list, universe reference, or file path."""
    list: Optional[List[str]] = Field(default=None, description="Direct list of symbols")
    universe: Optional[str] = Field(default=None, description="Universe reference (e.g., 'production.conservative')")
    file: Optional[str] = Field(default=None, description="Path to file with symbols")

    @model_validator(mode='after')
    def validate_symbols_source(self):
        """Ensure at least one symbol source is provided."""
        sources = [self.list, self.universe, self.file]
        non_none = [s for s in sources if s is not None]
        if len(non_none) == 0:
            raise ValueError("Must provide symbols via 'list', 'universe', or 'file'")
        if len(non_none) > 1:
            raise ValueError("Provide only one of 'list', 'universe', or 'file'")
        return self


class DatesConfig(BaseModel):
    """Date range selection - explicit dates or preset reference."""
    start: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    preset: Optional[str] = Field(default=None, description="Preset reference (e.g., 'full_periods.five_years')")

    @model_validator(mode='after')
    def validate_dates_source(self):
        """Ensure dates are provided via explicit values or preset."""
        if self.preset is not None:
            return self
        if self.start is None or self.end is None:
            raise ValueError("Must provide both 'start' and 'end' dates, or a 'preset'")
        return self


class BacktestSettings(BaseModel):
    """Core backtest execution settings."""
    initial_capital: float = Field(default=100000.0, gt=0, description="Starting capital")
    fees: float = Field(default=0.001, ge=0, le=0.1, description="Trading fees as decimal")
    slippage: float = Field(default=0.0005, ge=0, le=0.1, description="Slippage as decimal")
    benchmark: str = Field(default="SPY", description="Benchmark symbol for comparison")
    market_hours_only: bool = Field(default=True, description="Filter to market hours only")
    allow_shorts: bool = Field(default=False, description="Allow short selling")


class RiskSettings(BaseModel):
    """Risk management settings."""
    enabled: bool = Field(default=True, description="Enable risk management")
    position_sizing_method: PositionSizingMethod = Field(
        default=PositionSizingMethod.FIXED_PERCENT,
        description="Position sizing method"
    )
    position_size_pct: float = Field(default=0.10, gt=0, le=1.0, description="Position size as % of capital")
    max_positions: int = Field(default=5, ge=1, description="Maximum concurrent positions")
    stop_loss_pct: Optional[float] = Field(default=None, ge=0, le=1.0, description="Stop loss as % (None=disabled)")
    stop_loss_type: str = Field(default="fixed", description="Stop loss type: 'fixed' or 'trailing'")


class SweepSettings(BaseModel):
    """Settings for sweep mode (multi-symbol backtest)."""
    sort_by: str = Field(default="Sharpe Ratio", description="Column to sort results by")
    top_n: Optional[int] = Field(default=None, ge=1, description="Show only top N results")
    parallel: bool = Field(default=True, description="Run in parallel")
    max_workers: int = Field(default=4, ge=1, le=32, description="Max parallel workers")
    export_csv: bool = Field(default=True, description="Export results to CSV")
    export_html: bool = Field(default=True, description="Export results to HTML")


class OptimizationSettings(BaseModel):
    """Settings for optimization mode (parameter search)."""
    metric: str = Field(default="sharpe_ratio", description="Optimization metric")
    param_grid: Dict[str, List[Any]] = Field(default_factory=dict, description="Parameter grid for search")
    n_jobs: int = Field(default=4, ge=1, description="Parallel jobs for optimization")
    max_iterations: Optional[int] = Field(default=None, ge=1, description="Max iterations (None=unlimited)")

    @field_validator('metric')
    @classmethod
    def validate_metric(cls, v):
        valid_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']
        if v not in valid_metrics:
            raise ValueError(f"Invalid metric '{v}'. Valid: {valid_metrics}")
        return v


class WalkForwardSettings(BaseModel):
    """Settings for walk-forward validation mode."""
    train_months: int = Field(default=12, ge=1, description="Training period in months")
    test_months: int = Field(default=6, ge=1, description="Testing period in months")
    step_months: int = Field(default=6, ge=1, description="Step size in months")
    n_splits: Optional[int] = Field(default=None, ge=1, description="Number of splits (None=auto)")
    export_results: bool = Field(default=True, description="Export walk-forward results")


class OutputSettings(BaseModel):
    """Output and reporting settings."""
    directory: Optional[str] = Field(default=None, description="Output directory (None=auto-generate)")
    save_trades: bool = Field(default=True, description="Save trade log")
    save_reports: bool = Field(default=True, description="Save performance reports")
    quantstats: bool = Field(default=True, description="Generate QuantStats tearsheet")
    visualize: bool = Field(default=True, description="Generate charts and visualizations")
    timestamp_files: bool = Field(default=True, description="Add timestamps to output files")
    verbosity: int = Field(default=1, ge=0, le=3, description="Logging verbosity (0-3)")


class BacktestConfig(BaseModel):
    """
    Root configuration model for config-driven backtesting.

    Supports all backtest modes: single, sweep, optimize, walk_forward.
    Configuration can be loaded from YAML files with inheritance support.
    """
    mode: BacktestMode = Field(default=BacktestMode.SINGLE, description="Backtest mode")
    strategy: StrategyConfig = Field(..., description="Strategy configuration")
    symbols: SymbolsConfig = Field(..., description="Symbol configuration")
    dates: DatesConfig = Field(..., description="Date range configuration")
    backtest: BacktestSettings = Field(default_factory=BacktestSettings, description="Backtest settings")
    risk: RiskSettings = Field(default_factory=RiskSettings, description="Risk management settings")
    sweep: SweepSettings = Field(default_factory=SweepSettings, description="Sweep mode settings")
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings, description="Optimization settings")
    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings, description="Walk-forward settings")
    output: OutputSettings = Field(default_factory=OutputSettings, description="Output settings")

    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields for forward compatibility
