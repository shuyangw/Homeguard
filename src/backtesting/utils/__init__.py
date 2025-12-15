"""
Utility functions for backtesting.
"""

from .indicators import Indicators
from .validation import validate_parameters
from .risk_config import RiskConfig
from .position_sizer import (
    FixedPercentageSizer,
    FixedDollarSizer,
    VolatilityBasedSizer,
    KellyCriterionSizer,
    RiskParitySizer
)
from .risk_manager import (
    RiskManager,
    StopLoss,
    PercentageStopLoss,
    ATRStopLoss,
    TimeStopLoss,
    ProfitTargetStopLoss,
    Position
)
from .metrics_polars import (
    calculate_returns,
    calculate_total_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_all_metrics
)
from .performance import (
    PerformanceMonitor,
    TimingResult,
    MemorySnapshot,
    benchmark,
    compare_implementations,
    get_memory_usage_mb,
    estimate_dataframe_memory,
    memory_tracked
)

__all__ = [
    'Indicators',
    'validate_parameters',
    'RiskConfig',
    'FixedPercentageSizer',
    'FixedDollarSizer',
    'VolatilityBasedSizer',
    'KellyCriterionSizer',
    'RiskParitySizer',
    'RiskManager',
    'StopLoss',
    'PercentageStopLoss',
    'ATRStopLoss',
    'TimeStopLoss',
    'ProfitTargetStopLoss',
    'Position',
    # Polars-native metrics
    'calculate_returns',
    'calculate_total_return',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_all_metrics',
    # Performance utilities
    'PerformanceMonitor',
    'TimingResult',
    'MemorySnapshot',
    'benchmark',
    'compare_implementations',
    'get_memory_usage_mb',
    'estimate_dataframe_memory',
    'memory_tracked'
]
