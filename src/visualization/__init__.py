"""
Visualization engine for backtesting results.

This module provides comprehensive visualization and logging capabilities for
backtesting results, including:
- Candlestick charts with volume and trade count
- Trade signal overlays (buy/sell markers)
- Detailed trade logs with configurable verbosity
- Organized output directory structure
"""

from .config import VisualizationConfig
from .logger import TradeLogger
from .charts.candlestick import CandlestickChart
from .reports.report_generator import ReportGenerator
from .utils.output_manager import OutputManager

__all__ = [
    'VisualizationConfig',
    'TradeLogger',
    'CandlestickChart',
    'ReportGenerator',
    'OutputManager',
]
