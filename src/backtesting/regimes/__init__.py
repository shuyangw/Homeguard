"""
Regime detection and analysis for backtesting.

Analyzes strategy performance across different market regimes
to assess robustness and identify failure conditions.
"""

from backtesting.regimes.detector import (
    TrendDetector,
    VolatilityDetector,
    DrawdownDetector,
    RegimeLabel
)
from backtesting.regimes.analyzer import RegimeAnalyzer

__all__ = [
    'TrendDetector',
    'VolatilityDetector',
    'DrawdownDetector',
    'RegimeLabel',
    'RegimeAnalyzer'
]
