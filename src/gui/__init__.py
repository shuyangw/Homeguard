"""
Backtest Runner GUI - Flet-based desktop application for backtesting.

This package provides a modern, cross-platform GUI for configuring,
running, and analyzing backtests with real-time progress monitoring.

Usage:
    # As a module
    python -m gui

    # Or directly
    python src/gui/app.py

    # Or via launcher
    python run_gui.py
"""

__version__ = "2.0.0"
__author__ = "Homeguard Backtesting Team"

# Main application entry point
from gui.app import main, BacktestApp

# Views
from gui.views import SetupView, ExecutionView, ResultsView

# Workers
from gui.workers import GUIBacktestController

# Utils
from gui.utils import get_strategy_registry

__all__ = [
    'main',
    'BacktestApp',
    'SetupView',
    'ExecutionView',
    'ResultsView',
    'GUIBacktestController',
    'get_strategy_registry'
]
