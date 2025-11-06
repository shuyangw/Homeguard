"""
GUI optimization module for parameter optimization workflows.

This module provides GUI components for optimization:
- OptimizationDialog: Parameter grid configuration dialog
- OptimizationRunner: Execution and results display
"""

from gui.optimization.dialog import OptimizationDialog
from gui.optimization.runner import OptimizationRunner

__all__ = ['OptimizationDialog', 'OptimizationRunner']
