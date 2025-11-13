"""
Core strategy abstractions.

This module provides base classes and data structures for implementing
pure trading strategies that are independent of backtesting or live trading
infrastructure.

Key components:
- Signal: Pure signal data structure
- SignalBatch: Collection of signals
- StrategySignals: Abstract base class for pure strategies
- DataRequirements: Specification of data requirements
"""

from src.strategies.core.signal import Signal, SignalBatch
from src.strategies.core.base_strategy import StrategySignals, DataRequirements

__all__ = [
    'Signal',
    'SignalBatch',
    'StrategySignals',
    'DataRequirements',
]
