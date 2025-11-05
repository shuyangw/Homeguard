"""
Backtesting framework built on VectorBT for stock trading strategies.
"""

from .engine.backtest_engine import BacktestEngine
from .engine.data_loader import DataLoader
from .base.strategy import BaseStrategy

__all__ = ['BacktestEngine', 'DataLoader', 'BaseStrategy']
