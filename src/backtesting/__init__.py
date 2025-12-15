"""
Backtesting framework built on VectorBT for stock trading strategies.
"""

from .engine.backtest_engine import BacktestEngine
from .engine.streaming_data_loader import StreamingDataLoader
from .engine.rolling_results import RollingWindowResults
from .base.strategy import BaseStrategy

__all__ = ['BacktestEngine', 'StreamingDataLoader', 'RollingWindowResults', 'BaseStrategy']
