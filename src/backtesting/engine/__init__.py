"""
Core backtesting engine components.
"""

from .backtest_engine import BacktestEngine
from .base_portfolio import BasePortfolio
from .streaming_data_loader import StreamingDataLoader
from .metrics import PerformanceMetrics
from .rolling_results import RollingWindowResults
from .portfolio_simulator import Portfolio, from_signals

# Also export V2 portfolio for direct access to state tracking
try:
    from src.backtesting_v2.engine.portfolio_simulator import PortfolioV2
except ImportError:
    PortfolioV2 = None

__all__ = [
    'BacktestEngine',
    'BasePortfolio',
    'StreamingDataLoader',
    'PerformanceMetrics',
    'RollingWindowResults',
    'Portfolio',
    'PortfolioV2',
    'from_signals',
]
