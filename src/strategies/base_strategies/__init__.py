"""
Basic trading strategies.
"""

from src.strategies.base_strategies.moving_average import MovingAverageCrossover
from src.strategies.base_strategies.mean_reversion import MeanReversion
from src.strategies.base_strategies.momentum import MomentumStrategy

__all__ = ['MovingAverageCrossover', 'MeanReversion', 'MomentumStrategy']
