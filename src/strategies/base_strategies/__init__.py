"""
Basic trading strategies.
"""

from strategies.base_strategies.moving_average import MovingAverageCrossover
from strategies.base_strategies.mean_reversion import MeanReversion
from strategies.base_strategies.momentum import MomentumStrategy

__all__ = ['MovingAverageCrossover', 'MeanReversion', 'MomentumStrategy']
