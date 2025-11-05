"""
Trading strategy implementations for backtesting.
"""

from strategies.base_strategies.moving_average import MovingAverageCrossover, TripleMovingAverage
from strategies.base_strategies.mean_reversion import MeanReversion, RSIMeanReversion
from strategies.base_strategies.momentum import MomentumStrategy, BreakoutStrategy

from strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
from strategies.advanced.overnight_mean_reversion import OvernightMeanReversion
from strategies.advanced.cross_sectional_momentum import CrossSectionalMomentum
from strategies.advanced.pairs_trading import PairsTrading

__all__ = [
    'MovingAverageCrossover',
    'TripleMovingAverage',
    'MeanReversion',
    'RSIMeanReversion',
    'MomentumStrategy',
    'BreakoutStrategy',
    'VolatilityTargetedMomentum',
    'OvernightMeanReversion',
    'CrossSectionalMomentum',
    'PairsTrading'
]
