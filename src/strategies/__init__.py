"""
Trading strategy implementations for backtesting.
"""

from src.strategies.base_strategies.moving_average import MovingAverageCrossover, TripleMovingAverage
from src.strategies.base_strategies.mean_reversion import MeanReversion, RSIMeanReversion
from src.strategies.base_strategies.momentum import MomentumStrategy, BreakoutStrategy

from src.strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversion
from src.strategies.advanced.cross_sectional_momentum import CrossSectionalMomentum
from src.strategies.advanced.pairs_trading import PairsTrading

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
