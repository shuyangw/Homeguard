"""
Advanced trading strategy implementations.
"""

from src.strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversion
from src.strategies.advanced.cross_sectional_momentum import CrossSectionalMomentum
from src.strategies.advanced.pairs_trading import PairsTrading

__all__ = [
    'VolatilityTargetedMomentum',
    'OvernightMeanReversion',
    'CrossSectionalMomentum',
    'PairsTrading'
]
