"""
Research/Non-Production Strategies.

These strategies are for backtesting, research, and experimentation only.
They are NOT deployed to production.

Production strategies are in:
- src/strategies/advanced/overnight_mean_reversion.py (OMR)
- src/strategies/advanced/momentum_protection_strategy.py (MP)
- src/strategies/advanced/ramp_strategy.py (RAMP)
"""

from src.strategies.research.moving_average import MovingAverageCrossover, TripleMovingAverage
from src.strategies.research.mean_reversion import MeanReversion, RSIMeanReversion
from src.strategies.research.momentum import MomentumStrategy, BreakoutStrategy
from src.strategies.research.pairs_trading import PairsTrading

__all__ = [
    'MovingAverageCrossover',
    'TripleMovingAverage',
    'MeanReversion',
    'RSIMeanReversion',
    'MomentumStrategy',
    'BreakoutStrategy',
    'PairsTrading',
]
