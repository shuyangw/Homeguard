"""
Base trading strategies - DEPRECATED.

These strategies have been moved to src/strategies/research/.
Update imports to use the new location:

  # Old (deprecated):
  from src.strategies.base_strategies.moving_average import MovingAverageCrossover

  # New:
  from src.strategies.research.moving_average import MovingAverageCrossover
"""

# Re-export from research for backward compatibility
from src.strategies.research.moving_average import MovingAverageCrossover
from src.strategies.research.mean_reversion import MeanReversion
from src.strategies.research.momentum import MomentumStrategy

__all__ = ['MovingAverageCrossover', 'MeanReversion', 'MomentumStrategy']
