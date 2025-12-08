"""
Trading strategy implementations.

Production Strategies (deployed to EC2):
- OMR: src.strategies.advanced.overnight_mean_reversion
- MP: src.strategies.advanced.momentum_protection_strategy
- RAMP: src.strategies.advanced.ramp_strategy

Research Strategies (backtesting/experimentation only):
- src.strategies.research.moving_average
- src.strategies.research.mean_reversion
- src.strategies.research.momentum
- src.strategies.research.pairs_trading
- src.strategies.research.cross_sectional_momentum
- src.strategies.research.volatility_targeted_momentum
- src.strategies.research.breakout_strategies
- src.strategies.research.high52_breakout_strategy

Note: Imports are commented out to avoid import chain issues.
Import strategies directly when needed.
"""

__all__ = []
