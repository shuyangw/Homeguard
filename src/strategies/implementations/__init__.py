"""
Pure Strategy Implementations.

Pure signal generation strategies with no dependencies on
backtesting or live trading infrastructure.

Each implementation:
- Extends StrategySignals abstract class
- Generates signals based purely on market data
- Can be used by both backtest and live trading adapters

Directory structure:
- moving_average/: MA-based strategies
- momentum/: Momentum-based strategies
- mean_reversion/: Mean reversion strategies
- overnight/: Overnight mean reversion strategies
- pairs/: Pairs trading strategies
"""

from src.strategies.implementations.moving_average import (
    MACrossoverSignals,
    TripleMACrossoverSignals
)
from src.strategies.implementations.momentum import (
    MACDMomentumSignals,
    BreakoutMomentumSignals
)

__all__ = [
    'MACrossoverSignals',
    'TripleMACrossoverSignals',
    'MACDMomentumSignals',
    'BreakoutMomentumSignals'
]
