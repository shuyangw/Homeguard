"""
Momentum Strategy Implementations.

Pure signal generation strategies based on momentum indicators.
No dependencies on backtesting or live trading infrastructure.
"""

from src.strategies.implementations.momentum.momentum_signals import (
    MACDMomentumSignals,
    BreakoutMomentumSignals
)

__all__ = [
    'MACDMomentumSignals',
    'BreakoutMomentumSignals'
]
