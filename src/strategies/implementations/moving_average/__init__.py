"""
Moving Average Strategy Implementations.

Pure signal generation strategies based on moving average crossovers.
No dependencies on backtesting or live trading infrastructure.
"""

from src.strategies.implementations.moving_average.ma_crossover_signals import (
    MACrossoverSignals,
    TripleMACrossoverSignals
)

__all__ = [
    'MACrossoverSignals',
    'TripleMACrossoverSignals'
]
