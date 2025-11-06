"""
Chunking strategies for backtesting.

Provides walk-forward validation to prevent overfitting by
testing on truly out-of-sample data.
"""

from backtesting.chunking.walk_forward import (
    WalkForwardWindow,
    WalkForwardResults,
    WalkForwardValidator
)

__all__ = [
    'WalkForwardWindow',
    'WalkForwardResults',
    'WalkForwardValidator'
]
