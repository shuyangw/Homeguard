"""
Universe Management.

Organized lists of tradable symbols for different asset classes.
Replaces hardcoded symbol lists throughout the codebase.

Usage:
    ```python
    from src.strategies.universe import ETFUniverse, EquityUniverse

    # Get leveraged 3x ETFs
    etfs = ETFUniverse.LEVERAGED_3X

    # Get FAANG stocks
    stocks = EquityUniverse.FAANG

    # Get S&P 500 constituents dynamically
    sp500 = EquityUniverse.load_sp500()
    ```
"""

from src.strategies.universe.etf_universe import ETFUniverse
from src.strategies.universe.equity_universe import EquityUniverse

__all__ = [
    'ETFUniverse',
    'EquityUniverse',
]
