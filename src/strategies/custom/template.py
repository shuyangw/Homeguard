"""
Template for creating custom trading strategies.

Copy this file and implement your own strategy logic.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators
from backtesting.utils.validation import validate_positive_int


class CustomStrategyTemplate(LongOnlyStrategy):
    """
    Custom strategy template - replace with your strategy logic.

    Describe your strategy here:
    Entry: [Your entry condition]
    Exit: [Your exit condition]
    """

    def __init__(self, param1: int = 20, param2: float = 1.5):
        """
        Initialize strategy with parameters.

        Args:
            param1: Description of parameter 1 (default: 20)
            param2: Description of parameter 2 (default: 1.5)
        """
        super().__init__(param1=param1, param2=param2)

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Add your parameter validation logic here.
        """
        validate_positive_int(self.params['param1'], 'param1')

        if self.params['param2'] <= 0:
            raise ValueError("param2 must be positive")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        indicator1 = Indicators.sma(close, self.params['param1'])

        entries = close > indicator1
        exits = close < indicator1

        return entries, exits
