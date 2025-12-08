"""
Mean reversion trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.backtesting.base.strategy import LongOnlyStrategy
from src.backtesting.utils.indicators import Indicators
from src.backtesting.utils.validation import validate_positive_int, validate_positive_float


class MeanReversion(LongOnlyStrategy):
    """
    Bollinger Bands mean reversion strategy.

    Entry: Price crosses below lower band (oversold)
    Exit: Price crosses above middle band (mean reversion complete)
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        exit_at_middle: bool = True
    ):
        """
        Initialize strategy.

        Args:
            window: Bollinger Bands period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            exit_at_middle: Exit at middle band vs upper band (default: True)
        """
        super().__init__(
            window=window,
            num_std=num_std,
            exit_at_middle=exit_at_middle
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['window'], 'window')
        validate_positive_float(self.params['num_std'], 'num_std')

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.
        """
        close = data['close']

        upper, middle, lower = Indicators.bollinger_bands(
            close,
            window=self.params['window'],
            num_std=self.params['num_std']
        )

        entries = (close < lower) & (close.shift(1) >= lower.shift(1))

        if self.params['exit_at_middle']:
            exits = (close > middle) & (close.shift(1) <= middle.shift(1))
        else:
            exits = (close > upper) & (close.shift(1) <= upper.shift(1))

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits


class RSIMeanReversion(LongOnlyStrategy):
    """
    RSI-based mean reversion strategy.

    Entry: RSI crosses below oversold threshold
    Exit: RSI crosses above overbought threshold
    """

    def __init__(
        self,
        rsi_window: int = 14,
        oversold: int = 30,
        overbought: int = 70
    ):
        """
        Initialize strategy.

        Args:
            rsi_window: RSI calculation period (default: 14)
            oversold: Oversold threshold (default: 30)
            overbought: Overbought threshold (default: 70)
        """
        super().__init__(
            rsi_window=rsi_window,
            oversold=oversold,
            overbought=overbought
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['rsi_window'], 'rsi_window')

        if not (0 < self.params['oversold'] < 50):
            raise ValueError("oversold must be between 0 and 50")

        if not (50 < self.params['overbought'] < 100):
            raise ValueError("overbought must be between 50 and 100")

        if self.params['oversold'] >= self.params['overbought']:
            raise ValueError("oversold must be less than overbought")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.
        """
        close = data['close']

        rsi = Indicators.rsi(close, window=self.params['rsi_window'])

        entries = (rsi < self.params['oversold']) & (rsi.shift(1) >= self.params['oversold'])
        exits = (rsi > self.params['overbought']) & (rsi.shift(1) <= self.params['overbought'])

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits
