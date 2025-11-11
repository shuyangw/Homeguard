"""
Bollinger Bands mean reversion strategy with short selling capability.

This strategy uses a flip-flop model where it's always positioned (either long or short):
- When price crosses below lower band (oversold) → Go LONG
- When price crosses above upper band (overbought) → Go SHORT
- Trades flip between long and short positions based on Bollinger Band extremes
"""

import pandas as pd
import numpy as np
from typing import Tuple

from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators
from backtesting.utils.validation import validate_positive_int, validate_positive_float


class MeanReversionLongShort(LongOnlyStrategy):
    """
    Bollinger Bands mean reversion strategy with short selling (flip-flop model).

    Uses the portfolio simulator's flip-flop approach where:
    - Entry signals → Go LONG (or cover shorts and go long)
    - Exit signals → Go SHORT (or close longs and go short)

    Entry conditions:
    - Price crosses below lower Bollinger Band (oversold)
    - (Optional) Price crosses middle band from above when exit_at_middle=True

    Exit conditions:
    - Price crosses above upper Bollinger Band (overbought)
    - (Optional) Price crosses middle band from below when exit_at_middle=True

    When allow_shorts=True in the backtesting engine, this creates a flip-flop strategy
    that's always positioned (either long or short), capturing mean reversion in both directions.
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
            exit_at_middle: If True, exit/flip at middle band; if False, only flip at opposite extreme (default: True)
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
        Generate entry and exit signals for flip-flop long-short strategy.

        With allow_shorts=True:
        - Entry signals → Want to be LONG (buy if flat, or cover short and go long)
        - Exit signals → Want to be SHORT (sell if long, or open short if flat)

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        close = data['close']

        upper, middle, lower = Indicators.bollinger_bands(
            close,
            window=self.params['window'],
            num_std=self.params['num_std']
        )

        # ENTRY signals: Want to be LONG
        # 1. Price crosses below lower band (oversold)
        lower_cross = (close < lower) & (close.shift(1) >= lower.shift(1))

        if self.params['exit_at_middle']:
            # 2. Price crosses below middle band from above (exit short position, want to be long)
            middle_cross_down = (close < middle) & (close.shift(1) >= middle.shift(1))
            entries = lower_cross | middle_cross_down
        else:
            # Only enter on lower band (more conservative)
            entries = lower_cross

        # EXIT signals: Want to be SHORT
        # 1. Price crosses above upper band (overbought)
        upper_cross = (close > upper) & (close.shift(1) <= upper.shift(1))

        if self.params['exit_at_middle']:
            # 2. Price crosses above middle band from below (exit long position, want to be short)
            middle_cross_up = (close > middle) & (close.shift(1) <= middle.shift(1))
            exits = upper_cross | middle_cross_up
        else:
            # Only exit/flip at upper band (more conservative - stays positioned longer)
            exits = upper_cross

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits
