"""
Moving average based trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.backtesting.base.strategy import LongOnlyStrategy
from src.backtesting.utils.indicators import Indicators
from src.backtesting.utils.validation import validate_positive_int


class MovingAverageCrossover(LongOnlyStrategy):
    """
    Simple moving average crossover strategy.

    Entry: Fast MA crosses above slow MA
    Exit: Fast MA crosses below slow MA
    """

    def __init__(self, fast_window: int = 20, slow_window: int = 50, ma_type: str = 'sma'):
        """
        Initialize strategy.

        Args:
            fast_window: Fast moving average period (default: 20)
            slow_window: Slow moving average period (default: 50)
            ma_type: Type of moving average ('sma' or 'ema', default: 'sma')
        """
        super().__init__(
            fast_window=fast_window,
            slow_window=slow_window,
            ma_type=ma_type
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['fast_window'], 'fast_window')
        validate_positive_int(self.params['slow_window'], 'slow_window')

        if self.params['fast_window'] >= self.params['slow_window']:
            raise ValueError(
                f"fast_window ({self.params['fast_window']}) must be less than "
                f"slow_window ({self.params['slow_window']})"
            )

        if self.params['ma_type'] not in ['sma', 'ema']:
            raise ValueError(f"ma_type must be 'sma' or 'ema', got {self.params['ma_type']}")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Tuple of (long_entries, long_exits)
        """
        close = data['close']

        if self.params['ma_type'] == 'sma':
            fast_ma = Indicators.sma(close, self.params['fast_window'])
            slow_ma = Indicators.sma(close, self.params['slow_window'])
        else:
            fast_ma = Indicators.ema(close, self.params['fast_window'])
            slow_ma = Indicators.ema(close, self.params['slow_window'])

        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits


class TripleMovingAverage(LongOnlyStrategy):
    """
    Triple moving average strategy with trend filter.

    Entry: Fast MA > Medium MA > Slow MA (aligned trend)
    Exit: Trend no longer aligned
    """

    def __init__(
        self,
        fast_window: int = 10,
        medium_window: int = 20,
        slow_window: int = 50,
        ma_type: str = 'ema'
    ):
        """
        Initialize strategy.

        Args:
            fast_window: Fast MA period (default: 10)
            medium_window: Medium MA period (default: 20)
            slow_window: Slow MA period (default: 50)
            ma_type: Type of moving average (default: 'ema')
        """
        super().__init__(
            fast_window=fast_window,
            medium_window=medium_window,
            slow_window=slow_window,
            ma_type=ma_type
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['fast_window'], 'fast_window')
        validate_positive_int(self.params['medium_window'], 'medium_window')
        validate_positive_int(self.params['slow_window'], 'slow_window')

        if not (self.params['fast_window'] < self.params['medium_window'] < self.params['slow_window']):
            raise ValueError(
                "Windows must satisfy: fast_window < medium_window < slow_window"
            )

        if self.params['ma_type'] not in ['sma', 'ema']:
            raise ValueError(f"ma_type must be 'sma' or 'ema'")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.
        """
        close = data['close']

        if self.params['ma_type'] == 'sma':
            fast_ma = Indicators.sma(close, self.params['fast_window'])
            medium_ma = Indicators.sma(close, self.params['medium_window'])
            slow_ma = Indicators.sma(close, self.params['slow_window'])
        else:
            fast_ma = Indicators.ema(close, self.params['fast_window'])
            medium_ma = Indicators.ema(close, self.params['medium_window'])
            slow_ma = Indicators.ema(close, self.params['slow_window'])

        trend_aligned = (fast_ma > medium_ma) & (medium_ma > slow_ma)
        trend_not_aligned = ~trend_aligned

        entries = trend_aligned & ~trend_aligned.shift(1).fillna(False)
        exits = trend_not_aligned & trend_aligned.shift(1).fillna(False)

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits
