"""
Momentum-based trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.backtesting.base.strategy import LongOnlyStrategy
from src.backtesting.utils.indicators import Indicators
from src.backtesting.utils.validation import validate_positive_int


class MomentumStrategy(LongOnlyStrategy):
    """
    MACD momentum strategy.

    Entry: MACD line crosses above signal line
    Exit: MACD line crosses below signal line
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ):
        """
        Initialize strategy.

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
        """
        super().__init__(fast=fast, slow=slow, signal=signal)

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['fast'], 'fast')
        validate_positive_int(self.params['slow'], 'slow')
        validate_positive_int(self.params['signal'], 'signal')

        if self.params['fast'] >= self.params['slow']:
            raise ValueError("fast must be less than slow")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.
        """
        close = data['close']

        macd_line, signal_line, _ = Indicators.macd(
            close,
            fast=self.params['fast'],
            slow=self.params['slow'],
            signal=self.params['signal']
        )

        entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits


class BreakoutStrategy(LongOnlyStrategy):
    """
    Price breakout momentum strategy with optional filters.

    Entry: Price breaks above N-period high
    Exit: Price breaks below N-period low

    Optional filters:
    - Volatility filter: Only trade when volatility is within range
    - Volume confirmation: Require volume spike on breakout
    - ATR stop loss: Use ATR-based trailing stop
    """

    def __init__(
        self,
        breakout_window: int = 20,
        exit_window: int = 10,
        volatility_filter: bool = False,
        volatility_window: int = 20,
        min_volatility: float = 0.01,
        max_volatility: float = 0.10,
        volume_confirmation: bool = False,
        volume_threshold: float = 1.5,
        use_atr_stop: bool = False,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize strategy.

        Args:
            breakout_window: Period for breakout high (default: 20)
            exit_window: Period for exit low (default: 10)
            volatility_filter: Enable volatility filter to reduce chop (default: False)
            volatility_window: Window for volatility calculation (default: 20)
            min_volatility: Minimum annualized volatility to trade (default: 0.01 = 1%)
            max_volatility: Maximum annualized volatility to trade (default: 0.10 = 10%)
            volume_confirmation: Require volume spike on breakout (default: False)
            volume_threshold: Volume multiple vs average (default: 1.5x)
            use_atr_stop: Use ATR-based trailing stop (default: False)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
        """
        super().__init__(
            breakout_window=breakout_window,
            exit_window=exit_window,
            volatility_filter=volatility_filter,
            volatility_window=volatility_window,
            min_volatility=min_volatility,
            max_volatility=max_volatility,
            volume_confirmation=volume_confirmation,
            volume_threshold=volume_threshold,
            use_atr_stop=use_atr_stop,
            atr_multiplier=atr_multiplier
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['breakout_window'], 'breakout_window')
        validate_positive_int(self.params['exit_window'], 'exit_window')

        if self.params['volatility_filter']:
            validate_positive_int(self.params['volatility_window'], 'volatility_window')
            if self.params['min_volatility'] >= self.params['max_volatility']:
                raise ValueError("min_volatility must be less than max_volatility")

        if self.params['volume_confirmation']:
            if self.params['volume_threshold'] <= 1.0:
                raise ValueError("volume_threshold must be > 1.0")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.
        """
        high = data['high']
        low = data['low']
        close = data['close']

        highest = high.rolling(window=self.params['breakout_window']).max()
        lowest = low.rolling(window=self.params['exit_window']).min()

        entries = close > highest.shift(1)

        if self.params['volatility_filter']:
            volatility = Indicators.rolling_volatility(
                close,
                window=self.params['volatility_window'],
                annualize=True
            )
            vol_ok = (volatility >= self.params['min_volatility']) & (volatility <= self.params['max_volatility'])
            entries = entries & vol_ok

        if self.params['volume_confirmation']:
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=self.params['breakout_window']).mean()
                volume_spike = data['volume'] > (avg_volume * self.params['volume_threshold'])
                entries = entries & volume_spike

        if self.params['use_atr_stop']:
            atr = Indicators.atr(high, low, close, window=14)
            atr_stop = close - (atr * self.params['atr_multiplier'])

            exits = (close < lowest.shift(1)) | (close < atr_stop)
        else:
            exits = close < lowest.shift(1)

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits
