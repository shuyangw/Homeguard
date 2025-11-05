"""
Overnight mean reversion strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators
from backtesting.utils.validation import validate_positive_int, validate_positive_float


class OvernightMeanReversion(LongOnlyStrategy):
    """
    Short-horizon mean reversion using close/open prices.

    Entry: At close if stock has moved away from mean (VWAP, prior return)
    Exit: At next open (captures overnight mean reversion)

    This strategy exploits mean reversion in overnight returns. Stocks that close
    far from their intraday average often revert to the mean by next open.

    Signals:
    - Long at close if: distance from VWAP < -threshold OR prior return < -threshold
    - Exit at next open
    """

    def __init__(
        self,
        distance_threshold: float = 0.02,
        lookback_return_days: int = 1,
        use_vwap: bool = True,
        use_prior_return: bool = False,
        combine_signals: str = 'or',
        min_volume_ratio: float = 0.5
    ):
        """
        Initialize strategy.

        Args:
            distance_threshold: Distance from VWAP or prior return threshold (default: 0.02 = 2%)
            lookback_return_days: Days for prior return calculation (default: 1)
            use_vwap: Use distance from VWAP as signal (default: True)
            use_prior_return: Use prior day return as signal (default: False)
            combine_signals: How to combine if both enabled ('and' or 'or', default: 'or')
            min_volume_ratio: Minimum volume ratio vs average to trade (default: 0.5 = 50%)
        """
        super().__init__(
            distance_threshold=distance_threshold,
            lookback_return_days=lookback_return_days,
            use_vwap=use_vwap,
            use_prior_return=use_prior_return,
            combine_signals=combine_signals,
            min_volume_ratio=min_volume_ratio
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_float(self.params['distance_threshold'], 'distance_threshold')
        validate_positive_int(self.params['lookback_return_days'], 'lookback_return_days')
        validate_positive_float(self.params['min_volume_ratio'], 'min_volume_ratio')

        if not self.params['use_vwap'] and not self.params['use_prior_return']:
            raise ValueError("At least one signal type must be enabled (use_vwap or use_prior_return)")

        if self.params['combine_signals'] not in ['and', 'or']:
            raise ValueError("combine_signals must be 'and' or 'or'")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.

        Entry: At close if price is below mean reversion threshold
        Exit: At next open (overnight hold)

        Note: This assumes daily data. For intraday data, logic would need modification.
        """
        close = data['close']
        open_price = data['open'] if 'open' in data.columns else close

        signals = []

        if self.params['use_vwap']:
            if 'volume' in data.columns:
                distance = Indicators.distance_from_vwap(close, data['volume'])

                vwap_signal = distance < -self.params['distance_threshold']
                signals.append(vwap_signal)
            else:
                if not self.params['use_prior_return']:
                    raise ValueError("Volume data required for VWAP calculation when use_prior_return=False")

        if self.params['use_prior_return']:
            prior_return = Indicators.returns(close, self.params['lookback_return_days'])

            return_signal = prior_return < -self.params['distance_threshold']
            signals.append(return_signal)

        if len(signals) == 2:
            if self.params['combine_signals'] == 'and':
                entries = signals[0] & signals[1]
            else:
                entries = signals[0] | signals[1]
        else:
            entries = signals[0]

        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(window=20).mean()
            volume_ok = data['volume'] >= (avg_volume * self.params['min_volume_ratio'])
            entries = entries & volume_ok

        exits = entries.shift(1).fillna(False)

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

    def calculate_overnight_return(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate overnight return (close to next open).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series of overnight returns
        """
        close = data['close']
        open_next = data['open'].shift(-1)

        overnight_return = (open_next - close) / close

        return overnight_return
