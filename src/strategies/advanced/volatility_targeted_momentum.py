"""
Volatility-targeted momentum strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators
from backtesting.utils.validation import validate_positive_int, validate_positive_float


class VolatilityTargetedMomentum(LongOnlyStrategy):
    """
    Time-series momentum with volatility scaling.

    Entry: Price > MA or lookback return > 0
    Position size: Scaled inversely to volatility to target constant volatility

    This strategy adapts position size based on volatility:
    - Low volatility = higher leverage
    - High volatility = lower leverage

    Target is to achieve consistent portfolio volatility regardless of market conditions.
    """

    def __init__(
        self,
        lookback_period: int = 200,
        ma_window: int = 200,
        vol_window: int = 20,
        target_vol: float = 0.15,
        use_ma_filter: bool = True,
        use_return_filter: bool = False,
        combine_filters: str = 'or',
        max_leverage: float = 2.0
    ):
        """
        Initialize strategy.

        Args:
            lookback_period: Lookback period for return-based momentum (default: 200)
            ma_window: Moving average window for MA filter (default: 200)
            vol_window: Rolling window for volatility calculation (default: 20)
            target_vol: Target annualized volatility (default: 0.15 = 15%)
            use_ma_filter: Use MA filter (price > MA) (default: True)
            use_return_filter: Use return filter (lookback return > 0) (default: False)
            combine_filters: How to combine filters if both enabled ('and' or 'or', default: 'or')
            max_leverage: Maximum leverage to prevent excessive position sizing (default: 2.0)
        """
        super().__init__(
            lookback_period=lookback_period,
            ma_window=ma_window,
            vol_window=vol_window,
            target_vol=target_vol,
            use_ma_filter=use_ma_filter,
            use_return_filter=use_return_filter,
            combine_filters=combine_filters,
            max_leverage=max_leverage
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['lookback_period'], 'lookback_period')
        validate_positive_int(self.params['ma_window'], 'ma_window')
        validate_positive_int(self.params['vol_window'], 'vol_window')
        validate_positive_float(self.params['target_vol'], 'target_vol')
        validate_positive_float(self.params['max_leverage'], 'max_leverage')

        if not self.params['use_ma_filter'] and not self.params['use_return_filter']:
            raise ValueError("At least one filter must be enabled (use_ma_filter or use_return_filter)")

        if self.params['combine_filters'] not in ['and', 'or']:
            raise ValueError("combine_filters must be 'and' or 'or'")

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.

        Note: This strategy is designed to be always invested when signals are True.
        Position sizing is handled by volatility targeting.
        """
        close = data['close']

        if self.params['use_ma_filter'] and self.params['use_return_filter']:
            ma = Indicators.ema(close, self.params['ma_window'])
            ma_signal = close > ma

            lookback_return = Indicators.returns(close, self.params['lookback_period'])
            return_signal = lookback_return > 0

            if self.params['combine_filters'] == 'and':
                entries = ma_signal & return_signal
            else:
                entries = ma_signal | return_signal

        elif self.params['use_ma_filter']:
            ma = Indicators.ema(close, self.params['ma_window'])
            entries = close > ma

        elif self.params['use_return_filter']:
            lookback_return = Indicators.returns(close, self.params['lookback_period'])
            entries = lookback_return > 0

        else:
            raise ValueError("No filter enabled")

        exits = ~entries

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

    def calculate_position_size(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position size based on volatility targeting.

        Position size = target_vol / current_vol (capped at max_leverage)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series of position sizes (1.0 = 100% capital, 2.0 = 200% leverage)
        """
        close = data['close']

        current_vol = Indicators.rolling_volatility(
            close,
            window=self.params['vol_window'],
            annualize=True
        )

        position_size = self.params['target_vol'] / current_vol

        position_size = position_size.clip(upper=self.params['max_leverage'])

        position_size = position_size.fillna(1.0)

        return position_size
