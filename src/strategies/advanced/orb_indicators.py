"""
Opening Range Breakout (ORB) specific indicators.

Provides calculations for:
- Opening range (high, low, height)
- Relative Volume (RVOL)
- Gap direction detection
- Breakout detection
"""

from datetime import time, datetime
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ORBIndicators:
    """
    Indicators specific to Opening Range Breakout strategy.

    The opening range is defined as the high/low of the first N minutes
    after market open (default: 9:30-9:45 AM ET).
    """

    # Market hours in Eastern Time
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    @staticmethod
    def opening_range(
        df: pd.DataFrame,
        start_time: time = time(9, 30),
        end_time: time = time(9, 45),
    ) -> Dict[str, float]:
        """
        Calculate opening range (high, low, height) from minute data.

        Args:
            df: DataFrame with OHLCV data for a SINGLE DAY.
                Must have 'high', 'low' columns and DatetimeIndex.
            start_time: Opening range start (default 9:30 AM)
            end_time: Opening range end (default 9:45 AM)

        Returns:
            Dict with keys: 'or_high', 'or_low', 'or_height', 'or_midpoint'
            Returns empty dict if insufficient data.
        """
        if df.empty:
            return {}

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex")
            return {}

        # Filter to opening range time window
        idx_time = df.index.time
        or_mask = (idx_time >= start_time) & (idx_time < end_time)
        or_data = df[or_mask]

        if or_data.empty:
            return {}

        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_height = or_high - or_low
        or_midpoint = (or_high + or_low) / 2

        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_height': or_height,
            'or_midpoint': or_midpoint
        }

    @staticmethod
    def opening_range_by_day(
        df: pd.DataFrame,
        start_time: time = time(9, 30),
        end_time: time = time(9, 45),
    ) -> pd.DataFrame:
        """
        Calculate opening range for each day in the DataFrame.

        Args:
            df: DataFrame with OHLCV data spanning multiple days.
                Must have 'high', 'low' columns and DatetimeIndex.
            start_time: Opening range start (default 9:30 AM)
            end_time: Opening range end (default 9:45 AM)

        Returns:
            DataFrame indexed by date with columns:
            'or_high', 'or_low', 'or_height', 'or_midpoint'
        """
        if df.empty:
            return pd.DataFrame()

        # Group by date
        df = df.copy()
        df['date'] = df.index.date

        results = []
        for date, day_data in df.groupby('date'):
            or_data = ORBIndicators.opening_range(day_data, start_time, end_time)
            if or_data:
                or_data['date'] = date
                results.append(or_data)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df.set_index('date', inplace=True)
        return result_df

    @staticmethod
    def relative_volume(
        volume: pd.Series,
        lookback: int = 20,
    ) -> pd.Series:
        """
        Calculate Relative Volume (RVOL).

        RVOL = current volume / average volume over lookback period.

        Args:
            volume: Volume series
            lookback: Number of periods for average (default 20)

        Returns:
            RVOL series (1.0 = average, 2.0 = 2x average)
        """
        avg_volume = volume.rolling(window=lookback, min_periods=1).mean()
        rvol = volume / avg_volume
        return rvol.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    @staticmethod
    def relative_volume_intraday(
        df: pd.DataFrame,
        lookback_days: int = 20,
    ) -> pd.Series:
        """
        Calculate intraday RVOL comparing current cumulative volume
        to the average cumulative volume at the same time of day.

        This is more accurate for ORB as it accounts for volume patterns
        throughout the day (volume typically high at open, lower midday).

        Args:
            df: DataFrame with 'volume' column and DatetimeIndex
            lookback_days: Days to use for average (default 20)

        Returns:
            RVOL series comparing to same-time-of-day average
        """
        if df.empty or 'volume' not in df.columns:
            return pd.Series(dtype=float)

        df = df.copy()
        df['time'] = df.index.time
        df['date'] = df.index.date

        # Calculate cumulative volume from market open for each day
        df['cum_vol'] = df.groupby('date')['volume'].cumsum()

        # Calculate average cumulative volume at each time of day
        time_avg = df.groupby('time')['cum_vol'].transform(
            lambda x: x.rolling(window=lookback_days, min_periods=1).mean()
        )

        rvol = df['cum_vol'] / time_avg
        rvol = rvol.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Restore original index
        return pd.Series(rvol.values, index=df.index)

    @staticmethod
    def gap_direction(
        today_open: float,
        yesterday_close: float,
        threshold_pct: float = 0.001
    ) -> str:
        """
        Determine gap direction between sessions.

        Args:
            today_open: Today's opening price
            yesterday_close: Yesterday's closing price
            threshold_pct: Minimum gap size to classify (default 0.1%)

        Returns:
            'gap_up', 'gap_down', or 'flat'
        """
        if yesterday_close <= 0:
            return 'flat'

        gap_pct = (today_open - yesterday_close) / yesterday_close

        if gap_pct > threshold_pct:
            return 'gap_up'
        elif gap_pct < -threshold_pct:
            return 'gap_down'
        else:
            return 'flat'

    @staticmethod
    def detect_breakout(
        current_close: float,
        or_high: float,
        or_low: float,
        prev_close: float,
    ) -> Optional[str]:
        """
        Detect if current bar represents a breakout from the opening range.

        A breakout requires:
        - Current close above OR high (long) or below OR low (short)
        - Previous close was inside the range

        Args:
            current_close: Current bar's close price
            or_high: Opening range high
            or_low: Opening range low
            prev_close: Previous bar's close price

        Returns:
            'long' for upside breakout, 'short' for downside breakdown, None otherwise
        """
        prev_inside = or_low <= prev_close <= or_high

        # Upside breakout
        if current_close > or_high and prev_inside:
            return 'long'

        # Downside breakdown
        if current_close < or_low and prev_inside:
            return 'short'

        return None

    @staticmethod
    def is_breakout_confirmed(
        current_close: float,
        current_volume: float,
        or_high: float,
        or_low: float,
        vwap: float,
        rvol: float,
        fast_ema: float,
        slow_ema: float,
        rvol_threshold: float = 1.5,
        direction: str = 'long'
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if breakout is confirmed by all filters.

        Args:
            current_close: Current close price
            current_volume: Current bar volume (unused but kept for signature)
            or_high: Opening range high
            or_low: Opening range low
            vwap: Current VWAP
            rvol: Current relative volume
            fast_ema: Fast EMA value
            slow_ema: Slow EMA value
            rvol_threshold: Minimum RVOL required (default 1.5)
            direction: 'long' or 'short'

        Returns:
            Tuple of (is_confirmed, filter_results)
            filter_results is a dict showing which filters passed/failed
        """
        filters = {}

        if direction == 'long':
            # Long entry filters
            filters['price_breakout'] = current_close > or_high
            filters['vwap_above'] = current_close > vwap
            filters['trend_bullish'] = fast_ema > slow_ema
            filters['rvol_sufficient'] = rvol >= rvol_threshold
        else:
            # Short entry filters
            filters['price_breakout'] = current_close < or_low
            filters['vwap_below'] = current_close < vwap
            filters['trend_bearish'] = fast_ema < slow_ema
            filters['rvol_sufficient'] = rvol >= rvol_threshold

        is_confirmed = all(filters.values())
        return is_confirmed, filters

    @staticmethod
    def calculate_targets(
        entry_price: float,
        or_high: float,
        or_low: float,
        direction: str,
        target_multiplier: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate stop-loss and take-profit targets for a position.

        Args:
            entry_price: Entry price
            or_high: Opening range high
            or_low: Opening range low
            direction: 'long' or 'short'
            target_multiplier: Target as multiple of OR height (default 1.0)

        Returns:
            Dict with 'stop_loss', 'target', 'risk', 'reward', 'rr_ratio'
        """
        or_height = or_high - or_low

        if direction == 'long':
            stop_loss = or_low
            target = entry_price + (or_height * target_multiplier)
            risk = entry_price - stop_loss
            reward = target - entry_price
        else:
            stop_loss = or_high
            target = entry_price - (or_height * target_multiplier)
            risk = stop_loss - entry_price
            reward = entry_price - target

        rr_ratio = reward / risk if risk > 0 else 0.0

        return {
            'stop_loss': stop_loss,
            'target': target,
            'risk': risk,
            'reward': reward,
            'rr_ratio': rr_ratio
        }
