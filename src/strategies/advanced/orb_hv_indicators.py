"""
ORB High Volatility strategy indicators.

Provides calculations for:
- Stocks in Play (SIP) score: Opening volume vs historical average
- Pre-market gap detection with validation
- ATR-based volatility filters
- Confidence score aggregation
"""

from datetime import time, datetime, timedelta
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ORBHVIndicators:
    """
    Indicators specific to ORB High Volatility strategy.

    Key differentiator from standard ORB: "Stocks in Play" scoring
    based on abnormal opening volume relative to historical average.
    """

    # Market hours in Eastern Time
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    @staticmethod
    def stocks_in_play_score(
        df: pd.DataFrame,
        lookback_days: int = 14,
        opening_minutes: int = 5
    ) -> pd.Series:
        """
        Calculate "Stocks in Play" (SIP) score.

        SIP Score = (Volume in first N minutes today) /
                    (Avg volume in first N minutes, past K days)

        Interpretation:
        - Score < 1.0: Below average activity, skip
        - Score 1.0 - 2.0: Normal activity, weak candidate
        - Score 2.0 - 5.0: Elevated activity, good candidate
        - Score > 5.0: Extremely in play, best candidates

        Args:
            df: DataFrame with 'volume' column and DatetimeIndex
            lookback_days: Days for historical average (default 14)
            opening_minutes: Minutes to measure opening volume (default 5)

        Returns:
            Series of SIP scores indexed by date
        """
        if df.empty or 'volume' not in df.columns:
            return pd.Series(dtype=float)

        df = df.copy()

        # Extract date and time components
        df['date'] = df.index.date
        df['time'] = df.index.time

        # Define opening range end time
        opening_end = time(9, 30 + opening_minutes)

        # Calculate opening volume for each day
        opening_mask = (df['time'] >= time(9, 30)) & (df['time'] < opening_end)
        opening_volumes = df[opening_mask].groupby('date')['volume'].sum()

        if opening_volumes.empty:
            return pd.Series(dtype=float)

        # Calculate rolling average of opening volumes
        avg_opening_volume = opening_volumes.rolling(
            window=lookback_days,
            min_periods=1
        ).mean().shift(1)  # Shift to avoid lookahead bias

        # Calculate SIP score
        sip_score = opening_volumes / avg_opening_volume
        sip_score = sip_score.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        return sip_score

    @staticmethod
    def stocks_in_play_score_single_day(
        today_opening_volume: float,
        historical_opening_volumes: List[float]
    ) -> float:
        """
        Calculate SIP score for a single day given historical data.

        Args:
            today_opening_volume: Volume in first 5 minutes today
            historical_opening_volumes: List of opening volumes from past days

        Returns:
            SIP score (1.0 = average, 2.0 = 2x average, etc.)
        """
        if not historical_opening_volumes or today_opening_volume <= 0:
            return 1.0

        avg_volume = np.mean(historical_opening_volumes)
        if avg_volume <= 0:
            return 1.0

        return today_opening_volume / avg_volume

    @staticmethod
    def premarket_gap(
        today_open: float,
        yesterday_close: float,
        min_gap_pct: float = 0.02,
        max_gap_pct: float = 0.15
    ) -> Dict[str, any]:
        """
        Calculate pre-market gap and validate against thresholds.

        Args:
            today_open: Today's opening price
            yesterday_close: Yesterday's closing price
            min_gap_pct: Minimum gap percentage (default 2%)
            max_gap_pct: Maximum gap percentage (default 15%)

        Returns:
            Dict with:
            - gap_pct: Gap as decimal (0.05 = 5%)
            - gap_direction: 'up', 'down', or 'flat'
            - is_valid: True if gap within acceptable range
            - abs_gap_pct: Absolute gap percentage
        """
        if yesterday_close <= 0 or today_open <= 0:
            return {
                'gap_pct': 0.0,
                'gap_direction': 'flat',
                'is_valid': False,
                'abs_gap_pct': 0.0
            }

        gap_pct = (today_open - yesterday_close) / yesterday_close
        abs_gap_pct = abs(gap_pct)

        # Determine direction
        if gap_pct > 0.001:  # 0.1% threshold for "up"
            direction = 'up'
        elif gap_pct < -0.001:
            direction = 'down'
        else:
            direction = 'flat'

        # Validate gap is within acceptable range
        is_valid = min_gap_pct <= abs_gap_pct <= max_gap_pct

        return {
            'gap_pct': gap_pct,
            'gap_direction': direction,
            'is_valid': is_valid,
            'abs_gap_pct': abs_gap_pct
        }

    @staticmethod
    def gap_by_day(
        df: pd.DataFrame,
        min_gap_pct: float = 0.02,
        max_gap_pct: float = 0.15
    ) -> pd.DataFrame:
        """
        Calculate gap metrics for each day in the DataFrame.

        Args:
            df: DataFrame with 'open', 'close' columns and DatetimeIndex
            min_gap_pct: Minimum gap percentage
            max_gap_pct: Maximum gap percentage

        Returns:
            DataFrame indexed by date with gap metrics
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = df.index.date

        # Get first bar of each day (open) and last bar of previous day (close)
        daily_open = df.groupby('date')['open'].first()
        daily_close = df.groupby('date')['close'].last()

        # Shift close to align with next day's open
        prev_close = daily_close.shift(1)

        results = []
        for date in daily_open.index:
            if date not in prev_close.index or pd.isna(prev_close[date]):
                continue

            gap_info = ORBHVIndicators.premarket_gap(
                today_open=daily_open[date],
                yesterday_close=prev_close[date],
                min_gap_pct=min_gap_pct,
                max_gap_pct=max_gap_pct
            )
            gap_info['date'] = date
            results.append(gap_info)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df.set_index('date', inplace=True)
        return result_df

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.

        True Range = max(
            High - Low,
            |High - Previous Close|,
            |Low - Previous Close|
        )

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)

        Returns:
            ATR series (EMA-smoothed)
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Use EMA for responsiveness
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def atr_percent(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate ATR as percentage of price.

        ATR% = ATR / Close * 100

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)

        Returns:
            ATR percentage series
        """
        atr_value = ORBHVIndicators.atr(high, low, close, period)
        return (atr_value / close) * 100

    @staticmethod
    def volatility_filter(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        min_atr_pct: float = 2.0,
        max_atr_pct: float = 10.0
    ) -> pd.Series:
        """
        Filter for acceptable volatility range.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)
            min_atr_pct: Minimum ATR% (default 2.0%)
            max_atr_pct: Maximum ATR% (default 10.0%)

        Returns:
            Boolean series (True = within acceptable volatility)
        """
        atr_pct = ORBHVIndicators.atr_percent(high, low, close, period)
        return (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)

    @staticmethod
    def opening_range(
        df: pd.DataFrame,
        start_time: time = time(9, 30),
        end_time: time = time(9, 35)
    ) -> Dict[str, float]:
        """
        Calculate opening range (high, low, height) from minute data.

        Args:
            df: DataFrame with OHLCV data for a SINGLE DAY
            start_time: Opening range start (default 9:30 AM)
            end_time: Opening range end (default 9:35 AM for 5-min OR)

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
        or_volume = or_data['volume'].sum() if 'volume' in or_data.columns else 0.0

        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_height': or_height,
            'or_midpoint': or_midpoint,
            'or_volume': or_volume
        }

    @staticmethod
    def confidence_score(
        sip_score: float,
        sentiment_score: float = 0.0,
        gap_pct: float = 0.0,
        volume_ratio: float = 1.0,
        direction: str = 'long'
    ) -> float:
        """
        Calculate composite confidence score for position sizing.

        Components:
        - SIP score contribution: 40% weight
        - Sentiment alignment: 20% weight
        - Gap size: 25% weight
        - Volume confirmation: 15% weight

        Args:
            sip_score: Stocks in Play score (1.0 = average)
            sentiment_score: Sentiment score (-1 to 1)
            gap_pct: Gap percentage (as decimal, e.g., 0.05 for 5%)
            volume_ratio: Current volume vs average
            direction: Trade direction ('long' or 'short')

        Returns:
            Confidence score from 0.0 to 1.0
        """
        confidence = 0.0

        # SIP score contribution (40%)
        # Map SIP score: 2.0 -> 0.5, 5.0 -> 0.8, 10.0+ -> 1.0
        sip_contribution = min((sip_score - 1.0) / 9.0, 1.0) * 0.40
        confidence += max(sip_contribution, 0.0)

        # Sentiment alignment contribution (20%)
        if direction == 'long':
            # Positive sentiment helps longs
            if sentiment_score > 0.3:
                confidence += 0.20
            elif sentiment_score > 0.1:
                confidence += 0.10
            elif sentiment_score > -0.1:
                confidence += 0.05  # Neutral acceptable
        else:
            # Negative sentiment helps shorts
            if sentiment_score < -0.3:
                confidence += 0.20
            elif sentiment_score < -0.1:
                confidence += 0.10
            elif sentiment_score < 0.1:
                confidence += 0.05

        # Gap size contribution (25%)
        # Larger gaps (in right direction) increase confidence
        abs_gap = abs(gap_pct)
        if direction == 'long' and gap_pct > 0:
            gap_contribution = min(abs_gap / 0.10, 1.0) * 0.25
            confidence += gap_contribution
        elif direction == 'short' and gap_pct < 0:
            gap_contribution = min(abs_gap / 0.10, 1.0) * 0.25
            confidence += gap_contribution

        # Volume confirmation (15%)
        # Higher relative volume increases confidence
        vol_contribution = min((volume_ratio - 1.0) / 2.0, 1.0) * 0.15
        confidence += max(vol_contribution, 0.0)

        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def calculate_tiered_targets(
        entry_price: float,
        or_high: float,
        or_low: float,
        direction: str,
        target1_multiplier: float = 1.0,
        target2_multiplier: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate stop-loss and tiered take-profit targets.

        Args:
            entry_price: Entry price
            or_high: Opening range high
            or_low: Opening range low
            direction: 'long' or 'short'
            target1_multiplier: First target as multiple of OR height (default 1.0)
            target2_multiplier: Second target as multiple of OR height (default 2.0)

        Returns:
            Dict with 'stop_loss', 'target1', 'target2', 'risk', 'reward1', 'reward2'
        """
        or_height = or_high - or_low

        if direction == 'long':
            stop_loss = or_low
            target1 = entry_price + (or_height * target1_multiplier)
            target2 = entry_price + (or_height * target2_multiplier)
            risk = entry_price - stop_loss
            reward1 = target1 - entry_price
            reward2 = target2 - entry_price
        else:
            stop_loss = or_high
            target1 = entry_price - (or_height * target1_multiplier)
            target2 = entry_price - (or_height * target2_multiplier)
            risk = stop_loss - entry_price
            reward1 = entry_price - target1
            reward2 = entry_price - target2

        return {
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'risk': risk,
            'reward1': reward1,
            'reward2': reward2,
            'rr_ratio1': reward1 / risk if risk > 0 else 0.0,
            'rr_ratio2': reward2 / risk if risk > 0 else 0.0
        }

    @staticmethod
    def volatility_regime(
        spy_df: pd.DataFrame,
        lookback_bars: int = 12
    ) -> str:
        """
        Classify current volatility regime using SPY as proxy.

        Args:
            spy_df: SPY minute bars with 'close' column
            lookback_bars: Number of bars for volatility calculation (default 12)

        Returns:
            'low', 'normal', 'high', or 'extreme'
        """
        if spy_df.empty or len(spy_df) < lookback_bars:
            return 'normal'

        # Calculate recent returns volatility
        returns = spy_df['close'].pct_change().tail(lookback_bars)
        recent_vol = returns.std() * np.sqrt(252 * 78)  # Annualized

        # Baseline volatility (~16% annualized for SPY)
        baseline_vol = 0.16
        ratio = recent_vol / baseline_vol

        if ratio < 0.6:
            return 'low'
        elif ratio < 1.4:
            return 'normal'
        elif ratio < 2.5:
            return 'high'
        else:
            return 'extreme'

    @staticmethod
    def position_size_adjustment(
        confidence: float,
        volatility_regime: str
    ) -> float:
        """
        Calculate position size adjustment multiplier.

        Args:
            confidence: Confidence score (0.0 to 1.0)
            volatility_regime: 'low', 'normal', 'high', or 'extreme'

        Returns:
            Multiplier for base position size (0.2 to 1.2)
        """
        # Confidence adjustment (0.5 to 1.0)
        confidence_mult = 0.5 + (confidence * 0.5)

        # Volatility regime adjustment
        vol_mult = {
            'low': 1.2,      # Can size up slightly in calm markets
            'normal': 1.0,
            'high': 0.7,     # Reduce size in volatile markets
            'extreme': 0.4   # Significantly reduce in extreme vol
        }.get(volatility_regime, 1.0)

        return confidence_mult * vol_mult
