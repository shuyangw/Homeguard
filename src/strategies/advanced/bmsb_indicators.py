"""
Bull Market Support Band (BMSB) Indicators.

Implements the Bull Market Support Band indicator popularized in crypto trading.
Uses weekly 20 SMA and 21 EMA to identify bull/bear market conditions.

The band acts as:
- Support during bull markets (price bounces off band)
- Resistance during bear markets (price rejects at band)

Trading Logic:
- Price above band (close > both SMA and EMA) = Bullish, go long
- Price below band (close < both SMA and EMA) = Bearish, exit/go short
- Price inside band = Neutral, hold current position

Original TradingView indicator by zkdev.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BMSBSignal:
    """Bull Market Support Band signal state."""
    timestamp: pd.Timestamp
    close: float
    sma_20w: float
    ema_21w: float
    band_upper: float  # max(sma, ema)
    band_lower: float  # min(sma, ema)
    position: str  # 'above', 'below', 'inside'
    trend: str  # 'bullish', 'bearish', 'neutral'


class BMSBIndicators:
    """
    Bull Market Support Band indicator calculations.

    The BMSB is a weekly-timeframe indicator that identifies macro trend direction
    in crypto markets by using the relationship between price and a band formed
    by 20-week SMA and 21-week EMA.
    """

    @staticmethod
    def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample OHLCV data to weekly timeframe.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (any timeframe)

        Returns:
            Weekly OHLCV DataFrame
        """
        if df.empty:
            return df

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        # Resample to weekly (W-SUN for Sunday close, or W for default Sunday)
        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return weekly

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample OHLCV data to daily timeframe.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (any timeframe)

        Returns:
            Daily OHLCV DataFrame
        """
        if df.empty:
            return df

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        # Resample to daily
        daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return daily

    @staticmethod
    def calculate_band(
        df: pd.DataFrame,
        sma_period: int = 20,
        ema_period: int = 21,
        resample_to_weekly: bool = True,
        timeframe: str = 'weekly'
    ) -> pd.DataFrame:
        """
        Calculate the Bull Market Support Band.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            sma_period: SMA period (default 20)
            ema_period: EMA period (default 21)
            resample_to_weekly: If True, resample input data (deprecated, use timeframe)
            timeframe: 'weekly', 'daily', or 'raw' (no resampling)

        Returns:
            DataFrame with columns: close, sma, ema, band_upper, band_lower, position, trend
        """
        if df.empty:
            return pd.DataFrame()

        # Resample based on timeframe
        if timeframe == 'weekly' or (resample_to_weekly and timeframe != 'daily'):
            working_df = BMSBIndicators.resample_to_weekly(df)
        elif timeframe == 'daily':
            working_df = BMSBIndicators.resample_to_daily(df)
        else:
            working_df = df.copy()

        if len(working_df) < max(sma_period, ema_period):
            logger.warning(
                f"Insufficient data for BMSB: {len(working_df)} bars, "
                f"need at least {max(sma_period, ema_period)}"
            )
            return pd.DataFrame()

        # Calculate moving averages
        sma = BMSBIndicators.calculate_sma(working_df['close'], sma_period)
        ema = BMSBIndicators.calculate_ema(working_df['close'], ema_period)

        # Build result DataFrame
        result = pd.DataFrame(index=working_df.index)
        result['close'] = working_df['close']
        result['sma'] = sma
        result['ema'] = ema
        result['band_upper'] = np.maximum(sma, ema)
        result['band_lower'] = np.minimum(sma, ema)

        # Determine position relative to band
        result['position'] = 'inside'
        result.loc[result['close'] > result['band_upper'], 'position'] = 'above'
        result.loc[result['close'] < result['band_lower'], 'position'] = 'below'

        # Determine trend based on position
        result['trend'] = 'neutral'
        result.loc[result['position'] == 'above', 'trend'] = 'bullish'
        result.loc[result['position'] == 'below', 'trend'] = 'bearish'

        return result.dropna()

    @staticmethod
    def calculate_band_on_original_timeframe(
        df: pd.DataFrame,
        sma_period: int = 20,
        ema_period: int = 21,
        timeframe: str = 'weekly'
    ) -> pd.DataFrame:
        """
        Calculate BMSB and map signals back to original timeframe.

        This calculates the band on the specified timeframe and then forward-fills
        the signals to the original (e.g., minute) data for signal generation.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (any timeframe)
            sma_period: SMA period (default 20)
            ema_period: EMA period (default 21)
            timeframe: 'weekly', 'daily', or 'raw'

        Returns:
            DataFrame aligned to original index with band values and signals
        """
        if df.empty:
            return pd.DataFrame()

        # Get band on specified timeframe
        band = BMSBIndicators.calculate_band(
            df, sma_period, ema_period, resample_to_weekly=False, timeframe=timeframe
        )

        if band.empty:
            return pd.DataFrame()

        # Reindex to original timeframe with forward fill
        # This ensures we use band values at each bar without lookahead
        result = band.reindex(df.index, method='ffill')

        # Update position based on actual current close (not resampled close)
        current_close = df['close']
        result['current_close'] = current_close

        # Recalculate position using current close vs band levels
        result['position'] = 'inside'
        mask_above = current_close > result['band_upper']
        mask_below = current_close < result['band_lower']
        result.loc[mask_above, 'position'] = 'above'
        result.loc[mask_below, 'position'] = 'below'

        result['trend'] = 'neutral'
        result.loc[result['position'] == 'above', 'trend'] = 'bullish'
        result.loc[result['position'] == 'below', 'trend'] = 'bearish'

        return result.dropna()

    @staticmethod
    def generate_signals_vectorized(
        df: pd.DataFrame,
        sma_period: int = 20,
        ema_period: int = 21,
        use_weekly_close: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate long/short entry and exit signals from BMSB.

        Signal Logic:
        - Long Entry: Close crosses above band (position changes to 'above')
        - Long Exit: Close crosses below band (position changes to 'below')
        - Short Entry: Close crosses below band (position changes to 'below')
        - Short Exit: Close crosses above band (position changes to 'above')

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            sma_period: SMA period in weeks
            ema_period: EMA period in weeks
            use_weekly_close: If True, only generate signals on weekly close

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits) as boolean arrays
        """
        n = len(df)

        # Initialize output arrays
        long_entries = np.zeros(n, dtype=bool)
        long_exits = np.zeros(n, dtype=bool)
        short_entries = np.zeros(n, dtype=bool)
        short_exits = np.zeros(n, dtype=bool)

        if df.empty:
            return long_entries, long_exits, short_entries, short_exits

        # Calculate band on original timeframe
        band_df = BMSBIndicators.calculate_band_on_original_timeframe(
            df, sma_period, ema_period
        )

        if band_df.empty or len(band_df) < 2:
            return long_entries, long_exits, short_entries, short_exits

        # Get position series and detect changes
        position = band_df['position'].values

        # Find position changes
        for i in range(1, len(band_df)):
            prev_pos = position[i - 1]
            curr_pos = position[i]

            if prev_pos != curr_pos:
                # Get index in original DataFrame
                idx = df.index.get_loc(band_df.index[i])

                # Transition to bullish (above band)
                if curr_pos == 'above':
                    long_entries[idx] = True
                    short_exits[idx] = True

                # Transition to bearish (below band)
                elif curr_pos == 'below':
                    long_exits[idx] = True
                    short_entries[idx] = True

        return long_entries, long_exits, short_entries, short_exits

    @staticmethod
    def get_current_signal(
        df: pd.DataFrame,
        sma_period: int = 20,
        ema_period: int = 21
    ) -> BMSBSignal:
        """
        Get the current BMSB signal state for live trading.

        Args:
            df: OHLCV DataFrame with recent price history
            sma_period: SMA period in weeks
            ema_period: EMA period in weeks

        Returns:
            BMSBSignal with current state
        """
        band_df = BMSBIndicators.calculate_band_on_original_timeframe(
            df, sma_period, ema_period
        )

        if band_df.empty:
            raise ValueError("Insufficient data to calculate BMSB signal")

        # Get latest values
        latest = band_df.iloc[-1]

        return BMSBSignal(
            timestamp=band_df.index[-1],
            close=latest['current_close'],
            sma_20w=latest['sma'],
            ema_21w=latest['ema'],
            band_upper=latest['band_upper'],
            band_lower=latest['band_lower'],
            position=latest['position'],
            trend=latest['trend']
        )

    # =========================================================================
    # FILTER INDICATORS
    # =========================================================================

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            series: Price series (typically close)
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for volatility-based stops.

        Args:
            df: OHLCV DataFrame
            period: ATR period (default 14)

        Returns:
            ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()
        return atr

    @staticmethod
    def calculate_band_width(band_df: pd.DataFrame) -> pd.Series:
        """
        Calculate band width as percentage of price.

        Wider bands indicate trending markets, narrow bands indicate consolidation.

        Args:
            band_df: DataFrame with 'band_upper', 'band_lower', 'close' columns

        Returns:
            Band width as percentage
        """
        band_width = (band_df['band_upper'] - band_df['band_lower']) / band_df['close']
        return band_width

    @staticmethod
    def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate simple momentum (rate of change).

        Args:
            series: Price series
            period: Lookback period

        Returns:
            Momentum as percentage change
        """
        return series.pct_change(period)

    @staticmethod
    def get_higher_timeframe_trend(
        df: pd.DataFrame,
        sma_period: int = 20,
        ema_period: int = 21
    ) -> pd.Series:
        """
        Get weekly trend for use with daily signals.

        Args:
            df: OHLCV DataFrame (any timeframe)
            sma_period: SMA period for weekly band
            ema_period: EMA period for weekly band

        Returns:
            Series with trend values ('bullish', 'bearish', 'neutral')
            aligned to original timeframe
        """
        weekly_band = BMSBIndicators.calculate_band(
            df, sma_period, ema_period, resample_to_weekly=False, timeframe='weekly'
        )

        if weekly_band.empty:
            return pd.Series(index=df.index, data='neutral')

        # Forward-fill weekly trend to original timeframe
        trend = weekly_band['trend'].reindex(df.index, method='ffill')
        return trend.fillna('neutral')
