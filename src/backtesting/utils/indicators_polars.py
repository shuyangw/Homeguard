"""
Technical indicators implemented with Polars for improved performance.

This module provides Polars implementations of all indicators from the pandas-based
indicators.py module. These implementations are 3-5x faster for large datasets.

Usage:
    from src.backtesting.utils.indicators_polars import IndicatorsPolars

    # Direct usage with Polars Series
    sma = IndicatorsPolars.sma(pl_series, window=20)

    # Or use the hybrid interface in indicators.py which auto-detects input type
"""

import polars as pl
import numpy as np
from typing import Tuple


class IndicatorsPolars:
    """
    Collection of technical indicators using Polars for high performance.

    All methods accept pl.Series or pl.Expr and return pl.Series.
    """

    @staticmethod
    def sma(prices: pl.Series, window: int) -> pl.Series:
        """
        Simple Moving Average.

        Returns:
            SMA series
        """
        return prices.rolling_mean(window_size=window)

    @staticmethod
    def ema(prices: pl.Series, window: int) -> pl.Series:
        """
        Exponential Moving Average.

        Returns:
            EMA series
        """
        return prices.ewm_mean(span=window, adjust=False)

    @staticmethod
    def rsi(prices: pl.Series, window: int = 14) -> pl.Series:
        """
        Relative Strength Index.

        Args:
            prices: Price series
            window: Window size (default: 14)

        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()

        # Separate gains and losses using clip (native Polars, no map_elements)
        gain = delta.clip(lower_bound=0.0)
        loss = (-delta).clip(lower_bound=0.0)

        avg_gain = gain.rolling_mean(window_size=window)
        avg_loss = loss.rolling_mean(window_size=window)

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(
        prices: pl.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pl.Series, pl.Series, pl.Series]:
        """
        Bollinger Bands.

        Args:
            prices: Price series
            window: Window size (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling_mean(window_size=window)
        std = prices.rolling_std(window_size=window)

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def macd(
        prices: pl.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pl.Series, pl.Series, pl.Series]:
        """
        Moving Average Convergence Divergence.

        Args:
            prices: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm_mean(span=fast, adjust=False)
        ema_slow = prices.ewm_mean(span=slow, adjust=False)

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(span=signal, adjust=False)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def atr(high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14) -> pl.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size (default: 14)

        Returns:
            ATR series
        """
        prev_close = close.shift(1)

        high_low = high - low
        high_close = (high - prev_close).abs()
        low_close = (low - prev_close).abs()

        # Create DataFrame to find max across columns
        df = pl.DataFrame({
            'high_low': high_low,
            'high_close': high_close,
            'low_close': low_close
        })

        true_range = df.select(
            pl.max_horizontal('high_low', 'high_close', 'low_close').alias('tr')
        ).get_column('tr')

        atr = true_range.rolling_mean(window_size=window)

        return atr

    @staticmethod
    def stochastic(
        high: pl.Series,
        low: pl.Series,
        close: pl.Series,
        k_window: int = 14,
        d_window: int = 3
    ) -> Tuple[pl.Series, pl.Series]:
        """
        Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_window: %K period (default: 14)
            d_window: %D period (default: 3)

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling_min(window_size=k_window)
        highest_high = high.rolling_max(window_size=k_window)

        k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling_mean(window_size=d_window)

        return k, d

    @staticmethod
    def donchian_channel(high: pl.Series, low: pl.Series, window: int = 20) -> Tuple[pl.Series, pl.Series]:
        """
        Donchian Channel (N-period high/low breakout levels).

        Args:
            high: High prices
            low: Low prices
            window: Lookback period (default: 20)

        Returns:
            Tuple of (upper_channel, lower_channel)
        """
        upper = high.rolling_max(window_size=window)
        lower = low.rolling_min(window_size=window)

        return upper, lower

    @staticmethod
    def returns(prices: pl.Series, periods: int = 1) -> pl.Series:
        """
        Calculate percentage returns over specified periods.

        Returns:
            Returns series as decimal (e.g., 0.05 = 5% gain)
        """
        return prices.pct_change(n=periods)

    @staticmethod
    def rolling_volatility(prices: pl.Series, window: int = 20, annualize: bool = True) -> pl.Series:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            prices: Price series
            window: Rolling window size (default: 20)
            annualize: If True, annualize volatility assuming 252 trading days (default: True)

        Returns:
            Rolling volatility series
        """
        returns = prices.pct_change()
        volatility = returns.rolling_std(window_size=window)

        if annualize:
            volatility = volatility * np.sqrt(252)

        return volatility

    @staticmethod
    def vwap(prices: pl.Series, volume: pl.Series) -> pl.Series:
        """
        Volume-Weighted Average Price (cumulative for the day).

        Note: For intraday VWAP, this should be reset daily.
        For daily data, this calculates cumulative VWAP.

        Args:
            prices: Price series (typically close or typical price)
            volume: Volume series

        Returns:
            VWAP series
        """
        return (prices * volume).cum_sum() / volume.cum_sum()

    @staticmethod
    def distance_from_vwap(prices: pl.Series, volume: pl.Series) -> pl.Series:
        """
        Calculate percentage distance from VWAP.

        Returns:
            Distance from VWAP as decimal (e.g., -0.02 = 2% below VWAP)
        """
        vwap_series = IndicatorsPolars.vwap(prices, volume)
        distance = (prices - vwap_series) / vwap_series

        return distance

    @staticmethod
    def z_score(series: pl.Series, window: int = 20) -> pl.Series:
        """
        Calculate rolling Z-score.

        Returns:
            Z-score series
        """
        rolling_mean = series.rolling_mean(window_size=window)
        rolling_std = series.rolling_std(window_size=window)

        z_score = (series - rolling_mean) / rolling_std

        return z_score
