"""
Technical indicators for strategy development.

This module provides a hybrid interface that supports both pandas and polars inputs.
Input type is auto-detected and the appropriate implementation is used.

Usage:
    # Pandas input (original behavior)
    sma = Indicators.sma(pd_series, window=20)

    # Polars input (automatically routes to IndicatorsPolars)
    sma = Indicators.sma(pl_series, window=20)
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple

try:
    import polars as pl
    from .indicators_polars import IndicatorsPolars
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


def _is_polars(obj) -> bool:
    """Check if object is a Polars Series."""
    return POLARS_AVAILABLE and isinstance(obj, pl.Series)


class Indicators:
    """
    Collection of technical indicators for trading strategies.
    """

    @staticmethod
    def sma(prices: Union[pd.Series, "pl.Series"], window: int) -> Union[pd.Series, "pl.Series"]:
        """
        Simple Moving Average.

        Args:
            prices: Price series (pandas or polars)
            window: Window size

        Returns:
            SMA series (same type as input)
        """
        if _is_polars(prices):
            return IndicatorsPolars.sma(prices, window)
        return prices.rolling(window=window).mean()

    @staticmethod
    def ema(prices: Union[pd.Series, "pl.Series"], window: int) -> Union[pd.Series, "pl.Series"]:
        """
        Exponential Moving Average.

        Args:
            prices: Price series (pandas or polars)
            window: Window size

        Returns:
            EMA series (same type as input)
        """
        if _is_polars(prices):
            return IndicatorsPolars.ema(prices, window)
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(prices: Union[pd.Series, "pl.Series"], window: int = 14) -> Union[pd.Series, "pl.Series"]:
        """
        Relative Strength Index.

        Args:
            prices: Price series (pandas or polars)
            window: Window size (default: 14)

        Returns:
            RSI series (0-100, same type as input)
        """
        if _is_polars(prices):
            return IndicatorsPolars.rsi(prices, window)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(
        prices: Union[pd.Series, "pl.Series"],
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple:
        """
        Bollinger Bands.

        Args:
            prices: Price series (pandas or polars)
            window: Window size (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if _is_polars(prices):
            return IndicatorsPolars.bollinger_bands(prices, window, num_std)
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def macd(
        prices: Union[pd.Series, "pl.Series"],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple:
        """
        Moving Average Convergence Divergence.

        Args:
            prices: Price series (pandas or polars)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if _is_polars(prices):
            return IndicatorsPolars.macd(prices, fast, slow, signal)
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def atr(
        high: Union[pd.Series, "pl.Series"],
        low: Union[pd.Series, "pl.Series"],
        close: Union[pd.Series, "pl.Series"],
        window: int = 14
    ) -> Union[pd.Series, "pl.Series"]:
        """
        Average True Range.

        Args:
            high: High prices (pandas or polars)
            low: Low prices (pandas or polars)
            close: Close prices (pandas or polars)
            window: Window size (default: 14)

        Returns:
            ATR series (same type as input)
        """
        if _is_polars(high):
            return IndicatorsPolars.atr(high, low, close, window)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    @staticmethod
    def stochastic(
        high: Union[pd.Series, "pl.Series"],
        low: Union[pd.Series, "pl.Series"],
        close: Union[pd.Series, "pl.Series"],
        k_window: int = 14,
        d_window: int = 3
    ) -> Tuple:
        """
        Stochastic Oscillator.

        Args:
            high: High prices (pandas or polars)
            low: Low prices (pandas or polars)
            close: Close prices (pandas or polars)
            k_window: %K period (default: 14)
            d_window: %D period (default: 3)

        Returns:
            Tuple of (%K, %D)
        """
        if _is_polars(high):
            return IndicatorsPolars.stochastic(high, low, close, k_window, d_window)
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_window).mean()

        return k, d

    @staticmethod
    def donchian_channel(
        high: Union[pd.Series, "pl.Series"],
        low: Union[pd.Series, "pl.Series"],
        window: int = 20
    ) -> Tuple:
        """
        Donchian Channel (N-period high/low breakout levels).

        Args:
            high: High prices (pandas or polars)
            low: Low prices (pandas or polars)
            window: Lookback period (default: 20)

        Returns:
            Tuple of (upper_channel, lower_channel)
        """
        if _is_polars(high):
            return IndicatorsPolars.donchian_channel(high, low, window)
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()

        return upper, lower

    @staticmethod
    def returns(prices: Union[pd.Series, "pl.Series"], periods: int = 1) -> Union[pd.Series, "pl.Series"]:
        """
        Calculate percentage returns over specified periods.

        Args:
            prices: Price series (pandas or polars)
            periods: Number of periods for return calculation (default: 1)

        Returns:
            Returns series as decimal (e.g., 0.05 = 5% gain)
        """
        if _is_polars(prices):
            return IndicatorsPolars.returns(prices, periods)
        return prices.pct_change(periods=periods)

    @staticmethod
    def rolling_volatility(
        prices: Union[pd.Series, "pl.Series"],
        window: int = 20,
        annualize: bool = True
    ) -> Union[pd.Series, "pl.Series"]:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            prices: Price series (pandas or polars)
            window: Rolling window size (default: 20)
            annualize: If True, annualize volatility assuming 252 trading days (default: True)

        Returns:
            Rolling volatility series (same type as input)
        """
        if _is_polars(prices):
            return IndicatorsPolars.rolling_volatility(prices, window, annualize)
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std()

        if annualize:
            volatility = volatility * np.sqrt(252)

        return volatility

    @staticmethod
    def vwap(
        prices: Union[pd.Series, "pl.Series"],
        volume: Union[pd.Series, "pl.Series"]
    ) -> Union[pd.Series, "pl.Series"]:
        """
        Volume-Weighted Average Price (cumulative for the day).

        Note: For intraday VWAP, this should be reset daily.
        For daily data, this calculates cumulative VWAP.

        Args:
            prices: Price series (typically close or typical price, pandas or polars)
            volume: Volume series (pandas or polars)

        Returns:
            VWAP series (same type as input)
        """
        if _is_polars(prices):
            return IndicatorsPolars.vwap(prices, volume)
        typical_price = prices
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def distance_from_vwap(
        prices: Union[pd.Series, "pl.Series"],
        volume: Union[pd.Series, "pl.Series"]
    ) -> Union[pd.Series, "pl.Series"]:
        """
        Calculate percentage distance from VWAP.

        Args:
            prices: Price series (pandas or polars)
            volume: Volume series (pandas or polars)

        Returns:
            Distance from VWAP as decimal (e.g., -0.02 = 2% below VWAP)
        """
        if _is_polars(prices):
            return IndicatorsPolars.distance_from_vwap(prices, volume)
        vwap_series = Indicators.vwap(prices, volume)
        distance = (prices - vwap_series) / vwap_series

        return distance

    @staticmethod
    def z_score(series: Union[pd.Series, "pl.Series"], window: int = 20) -> Union[pd.Series, "pl.Series"]:
        """
        Calculate rolling Z-score.

        Args:
            series: Data series (pandas or polars)
            window: Rolling window for mean and std calculation (default: 20)

        Returns:
            Z-score series (same type as input)
        """
        if _is_polars(series):
            return IndicatorsPolars.z_score(series, window)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        z_score = (series - rolling_mean) / rolling_std

        return z_score
