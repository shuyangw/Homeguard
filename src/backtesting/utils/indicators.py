"""
Technical indicators for strategy development.
"""

import pandas as pd
import numpy as np
from typing import Union


class Indicators:
    """
    Collection of technical indicators for trading strategies.
    """

    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            prices: Price series
            window: Window size

        Returns:
            SMA series
        """
        return prices.rolling(window=window).mean()

    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            prices: Price series
            window: Window size

        Returns:
            EMA series
        """
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            prices: Price series
            window: Window size (default: 14)

        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> tuple:
        """
        Bollinger Bands.

        Args:
            prices: Price series
            window: Window size (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
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
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
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
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3
    ) -> tuple:
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
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_window).mean()

        return k, d

    @staticmethod
    def donchian_channel(high: pd.Series, low: pd.Series, window: int = 20) -> tuple:
        """
        Donchian Channel (N-period high/low breakout levels).

        Args:
            high: High prices
            low: Low prices
            window: Lookback period (default: 20)

        Returns:
            Tuple of (upper_channel, lower_channel)
        """
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()

        return upper, lower

    @staticmethod
    def returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate percentage returns over specified periods.

        Args:
            prices: Price series
            periods: Number of periods for return calculation (default: 1)

        Returns:
            Returns series as decimal (e.g., 0.05 = 5% gain)
        """
        return prices.pct_change(periods=periods)

    @staticmethod
    def rolling_volatility(prices: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
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
        volatility = returns.rolling(window=window).std()

        if annualize:
            volatility = volatility * np.sqrt(252)

        return volatility

    @staticmethod
    def vwap(prices: pd.Series, volume: pd.Series) -> pd.Series:
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
        typical_price = prices
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def distance_from_vwap(prices: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate percentage distance from VWAP.

        Args:
            prices: Price series
            volume: Volume series

        Returns:
            Distance from VWAP as decimal (e.g., -0.02 = 2% below VWAP)
        """
        vwap_series = Indicators.vwap(prices, volume)
        distance = (prices - vwap_series) / vwap_series

        return distance

    @staticmethod
    def z_score(series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling Z-score.

        Args:
            series: Data series
            window: Rolling window for mean and std calculation (default: 20)

        Returns:
            Z-score series
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        z_score = (series - rolling_mean) / rolling_std

        return z_score
