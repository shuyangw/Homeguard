"""
Technical Indicators for Stock Screener.

Provides calculation of common technical indicators from OHLCV data:
- RSI (Relative Strength Index)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Historical Volatility

All calculations are vectorized using pandas/numpy for performance.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""

    rsi_14: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None
    historical_volatility_20d: Optional[float] = None


class TechnicalIndicators:
    """
    Calculator for technical indicators from OHLCV DataFrame.

    Usage:
        indicators = TechnicalIndicators(ohlcv_df)

        rsi = indicators.rsi(14)
        sma_50, sma_200 = indicators.sma(50), indicators.sma(200)
        is_golden_cross = sma_50 > sma_200

        # Get all indicators at once
        result = indicators.compute_all()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime
        """
        self.df = df.copy()

        # Normalize column names to lowercase
        self.df.columns = [c.lower() for c in self.df.columns]

        # Validate required columns
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort by index (date) ascending
        self.df.sort_index(inplace=True)

    # -------------------------------------------------------------------------
    # RSI (Relative Strength Index)
    # -------------------------------------------------------------------------

    def rsi(self, period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            period: Lookback period (default: 14)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(self.df) < period + 1:
            return None

        close = self.df["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # Use Wilder's smoothing (exponential moving average)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi_series = 100 - (100 / (1 + rs))

        # Return the most recent value
        return float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None

    def rsi_series(self, period: int = 14) -> pd.Series:
        """
        Calculate RSI series for all periods.

        Args:
            period: Lookback period (default: 14)

        Returns:
            Series of RSI values
        """
        close = self.df["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    # -------------------------------------------------------------------------
    # Moving Averages
    # -------------------------------------------------------------------------

    def sma(self, period: int) -> Optional[float]:
        """
        Calculate Simple Moving Average.

        Args:
            period: Number of periods

        Returns:
            SMA value or None if insufficient data
        """
        if len(self.df) < period:
            return None

        sma_series = self.df["close"].rolling(window=period).mean()
        return float(sma_series.iloc[-1]) if not pd.isna(sma_series.iloc[-1]) else None

    def sma_series(self, period: int) -> pd.Series:
        """Calculate SMA series for all periods."""
        return self.df["close"].rolling(window=period).mean()

    def ema(self, period: int) -> Optional[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            period: Number of periods

        Returns:
            EMA value or None if insufficient data
        """
        if len(self.df) < period:
            return None

        ema_series = self.df["close"].ewm(span=period, adjust=False).mean()
        return float(ema_series.iloc[-1]) if not pd.isna(ema_series.iloc[-1]) else None

    def ema_series(self, period: int) -> pd.Series:
        """Calculate EMA series for all periods."""
        return self.df["close"].ewm(span=period, adjust=False).mean()

    def is_sma_crossover(
        self, fast_period: int, slow_period: int, direction: str = "above"
    ) -> bool:
        """
        Check if fast SMA is above/below slow SMA.

        Args:
            fast_period: Fast SMA period (e.g., 50)
            slow_period: Slow SMA period (e.g., 200)
            direction: 'above' or 'below'

        Returns:
            True if crossover condition is met
        """
        fast_sma = self.sma(fast_period)
        slow_sma = self.sma(slow_period)

        if fast_sma is None or slow_sma is None:
            return False

        if direction.lower() == "above":
            return fast_sma > slow_sma
        else:
            return fast_sma < slow_sma

    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------

    def macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(self.df) < slow + signal:
            return None, None, None

        close = self.df["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    def macd_signal_type(self) -> Optional[str]:
        """
        Determine MACD signal type.

        Returns:
            'bullish' if MACD > signal, 'bearish' if MACD < signal, None if insufficient data
        """
        macd_line, signal_line, _ = self.macd()
        if macd_line is None or signal_line is None:
            return None

        if macd_line > signal_line:
            return "bullish"
        else:
            return "bearish"

    # -------------------------------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------------------------------

    def bollinger_bands(
        self, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate Bollinger Bands.

        Args:
            period: SMA period (default: 20)
            std_dev: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(self.df) < period:
            return None, None, None

        close = self.df["close"]
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return (
            float(upper.iloc[-1]),
            float(middle.iloc[-1]),
            float(lower.iloc[-1]),
        )

    def bollinger_position(self, period: int = 20, std_dev: float = 2.0) -> Optional[str]:
        """
        Determine price position relative to Bollinger Bands.

        Returns:
            'above_upper', 'below_lower', or 'middle', None if insufficient data
        """
        upper, middle, lower = self.bollinger_bands(period, std_dev)
        if upper is None:
            return None

        current_price = float(self.df["close"].iloc[-1])

        if current_price > upper:
            return "above_upper"
        elif current_price < lower:
            return "below_lower"
        else:
            return "middle"

    # -------------------------------------------------------------------------
    # ATR (Average True Range)
    # -------------------------------------------------------------------------

    def atr(self, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range.

        Args:
            period: ATR period (default: 14)

        Returns:
            ATR value or None if insufficient data
        """
        if len(self.df) < period + 1:
            return None

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # True Range = max(H-L, |H-Pc|, |L-Pc|)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the EMA of True Range
        atr_series = true_range.ewm(span=period, adjust=False).mean()

        return float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

    def atr_series(self, period: int = 14) -> pd.Series:
        """Calculate ATR series for all periods."""
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()

    def atr_percent(self, period: int = 14) -> Optional[float]:
        """
        Calculate ATR as percentage of current price.

        Returns:
            ATR% or None if insufficient data
        """
        atr_value = self.atr(period)
        if atr_value is None:
            return None

        current_price = float(self.df["close"].iloc[-1])
        if current_price == 0:
            return None

        return (atr_value / current_price) * 100

    # -------------------------------------------------------------------------
    # Volatility
    # -------------------------------------------------------------------------

    def historical_volatility(self, period: int = 20) -> Optional[float]:
        """
        Calculate historical volatility (annualized standard deviation of returns).

        Args:
            period: Lookback period (default: 20)

        Returns:
            Annualized volatility as percentage, or None if insufficient data
        """
        if len(self.df) < period + 1:
            return None

        close = self.df["close"]
        returns = close.pct_change().dropna()

        if len(returns) < period:
            return None

        # Rolling standard deviation of returns
        std = returns.rolling(window=period).std()

        # Annualize (assuming daily data, 252 trading days)
        annualized_vol = std.iloc[-1] * np.sqrt(252) * 100

        return float(annualized_vol) if not pd.isna(annualized_vol) else None

    # -------------------------------------------------------------------------
    # Volume Metrics
    # -------------------------------------------------------------------------

    def avg_volume(self, period: int = 20) -> Optional[float]:
        """
        Calculate average volume.

        Args:
            period: Lookback period (default: 20)

        Returns:
            Average volume or None if insufficient data
        """
        if "volume" not in self.df.columns:
            return None
        if len(self.df) < period:
            return None

        avg = self.df["volume"].rolling(window=period).mean()
        return float(avg.iloc[-1]) if not pd.isna(avg.iloc[-1]) else None

    def relative_volume(self, period: int = 20) -> Optional[float]:
        """
        Calculate relative volume (current volume / average volume).

        Args:
            period: Lookback period for average (default: 20)

        Returns:
            Relative volume ratio or None if insufficient data
        """
        if "volume" not in self.df.columns:
            return None

        avg_vol = self.avg_volume(period)
        if avg_vol is None or avg_vol == 0:
            return None

        current_volume = float(self.df["volume"].iloc[-1])
        return current_volume / avg_vol

    # -------------------------------------------------------------------------
    # Compute All
    # -------------------------------------------------------------------------

    def compute_all(self) -> IndicatorResult:
        """
        Compute all indicators at once.

        Returns:
            IndicatorResult with all computed values
        """
        macd_line, macd_signal, macd_histogram = self.macd()
        bb_upper, bb_middle, bb_lower = self.bollinger_bands()

        return IndicatorResult(
            rsi_14=self.rsi(14),
            sma_20=self.sma(20),
            sma_50=self.sma(50),
            sma_200=self.sma(200),
            ema_12=self.ema(12),
            ema_26=self.ema(26),
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bollinger_upper=bb_upper,
            bollinger_middle=bb_middle,
            bollinger_lower=bb_lower,
            atr_14=self.atr(14),
            atr_pct=self.atr_percent(14),
            historical_volatility_20d=self.historical_volatility(20),
        )


def compute_indicators_for_screening(
    df: pd.DataFrame,
) -> IndicatorResult:
    """
    Convenience function to compute all indicators for a symbol.

    Args:
        df: OHLCV DataFrame

    Returns:
        IndicatorResult with all computed values
    """
    try:
        indicators = TechnicalIndicators(df)
        return indicators.compute_all()
    except Exception:
        return IndicatorResult()
