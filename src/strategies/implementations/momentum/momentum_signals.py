"""
Pure Momentum Signal Generation.

Generates trading signals based on momentum indicators with no
dependencies on backtesting or live trading infrastructure.

Includes:
- MACD momentum strategy
- Price breakout strategy
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from src.strategies.core import StrategySignals, Signal
from src.utils.logger import logger


class MACDMomentumSignals(StrategySignals):
    """
    Pure MACD momentum signal generation.

    Generates BUY signal when MACD line crosses above signal line.
    Generates SELL signal when MACD line crosses below signal line.

    MACD (Moving Average Convergence Divergence) is a trend-following
    momentum indicator that shows the relationship between two EMAs.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        min_confidence: float = 0.6
    ):
        """
        Initialize MACD momentum strategy.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            min_confidence: Minimum confidence threshold (default: 0.6)
        """
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_confidence = min_confidence

    def get_required_lookback(self) -> int:
        """Return number of periods needed."""
        return self.slow_period + self.signal_period + 10

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate MACD momentum signals for all symbols.

        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
            timestamp: Current timestamp for signal generation

        Returns:
            List of trading signals
        """
        signals = []

        for symbol, df in market_data.items():
            is_valid, error = self.validate_data(df, symbol)
            if not is_valid:
                continue

            signal = self._generate_symbol_signal(symbol, df, timestamp)
            if signal:
                signals.append(signal)

        logger.info(f"Generated {len(signals)} MACD momentum signals at {timestamp}")
        return signals

    def _calculate_macd(
        self,
        close: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator using pandas.

        Args:
            close: Close price series

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # MACD line = fast EMA - slow EMA
        macd_line = fast_ema - slow_ema

        # Signal line = EMA of MACD line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram = MACD line - signal line
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _generate_symbol_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate signal for single symbol."""
        try:
            close = df['close']

            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(close)

            # Current and previous values
            curr_macd = macd_line.iloc[-1]
            curr_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]

            # Check for NaN
            if pd.isna(curr_macd) or pd.isna(curr_signal) or \
               pd.isna(prev_macd) or pd.isna(prev_signal):
                return None

            # Detect crossovers
            bullish_cross = (curr_macd > curr_signal) and (prev_macd <= prev_signal)
            bearish_cross = (curr_macd < curr_signal) and (prev_macd >= prev_signal)

            if not bullish_cross and not bearish_cross:
                return None

            direction = 'BUY' if bullish_cross else 'SELL'

            # Calculate confidence
            confidence = self._calculate_confidence(
                macd_line, signal_line, histogram, direction
            )

            if confidence < self.min_confidence:
                return None

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                price=df['close'].iloc[-1],
                metadata={
                    'strategy': 'MACD_Momentum',
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'signal_period': self.signal_period,
                    'macd': float(curr_macd),
                    'signal': float(curr_signal),
                    'histogram': float(histogram.iloc[-1])
                }
            )

        except Exception as e:
            logger.error(f"Error generating MACD signal for {symbol}: {e}")
            return None

    def _calculate_confidence(
        self,
        macd_line: pd.Series,
        signal_line: pd.Series,
        histogram: pd.Series,
        direction: str
    ) -> float:
        """
        Calculate signal confidence (0-1).

        Factors:
        - Histogram magnitude (40% weight)
        - MACD trend strength (30% weight)
        - Consistency (30% weight)
        """
        # 1. Histogram magnitude
        # Stronger histogram = higher confidence
        hist_magnitude = abs(histogram.iloc[-1])
        avg_hist = histogram.abs().rolling(20).mean().iloc[-1]
        if pd.notna(avg_hist) and avg_hist > 0:
            hist_score = min(hist_magnitude / (avg_hist * 2), 1.0)
        else:
            hist_score = 0.5

        # 2. MACD trend strength
        # Check if MACD has been trending consistently
        lookback = min(5, len(macd_line))
        macd_slope = (macd_line.iloc[-1] - macd_line.iloc[-lookback]) / lookback

        if direction == 'BUY':
            # Want positive MACD slope for bullish signal
            trend_score = min(max(macd_slope / 0.5, 0), 1.0)
        else:
            # Want negative MACD slope for bearish signal
            trend_score = min(max(-macd_slope / 0.5, 0), 1.0)

        # 3. Consistency
        # Check if MACD has been consistently above/below signal recently
        lookback = min(3, len(macd_line))
        if direction == 'BUY':
            consistency = (macd_line.iloc[-lookback:] > signal_line.iloc[-lookback:]).sum() / lookback
        else:
            consistency = (macd_line.iloc[-lookback:] < signal_line.iloc[-lookback:]).sum() / lookback

        confidence = (
            hist_score * 0.40 +
            trend_score * 0.30 +
            consistency * 0.30
        )

        return confidence

    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'min_confidence': self.min_confidence
        }


class BreakoutMomentumSignals(StrategySignals):
    """
    Pure price breakout momentum signal generation.

    Generates BUY when price breaks above N-period high.
    Generates SELL when price breaks below N-period low.

    Optional filters:
    - Volatility filter (only trade within volatility range)
    - Volume confirmation (require volume spike)
    """

    def __init__(
        self,
        breakout_window: int = 20,
        exit_window: int = 10,
        min_confidence: float = 0.65,
        volatility_filter: bool = False,
        volatility_window: int = 20,
        min_volatility: float = 0.01,
        max_volatility: float = 0.10,
        volume_confirmation: bool = False,
        volume_threshold: float = 1.5
    ):
        """
        Initialize breakout momentum strategy.

        Args:
            breakout_window: Period for breakout high/low (default: 20)
            exit_window: Period for exit low/high (default: 10)
            min_confidence: Minimum confidence threshold (default: 0.65)
            volatility_filter: Enable volatility filter (default: False)
            volatility_window: Window for volatility (default: 20)
            min_volatility: Min annualized volatility (default: 0.01)
            max_volatility: Max annualized volatility (default: 0.10)
            volume_confirmation: Require volume spike (default: False)
            volume_threshold: Volume multiple (default: 1.5x)
        """
        if volatility_filter and min_volatility >= max_volatility:
            raise ValueError("min_volatility must be < max_volatility")

        if volume_confirmation and volume_threshold <= 1.0:
            raise ValueError("volume_threshold must be > 1.0")

        self.breakout_window = breakout_window
        self.exit_window = exit_window
        self.min_confidence = min_confidence
        self.volatility_filter = volatility_filter
        self.volatility_window = volatility_window
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold

    def get_required_lookback(self) -> int:
        """Return number of periods needed."""
        return max(self.breakout_window, self.volatility_window) + 10

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """Generate breakout signals for all symbols."""
        signals = []

        for symbol, df in market_data.items():
            is_valid, error = self.validate_data(df, symbol)
            if not is_valid:
                continue

            signal = self._generate_symbol_signal(symbol, df, timestamp)
            if signal:
                signals.append(signal)

        logger.info(f"Generated {len(signals)} breakout signals at {timestamp}")
        return signals

    def _generate_symbol_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate signal for single symbol."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # Calculate breakout levels
            highest = high.rolling(window=self.breakout_window).max()
            lowest = low.rolling(window=self.breakout_window).min()

            # Current price vs breakout levels
            curr_close = close.iloc[-1]
            prev_high = highest.iloc[-2] if len(highest) >= 2 else None
            prev_low = lowest.iloc[-2] if len(lowest) >= 2 else None

            if pd.isna(prev_high) or pd.isna(prev_low):
                return None

            # Detect breakouts
            bullish_breakout = curr_close > prev_high
            bearish_breakout = curr_close < prev_low

            if not bullish_breakout and not bearish_breakout:
                return None

            # Apply filters
            if self.volatility_filter:
                volatility = self._calculate_volatility(close)
                if volatility is None or \
                   volatility < self.min_volatility or \
                   volatility > self.max_volatility:
                    return None

            if self.volume_confirmation and 'volume' in df.columns:
                if not self._check_volume_confirmation(df):
                    return None

            direction = 'BUY' if bullish_breakout else 'SELL'

            # Calculate confidence
            confidence = self._calculate_confidence(
                df, direction, prev_high, prev_low
            )

            if confidence < self.min_confidence:
                return None

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                price=curr_close,
                metadata={
                    'strategy': 'Breakout_Momentum',
                    'breakout_window': self.breakout_window,
                    'breakout_high': float(prev_high),
                    'breakout_low': float(prev_low),
                    'volatility_filter': self.volatility_filter,
                    'volume_confirmation': self.volume_confirmation
                }
            )

        except Exception as e:
            logger.error(f"Error generating breakout signal for {symbol}: {e}")
            return None

    def _calculate_volatility(self, close: pd.Series) -> Optional[float]:
        """Calculate annualized volatility."""
        returns = close.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        annualized_vol = volatility.iloc[-1] * np.sqrt(252)
        return annualized_vol if pd.notna(annualized_vol) else None

    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume spike confirms breakout."""
        volume = df['volume']
        avg_volume = volume.rolling(window=self.breakout_window).mean()

        curr_volume = volume.iloc[-1]
        avg_vol = avg_volume.iloc[-1]

        if pd.isna(curr_volume) or pd.isna(avg_vol):
            return False

        return curr_volume > (avg_vol * self.volume_threshold)

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        direction: str,
        breakout_high: float,
        breakout_low: float
    ) -> float:
        """
        Calculate confidence (0-1).

        Factors:
        - Breakout strength (40% weight)
        - Momentum (30% weight)
        - Range expansion (30% weight)
        """
        close = df['close']
        curr_close = close.iloc[-1]

        # 1. Breakout strength
        # How far beyond the breakout level
        if direction == 'BUY':
            breakout_pct = (curr_close - breakout_high) / breakout_high
        else:
            breakout_pct = (breakout_low - curr_close) / breakout_low

        strength_score = min(breakout_pct / 0.02, 1.0)  # 2% = max

        # 2. Momentum
        # Recent price momentum
        lookback = min(5, len(close))
        price_change = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]

        if direction == 'BUY':
            momentum_score = min(max(price_change / 0.05, 0), 1.0)  # 5% = max
        else:
            momentum_score = min(max(-price_change / 0.05, 0), 1.0)

        # 3. Range expansion
        # Check if recent range is expanding
        atr_recent = self._calculate_atr(df, 5)
        atr_long = self._calculate_atr(df, self.breakout_window)

        if atr_recent and atr_long and atr_long > 0:
            range_expansion = (atr_recent - atr_long) / atr_long
            range_score = min(max(range_expansion, 0), 1.0)
        else:
            range_score = 0.5

        confidence = (
            strength_score * 0.40 +
            momentum_score * 0.30 +
            range_score * 0.30
        )

        return confidence

    def _calculate_atr(self, df: pd.DataFrame, window: int) -> Optional[float]:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean().iloc[-1]

        return atr if pd.notna(atr) else None

    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        return {
            'breakout_window': self.breakout_window,
            'exit_window': self.exit_window,
            'min_confidence': self.min_confidence,
            'volatility_filter': self.volatility_filter,
            'volume_confirmation': self.volume_confirmation
        }


if __name__ == "__main__":
    logger.info("Momentum Signal Generators")
    logger.info("=" * 60)
    logger.info("Pure strategy implementations for momentum trading")
    logger.info("")
    logger.info("MACDMomentumSignals:")
    logger.info("  - MACD line crosses signal line")
    logger.info("  - Default: 12/26/9 EMA")
    logger.info("")
    logger.info("BreakoutMomentumSignals:")
    logger.info("  - Price breaks N-period high/low")
    logger.info("  - Default: 20-period breakout")
    logger.info("  - Optional volatility and volume filters")
    logger.info("")
    logger.info("Both strategies:")
    logger.info("  - Asset agnostic")
    logger.info("  - Return Signal objects")
    logger.info("  - Include confidence scores")
    logger.info("  - No infrastructure dependencies")
