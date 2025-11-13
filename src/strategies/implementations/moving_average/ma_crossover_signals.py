"""
Pure MA Crossover Signal Generation.

Generates trading signals based on moving average crossovers with no
dependencies on backtesting or live trading infrastructure.

This is a pure strategy implementation that can be used by both:
- Backtest adapters (via src/backtesting/adapters)
- Live trading adapters (via src/trading/adapters)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from src.strategies.core import StrategySignals, Signal
from src.utils.logger import logger


class MACrossoverSignals(StrategySignals):
    """
    Pure moving average crossover signal generation.

    Generates BUY signal when fast MA crosses above slow MA.
    Generates SELL signal when fast MA crosses below slow MA.

    This is a pure strategy that only depends on pandas DataFrames.
    It can be used with any universe (ETFs, stocks, options, etc.).
    """

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        ma_type: str = 'sma',
        min_confidence: float = 0.7
    ):
        """
        Initialize MA crossover strategy.

        Args:
            fast_period: Fast moving average period (default: 50)
            slow_period: Slow moving average period (default: 200)
            ma_type: Type of MA ('sma' or 'ema', default: 'sma')
            min_confidence: Minimum confidence threshold for signals (default: 0.7)
        """
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        if ma_type not in ['sma', 'ema']:
            raise ValueError(f"ma_type must be 'sma' or 'ema', got {ma_type}")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.min_confidence = min_confidence

    def get_required_lookback(self) -> int:
        """Return number of periods needed for calculation."""
        return self.slow_period + 10  # Extra buffer for crossover detection

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate MA crossover signals for all symbols.

        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
            timestamp: Current timestamp for signal generation

        Returns:
            List of trading signals
        """
        signals = []

        for symbol, df in market_data.items():
            # Validate data
            is_valid, error = self.validate_data(df, symbol)
            if not is_valid:
                # Skip invalid data silently (validation handles logging)
                continue

            # Generate signal for this symbol
            signal = self._generate_symbol_signal(symbol, df, timestamp)
            if signal:
                signals.append(signal)

        logger.info(f"Generated {len(signals)} MA crossover signals at {timestamp}")
        return signals

    def _generate_symbol_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Generate signal for a single symbol.

        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal or None if no signal
        """
        try:
            # Calculate moving averages
            close = df['close']

            if self.ma_type == 'sma':
                fast_ma = close.rolling(window=self.fast_period).mean()
                slow_ma = close.rolling(window=self.slow_period).mean()
            else:  # ema
                fast_ma = close.ewm(span=self.fast_period, adjust=False).mean()
                slow_ma = close.ewm(span=self.slow_period, adjust=False).mean()

            # Get current and previous values
            current_fast = fast_ma.iloc[-1]
            current_slow = slow_ma.iloc[-1]
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]

            # Check for NaN (insufficient data)
            if pd.isna(current_fast) or pd.isna(current_slow) or \
               pd.isna(prev_fast) or pd.isna(prev_slow):
                return None

            # Detect crossovers
            golden_cross = (current_fast > current_slow) and (prev_fast <= prev_slow)
            death_cross = (current_fast < current_slow) and (prev_fast >= prev_slow)

            # No crossover detected
            if not golden_cross and not death_cross:
                return None

            # Determine direction
            direction = 'BUY' if golden_cross else 'SELL'

            # Calculate confidence based on:
            # 1. Separation between MAs (more separation = higher confidence)
            # 2. Trend strength (slope of MAs)
            confidence = self._calculate_confidence(
                fast_ma, slow_ma, direction
            )

            # Filter by minimum confidence
            if confidence < self.min_confidence:
                # Signal filtered out by confidence threshold
                return None

            # Get current price
            current_price = df['close'].iloc[-1]

            # Create signal
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                price=current_price,
                metadata={
                    'strategy': 'MA_Crossover',
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'ma_type': self.ma_type,
                    'fast_ma': float(current_fast),
                    'slow_ma': float(current_slow),
                    'crossover_type': 'golden' if golden_cross else 'death'
                }
            )

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _calculate_confidence(
        self,
        fast_ma: pd.Series,
        slow_ma: pd.Series,
        direction: str
    ) -> float:
        """
        Calculate signal confidence score (0-1).

        Factors:
        - MA separation (40% weight)
        - Trend strength (30% weight)
        - Consistency (30% weight)

        Args:
            fast_ma: Fast MA series
            slow_ma: Slow MA series
            direction: Signal direction ('BUY' or 'SELL')

        Returns:
            Confidence score [0, 1]
        """
        # 1. MA Separation (0-1)
        # Normalize by slow MA value
        separation = abs(fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]
        separation_score = min(separation / 0.02, 1.0)  # 2% separation = max score

        # 2. Trend Strength (0-1)
        # Calculate slope of fast MA over last 5 periods
        lookback = min(5, len(fast_ma))
        fast_slope = (fast_ma.iloc[-1] - fast_ma.iloc[-lookback]) / fast_ma.iloc[-lookback]

        # For BUY signals, want positive slope; for SELL, want negative slope
        if direction == 'BUY':
            trend_score = min(max(fast_slope / 0.01, 0), 1.0)  # 1% rise = max score
        else:  # SELL
            trend_score = min(max(-fast_slope / 0.01, 0), 1.0)  # 1% fall = max score

        # 3. Consistency (0-1)
        # Check if fast MA has been consistently above/below slow MA recently
        lookback = min(3, len(fast_ma))
        if direction == 'BUY':
            consistency = (fast_ma.iloc[-lookback:] > slow_ma.iloc[-lookback:]).sum() / lookback
        else:
            consistency = (fast_ma.iloc[-lookback:] < slow_ma.iloc[-lookback:]).sum() / lookback

        # Weighted combination
        confidence = (
            separation_score * 0.40 +
            trend_score * 0.30 +
            consistency * 0.30
        )

        return confidence

    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'ma_type': self.ma_type,
            'min_confidence': self.min_confidence
        }


class TripleMACrossoverSignals(StrategySignals):
    """
    Pure triple moving average signal generation.

    Generates BUY when: Fast > Medium > Slow (aligned uptrend)
    Generates SELL when: Fast < Medium < Slow (aligned downtrend)

    More conservative than dual MA crossover - requires all three MAs aligned.
    """

    def __init__(
        self,
        fast_period: int = 10,
        medium_period: int = 20,
        slow_period: int = 50,
        ma_type: str = 'ema',
        min_confidence: float = 0.75
    ):
        """
        Initialize triple MA strategy.

        Args:
            fast_period: Fast MA period (default: 10)
            medium_period: Medium MA period (default: 20)
            slow_period: Slow MA period (default: 50)
            ma_type: Type of MA (default: 'ema')
            min_confidence: Minimum confidence threshold (default: 0.75)
        """
        if not (fast_period < medium_period < slow_period):
            raise ValueError(
                "Periods must satisfy: fast < medium < slow"
            )

        if ma_type not in ['sma', 'ema']:
            raise ValueError(f"ma_type must be 'sma' or 'ema', got {ma_type}")

        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.min_confidence = min_confidence

    def get_required_lookback(self) -> int:
        """Return number of periods needed."""
        return self.slow_period + 10

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """Generate triple MA signals for all symbols."""
        signals = []

        for symbol, df in market_data.items():
            is_valid, error = self.validate_data(df, symbol)
            if not is_valid:
                # Skip invalid data silently
                continue

            signal = self._generate_symbol_signal(symbol, df, timestamp)
            if signal:
                signals.append(signal)

        logger.info(f"Generated {len(signals)} triple MA signals at {timestamp}")
        return signals

    def _generate_symbol_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate signal for single symbol."""
        try:
            # Calculate three MAs
            close = df['close']

            if self.ma_type == 'sma':
                fast_ma = close.rolling(window=self.fast_period).mean()
                medium_ma = close.rolling(window=self.medium_period).mean()
                slow_ma = close.rolling(window=self.slow_period).mean()
            else:  # ema
                fast_ma = close.ewm(span=self.fast_period, adjust=False).mean()
                medium_ma = close.ewm(span=self.medium_period, adjust=False).mean()
                slow_ma = close.ewm(span=self.slow_period, adjust=False).mean()

            # Current values
            curr_fast = fast_ma.iloc[-1]
            curr_medium = medium_ma.iloc[-1]
            curr_slow = slow_ma.iloc[-1]

            # Previous values
            prev_fast = fast_ma.iloc[-2]
            prev_medium = medium_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]

            # Check for NaN
            if any(pd.isna([curr_fast, curr_medium, curr_slow,
                           prev_fast, prev_medium, prev_slow])):
                return None

            # Check for trend alignment
            uptrend_aligned = (curr_fast > curr_medium > curr_slow)
            downtrend_aligned = (curr_fast < curr_medium < curr_slow)

            # Check if trend just became aligned
            prev_uptrend = (prev_fast > prev_medium > prev_slow)
            prev_downtrend = (prev_fast < prev_medium < prev_slow)

            new_uptrend = uptrend_aligned and not prev_uptrend
            new_downtrend = downtrend_aligned and not prev_downtrend

            # No new alignment
            if not new_uptrend and not new_downtrend:
                return None

            direction = 'BUY' if new_uptrend else 'SELL'

            # Calculate confidence
            confidence = self._calculate_confidence(
                fast_ma, medium_ma, slow_ma, direction
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
                    'strategy': 'Triple_MA',
                    'fast_period': self.fast_period,
                    'medium_period': self.medium_period,
                    'slow_period': self.slow_period,
                    'ma_type': self.ma_type,
                    'fast_ma': float(curr_fast),
                    'medium_ma': float(curr_medium),
                    'slow_ma': float(curr_slow)
                }
            )

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _calculate_confidence(
        self,
        fast_ma: pd.Series,
        medium_ma: pd.Series,
        slow_ma: pd.Series,
        direction: str
    ) -> float:
        """Calculate confidence based on MA separation and slope."""
        # MA separation uniformity
        sep1 = abs(fast_ma.iloc[-1] - medium_ma.iloc[-1]) / medium_ma.iloc[-1]
        sep2 = abs(medium_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]
        avg_separation = (sep1 + sep2) / 2
        separation_score = min(avg_separation / 0.015, 1.0)

        # Trend strength (slope of fast MA)
        lookback = min(5, len(fast_ma))
        fast_slope = (fast_ma.iloc[-1] - fast_ma.iloc[-lookback]) / fast_ma.iloc[-lookback]

        if direction == 'BUY':
            trend_score = min(max(fast_slope / 0.015, 0), 1.0)
        else:
            trend_score = min(max(-fast_slope / 0.015, 0), 1.0)

        # Alignment consistency
        lookback = min(3, len(fast_ma))
        if direction == 'BUY':
            consistency = ((fast_ma.iloc[-lookback:] > medium_ma.iloc[-lookback:]) &
                          (medium_ma.iloc[-lookback:] > slow_ma.iloc[-lookback:])).sum() / lookback
        else:
            consistency = ((fast_ma.iloc[-lookback:] < medium_ma.iloc[-lookback:]) &
                          (medium_ma.iloc[-lookback:] < slow_ma.iloc[-lookback:])).sum() / lookback

        confidence = (
            separation_score * 0.40 +
            trend_score * 0.35 +
            consistency * 0.25
        )

        return confidence

    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'medium_period': self.medium_period,
            'slow_period': self.slow_period,
            'ma_type': self.ma_type,
            'min_confidence': self.min_confidence
        }


if __name__ == "__main__":
    logger.info("MA Crossover Signal Generators")
    logger.info("=" * 60)
    logger.info("Pure strategy implementations for MA crossovers")
    logger.info("")
    logger.info("MACrossoverSignals:")
    logger.info("  - Dual MA crossover (fast/slow)")
    logger.info("  - Default: 50/200 SMA (Golden Cross)")
    logger.info("")
    logger.info("TripleMACrossoverSignals:")
    logger.info("  - Triple MA alignment (fast/medium/slow)")
    logger.info("  - Default: 10/20/50 EMA")
    logger.info("")
    logger.info("Both strategies:")
    logger.info("  - Asset agnostic (works with ETFs, stocks, options)")
    logger.info("  - Return Signal objects (not boolean series)")
    logger.info("  - Include confidence scores")
    logger.info("  - No infrastructure dependencies")
