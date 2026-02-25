"""
EVR (Effort vs. Result) Strategy.

A Volume Spread Analysis (VSA) based mean reversion strategy that detects
absorption candles - when massive volume produces minimal price movement,
indicating smart money passively absorbing aggressive orders at trend
exhaustion points.

Core Concept:
- Normal: Heavy volume -> Large price movement
- Anomaly: Heavy volume -> Tiny price movement (ABSORPTION)

Signal Logic:
1. Context: Market must be extended (oversold for longs, overbought for shorts)
2. Absorption: High relative volume + Low relative spread
3. Confirmation: Next candle closes above/below absorption candle
4. Exit: Mean reversion target (middle Bollinger Band)

Usage:
    from src.strategies.advanced.evr_strategy import EVRStrategy

    strategy = EVRStrategy(
        vol_threshold=2.0,
        spread_threshold=0.8,
        require_confirmation=True,
    )

    long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.backtesting.base.strategy import LongShortStrategy
from src.strategies.advanced.evr_indicators import EVRIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EVRStrategy(LongShortStrategy):
    """
    Effort vs. Result Strategy - Volume Spread Analysis for absorption detection.

    Detects absorption candles where high volume produces minimal price movement,
    often signaling trend exhaustion and reversal opportunities.

    Parameters:
        vol_lookback: Lookback period for volume SMA (default 20)
        spread_lookback: Lookback period for spread SMA (default 20)
        vol_threshold: Minimum relative volume for absorption (default 2.0)
        spread_threshold: Maximum relative spread for absorption (default 0.8)
        rsi_period: RSI period (default 14)
        rsi_oversold: RSI oversold threshold (default 30)
        rsi_overbought: RSI overbought threshold (default 70)
        bb_period: Bollinger Band period (default 20)
        bb_std: Bollinger Band standard deviations (default 2.0)
        require_confirmation: Require next bar confirmation (default True)
    """

    def __init__(
        self,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        vol_threshold: float = 2.0,
        spread_threshold: float = 0.8,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
        require_confirmation: bool = True,
        **params
    ):
        """
        Initialize EVR strategy.

        Args:
            vol_lookback: Lookback period for volume normalization
            spread_lookback: Lookback period for spread normalization
            vol_threshold: Min relative volume to detect absorption (2.0 = 200%)
            spread_threshold: Max relative spread for absorption (0.8 = 80%)
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI threshold for oversold context
            rsi_overbought: RSI threshold for overbought context
            bb_period: Period for Bollinger Bands
            bb_std: Standard deviations for Bollinger Bands
            require_confirmation: Whether to require confirmation candle
        """
        self.vol_lookback = vol_lookback
        self.spread_lookback = spread_lookback
        self.vol_threshold = vol_threshold
        self.spread_threshold = spread_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.require_confirmation = require_confirmation

        super().__init__(
            vol_lookback=vol_lookback,
            spread_lookback=spread_lookback,
            vol_threshold=vol_threshold,
            spread_threshold=spread_threshold,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            bb_period=bb_period,
            bb_std=bb_std,
            require_confirmation=require_confirmation,
            **params
        )

        # Validate parameters after super().__init__
        self.validate_parameters()

        logger.info("Initialized EVR Strategy")
        logger.info(f"  Vol Lookback: {vol_lookback}, Spread Lookback: {spread_lookback}")
        logger.info(f"  Vol Threshold: {vol_threshold}, Spread Threshold: {spread_threshold}")
        logger.info(f"  RSI Period: {rsi_period}, OS: {rsi_oversold}, OB: {rsi_overbought}")
        logger.info(f"  BB Period: {bb_period}, Std: {bb_std}")
        logger.info(f"  Require Confirmation: {require_confirmation}")

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.vol_lookback < 5:
            raise ValueError(f"vol_lookback must be >= 5, got {self.vol_lookback}")

        if self.spread_lookback < 5:
            raise ValueError(f"spread_lookback must be >= 5, got {self.spread_lookback}")

        if self.vol_threshold <= 1.0:
            raise ValueError(
                f"vol_threshold must be > 1.0, got {self.vol_threshold}"
            )

        if not (0 < self.spread_threshold < 1.0):
            raise ValueError(
                f"spread_threshold must be in (0, 1.0), got {self.spread_threshold}"
            )

        if self.rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {self.rsi_period}")

        if not (0 < self.rsi_oversold < self.rsi_overbought < 100):
            raise ValueError(
                f"RSI thresholds must satisfy 0 < oversold < overbought < 100, "
                f"got oversold={self.rsi_oversold}, overbought={self.rsi_overbought}"
            )

        if self.bb_period < 5:
            raise ValueError(f"bb_period must be >= 5, got {self.bb_period}")

        if self.bb_std <= 0:
            raise ValueError(f"bb_std must be > 0, got {self.bb_std}")

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry and exit signals.

        Args:
            data: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        if data.empty:
            empty = pd.Series(dtype=bool)
            return empty, empty, empty, empty

        # Normalize column names to lowercase
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(
                df=df,
                vol_lookback=self.vol_lookback,
                spread_lookback=self.spread_lookback,
                vol_threshold=self.vol_threshold,
                spread_threshold=self.spread_threshold,
                rsi_period=self.rsi_period,
                rsi_oversold=self.rsi_oversold,
                rsi_overbought=self.rsi_overbought,
                bb_period=self.bb_period,
                bb_std=self.bb_std,
                require_confirmation=self.require_confirmation,
            )

        return long_entries, long_exits, short_entries, short_exits

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'vol_lookback': self.vol_lookback,
            'spread_lookback': self.spread_lookback,
            'vol_threshold': self.vol_threshold,
            'spread_threshold': self.spread_threshold,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'require_confirmation': self.require_confirmation,
        }

    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get all indicators for analysis/visualization.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all EVR indicators
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        return EVRIndicators.calculate_all_indicators(
            df=df,
            vol_lookback=self.vol_lookback,
            spread_lookback=self.spread_lookback,
            rsi_period=self.rsi_period,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
        )

    def get_current_signal(self, data: pd.DataFrame):
        """
        Get current EVR signal state for live trading.

        Args:
            data: OHLCV DataFrame (must have enough history)

        Returns:
            EVRSignal with current state
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        return EVRIndicators.get_current_signal(
            df=df,
            vol_lookback=self.vol_lookback,
            spread_lookback=self.spread_lookback,
            vol_threshold=self.vol_threshold,
            spread_threshold=self.spread_threshold,
            rsi_period=self.rsi_period,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
        )

    def __repr__(self) -> str:
        """String representation of strategy."""
        return (
            f"EVRStrategy("
            f"vol_lookback={self.vol_lookback}, "
            f"spread_lookback={self.spread_lookback}, "
            f"vol_threshold={self.vol_threshold}, "
            f"spread_threshold={self.spread_threshold}, "
            f"rsi_period={self.rsi_period}, "
            f"require_confirmation={self.require_confirmation})"
        )
