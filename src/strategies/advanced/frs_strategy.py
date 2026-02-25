"""
Fractal Regime Switching (FRS) Strategy.

A meta-strategy that uses the Hurst Exponent to detect market regimes
and routes to the appropriate sub-strategy:

- H > 0.60 (Trending): Deploy Donchian Breakout
- H < 0.40 (Mean-Reverting): Deploy Bollinger + RSI Mean Reversion
- 0.40 <= H <= 0.60 (Noise): Go to cash, do not trade

The Hurst Exponent measures long-term memory in a time series:
- H < 0.5: Anti-persistent (mean-reverting)
- H = 0.5: Random walk (Brownian motion)
- H > 0.5: Persistent (trending)

Usage:
    from src.strategies.advanced.frs_strategy import FRSStrategy

    strategy = FRSStrategy(
        hurst_window=100,
        trend_threshold=0.60,
        mr_threshold=0.40,
    )

    # For backtesting (long-short)
    long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(data)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.backtesting.base.strategy import LongShortStrategy
from src.strategies.advanced.frs_indicators import FRSIndicators, FRSRegime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FRSStrategy(LongShortStrategy):
    """
    Fractal Regime Switching Strategy - Hurst-based regime routing.

    Routes to Donchian Breakout (trending) or Bollinger+RSI (mean-reverting)
    based on rolling Hurst exponent. Goes to cash during noise regime.

    Parameters:
        hurst_window: Window for Hurst calculation (default 100)
        hurst_smooth_span: EMA span for Hurst smoothing (default 10)
        trend_threshold: Hurst above this = TRENDING (default 0.60)
        mr_threshold: Hurst below this = MEAN_REVERTING (default 0.40)
        donchian_period: Donchian channel period (default 20)
        bollinger_period: Bollinger band period (default 20)
        bollinger_std: Bollinger standard deviations (default 2.0)
        rsi_period: RSI period (default 14)
        rsi_overbought: RSI level for overbought (default 70)
        rsi_oversold: RSI level for oversold (default 30)
    """

    def __init__(
        self,
        hurst_window: int = 100,
        hurst_smooth_span: int = 10,
        trend_threshold: float = 0.60,
        mr_threshold: float = 0.40,
        donchian_period: int = 20,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        **params
    ):
        """
        Initialize FRS strategy.

        Args:
            hurst_window: Window for Hurst exponent calculation
            hurst_smooth_span: EMA span for smoothing Hurst values
            trend_threshold: Hurst above this triggers trend-following
            mr_threshold: Hurst below this triggers mean-reversion
            donchian_period: Period for Donchian channel
            bollinger_period: Period for Bollinger bands
            bollinger_std: Number of standard deviations for Bollinger
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI threshold for overbought
            rsi_oversold: RSI threshold for oversold
        """
        self.hurst_window = hurst_window
        self.hurst_smooth_span = hurst_smooth_span
        self.trend_threshold = trend_threshold
        self.mr_threshold = mr_threshold
        self.donchian_period = donchian_period
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

        super().__init__(
            hurst_window=hurst_window,
            hurst_smooth_span=hurst_smooth_span,
            trend_threshold=trend_threshold,
            mr_threshold=mr_threshold,
            donchian_period=donchian_period,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            **params
        )

        logger.info("Initialized FRS Strategy")
        logger.info(f"  Hurst Window: {hurst_window}, Smooth Span: {hurst_smooth_span}")
        logger.info(f"  Trend Threshold: {trend_threshold}, MR Threshold: {mr_threshold}")
        logger.info(f"  Donchian Period: {donchian_period}")
        logger.info(f"  Bollinger Period: {bollinger_period}, Std: {bollinger_std}")
        logger.info(f"  RSI Period: {rsi_period}, OB: {rsi_overbought}, OS: {rsi_oversold}")

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.hurst_window < 20:
            raise ValueError(f"hurst_window must be >= 20, got {self.hurst_window}")

        if self.trend_threshold <= self.mr_threshold:
            raise ValueError(
                f"trend_threshold ({self.trend_threshold}) must be > "
                f"mr_threshold ({self.mr_threshold})"
            )

        if not (0 < self.mr_threshold < 0.5):
            raise ValueError(
                f"mr_threshold should be in (0, 0.5), got {self.mr_threshold}"
            )

        if not (0.5 < self.trend_threshold < 1.0):
            raise ValueError(
                f"trend_threshold should be in (0.5, 1.0), got {self.trend_threshold}"
            )

        if self.donchian_period < 5:
            raise ValueError(f"donchian_period must be >= 5, got {self.donchian_period}")

        if self.bollinger_period < 5:
            raise ValueError(f"bollinger_period must be >= 5, got {self.bollinger_period}")

        if self.rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {self.rsi_period}")

        if not (0 < self.rsi_oversold < self.rsi_overbought < 100):
            raise ValueError(
                f"RSI thresholds must satisfy 0 < oversold < overbought < 100, "
                f"got oversold={self.rsi_oversold}, overbought={self.rsi_overbought}"
            )

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry and exit signals.

        Args:
            data: OHLCV DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        if data.empty:
            empty = pd.Series(dtype=bool)
            return empty, empty, empty, empty

        # Normalize column names to lowercase
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        long_entries, long_exits, short_entries, short_exits, regime = \
            FRSIndicators.generate_signals(
                df=df,
                hurst_window=self.hurst_window,
                hurst_smooth_span=self.hurst_smooth_span,
                trend_threshold=self.trend_threshold,
                mr_threshold=self.mr_threshold,
                donchian_period=self.donchian_period,
                bollinger_period=self.bollinger_period,
                bollinger_std=self.bollinger_std,
                rsi_period=self.rsi_period,
                rsi_overbought=self.rsi_overbought,
                rsi_oversold=self.rsi_oversold,
            )

        return long_entries, long_exits, short_entries, short_exits

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'hurst_window': self.hurst_window,
            'hurst_smooth_span': self.hurst_smooth_span,
            'trend_threshold': self.trend_threshold,
            'mr_threshold': self.mr_threshold,
            'donchian_period': self.donchian_period,
            'bollinger_period': self.bollinger_period,
            'bollinger_std': self.bollinger_std,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
        }

    def get_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Get regime classification series for analysis/visualization.

        Args:
            data: OHLCV DataFrame with 'close' column

        Returns:
            Series of regime classifications (TRENDING, MEAN_REVERTING, NOISE)
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        hurst = FRSIndicators.calculate_hurst(df['close'], self.hurst_window)
        hurst_smoothed = FRSIndicators.smooth_hurst(hurst, self.hurst_smooth_span)
        regime = FRSIndicators.classify_regime_series(
            hurst_smoothed,
            self.trend_threshold,
            self.mr_threshold
        )
        return regime

    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get all indicators for analysis/visualization.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all FRS indicators
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        return FRSIndicators.calculate_all_indicators(
            df=df,
            hurst_window=self.hurst_window,
            hurst_smooth_span=self.hurst_smooth_span,
            donchian_period=self.donchian_period,
            bollinger_period=self.bollinger_period,
            bollinger_std=self.bollinger_std,
            rsi_period=self.rsi_period,
        )

    def get_current_signal(self, data: pd.DataFrame):
        """
        Get current FRS signal state for live trading.

        Args:
            data: OHLCV DataFrame (must have enough history)

        Returns:
            FRSSignal with current state
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        return FRSIndicators.get_current_signal(
            df=df,
            hurst_window=self.hurst_window,
            hurst_smooth_span=self.hurst_smooth_span,
            trend_threshold=self.trend_threshold,
            mr_threshold=self.mr_threshold,
            donchian_period=self.donchian_period,
            bollinger_period=self.bollinger_period,
            bollinger_std=self.bollinger_std,
            rsi_period=self.rsi_period,
        )

    def get_regime_distribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get percentage of time spent in each regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary with regime percentages
        """
        regime = self.get_regime(data)
        counts = regime.value_counts(normalize=True)

        return {
            'TRENDING': counts.get(FRSRegime.TRENDING.value, 0.0),
            'MEAN_REVERTING': counts.get(FRSRegime.MEAN_REVERTING.value, 0.0),
            'NOISE': counts.get(FRSRegime.NOISE.value, 0.0),
        }
