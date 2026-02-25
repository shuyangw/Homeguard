"""
Dual Signal Trend Sentinel (DSTS) Strategy.

Volatility-normalized trend following for Bitcoin using Z-score deviation from EMA.

Strategy Logic:
- Baseline: EMA(65)
- Volatility: StdDev(65)
- Signal: Z-Score = (Close - EMA) / StdDev
- Entry: Z-Score > bull_threshold (default 0.75)
- Exit: Z-Score < bear_threshold (default 0.0)
- Position: 100% long or 100% cash

Key Insight:
- Complexity hurts - adding filters reduced performance
- Simple thresholds outperform grid search optimization
- State machine with hysteresis reduces whipsaws

Target Performance:
- CAGR: ~66%
- Max DD: ~27%
- Win Rate: ~47%
- Win/Loss Ratio: >8:1
- Trades/Year: ~4-5

Usage:
    from src.strategies.advanced.dsts_strategy import DSTSStrategy

    strategy = DSTSStrategy(
        period=65,
        bull_threshold=0.75,
        bear_threshold=0.0,
    )

    # For backtesting
    entries, exits = strategy.generate_signals(btc_data)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.backtesting.base.strategy import LongOnlyStrategy
from src.strategies.advanced.dsts_indicators import DSTSIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DSTSStrategy(LongOnlyStrategy):
    """
    Dual Signal Trend Sentinel - Z-score based trend following.

    Uses volatility-normalized price deviation to detect trends.
    State machine with hysteresis reduces whipsaws.

    Parameters:
        period: Period for EMA and StdDev calculations (default 65)
        bull_threshold: Z-score threshold to enter long (default 0.75)
        bear_threshold: Z-score threshold to exit long (default 0.0)
    """

    def __init__(
        self,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
        **params
    ):
        """
        Initialize DSTS strategy.

        Args:
            period: Period for EMA and StdDev calculations
            bull_threshold: Z-score threshold to enter long position
            bear_threshold: Z-score threshold to exit long position
        """
        self.period = period
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

        super().__init__(
            period=period,
            bull_threshold=bull_threshold,
            bear_threshold=bear_threshold,
            **params
        )

        logger.info(f"Initialized DSTS Strategy")
        logger.info(f"  Period: {period}")
        logger.info(f"  Bull Threshold: {bull_threshold}")
        logger.info(f"  Bear Threshold: {bear_threshold}")

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.period < 10:
            raise ValueError(f"period must be >= 10, got {self.period}")

        if self.bull_threshold <= self.bear_threshold:
            raise ValueError(
                f"bull_threshold ({self.bull_threshold}) must be > "
                f"bear_threshold ({self.bear_threshold})"
            )

    def generate_long_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.

        Args:
            data: OHLCV DataFrame with 'close' column

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        if data.empty:
            return pd.Series(dtype=bool), pd.Series(dtype=bool)

        # Normalize column names
        close_col = 'close' if 'close' in data.columns else 'Close'
        if close_col not in data.columns:
            raise ValueError("Data must contain 'close' column")

        close = data[close_col]

        entries, exits, _ = DSTSIndicators.generate_signals(
            close=close,
            period=self.period,
            bull_threshold=self.bull_threshold,
            bear_threshold=self.bear_threshold,
        )

        return entries, exits

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'period': self.period,
            'bull_threshold': self.bull_threshold,
            'bear_threshold': self.bear_threshold,
        }

    def get_zscore(self, data: pd.DataFrame) -> pd.Series:
        """
        Get Z-score series for analysis/visualization.

        Args:
            data: OHLCV DataFrame with 'close' column

        Returns:
            Z-score series
        """
        close_col = 'close' if 'close' in data.columns else 'Close'
        return DSTSIndicators.calculate_zscore(data[close_col], self.period)

    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get all indicators for analysis/visualization.

        Args:
            data: OHLCV DataFrame with 'close' column

        Returns:
            DataFrame with close, ema, std, zscore columns
        """
        close_col = 'close' if 'close' in data.columns else 'Close'
        return DSTSIndicators.calculate_all_indicators(data[close_col], self.period)
