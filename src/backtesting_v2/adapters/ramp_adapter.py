"""
RAMP Strategy Adapter for v2 backtesting framework.

Provides trade enrichment with RAMP-specific columns:
- regime: Market regime at entry
- momentum_score: Momentum score at entry
- rank: Ranking among universe
- weight: Portfolio weight
"""

from datetime import time
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import numpy as np

from src.backtesting_v2.base.strategy import BaseStrategyV2
from src.utils.logger import get_logger

logger = get_logger(__name__)

# RAMP regime parameters (from walk-forward validation)
REGIME_PARAMS = {
    "STRONG_BULL": {"long_p": 21, "short_p": 5, "long_w": 0.3, "pen_w": 5.0, "top_n": 20},
    "WEAK_BULL": {"long_p": 21, "short_p": 5, "long_w": 0.3, "pen_w": 5.0, "top_n": 10},
    "SIDEWAYS": {"long_p": 21, "short_p": 5, "long_w": 0.5, "pen_w": 2.0, "top_n": 5},
    "UNPREDICTABLE": {"long_p": 42, "short_p": 21, "long_w": 0.5, "pen_w": 4.0, "top_n": 10},
    "BEAR": {"long_p": 21, "short_p": 5, "long_w": 0.3, "pen_w": 3.0, "top_n": 10},
}

DEFAULT_PARAMS = {"long_p": 21, "short_p": 5, "long_w": 0.3, "pen_w": 4.0, "top_n": 10}


class RAMPStrategyAdapter(BaseStrategyV2):
    """
    V2 adapter for Regime-Aware Momentum Protection strategy.

    This adapter wraps RAMP logic to work with BacktestEngineV2 and provides
    trade enrichment with RAMP-specific columns.

    Note: RAMP is a multi-asset strategy. This adapter is for single-symbol
    backtesting (e.g., testing individual momentum signals).

    Usage:
        from src.backtesting_v2.adapters import RAMPStrategyAdapter
        from src.backtesting_v2.engine import BacktestEngineV2

        adapter = RAMPStrategyAdapter(
            rebalance_time="15:55",
            momentum_long_period=21,
            momentum_short_period=5,
        )
        engine = BacktestEngineV2()
        portfolio = engine.run(adapter, "AAPL", "2023-01-01", "2023-12-31")
    """

    def __init__(
        self,
        rebalance_time: str = "15:55",
        momentum_long_period: int = 21,
        momentum_short_period: int = 5,
        long_weight: float = 0.3,
        penalty_weight: float = 4.0,
        top_n: int = 10,
        regime_detector=None,
        **kwargs,
    ):
        """
        Initialize RAMP adapter.

        Args:
            rebalance_time: Daily rebalance time (HH:MM format)
            momentum_long_period: Long-term momentum lookback
            momentum_short_period: Short-term penalty lookback
            long_weight: Weight for long-term return
            penalty_weight: Weight for short-term penalty
            top_n: Number of top momentum stocks
            regime_detector: Optional MarketRegimeDetector instance
        """
        super().__init__(name="RAMPStrategyAdapter", **kwargs)

        self.rebalance_hour, self.rebalance_minute = map(int, rebalance_time.split(":"))
        self.rebalance_time = time(self.rebalance_hour, self.rebalance_minute)

        self.momentum_long_period = momentum_long_period
        self.momentum_short_period = momentum_short_period
        self.long_weight = long_weight
        self.penalty_weight = penalty_weight
        self.top_n = top_n
        self.regime_detector = regime_detector

        self._context_cache: Dict[pd.Timestamp, Dict[str, Any]] = {}

    def generate_signals(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate momentum-based signals at rebalance time.

        Entry: When momentum score > 0 at rebalance time
        Exit: When momentum score <= 0 or next rebalance

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Calculate momentum scores
        momentum_scores = self._calculate_momentum(data)

        # Generate signals at rebalance times
        prev_signal = 0  # 0 = flat, 1 = long

        for ts in data.index:
            if ts.time() == self.rebalance_time:
                score = momentum_scores.get(ts, 0)

                if score > 0 and prev_signal == 0:
                    entries.loc[ts] = True
                    prev_signal = 1
                elif score <= 0 and prev_signal == 1:
                    exits.loc[ts] = True
                    prev_signal = 0

        return entries, exits

    def _calculate_momentum(self, data: pd.DataFrame) -> Dict[pd.Timestamp, float]:
        """Calculate RAMP momentum scores."""
        momentum = {}

        if "close" not in data.columns:
            return momentum

        close = data["close"]

        for ts in data.index:
            if ts.time() != self.rebalance_time:
                continue

            idx = data.index.get_loc(ts)
            if idx < self.momentum_long_period:
                continue

            # Calculate returns
            long_ret = (close.iloc[idx] - close.iloc[idx - self.momentum_long_period]) / close.iloc[idx - self.momentum_long_period]
            short_ret = (close.iloc[idx] - close.iloc[idx - self.momentum_short_period]) / close.iloc[idx - self.momentum_short_period]

            # RAMP formula: long_weight * long_ret - penalty_weight * short_ret
            score = self.long_weight * long_ret - self.penalty_weight * short_ret
            momentum[ts] = score

        return momentum

    def get_signals_context(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Cache RAMP signal context for trade enrichment.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dict mapping timestamps to signal metadata
        """
        self._context_cache = {}

        momentum = self._calculate_momentum(data)

        for ts, score in momentum.items():
            # Detect regime if detector available
            regime = "SIDEWAYS"
            if self.regime_detector is not None:
                try:
                    regime = self.regime_detector.detect_regime(ts.date(), data)
                except (ValueError, AttributeError, KeyError):
                    pass

            # Get regime-specific parameters
            params = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)

            self._context_cache[ts] = {
                "regime": regime,
                "momentum_score": score,
                "rank": 1,  # Single symbol = rank 1
                "weight": 1.0 / params["top_n"],
                "long_period": params["long_p"],
                "short_period": params["short_p"],
            }

        logger.info(f"RAMP context cached for {len(self._context_cache)} rebalance points")
        return self._context_cache

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Add RAMP-specific columns to trades.

        Adds:
        - regime: Market regime at entry
        - momentum_score: Momentum score at entry
        - rank: Ranking among universe (1 for single symbol)
        - weight: Portfolio weight

        Args:
            trades_df: Base trades DataFrame
            price_data: Original OHLCV data
            signals_context: Cached context from get_signals_context

        Returns:
            DataFrame with additional RAMP columns
        """
        if trades_df.empty:
            return trades_df

        context = signals_context or self._context_cache

        if not context:
            logger.warning("No signals context available for RAMP enrichment")
            return trades_df

        # Add columns from context
        ramp_columns = ["regime", "momentum_score", "rank", "weight"]

        for col in ramp_columns:
            trades_df[col] = trades_df["timestamp"].apply(
                lambda ts: context.get(ts, {}).get(col)
            )

        logger.info(f"Enriched {len(trades_df)} trades with RAMP columns")
        return trades_df


class SimpleMomentumAdapter(BaseStrategyV2):
    """
    Simplified momentum adapter without regime detection.

    Uses basic momentum: (price - price_n_bars_ago) / price_n_bars_ago
    """

    def __init__(
        self,
        lookback_period: int = 21,
        rebalance_time: str = "15:55",
        **kwargs,
    ):
        super().__init__(name="SimpleMomentumAdapter", **kwargs)

        self.lookback_period = lookback_period
        self.rebalance_hour, self.rebalance_minute = map(int, rebalance_time.split(":"))
        self.rebalance_time = time(self.rebalance_hour, self.rebalance_minute)

    def generate_signals(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate momentum signals."""
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        close = data["close"]
        prev_signal = 0

        for i, ts in enumerate(data.index):
            if ts.time() != self.rebalance_time:
                continue

            if i < self.lookback_period:
                continue

            momentum = (close.iloc[i] - close.iloc[i - self.lookback_period]) / close.iloc[i - self.lookback_period]

            if momentum > 0 and prev_signal == 0:
                entries.iloc[i] = True
                prev_signal = 1
            elif momentum <= 0 and prev_signal == 1:
                exits.iloc[i] = True
                prev_signal = 0

        return entries, exits

    def get_signals_context(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Cache momentum scores."""
        context = {}
        close = data["close"]

        for i, ts in enumerate(data.index):
            if ts.time() != self.rebalance_time:
                continue
            if i < self.lookback_period:
                continue

            momentum = (close.iloc[i] - close.iloc[i - self.lookback_period]) / close.iloc[i - self.lookback_period]
            context[ts] = {"momentum_score": momentum}

        return context

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Add momentum score to trades."""
        if trades_df.empty or not signals_context:
            return trades_df

        trades_df["momentum_score"] = trades_df["timestamp"].apply(
            lambda ts: signals_context.get(ts, {}).get("momentum_score")
        )

        return trades_df
