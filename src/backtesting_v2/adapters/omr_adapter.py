"""
OMR Strategy Adapter for v2 backtesting framework.

Wraps the OvernightMeanReversionStrategy to work with BacktestEngineV2
and provides trade enrichment with OMR-specific columns.
"""

from datetime import time
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np

from src.backtesting_v2.base.strategy import BaseStrategyV2
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OMRStrategyAdapter(BaseStrategyV2):
    """
    V2 adapter for Overnight Mean Reversion strategy.

    Provides:
    - generate_signals(): Entry/exit as boolean Series
    - get_signals_context(): Cache regime, probability, expected_return
    - enrich_trades(): Add OMR-specific columns to trades

    Usage:
        from src.backtesting_v2.adapters import OMRStrategyAdapter
        from src.backtesting_v2.engine import BacktestEngineV2

        adapter = OMRStrategyAdapter(
            entry_time="15:50",
            exit_time="09:31",
            regime_detector=...,  # Optional
            bayesian_model=...,   # Optional
        )
        engine = BacktestEngineV2()
        portfolio = engine.run(adapter, "TQQQ", "2023-01-01", "2023-12-31")
    """

    def __init__(
        self,
        entry_time: str = "15:50",
        exit_time: str = "09:31",
        regime_detector=None,
        bayesian_model=None,
        min_probability: float = 0.55,
        min_expected_return: float = 0.002,
        **kwargs,
    ):
        """
        Initialize OMR adapter.

        Args:
            entry_time: Entry time in HH:MM format (default: 15:50)
            exit_time: Exit time in HH:MM format (default: 09:31)
            regime_detector: Optional MarketRegimeDetector instance
            bayesian_model: Optional BayesianReversionModel instance
            min_probability: Minimum win rate threshold
            min_expected_return: Minimum expected return threshold
        """
        super().__init__(name="OMRStrategyAdapter", **kwargs)

        self.entry_hour, self.entry_minute = map(int, entry_time.split(":"))
        self.exit_hour, self.exit_minute = map(int, exit_time.split(":"))
        self.entry_time = time(self.entry_hour, self.entry_minute)
        self.exit_time = time(self.exit_hour, self.exit_minute)

        self.regime_detector = regime_detector
        self.bayesian_model = bayesian_model
        self.min_probability = min_probability
        self.min_expected_return = min_expected_return

        # Cache for signals context
        self._context_cache: Dict[pd.Timestamp, Dict[str, Any]] = {}

    def generate_signals(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry/exit signals based on time of day.

        Entry: At entry_time (default 3:50 PM)
        Exit: At exit_time (default 9:31 AM next day)

        Args:
            data: DataFrame with OHLCV data and datetime index

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        for ts in data.index:
            ts_time = ts.time()

            # Entry at 3:50 PM
            if ts_time == self.entry_time:
                entries.loc[ts] = True

            # Exit at 9:31 AM
            if ts_time == self.exit_time:
                exits.loc[ts] = True

        return entries, exits

    def get_signals_context(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Cache OMR signal context for trade enrichment.

        Computes regime, probability, and expected return at each entry point.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dict mapping timestamps to signal metadata
        """
        self._context_cache = {}

        for ts in data.index:
            if ts.time() != self.entry_time:
                continue

            # Get intraday return (close - open) / open
            date_data = data[data.index.date == ts.date()]
            if not date_data.empty:
                intraday_return = (date_data["close"].iloc[-1] - date_data["open"].iloc[0]) / date_data["open"].iloc[0]
            else:
                intraday_return = 0.0

            # Detect regime if detector available
            regime = "UNKNOWN"
            if self.regime_detector is not None:
                try:
                    regime = self.regime_detector.detect_regime(ts.date(), data)
                except (ValueError, AttributeError, KeyError):
                    regime = "UNKNOWN"

            # Get Bayesian probability if model available
            probability = 0.5
            expected_return = 0.0
            sample_size = 0

            if self.bayesian_model is not None:
                try:
                    prob_data = self.bayesian_model.get_probability(
                        regime=regime,
                        intraday_return=intraday_return,
                    )
                    probability = prob_data.get("probability", 0.5)
                    expected_return = prob_data.get("expected_return", 0.0)
                    sample_size = prob_data.get("sample_size", 0)
                except (KeyError, ValueError, TypeError):
                    pass

            self._context_cache[ts] = {
                "regime": regime,
                "win_rate": probability,
                "expected_return": expected_return,
                "sample_size": sample_size,
                "intraday_return": intraday_return,
            }

        logger.info(f"OMR context cached for {len(self._context_cache)} entry points")
        return self._context_cache

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Add OMR-specific columns to trades.

        Adds:
        - regime: Market regime at entry
        - win_rate: Bayesian win probability
        - expected_return: Expected overnight return
        - sample_size: Historical sample count
        - intraday_return: Intraday price change at entry

        Args:
            trades_df: Base trades DataFrame
            price_data: Original OHLCV data
            signals_context: Cached context from get_signals_context

        Returns:
            DataFrame with additional OMR columns
        """
        if trades_df.empty:
            return trades_df

        context = signals_context or self._context_cache

        if not context:
            logger.warning("No signals context available for OMR enrichment")
            return trades_df

        # Add columns from context
        omr_columns = ["regime", "win_rate", "expected_return", "sample_size", "intraday_return"]

        for col in omr_columns:
            trades_df[col] = trades_df["timestamp"].apply(
                lambda ts: context.get(ts, {}).get(col)
            )

        # Calculate actual overnight return for exit trades
        if "pnl_pct" not in trades_df.columns:
            trades_df["pnl_pct"] = None

        logger.info(f"Enriched {len(trades_df)} trades with OMR columns")
        return trades_df


class SimpleOMRAdapter(BaseStrategyV2):
    """
    Simplified OMR adapter without external dependencies.

    Use this for quick backtests when you don't need
    regime detection or Bayesian probability.
    """

    def __init__(
        self,
        entry_time: str = "15:50",
        exit_time: str = "09:31",
        **kwargs,
    ):
        super().__init__(name="SimpleOMRAdapter", **kwargs)

        self.entry_hour, self.entry_minute = map(int, entry_time.split(":"))
        self.exit_hour, self.exit_minute = map(int, exit_time.split(":"))
        self.entry_time = time(self.entry_hour, self.entry_minute)
        self.exit_time = time(self.exit_hour, self.exit_minute)

    def generate_signals(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals based on time."""
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        for ts in data.index:
            ts_time = ts.time()
            if ts_time == self.entry_time:
                entries.loc[ts] = True
            if ts_time == self.exit_time:
                exits.loc[ts] = True

        return entries, exits

    def get_signals_context(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Simple context with intraday returns only."""
        context = {}

        for ts in data.index:
            if ts.time() != self.entry_time:
                continue

            date_data = data[data.index.date == ts.date()]
            if not date_data.empty:
                intraday_return = (
                    (date_data["close"].iloc[-1] - date_data["open"].iloc[0])
                    / date_data["open"].iloc[0]
                )
            else:
                intraday_return = 0.0

            context[ts] = {
                "intraday_return": intraday_return,
            }

        return context

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Add intraday return to trades."""
        if trades_df.empty or not signals_context:
            return trades_df

        trades_df["intraday_return"] = trades_df["timestamp"].apply(
            lambda ts: signals_context.get(ts, {}).get("intraday_return")
        )

        return trades_df
