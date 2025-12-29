"""
Base strategy class for backtesting v2 with trade enrichment support.

Extends the original BaseStrategy pattern with:
- get_signals_context(): Cache intermediate computations for enrichment
- enrich_trades(): Add strategy-specific columns to trades post-simulation
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

import pandas as pd


class BaseStrategyV2(ABC):
    """
    Extended strategy base with trade enrichment support.

    Strategies inherit from this class and implement:
    - generate_signals(): Required - produces entry/exit signals
    - get_signals_context(): Optional - caches data for enrichment
    - enrich_trades(): Optional - adds custom columns to trades

    Example:
        class MyStrategy(BaseStrategyV2):
            def generate_signals(self, data):
                entries = data['close'] > data['close'].shift(1)
                exits = data['close'] < data['close'].shift(1)
                return entries, exits

            def get_signals_context(self, data):
                return {'momentum': data['close'].pct_change()}

            def enrich_trades(self, trades_df, price_data, context):
                trades_df['momentum'] = trades_df['timestamp'].map(
                    lambda ts: context['momentum'].get(ts)
                )
                return trades_df
    """

    def __init__(self, name: str = "BaseStrategyV2", **params):
        """
        Initialize strategy with name and parameters.

        Args:
            name: Strategy identifier
            **params: Strategy-specific parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals from price data.

        Args:
            data: DataFrame with OHLCV columns (timestamp index)

        Returns:
            Tuple of (entries, exits) boolean Series aligned to data index
        """
        pass

    def get_signals_context(
        self, data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Cache intermediate signals/computations for trade enrichment.

        Override this method to store any data needed by enrich_trades().
        This is called ONCE before simulation, so computations here don't
        affect per-bar performance.

        Args:
            data: DataFrame with OHLCV columns (timestamp index)

        Returns:
            Dictionary of cached data, or None if no caching needed

        Example:
            def get_signals_context(self, data):
                return {
                    'regime': self.detect_regime(data),
                    'momentum': self.calculate_momentum(data),
                }
        """
        return None

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Add strategy-specific columns to trades post-simulation.

        Override this method to add custom metadata to each trade.
        This is called ONCE after simulation completes, so enrichment
        logic doesn't affect simulation performance.

        Args:
            trades_df: DataFrame of trades with base columns
            price_data: Original OHLCV DataFrame
            signals_context: Cached data from get_signals_context()

        Returns:
            DataFrame with additional strategy-specific columns

        Example:
            def enrich_trades(self, trades_df, price_data, context):
                if trades_df.empty or not context:
                    return trades_df

                trades_df['regime'] = trades_df['timestamp'].map(
                    lambda ts: context['regime'].get(ts)
                )
                return trades_df
        """
        return trades_df

    def get_required_columns(self) -> list:
        """
        Return list of required data columns.

        Override if strategy needs non-standard columns.

        Returns:
            List of column names required in price data
        """
        return ["open", "high", "low", "close", "volume"]

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        required = self.get_required_columns()
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(
                f"Strategy '{self.name}' requires columns {missing} "
                f"but data only has {list(data.columns)}"
            )
        return True

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
