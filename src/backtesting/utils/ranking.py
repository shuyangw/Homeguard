"""
Ranking utilities for cross-sectional strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List


class Ranking:
    """
    Utilities for cross-sectional ranking and filtering.
    """

    @staticmethod
    def cross_sectional_rank(
        data_dict: Dict[str, pd.DataFrame],
        metric: str = 'close',
        ascending: bool = False,
        method: str = 'average'
    ) -> pd.DataFrame:
        """
        Rank assets cross-sectionally at each timestamp.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data
            metric: Column name to rank on (default: 'close')
            ascending: Rank ascending if True (default: False for momentum)
            method: Ranking method ('average', 'min', 'max', 'first', 'dense')

        Returns:
            DataFrame with symbols as columns, timestamps as index, values are ranks
            Lower rank = better (rank 1 is best if ascending=False)
        """
        metric_dict = {}

        for symbol, df in data_dict.items():
            if metric in df.columns:
                metric_dict[symbol] = df[metric]

        combined = pd.DataFrame(metric_dict)

        ranks = combined.rank(axis=1, ascending=ascending, method=method)

        return ranks

    @staticmethod
    def percentile_filter(
        ranks: pd.DataFrame,
        top_pct: Optional[float] = None,
        bottom_pct: Optional[float] = None,
        top_n: Optional[int] = None,
        bottom_n: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Filter ranks to select top and/or bottom performers.

        Args:
            ranks: Rank DataFrame from cross_sectional_rank()
            top_pct: Top percentile to select (e.g., 0.2 = top 20%)
            bottom_pct: Bottom percentile to select (e.g., 0.2 = bottom 20%)
            top_n: Top N assets to select
            bottom_n: Bottom N assets to select

        Returns:
            Dictionary with 'long' and 'short' keys, each containing boolean DataFrame
            True where asset should be held long/short
        """
        n_assets = ranks.shape[1]

        long_mask = pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
        short_mask = pd.DataFrame(False, index=ranks.index, columns=ranks.columns)

        if top_pct is not None:
            threshold = int(n_assets * top_pct)
            long_mask = ranks <= threshold
        elif top_n is not None:
            long_mask = ranks <= top_n

        if bottom_pct is not None:
            threshold = int(n_assets * (1 - bottom_pct))
            short_mask = ranks > threshold
        elif bottom_n is not None:
            short_mask = ranks > (n_assets - bottom_n)

        return {'long': long_mask, 'short': short_mask}

    @staticmethod
    def calculate_momentum_score(
        data_dict: Dict[str, pd.DataFrame],
        lookback_periods: List[int] = [63, 126, 252],
        weights: Optional[List[float]] = None,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Calculate composite momentum scores across multiple lookback periods.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with price data
            lookback_periods: List of lookback periods in days (default: [63, 126, 252] = 3m, 6m, 12m)
            weights: Weights for each lookback period (default: equal weight)
            price_column: Price column to use (default: 'close')

        Returns:
            DataFrame with symbols as columns, timestamps as index, values are momentum scores
        """
        if weights is None:
            weights = [1.0 / len(lookback_periods)] * len(lookback_periods)

        if len(weights) != len(lookback_periods):
            raise ValueError("Number of weights must match number of lookback periods")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        momentum_dict = {}

        for symbol, df in data_dict.items():
            if price_column not in df.columns:
                continue

            prices = df[price_column]

            momentum_components = []
            for period in lookback_periods:
                returns = prices.pct_change(periods=period)
                momentum_components.append(returns)

            weighted_momentum = pd.Series(0.0, index=prices.index)
            for component, weight in zip(momentum_components, weights):
                weighted_momentum += component * weight

            momentum_dict[symbol] = weighted_momentum

        momentum_df = pd.DataFrame(momentum_dict)

        return momentum_df

    @staticmethod
    def rebalance_on_schedule(
        signals: pd.DataFrame,
        rebalance_period: str = 'monthly',
        offset: int = 0
    ) -> pd.DataFrame:
        """
        Convert continuous signals to periodic rebalancing signals.

        Args:
            signals: Boolean DataFrame of continuous signals
            rebalance_period: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
            offset: Offset for rebalancing (e.g., 0 = first day, 1 = second day)

        Returns:
            Boolean DataFrame where True only on rebalancing days
        """
        rebalanced = signals.copy()

        if rebalance_period == 'daily':
            return rebalanced

        rebalance_mask = pd.Series(False, index=signals.index)

        if rebalance_period == 'weekly':
            rebalance_mask = (signals.index.dayofweek == offset)
        elif rebalance_period == 'monthly':
            is_month_start = (signals.index.to_series().diff().dt.days > 1) | (signals.index == signals.index[0])

            month_changes = signals.index.to_series().dt.month.diff() != 0
            month_changes.iloc[0] = True

            rebalance_dates = signals.index[month_changes]

            if offset > 0 and len(rebalance_dates) > 0:
                shifted_dates = []
                for date in rebalance_dates:
                    mask = signals.index > date
                    if mask.sum() >= offset:
                        shifted_dates.append(signals.index[mask][offset - 1])
                rebalance_mask = signals.index.isin(shifted_dates)
            else:
                rebalance_mask = month_changes

        elif rebalance_period == 'quarterly':
            quarter_changes = signals.index.to_series().dt.quarter.diff() != 0
            quarter_changes.iloc[0] = True

            rebalance_dates = signals.index[quarter_changes]

            if offset > 0 and len(rebalance_dates) > 0:
                shifted_dates = []
                for date in rebalance_dates:
                    mask = signals.index > date
                    if mask.sum() >= offset:
                        shifted_dates.append(signals.index[mask][offset - 1])
                rebalance_mask = signals.index.isin(shifted_dates)
            else:
                rebalance_mask = quarter_changes

        rebalanced[~rebalance_mask] = False

        return rebalanced

    @staticmethod
    def equal_weight_allocation(
        selection: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate equal-weight allocations for selected assets.

        Args:
            selection: Boolean DataFrame where True = asset is selected

        Returns:
            DataFrame with weights (e.g., if 5 assets selected, each gets 0.2)
        """
        weights = selection.astype(float)

        n_selected = selection.sum(axis=1)
        n_selected = n_selected.replace(0, np.nan)

        for col in weights.columns:
            weights[col] = weights[col] / n_selected

        weights = weights.fillna(0.0)

        return weights
