"""
Cross-sectional momentum strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from src.backtesting.base.strategy import BaseStrategy
from src.backtesting.utils.indicators import Indicators
from src.backtesting.utils.ranking import Ranking
from src.backtesting.utils.validation import validate_positive_int, validate_positive_float


class CrossSectionalMomentum(BaseStrategy):
    """
    Rank stocks by momentum cross-sectionally, long top performers.

    This strategy:
    1. Calculates momentum scores for each stock (weighted average of 3m, 6m, 12m returns)
    2. Ranks stocks cross-sectionally at each rebalance period
    3. Goes long top N% or top N stocks
    4. Rebalances on fixed schedule (weekly/monthly)

    IMPORTANT: This strategy requires multi-symbol backtesting.
    Use with BacktestEngine.run(symbols=[...]) with multiple symbols.

    Backtesting considerations:
    - Watch for survivorship bias: Include delisted stocks if possible
    - Watch for lookahead bias: Ensure rankings use only data available at rebalance time
    - Rebalancing costs can be significant with monthly rebalancing
    """

    def __init__(
        self,
        lookback_periods: List[int] = None,
        weights: List[float] = None,
        rebalance_period: str = 'monthly',
        top_percentile: float = 0.2,
        top_n: Optional[int] = None,
        equal_weight: bool = True
    ):
        """
        Initialize strategy.

        Args:
            lookback_periods: Lookback periods for momentum (default: [63, 126, 252] = 3m, 6m, 12m)
            weights: Weights for each lookback period (default: [0.3, 0.3, 0.4])
            rebalance_period: Rebalancing frequency ('weekly', 'monthly', 'quarterly')
            top_percentile: Top percentile to select (default: 0.2 = top 20%)
            top_n: Alternative to percentile - select top N stocks
            equal_weight: Use equal weighting vs momentum weighting (default: True)
        """
        if lookback_periods is None:
            lookback_periods = [63, 126, 252]

        if weights is None:
            weights = [0.3, 0.3, 0.4]

        super().__init__(
            lookback_periods=lookback_periods,
            weights=weights,
            rebalance_period=rebalance_period,
            top_percentile=top_percentile,
            top_n=top_n,
            equal_weight=equal_weight
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        lookback_periods = self.params['lookback_periods']
        weights = self.params['weights']

        if len(lookback_periods) != len(weights):
            raise ValueError("lookback_periods and weights must have same length")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")

        for period in lookback_periods:
            validate_positive_int(period, 'lookback_period')

        if self.params['rebalance_period'] not in ['daily', 'weekly', 'monthly', 'quarterly']:
            raise ValueError("rebalance_period must be 'daily', 'weekly', 'monthly', or 'quarterly'")

        if self.params['top_percentile'] is not None:
            if not (0 < self.params['top_percentile'] <= 1.0):
                raise ValueError("top_percentile must be between 0 and 1.0")

        if self.params['top_n'] is not None:
            validate_positive_int(self.params['top_n'], 'top_n')

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals for SINGLE symbol.

        NOTE: This is a simplified implementation for single-symbol backtests.
        For true cross-sectional momentum, use generate_multi_symbol_signals()
        with multiple symbols.

        For single symbol, this just checks if momentum is positive.
        """
        close = data['close']

        momentum_components = []
        for period in self.params['lookback_periods']:
            returns = Indicators.returns(close, period)
            momentum_components.append(returns)

        weighted_momentum = pd.Series(0.0, index=close.index)
        for component, weight in zip(momentum_components, self.params['weights']):
            weighted_momentum += component * weight

        entries = weighted_momentum > 0

        rebalanced_entries = Ranking.rebalance_on_schedule(
            pd.DataFrame({'signal': entries}),
            rebalance_period=self.params['rebalance_period']
        )['signal']

        exits = ~rebalanced_entries

        entries = rebalanced_entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

    def generate_multi_symbol_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """
        Generate cross-sectional signals for multiple symbols.

        This is the proper implementation for cross-sectional momentum.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data

        Returns:
            Dictionary of {symbol: (entries, exits)} signals
        """
        momentum_df = Ranking.calculate_momentum_score(
            data_dict,
            lookback_periods=self.params['lookback_periods'],
            weights=self.params['weights'],
            price_column='close'
        )

        ranks = Ranking.cross_sectional_rank(
            {sym: df for sym, df in data_dict.items()},
            metric='close',
            ascending=False
        )

        selection = Ranking.percentile_filter(
            ranks,
            top_pct=self.params['top_percentile'],
            top_n=self.params['top_n']
        )

        long_selection = selection['long']

        rebalanced_selection = Ranking.rebalance_on_schedule(
            long_selection,
            rebalance_period=self.params['rebalance_period']
        )

        signals_dict = {}
        for symbol in data_dict.keys():
            if symbol in rebalanced_selection.columns:
                entries = rebalanced_selection[symbol]
                exits = ~entries

                entries = entries.fillna(False)
                exits = exits.fillna(False)

                signals_dict[symbol] = (entries, exits)

        return signals_dict
