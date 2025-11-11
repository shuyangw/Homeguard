"""
Pairs trading / statistical arbitrage strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List

from backtesting.base.pairs_strategy import PairsStrategy
from backtesting.utils.pairs import PairsUtils
from backtesting.utils.indicators import Indicators
from backtesting.utils.validation import validate_positive_int, validate_positive_float


class PairsTrading(PairsStrategy):
    """
    Statistical arbitrage via cointegrated pairs.

    This strategy:
    1. Tests for cointegration between two price series
    2. Calculates hedge ratio and spread
    3. Generates signals when spread Z-score exceeds thresholds
    4. Trades mean reversion of the spread

    Position logic:
    - Long spread: Short asset1, Long asset2 (when spread is too low)
    - Short spread: Long asset1, Short asset2 (when spread is too high)

    IMPORTANT: This strategy requires two symbols and trades them simultaneously.
    Use with BacktestEngine.run(symbols=['SYM1', 'SYM2'])

    Backtesting considerations:
    - Selection bias: Don't backtest only pairs that worked historically
    - Retraining leakage: Don't use future data for pair selection or hedge ratio
    - Regime changes: Cointegration can break down over time
    - Transaction costs: Pairs trading involves more trades than directional strategies
    """

    def __init__(
        self,
        pair_selection_window: int = 252,
        cointegration_pvalue: float = 0.05,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: Optional[float] = 3.5,
        hedge_ratio_method: str = 'ols',
        rolling_hedge_window: Optional[int] = None,
        zscore_window: int = 20
    ):
        """
        Initialize strategy.

        Args:
            pair_selection_window: Window for initial cointegration test (default: 252 days)
            cointegration_pvalue: P-value threshold for cointegration (default: 0.05)
            entry_zscore: Z-score threshold for entry (default: 2.0)
            exit_zscore: Z-score threshold for exit (default: 0.5)
            stop_loss_zscore: Z-score threshold for stop loss (default: 3.5, None to disable)
            hedge_ratio_method: Method for hedge ratio ('ols' or 'rolling')
            rolling_hedge_window: Window for rolling hedge ratio (required if method='rolling')
            zscore_window: Window for spread Z-score calculation (default: 20)
        """
        super().__init__(
            pair_selection_window=pair_selection_window,
            cointegration_pvalue=cointegration_pvalue,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_loss_zscore=stop_loss_zscore,
            hedge_ratio_method=hedge_ratio_method,
            rolling_hedge_window=rolling_hedge_window,
            zscore_window=zscore_window
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['pair_selection_window'], 'pair_selection_window')
        validate_positive_float(self.params['cointegration_pvalue'], 'cointegration_pvalue')
        validate_positive_float(self.params['entry_zscore'], 'entry_zscore')
        validate_positive_float(self.params['exit_zscore'], 'exit_zscore')
        validate_positive_int(self.params['zscore_window'], 'zscore_window')

        if self.params['entry_zscore'] <= self.params['exit_zscore']:
            raise ValueError("entry_zscore must be greater than exit_zscore")

        if self.params['stop_loss_zscore'] is not None:
            validate_positive_float(self.params['stop_loss_zscore'], 'stop_loss_zscore')
            if self.params['stop_loss_zscore'] <= self.params['entry_zscore']:
                raise ValueError("stop_loss_zscore must be greater than entry_zscore")

        if self.params['hedge_ratio_method'] not in ['ols', 'rolling']:
            raise ValueError("hedge_ratio_method must be 'ols' or 'rolling'")

        if self.params['hedge_ratio_method'] == 'rolling':
            if self.params['rolling_hedge_window'] is None:
                raise ValueError("rolling_hedge_window required when hedge_ratio_method='rolling'")
            validate_positive_int(self.params['rolling_hedge_window'], 'rolling_hedge_window')

    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[bool, float]:
        """
        Test if two series are cointegrated.

        Args:
            series1: First price series
            series2: Second price series

        Returns:
            Tuple of (is_cointegrated, p_value)
        """
        is_coint, p_value, _ = PairsUtils.test_cointegration(
            series1,
            series2,
            significance_level=self.params['cointegration_pvalue']
        )

        return is_coint, p_value

    def generate_pairs_signals(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol1: str = 'asset1',
        symbol2: str = 'asset2'
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """
        Generate trading signals for a pair of assets.

        Args:
            data1: DataFrame for first asset (OHLCV)
            data2: DataFrame for second asset (OHLCV)
            symbol1: Name of first asset (for return dict)
            symbol2: Name of second asset (for return dict)

        Returns:
            Dictionary with keys (symbol1, symbol2) and values:
            (long_entries, long_exits, short_entries, short_exits)

        Note: For spread trading:
        - When we go "long spread": short asset1, long asset2
        - When we go "short spread": long asset1, short asset2
        """
        close1 = data1['close']
        close2 = data2['close']

        is_coint, p_value = self.test_cointegration(close1, close2)

        if not is_coint:
            empty_series = pd.Series(False, index=close1.index)
            return {
                symbol1: (empty_series.copy(), empty_series.copy(), empty_series.copy(), empty_series.copy()),
                symbol2: (empty_series.copy(), empty_series.copy(), empty_series.copy(), empty_series.copy())
            }

        hedge_ratio = PairsUtils.calculate_hedge_ratio(
            close1,
            close2,
            method=self.params['hedge_ratio_method'],
            window=self.params.get('rolling_hedge_window')
        )

        spread = PairsUtils.calculate_spread(close1, close2, hedge_ratio)

        spread_zscore = PairsUtils.spread_zscore(spread, window=self.params['zscore_window'])

        long_spread_entries, long_spread_exits, short_spread_entries, short_spread_exits = \
            PairsUtils.generate_pairs_signals(
                spread_zscore,
                entry_threshold=self.params['entry_zscore'],
                exit_threshold=self.params['exit_zscore'],
                stop_loss_threshold=self.params['stop_loss_zscore']
            )

        asset1_long_entries = short_spread_entries
        asset1_long_exits = short_spread_exits
        asset1_short_entries = long_spread_entries
        asset1_short_exits = long_spread_exits

        asset2_long_entries = long_spread_entries
        asset2_long_exits = long_spread_exits
        asset2_short_entries = short_spread_entries
        asset2_short_exits = short_spread_exits

        return {
            symbol1: (asset1_long_entries, asset1_long_exits, asset1_short_entries, asset1_short_exits),
            symbol2: (asset2_long_entries, asset2_long_exits, asset2_short_entries, asset2_short_exits)
        }

    def find_best_pairs(
        self,
        data_dict: Dict[str, pd.DataFrame],
        top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Find the best cointegrated pairs from a universe of assets.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with price data
            top_n: Number of top pairs to return (default: 5)

        Returns:
            List of tuples (symbol1, symbol2, p_value) for top N cointegrated pairs
        """
        pairs = PairsUtils.find_cointegrated_pairs(
            data_dict,
            price_column='close',
            significance_level=self.params['cointegration_pvalue'],
            min_observations=self.params['pair_selection_window']
        )

        return pairs[:top_n]
