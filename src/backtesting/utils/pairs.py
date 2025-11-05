"""
Pairs trading and statistical arbitrage utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict, List


class PairsUtils:
    """
    Utilities for pairs trading and statistical arbitrage.
    """

    @staticmethod
    def test_cointegration(
        series1: pd.Series,
        series2: pd.Series,
        significance_level: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Test if two price series are cointegrated using Augmented Dickey-Fuller test.

        Args:
            series1: First price series
            series2: Second price series
            significance_level: P-value threshold for cointegration (default: 0.05)

        Returns:
            Tuple of (is_cointegrated, p_value, test_statistic)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.regression.linear_model import OLS
        except ImportError:
            raise ImportError("statsmodels is required for cointegration testing. Install with: pip install statsmodels")

        if len(series1) != len(series2):
            raise ValueError("Series must have same length")

        aligned_s1 = series1.dropna()
        aligned_s2 = series2.dropna()

        common_index = aligned_s1.index.intersection(aligned_s2.index)
        s1 = aligned_s1.loc[common_index]
        s2 = aligned_s2.loc[common_index]

        if len(s1) < 30:
            raise ValueError("Need at least 30 observations for cointegration test")

        s1_values = s1.values.reshape(-1, 1)
        s2_values = s2.values

        model = OLS(s2_values, s1_values).fit()
        spread = s2_values - model.predict(s1_values)

        adf_result = adfuller(spread, maxlag=1, regression='c', autolag=None)

        test_statistic = adf_result[0]
        p_value = adf_result[1]
        is_cointegrated = p_value < significance_level

        return is_cointegrated, p_value, test_statistic

    @staticmethod
    def calculate_hedge_ratio(
        series1: pd.Series,
        series2: pd.Series,
        method: str = 'ols',
        window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate hedge ratio between two price series.

        Args:
            series1: First price series (independent variable)
            series2: Second price series (dependent variable)
            method: Method for calculating hedge ratio ('ols' or 'rolling')
            window: Window size for rolling hedge ratio (required if method='rolling')

        Returns:
            Hedge ratio as float (if method='ols') or Series (if method='rolling')
        """
        try:
            from statsmodels.regression.linear_model import OLS
        except ImportError:
            raise ImportError("statsmodels is required for hedge ratio calculation. Install with: pip install statsmodels")

        aligned_s1 = series1.dropna()
        aligned_s2 = series2.dropna()

        common_index = aligned_s1.index.intersection(aligned_s2.index)
        s1 = aligned_s1.loc[common_index]
        s2 = aligned_s2.loc[common_index]

        if method == 'ols':
            s1_values = s1.values.reshape(-1, 1)
            s2_values = s2.values

            model = OLS(s2_values, s1_values).fit()
            hedge_ratio = model.params[0]

            return hedge_ratio

        elif method == 'rolling':
            if window is None:
                raise ValueError("window parameter required for rolling hedge ratio")

            hedge_ratios = pd.Series(index=s1.index, dtype=float)

            for i in range(window, len(s1) + 1):
                s1_window = s1.iloc[i - window:i].values.reshape(-1, 1)
                s2_window = s2.iloc[i - window:i].values

                model = OLS(s2_window, s1_window).fit()
                hedge_ratios.iloc[i - 1] = model.params[0]

            return hedge_ratios

        else:
            raise ValueError(f"Unknown method: {method}. Use 'ols' or 'rolling'")

    @staticmethod
    def calculate_spread(
        series1: pd.Series,
        series2: pd.Series,
        hedge_ratio: Union[float, pd.Series]
    ) -> pd.Series:
        """
        Calculate spread between two price series using hedge ratio.

        Spread = series2 - hedge_ratio * series1

        Args:
            series1: First price series
            series2: Second price series
            hedge_ratio: Hedge ratio as float or Series

        Returns:
            Spread series
        """
        aligned_s1 = series1.dropna()
        aligned_s2 = series2.dropna()

        common_index = aligned_s1.index.intersection(aligned_s2.index)
        s1 = aligned_s1.loc[common_index]
        s2 = aligned_s2.loc[common_index]

        if isinstance(hedge_ratio, pd.Series):
            common_index = common_index.intersection(hedge_ratio.index)
            s1 = s1.loc[common_index]
            s2 = s2.loc[common_index]
            hr = hedge_ratio.loc[common_index]
            spread = s2 - hr * s1
        else:
            spread = s2 - hedge_ratio * s1

        return spread

    @staticmethod
    def spread_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Z-score of spread.

        Args:
            spread: Spread series
            window: Rolling window for mean and std (default: 20)

        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        zscore = (spread - rolling_mean) / rolling_std

        return zscore

    @staticmethod
    def generate_pairs_signals(
        spread_zscore: pd.Series,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: Optional[float] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate entry and exit signals for pairs trading based on spread Z-score.

        Logic:
        - Long spread (short asset1, long asset2) when zscore < -entry_threshold
        - Short spread (long asset1, short asset2) when zscore > +entry_threshold
        - Exit when zscore crosses back to exit_threshold
        - Stop loss when zscore exceeds stop_loss_threshold

        Args:
            spread_zscore: Z-score of spread
            entry_threshold: Z-score threshold for entry (default: 2.0)
            exit_threshold: Z-score threshold for exit (default: 0.5)
            stop_loss_threshold: Z-score threshold for stop loss (default: None)

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        long_entries = spread_zscore < -entry_threshold
        short_entries = spread_zscore > entry_threshold

        long_exits = spread_zscore > -exit_threshold
        short_exits = spread_zscore < exit_threshold

        if stop_loss_threshold is not None:
            long_stop_loss = spread_zscore > stop_loss_threshold
            short_stop_loss = spread_zscore < -stop_loss_threshold

            long_exits = long_exits | long_stop_loss
            short_exits = short_exits | short_stop_loss

        long_entries = long_entries.fillna(False)
        long_exits = long_exits.fillna(False)
        short_entries = short_entries.fillna(False)
        short_exits = short_exits.fillna(False)

        return long_entries, long_exits, short_entries, short_exits

    @staticmethod
    def find_cointegrated_pairs(
        data_dict: Dict[str, pd.DataFrame],
        price_column: str = 'close',
        significance_level: float = 0.05,
        min_observations: int = 100
    ) -> List[Tuple[str, str, float]]:
        """
        Find all cointegrated pairs from a universe of assets.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with price data
            price_column: Price column to use (default: 'close')
            significance_level: P-value threshold (default: 0.05)
            min_observations: Minimum number of observations required (default: 100)

        Returns:
            List of tuples (symbol1, symbol2, p_value) for cointegrated pairs
        """
        symbols = list(data_dict.keys())
        cointegrated_pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]

                df1 = data_dict[sym1]
                df2 = data_dict[sym2]

                if price_column not in df1.columns or price_column not in df2.columns:
                    continue

                s1 = df1[price_column]
                s2 = df2[price_column]

                common_index = s1.dropna().index.intersection(s2.dropna().index)

                if len(common_index) < min_observations:
                    continue

                try:
                    is_coint, p_value, _ = PairsUtils.test_cointegration(
                        s1, s2, significance_level
                    )

                    if is_coint:
                        cointegrated_pairs.append((sym1, sym2, p_value))

                except (ValueError, Exception):
                    continue

        cointegrated_pairs.sort(key=lambda x: x[2])

        return cointegrated_pairs
