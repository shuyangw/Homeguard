"""
Cross-Sectional Crypto Momentum (CSCM) Indicators.

Implements momentum ranking and regime filtering for crypto markets.

Strategy Logic:
- Weekly rebalance (Sunday 0:00 UTC)
- 14-day momentum lookback
- BTC regime filter: BTC > 20-day SMA = bullish (hold), else go to cash
- Top N coins by momentum ranking
- Equal weight allocation

Reference:
- Crypto markets exhibit stronger momentum than equities
- Shorter lookback (14 days) captures crypto's faster cycles
- Weekly rebalance balances turnover vs responsiveness
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CSCMSignal:
    """CSCM signal state for a single timestamp."""
    timestamp: pd.Timestamp
    regime: str  # 'bullish', 'bearish'
    btc_price: float
    btc_sma: float
    top_symbols: List[str]  # Ranked by momentum
    momentum_scores: Dict[str, float]  # symbol -> momentum
    is_rebalance_day: bool


class CSCMIndicators:
    """
    Cross-Sectional Crypto Momentum indicator calculations.

    Provides momentum ranking across crypto universe with BTC regime filter.
    Designed for weekly rebalancing on crypto's 24/7 markets.
    """

    @staticmethod
    def calculate_momentum(
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate momentum as return over lookback period.

        Args:
            prices: Price series (close prices)
            period: Lookback period in days (default 14)

        Returns:
            Momentum (return) series
        """
        return prices.pct_change(periods=period)

    @staticmethod
    def calculate_sma(
        prices: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            prices: Price series
            period: SMA period (default 20)

        Returns:
            SMA series
        """
        return prices.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_volatility(
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate annualized volatility.

        Args:
            prices: Price series
            period: Lookback period

        Returns:
            Annualized volatility series
        """
        returns = prices.pct_change()
        vol = returns.rolling(window=period, min_periods=period).std()
        return vol * np.sqrt(365)  # Annualize for 24/7 crypto

    @staticmethod
    def get_btc_regime(
        btc_prices: pd.Series,
        sma_period: int = 20
    ) -> pd.Series:
        """
        Calculate BTC regime filter.

        Args:
            btc_prices: BTC close prices
            sma_period: SMA period for regime detection

        Returns:
            Series with 'bullish' or 'bearish' values
        """
        sma = CSCMIndicators.calculate_sma(btc_prices, sma_period)
        regime = pd.Series(index=btc_prices.index, data='bearish')
        regime[btc_prices > sma] = 'bullish'
        return regime

    @staticmethod
    def calculate_all_momentum(
        data_dict: Dict[str, pd.DataFrame],
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate momentum for all symbols in universe.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data
            period: Momentum lookback period

        Returns:
            DataFrame with momentum values, indexed by date, columns are symbols
        """
        momentum_dict = {}
        for symbol, df in data_dict.items():
            if df is None or df.empty:
                continue
            if 'close' not in df.columns:
                logger.warning(f"No 'close' column for {symbol}")
                continue
            momentum_dict[symbol] = CSCMIndicators.calculate_momentum(df['close'], period)

        if not momentum_dict:
            return pd.DataFrame()

        return pd.DataFrame(momentum_dict)

    @staticmethod
    def rank_by_momentum(
        momentum_df: pd.DataFrame,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank symbols cross-sectionally by momentum.

        Args:
            momentum_df: DataFrame with momentum values (columns = symbols)
            ascending: If False, highest momentum gets rank 1

        Returns:
            DataFrame with ranks (1 = best)
        """
        # Rank across columns (symbols) for each row (date)
        return momentum_df.rank(axis=1, ascending=ascending, na_option='bottom')

    @staticmethod
    def select_top_n(
        momentum_df: pd.DataFrame,
        n: int = 5
    ) -> pd.DataFrame:
        """
        Select top N symbols by momentum at each date.

        Args:
            momentum_df: DataFrame with momentum values
            n: Number of top symbols to select

        Returns:
            DataFrame with True/False for selected symbols
        """
        ranks = CSCMIndicators.rank_by_momentum(momentum_df)
        return ranks <= n

    @staticmethod
    def is_weekly_rebalance_day(
        date: datetime,
        rebalance_day: str = 'sunday'
    ) -> bool:
        """
        Check if date is a rebalance day.

        Args:
            date: Date to check
            rebalance_day: Day of week for rebalancing

        Returns:
            True if it's a rebalance day
        """
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        target_day = day_map.get(rebalance_day.lower(), 6)  # Default Sunday
        return date.weekday() == target_day

    @staticmethod
    def get_rebalance_dates(
        date_index: pd.DatetimeIndex,
        rebalance_day: str = 'sunday'
    ) -> pd.DatetimeIndex:
        """
        Get all rebalance dates from a date range.

        Args:
            date_index: DatetimeIndex to filter
            rebalance_day: Day of week for rebalancing

        Returns:
            DatetimeIndex of rebalance days only
        """
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        target_day = day_map.get(rebalance_day.lower(), 6)
        mask = date_index.dayofweek == target_day
        return date_index[mask]

    @staticmethod
    def generate_signals(
        data_dict: Dict[str, pd.DataFrame],
        btc_data: pd.DataFrame,
        momentum_period: int = 14,
        btc_sma_period: int = 20,
        top_n: int = 5,
        rebalance_day: str = 'sunday',
        go_to_cash_in_bear: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Generate CSCM trading signals.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data
            btc_data: BTC price data for regime filter
            momentum_period: Lookback for momentum (default 14 days)
            btc_sma_period: SMA period for BTC regime (default 20)
            top_n: Number of top coins to hold
            rebalance_day: Day of week for rebalancing
            go_to_cash_in_bear: Exit all in bearish regime

        Returns:
            Tuple of:
            - selections: DataFrame with True/False for selected symbols
            - weights: DataFrame with position weights (0.0 to 1.0)
            - regime: Series with 'bullish'/'bearish' regime
        """
        # Calculate momentum for all symbols
        momentum_df = CSCMIndicators.calculate_all_momentum(data_dict, momentum_period)

        if momentum_df.empty:
            logger.warning("No momentum data available")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=str)

        # Get BTC regime
        btc_close = btc_data['close'] if 'close' in btc_data.columns else btc_data['Close']
        regime = CSCMIndicators.get_btc_regime(btc_close, btc_sma_period)

        # Align regime to momentum dates
        regime = regime.reindex(momentum_df.index, method='ffill')

        # Get rebalance dates
        rebalance_dates = CSCMIndicators.get_rebalance_dates(
            momentum_df.index, rebalance_day
        )

        # Initialize selections and weights
        selections = pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)
        weights = pd.DataFrame(0.0, index=momentum_df.index, columns=momentum_df.columns)

        # Track current holdings
        current_selection = set()

        for date in momentum_df.index:
            is_rebalance = date in rebalance_dates
            current_regime = regime.loc[date] if date in regime.index else 'bearish'

            if is_rebalance:
                if go_to_cash_in_bear and current_regime == 'bearish':
                    # Go to cash - no holdings
                    current_selection = set()
                else:
                    # Select top N by momentum
                    mom_row = momentum_df.loc[date].dropna()
                    if len(mom_row) > 0:
                        top_symbols = mom_row.nlargest(min(top_n, len(mom_row))).index.tolist()
                        current_selection = set(top_symbols)
                    else:
                        current_selection = set()

            # Apply current selection
            for sym in current_selection:
                if sym in selections.columns:
                    selections.loc[date, sym] = True
                    # Equal weight
                    weights.loc[date, sym] = 1.0 / top_n if current_selection else 0.0

        return selections, weights, regime

    @staticmethod
    def get_current_signal(
        data_dict: Dict[str, pd.DataFrame],
        btc_data: pd.DataFrame,
        momentum_period: int = 14,
        btc_sma_period: int = 20,
        top_n: int = 5
    ) -> CSCMSignal:
        """
        Get current CSCM signal state for live trading.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with recent OHLCV data
            btc_data: BTC price data for regime filter
            momentum_period: Lookback for momentum
            btc_sma_period: SMA period for BTC regime
            top_n: Number of top coins to select

        Returns:
            CSCMSignal with current state
        """
        # Get BTC regime
        btc_close = btc_data['close'] if 'close' in btc_data.columns else btc_data['Close']
        btc_sma = CSCMIndicators.calculate_sma(btc_close, btc_sma_period)
        current_btc = btc_close.iloc[-1]
        current_sma = btc_sma.iloc[-1]
        regime = 'bullish' if current_btc > current_sma else 'bearish'

        # Calculate momentum for all symbols
        momentum_scores = {}
        for symbol, df in data_dict.items():
            if df is None or df.empty or 'close' not in df.columns:
                continue
            mom = CSCMIndicators.calculate_momentum(df['close'], momentum_period)
            if len(mom) > 0 and not pd.isna(mom.iloc[-1]):
                momentum_scores[symbol] = mom.iloc[-1]

        # Rank and select top N
        sorted_symbols = sorted(
            momentum_scores.keys(),
            key=lambda x: momentum_scores[x],
            reverse=True
        )
        top_symbols = sorted_symbols[:top_n]

        # Determine if today is rebalance day
        current_date = btc_close.index[-1]
        is_rebalance = CSCMIndicators.is_weekly_rebalance_day(current_date)

        return CSCMSignal(
            timestamp=current_date,
            regime=regime,
            btc_price=current_btc,
            btc_sma=current_sma,
            top_symbols=top_symbols,
            momentum_scores=momentum_scores,
            is_rebalance_day=is_rebalance
        )

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Sharpe ratio for a return series.

        Args:
            returns: Daily returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (365 for crypto)

        Returns:
            Annualized Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        excess_return = returns.mean() - (risk_free_rate / periods_per_year)
        sharpe = excess_return / returns.std() * np.sqrt(periods_per_year)
        return sharpe

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            cumulative_returns: Cumulative return series (1.0 = starting value)

        Returns:
            Maximum drawdown as positive decimal (e.g., 0.20 = 20% drawdown)
        """
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown.min())
