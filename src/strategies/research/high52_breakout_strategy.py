"""
52-Week High Monthly Breakout Strategy.

This strategy selects stocks trading near their 52-week highs, based on
research showing these stocks tend to outperform due to momentum and
anchoring bias effects.

Strategy Overview:
1. Universe: S&P 500 stocks
2. Selection: Top N stocks closest to 52-week highs
3. Rebalance: MONTHLY (first trading day of each month)
4. Position sizing: Equal weight (e.g., 10% each for top 10)

Research Foundation:
- George and Hwang (2004): "The 52-Week High and Momentum Investing"
- Monthly rebalancing is CRITICAL - weekly loses money due to churn

Key Insight:
- Weekly rebalancing: CAGR ~0%, MaxDD -39%
- Monthly rebalancing: Outperforms S&P 500, MaxDD ~half of buy-and-hold
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class High52Signal:
    """Signal for a single stock."""
    symbol: str
    distance_to_high: float  # 0.0 = at high, 0.05 = 5% below
    high_52w: float
    current_price: float
    rank: int
    weight: float
    action: str  # 'buy', 'hold', 'sell'


class High52BreakoutSignals:
    """
    Pure signal generator for 52-week high breakout strategy.

    Can be used standalone or injected into live adapter.
    """

    def __init__(
        self,
        symbols: List[str],
        top_n: int = 10,
        max_distance_pct: float = 0.05,
        lookback_days: int = 252
    ):
        """
        Initialize signal generator.

        Args:
            symbols: List of symbols to trade
            top_n: Number of top stocks to hold
            max_distance_pct: Maximum distance from 52-week high (0.05 = 5%)
            lookback_days: Days for 52-week high calculation (252 = 1 year)
        """
        self.symbols = symbols
        self.top_n = top_n
        self.max_distance_pct = max_distance_pct
        self.lookback_days = lookback_days

        # Cache for historical data
        self._prices_cache: Optional[pd.DataFrame] = None

        logger.info("[H52] Initialized 52-Week High Breakout Signals")
        logger.info(f"[H52]   Universe: {len(symbols)} symbols")
        logger.info(f"[H52]   Top N: {top_n}")
        logger.info(f"[H52]   Max distance: {max_distance_pct:.0%}")
        logger.info(f"[H52]   Lookback: {lookback_days} days")

    def update_historical_data(self, prices_df: pd.DataFrame):
        """
        Update cached historical data.

        Args:
            prices_df: DataFrame with stock prices (columns=symbols, index=dates)
        """
        self._prices_cache = prices_df
        logger.debug(f"[H52] Updated historical cache: {len(prices_df)} days")

    def calculate_52_week_high(
        self,
        prices_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate 52-week high for all symbols.

        Args:
            prices_df: Optional prices DataFrame. Uses cache if not provided.

        Returns:
            Series of 52-week high prices indexed by symbol
        """
        if prices_df is None:
            prices_df = self._prices_cache

        if prices_df is None or len(prices_df) < self.lookback_days:
            logger.warning("[H52] Insufficient price history for 52-week high")
            return pd.Series(dtype=float)

        # Calculate rolling 252-day high
        high_52w = prices_df.rolling(window=self.lookback_days).max()

        # Get latest 52-week highs
        return high_52w.iloc[-1].dropna()

    def calculate_distance_to_high(
        self,
        prices_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate distance from current price to 52-week high for all symbols.

        Formula: distance = (high_52w - current) / high_52w

        Values:
        - 0.00 = at 52-week high
        - 0.05 = 5% below high
        - 0.10 = 10% below high

        Args:
            prices_df: Optional prices DataFrame. Uses cache if not provided.

        Returns:
            Series of distance values indexed by symbol (lower = closer to high)
        """
        if prices_df is None:
            prices_df = self._prices_cache

        if prices_df is None or len(prices_df) < self.lookback_days:
            logger.warning("[H52] Insufficient price history for distance calculation")
            return pd.Series(dtype=float)

        # Calculate 52-week high
        high_52w = prices_df.rolling(window=self.lookback_days).max()

        # Current prices (latest row)
        current_prices = prices_df.iloc[-1]

        # Distance = (high - current) / high
        distance = (high_52w.iloc[-1] - current_prices) / high_52w.iloc[-1]

        # Drop NaN values
        distance = distance.dropna()

        return distance

    def generate_signals(
        self,
        current_positions: Dict[str, float],
        prices_df: Optional[pd.DataFrame] = None
    ) -> List[High52Signal]:
        """
        Generate trading signals.

        Selects top N stocks closest to their 52-week highs.

        Args:
            current_positions: Dict of current holdings {symbol: shares}
            prices_df: Optional prices DataFrame. Uses cache if not provided.

        Returns:
            List of High52Signal objects with buy/sell/hold actions
        """
        if prices_df is None:
            prices_df = self._prices_cache

        if prices_df is None or len(prices_df) < self.lookback_days:
            logger.warning("[H52] Insufficient data for signal generation")
            return []

        # Calculate distances to 52-week high
        distances = self.calculate_distance_to_high(prices_df)

        if len(distances) == 0:
            logger.warning("[H52] No valid distance calculations")
            return []

        # Optional: Filter to only stocks within max_distance_pct of high
        if self.max_distance_pct > 0:
            distances = distances[distances <= self.max_distance_pct]

        if len(distances) < self.top_n:
            logger.warning(f"[H52] Only {len(distances)} stocks within {self.max_distance_pct:.0%} of high")

        # Rank by distance (ascending - closest to high first)
        distances_sorted = distances.sort_values(ascending=True)

        # Select top N
        top_stocks = distances_sorted.head(self.top_n)

        # Get 52-week highs and current prices
        high_52w = self.calculate_52_week_high(prices_df)
        current_prices = prices_df.iloc[-1]

        # Calculate equal weight
        weight = 1.0 / self.top_n

        # Current holdings set
        current_set = set(current_positions.keys())
        target_set = set(top_stocks.index)

        # Generate signals
        signals = []

        for rank, (symbol, distance) in enumerate(top_stocks.items(), 1):
            if symbol in current_set:
                action = 'hold'
            else:
                action = 'buy'

            signals.append(High52Signal(
                symbol=symbol,
                distance_to_high=distance,
                high_52w=high_52w.get(symbol, 0),
                current_price=current_prices.get(symbol, 0),
                rank=rank,
                weight=weight,
                action=action
            ))

        # Add sell signals for positions not in top N
        for symbol in current_set - target_set:
            signals.append(High52Signal(
                symbol=symbol,
                distance_to_high=distances.get(symbol, 1.0),
                high_52w=high_52w.get(symbol, 0),
                current_price=current_prices.get(symbol, 0),
                rank=0,
                weight=0,
                action='sell'
            ))

        return signals


class High52BreakoutStrategy:
    """
    52-Week High Monthly Breakout Strategy.

    Research-backed implementation:
    - Monthly rebalancing (weekly loses money due to churn)
    - Top 10 stocks closest to 52-week highs
    - Equal weight allocation (10% each)

    Walk-Forward Target:
    - Sharpe degradation < 30%
    - Beat SPY in > 50% of years
    """

    def __init__(
        self,
        symbols: List[str],
        top_n: int = 10,
        position_size_pct: float = 0.10,
        max_distance_pct: float = 0.05,
        lookback_days: int = 252
    ):
        """
        Initialize strategy.

        Args:
            symbols: List of symbols to trade
            top_n: Number of positions to hold
            position_size_pct: Position size per stock (default 10%)
            max_distance_pct: Max distance from 52-week high (default 5%)
            lookback_days: Days for 52-week high calculation
        """
        self.symbols = symbols
        self.top_n = top_n
        self.position_size_pct = position_size_pct
        self.max_distance_pct = max_distance_pct
        self.lookback_days = lookback_days

        # Initialize signal generator
        self.signal_generator = High52BreakoutSignals(
            symbols=symbols,
            top_n=top_n,
            max_distance_pct=max_distance_pct,
            lookback_days=lookback_days
        )

        # Track current positions
        self.positions: Dict[str, float] = {}

        # Track last rebalance date
        self._last_rebalance_month: Optional[int] = None

        logger.info("[H52] Initialized 52-Week High Breakout Strategy")
        logger.info(f"[H52]   Position size: {position_size_pct:.0%}")

    def is_rebalance_day(self, date: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> bool:
        """
        Check if date is the first trading day of the month.

        CRITICAL: Monthly rebalancing only.

        Args:
            date: Current date
            trading_dates: Index of all trading dates

        Returns:
            True if first trading day of month
        """
        current_month = date.month
        current_year = date.year

        # Skip if already rebalanced this month
        if self._last_rebalance_month == (current_year, current_month):
            return False

        # Check if this is the first trading day of the month
        month_dates = trading_dates[(trading_dates.month == current_month) &
                                     (trading_dates.year == current_year)]

        if len(month_dates) == 0:
            return False

        first_trading_day = month_dates[0]

        return date == first_trading_day

    def update_data(self, prices_df: pd.DataFrame):
        """Update historical data for signal generation."""
        self.signal_generator.update_historical_data(prices_df)

    def generate_signals(self, date: pd.Timestamp) -> List[High52Signal]:
        """
        Generate trading signals.

        Args:
            date: Current date

        Returns:
            List of High52Signal objects
        """
        signals = self.signal_generator.generate_signals(
            current_positions=self.positions
        )

        # Update last rebalance month
        self._last_rebalance_month = (date.year, date.month)

        return signals

    def get_target_positions(
        self,
        portfolio_value: float,
        signals: List[High52Signal]
    ) -> Dict[str, int]:
        """
        Calculate target position sizes (shares) based on signals.

        Args:
            portfolio_value: Current portfolio value in dollars
            signals: List of signals from generate_signals()

        Returns:
            Dict of target positions {symbol: shares}
        """
        targets = {}

        for signal in signals:
            if signal.action in ('buy', 'hold'):
                # Calculate dollar amount
                dollar_amount = portfolio_value * signal.weight

                # Calculate shares (floor to avoid fractional)
                if signal.current_price > 0:
                    shares = int(dollar_amount / signal.current_price)
                    if shares > 0:
                        targets[signal.symbol] = shares

        return targets

    def update_positions(self, new_positions: Dict[str, float]):
        """Update current positions after execution."""
        self.positions = new_positions.copy()
