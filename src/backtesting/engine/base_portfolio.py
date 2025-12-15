"""
Base portfolio class with common functionality.

This module provides the BasePortfolio class that contains shared logic
used by Portfolio, MultiAssetPortfolio, and PairsPortfolio.
"""

import pandas as pd
import pytz
from abc import ABC, abstractmethod
from datetime import time
from typing import Optional, Dict, Any


class BasePortfolio(ABC):
    """
    Abstract base class for portfolio simulators.

    Provides common functionality:
    - Market hours checking
    - Period annualization helpers
    - Standard attribute initialization

    Subclasses must implement:
    - _simulate(): Run the portfolio simulation
    - stats(): Calculate performance statistics
    - returns(): Get returns series
    """

    # Default market hours (EST/EDT)
    DEFAULT_MARKET_OPEN = time(9, 35)   # 9:35 AM EST
    DEFAULT_MARKET_CLOSE = time(15, 55)  # 3:55 PM EST

    # Frequency to periods-per-year mapping
    FREQ_TO_PERIODS = {
        '1min': 252 * 6.5 * 60,   # 252 days * 6.5 hours * 60 minutes
        '1h': 252 * 6.5,          # 252 days * 6.5 hours
        '1d': 252,
        'D': 252,
        'daily': 252,
        '1D': 252,
    }

    def __init__(
        self,
        init_cash: float,
        fees: float,
        slippage: float = 0.0,
        freq: str = '1min',
        market_hours_only: bool = True,
        market_open: Optional[time] = None,
        market_close: Optional[time] = None
    ):
        """
        Initialize base portfolio attributes.

        Args:
            init_cash: Initial capital
            fees: Trading fees as decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as decimal
            freq: Data frequency ('1min', '1h', '1d', etc.)
            market_hours_only: If True, only trade during market hours
            market_open: Override market open time (default: 9:35 AM EST)
            market_close: Override market close time (default: 3:55 PM EST)
        """
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.market_hours_only = market_hours_only

        # Market hours configuration
        self.market_open = market_open or self.DEFAULT_MARKET_OPEN
        self.market_close = market_close or self.DEFAULT_MARKET_CLOSE
        self.eastern_tz = pytz.timezone('US/Eastern')

        # Common state
        self.equity_curve: Optional[pd.Series] = None
        self._stats: Optional[Dict[str, Any]] = None

    def _is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if the given timestamp is within market trading hours.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if within market hours, False otherwise
        """
        if not self.market_hours_only:
            return True

        # Convert timestamp to Eastern time if it has timezone info
        if timestamp.tz is None:
            # Assume the timestamp is already in Eastern time if no timezone
            eastern_time = timestamp
        else:
            # Convert to Eastern timezone
            eastern_time = timestamp.tz_convert(self.eastern_tz)

        # Check if it's a weekday (Monday=0, Sunday=6)
        if eastern_time.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if within trading hours
        current_time = eastern_time.time()
        return self.market_open <= current_time <= self.market_close

    def _get_periods_per_year(self) -> int:
        """
        Get number of periods per year based on data frequency.

        Returns:
            Number of periods per year for annualization
        """
        return self.FREQ_TO_PERIODS.get(self.freq, 252)

    def _calculate_basic_metrics(
        self,
        equity_series: pd.Series,
        closed_trades: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Calculate basic performance metrics common to all portfolios.

        Args:
            equity_series: Series of equity values over time
            closed_trades: Optional list of closed trade dicts with 'pnl' key

        Returns:
            Dictionary of basic metrics
        """
        if equity_series is None or len(equity_series) == 0:
            return {}

        # Total return
        total_return_pct = ((equity_series.iloc[-1] - self.init_cash) / self.init_cash) * 100

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0:
            return {
                'Total Return [%]': total_return_pct,
                'Start Value': self.init_cash,
                'End Value': equity_series.iloc[-1],
            }

        # Mean and std of returns
        mean_return = returns.mean()
        std_return = returns.std()

        # Annualization
        periods_per_year = self._get_periods_per_year()

        # Sharpe ratio
        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * (periods_per_year ** 0.5)
        else:
            sharpe_ratio = 0

        # Max drawdown
        cummax = equity_series.cummax()
        # Prevent division by zero
        cummax = cummax.replace(0, 1)
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()

        metrics = {
            'Total Return [%]': total_return_pct,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown [%]': max_drawdown_pct,
            'Start Value': self.init_cash,
            'End Value': equity_series.iloc[-1],
        }

        # Win rate if trades provided
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            total_trades = len(closed_trades)
            win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            metrics['Win Rate [%]'] = win_rate_pct
            metrics['Total Trades'] = total_trades

        return metrics

    @abstractmethod
    def _simulate(self):
        """
        Run the portfolio simulation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def stats(self) -> Optional[Dict[str, Any]]:
        """
        Calculate portfolio statistics.

        Must be implemented by subclasses.

        Returns:
            Dictionary of performance metrics
        """
        pass

    @abstractmethod
    def returns(self, freq: Optional[str] = None) -> pd.Series:
        """
        Get returns series.

        Args:
            freq: Optional frequency for resampling

        Returns:
            pd.Series with returns indexed by timestamp
        """
        pass

    def plot(self, **kwargs):
        """
        Plot equity curve (basic implementation).

        Subclasses can override for more sophisticated plots.
        """
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve.index, self.equity_curve.values)
            plt.title('Portfolio Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            return plt.gcf()
        return None
