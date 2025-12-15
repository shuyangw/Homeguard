"""
Rolling window backtest results container.

Holds per-window results and provides methods to stitch them together
into a continuous equity curve with combined statistics.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class RollingWindowResults:
    """
    Results from a rolling window backtest.

    Stores per-window portfolios and provides methods to:
    - Stitch equity curves into continuous series
    - Compute combined statistics across all windows
    - Generate window-by-window summary reports

    Attributes:
        windows: List of (start_date, end_date) tuples for each window
        window_portfolios: List of Portfolio objects, one per window
        window_stats: List of statistics dicts, one per window
        combined_equity: Stitched equity curve across all windows
        combined_trades: All trades from all windows
        combined_stats: Overall statistics
    """

    windows: List[Tuple[str, str]] = field(default_factory=list)
    window_portfolios: List[Any] = field(default_factory=list)  # List[Portfolio]
    window_stats: List[Dict[str, float]] = field(default_factory=list)

    # Combined results (populated by stitch/compute methods)
    combined_equity: Optional[pd.Series] = field(default=None)
    combined_trades: List[Dict] = field(default_factory=list)
    combined_stats: Dict[str, float] = field(default_factory=dict)

    def add_window(
        self,
        start_date: str,
        end_date: str,
        portfolio: Any,
        stats: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add a window's results.

        Args:
            start_date: Window start date (YYYY-MM-DD)
            end_date: Window end date (YYYY-MM-DD)
            portfolio: Portfolio object for this window
            stats: Optional pre-computed stats (computed if not provided)
        """
        self.windows.append((start_date, end_date))
        self.window_portfolios.append(portfolio)

        if stats is None:
            stats = self._compute_window_stats(portfolio)
        self.window_stats.append(stats)

        # Collect trades
        if hasattr(portfolio, 'trades') and portfolio.trades:
            self.combined_trades.extend(portfolio.trades)

    def _get_equity_series(self, portfolio: Any) -> Optional[pd.Series]:
        """Extract equity curve as a pandas Series from various portfolio types."""
        # MultiAssetPortfolio has equity_curve as list, equity_curve_series as pd.Series
        if hasattr(portfolio, 'equity_curve_series'):
            return portfolio.equity_curve_series

        # Standard Portfolio has equity_curve as pd.Series
        if hasattr(portfolio, 'equity_curve'):
            equity = portfolio.equity_curve
            if isinstance(equity, pd.Series):
                return equity
            elif isinstance(equity, list) and len(equity) > 0:
                # Convert list to Series if possible
                if hasattr(portfolio, 'equity_timestamps'):
                    return pd.Series(equity, index=portfolio.equity_timestamps)
                return pd.Series(equity)

        return None

    def _compute_window_stats(self, portfolio: Any) -> Dict[str, float]:
        """Compute statistics for a single window's portfolio."""
        stats = {}

        equity = self._get_equity_series(portfolio)

        if equity is None or len(equity) < 2:
            return stats

        # Returns
        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            return stats

        # Total return
        stats['total_return'] = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            stats['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            stats['sharpe'] = 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        stats['max_drawdown'] = drawdown.min()

        # Win rate (if trades available)
        if hasattr(portfolio, 'trades') and portfolio.trades:
            wins = sum(1 for t in portfolio.trades if t.get('pnl', 0) > 0)
            total = len(portfolio.trades)
            stats['win_rate'] = wins / total if total > 0 else 0.0
            stats['n_trades'] = total

        return stats

    def stitch_equity_curves(self) -> pd.Series:
        """
        Combine window equity curves into a continuous series.

        Each window starts at init_cash. To create continuous equity:
        - Window 0: Use as-is
        - Window N: Scale to start where window N-1 ended

        Returns:
            Combined equity curve as pd.Series
        """
        if not self.window_portfolios:
            self.combined_equity = pd.Series(dtype=float)
            return self.combined_equity

        stitched_parts = []
        prev_end_value = None

        for i, portfolio in enumerate(self.window_portfolios):
            equity = self._get_equity_series(portfolio)

            if equity is None or len(equity) == 0:
                continue

            equity = equity.copy()

            if prev_end_value is None:
                # First window - use as-is
                stitched_parts.append(equity)
            else:
                # Scale to continue from previous end value
                start_value = equity.iloc[0]
                if start_value > 0:
                    scale = prev_end_value / start_value
                    scaled_equity = equity * scale
                    # Skip first point to avoid duplicate timestamp
                    stitched_parts.append(scaled_equity.iloc[1:])
                else:
                    stitched_parts.append(equity.iloc[1:])

            if len(stitched_parts[-1]) > 0:
                prev_end_value = stitched_parts[-1].iloc[-1]

        if stitched_parts:
            self.combined_equity = pd.concat(stitched_parts)
            # Remove any duplicate indices
            self.combined_equity = self.combined_equity[
                ~self.combined_equity.index.duplicated(keep='last')
            ]
        else:
            self.combined_equity = pd.Series(dtype=float)

        return self.combined_equity

    def compute_combined_stats(self) -> Dict[str, float]:
        """
        Compute overall statistics from the stitched equity curve.

        Returns:
            Dictionary of combined statistics
        """
        if self.combined_equity is None:
            self.stitch_equity_curves()

        stats = {}

        if self.combined_equity is None or len(self.combined_equity) < 2:
            self.combined_stats = stats
            return stats

        equity = self.combined_equity
        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            self.combined_stats = stats
            return stats

        # Total return
        stats['total_return'] = (equity.iloc[-1] / equity.iloc[0]) - 1

        # CAGR
        n_years = len(returns) / 252
        if n_years > 0:
            stats['cagr'] = (1 + stats['total_return']) ** (1 / n_years) - 1
        else:
            stats['cagr'] = 0.0

        # Sharpe ratio
        if returns.std() > 0:
            stats['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            stats['sharpe'] = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            stats['sortino'] = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            stats['sortino'] = 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        stats['max_drawdown'] = drawdown.min()

        # Calmar ratio
        if stats['max_drawdown'] < 0:
            stats['calmar'] = stats['cagr'] / abs(stats['max_drawdown'])
        else:
            stats['calmar'] = 0.0

        # Win rate from combined trades
        if self.combined_trades:
            wins = sum(1 for t in self.combined_trades if t.get('pnl', 0) > 0)
            total = len(self.combined_trades)
            stats['win_rate'] = wins / total if total > 0 else 0.0
            stats['n_trades'] = total

        # Window statistics
        stats['n_windows'] = len(self.windows)
        if self.window_stats:
            window_sharpes = [s.get('sharpe', 0) for s in self.window_stats]
            stats['avg_window_sharpe'] = np.mean(window_sharpes)
            stats['std_window_sharpe'] = np.std(window_sharpes)
            stats['min_window_sharpe'] = np.min(window_sharpes)
            stats['max_window_sharpe'] = np.max(window_sharpes)

        self.combined_stats = stats
        return stats

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame summarizing window-by-window results.

        Returns:
            DataFrame with one row per window
        """
        rows = []
        for i, (start, end) in enumerate(self.windows):
            row = {
                'window': i,
                'start_date': start,
                'end_date': end,
            }
            if i < len(self.window_stats):
                row.update(self.window_stats[i])
            rows.append(row)

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        n_windows = len(self.windows)
        if self.combined_stats:
            sharpe = self.combined_stats.get('sharpe', 0)
            total_ret = self.combined_stats.get('total_return', 0)
            return f"RollingWindowResults({n_windows} windows, sharpe={sharpe:.2f}, return={total_ret:.1%})"
        return f"RollingWindowResults({n_windows} windows)"
