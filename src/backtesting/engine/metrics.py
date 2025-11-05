"""
Performance metrics and reporting for backtesting results.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Optional
from pathlib import Path

from utils import logger


class PerformanceMetrics:
    """
    Calculate and report performance metrics for backtest results.
    """

    @staticmethod
    def calculate_all_metrics(portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio: VectorBT Portfolio object

        Returns:
            Dictionary of performance metrics
        """
        try:
            stats = portfolio.stats()

            if stats is None:
                return {}

            metrics = {
                'total_return_pct': stats.get('Total Return [%]', 0),
                'annual_return_pct': stats.get('Annual Return [%]', 0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown_pct': stats.get('Max Drawdown [%]', 0),
                'win_rate_pct': stats.get('Win Rate [%]', 0),
                'total_trades': stats.get('Total Trades', 0),
                'start_value': stats.get('Start Value', 0),
                'end_value': stats.get('End Value', 0),
                'profit_factor': PerformanceMetrics._calculate_profit_factor(portfolio),
                'avg_win': PerformanceMetrics._calculate_avg_win(portfolio),
                'avg_loss': PerformanceMetrics._calculate_avg_loss(portfolio),
                'largest_win': PerformanceMetrics._calculate_largest_win(portfolio),
                'largest_loss': PerformanceMetrics._calculate_largest_loss(portfolio),
            }

            return metrics
        except Exception as e:
            logger.warning(f"Could not calculate all metrics: {e}")
            return {}

    @staticmethod
    def _calculate_profit_factor(portfolio: vbt.Portfolio) -> float:
        """
        Calculate profit factor (total wins / total losses).
        """
        try:
            trades = portfolio.trades.records_readable  # type: ignore[attr-defined]

            if len(trades) == 0:
                return 0.0

            wins = trades[trades['PnL'] > 0]['PnL'].sum()
            losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())

            if losses == 0:
                return float('inf') if wins > 0 else 0.0

            return float(wins / losses)
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_avg_win(portfolio: vbt.Portfolio) -> float:
        """
        Calculate average winning trade.
        """
        try:
            trades = portfolio.trades.records_readable  # type: ignore[attr-defined]
            wins = trades[trades['PnL'] > 0]['PnL']
            return float(wins.mean()) if len(wins) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_avg_loss(portfolio: vbt.Portfolio) -> float:
        """
        Calculate average losing trade.
        """
        try:
            trades = portfolio.trades.records_readable  # type: ignore[attr-defined]
            losses = trades[trades['PnL'] < 0]['PnL']
            return float(losses.mean()) if len(losses) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_largest_win(portfolio: vbt.Portfolio) -> float:
        """
        Calculate largest winning trade.
        """
        try:
            trades = portfolio.trades.records_readable  # type: ignore[attr-defined]
            wins = trades[trades['PnL'] > 0]['PnL']
            return float(wins.max()) if len(wins) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_largest_loss(portfolio: vbt.Portfolio) -> float:
        """
        Calculate largest losing trade.
        """
        try:
            trades = portfolio.trades.records_readable  # type: ignore[attr-defined]
            losses = trades[trades['PnL'] < 0]['PnL']
            return float(losses.min()) if len(losses) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def generate_report(
        portfolio: vbt.Portfolio,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate detailed performance report.

        Args:
            portfolio: VectorBT Portfolio object
            output_path: Optional path to save report as CSV

        Returns:
            DataFrame with performance metrics
        """
        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)

        report_df = pd.DataFrame([metrics]).T
        report_df.columns = ['Value']

        if output_path:
            report_df.to_csv(output_path)
            logger.success(f"Report saved to: {output_path}")

        return report_df

    @staticmethod
    def compare_strategies(
        portfolios: Dict[str, vbt.Portfolio]
    ) -> pd.DataFrame:
        """
        Compare performance metrics across multiple strategies.

        Args:
            portfolios: Dictionary mapping strategy names to Portfolio objects

        Returns:
            DataFrame comparing strategies across key metrics
        """
        comparison_data = {}

        for name, portfolio in portfolios.items():
            metrics = PerformanceMetrics.calculate_all_metrics(portfolio)
            comparison_data[name] = metrics

        comparison_df = pd.DataFrame(comparison_data).T

        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)

        return comparison_df

    @staticmethod
    def plot_equity_curve(portfolio: vbt.Portfolio, title: Optional[str] = None):
        """
        Plot equity curve using matplotlib.

        Args:
            portfolio: VectorBT Portfolio object
            title: Optional chart title
        """
        import matplotlib.pyplot as plt

        fig = portfolio.plot(subplots=['cum_returns'])

        if title:
            # VectorBT returns matplotlib Figure - use matplotlib methods
            if hasattr(fig, 'suptitle'):
                fig.suptitle(title)
            elif hasattr(fig, 'update_layout'):
                # Plotly figure
                fig.update_layout(title=title)

        plt.show()

    @staticmethod
    def plot_drawdown(portfolio: vbt.Portfolio, title: Optional[str] = None):
        """
        Plot drawdown chart using matplotlib.

        Args:
            portfolio: VectorBT Portfolio object
            title: Optional chart title
        """
        import matplotlib.pyplot as plt

        fig = portfolio.plot(subplots=['drawdown'])

        if title:
            # VectorBT returns matplotlib Figure - use matplotlib methods
            if hasattr(fig, 'suptitle'):
                fig.suptitle(title)
            elif hasattr(fig, 'update_layout'):
                # Plotly figure
                fig.update_layout(title=title)

        plt.show()
