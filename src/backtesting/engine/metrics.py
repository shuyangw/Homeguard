"""
Performance metrics and reporting for backtesting results.

This module provides the PerformanceMetrics class for calculating and reporting
performance metrics from Portfolio objects. No external dependencies required.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable

from src.utils import logger


@runtime_checkable
class PortfolioProtocol(Protocol):
    """Protocol defining the expected interface for Portfolio objects."""
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]

    def stats(self) -> Dict[str, Any]: ...


class PerformanceMetrics:
    """
    Calculate and report performance metrics for backtest results.
    """

    @staticmethod
    def _get_exit_trades(portfolio: PortfolioProtocol) -> List[Dict[str, Any]]:
        """
        Get all exit trades (trades with P&L).

        Args:
            portfolio: Portfolio object with trades list

        Returns:
            List of exit trade dictionaries
        """
        return [t for t in portfolio.trades if t.get('type') in ['exit', 'cover_short']]

    @staticmethod
    def _get_pnl_values(portfolio: PortfolioProtocol) -> List[float]:
        """
        Extract P&L values from exit trades.

        Args:
            portfolio: Portfolio object with trades list

        Returns:
            List of P&L values
        """
        exit_trades = PerformanceMetrics._get_exit_trades(portfolio)
        return [t.get('pnl', 0.0) for t in exit_trades if 'pnl' in t]

    @staticmethod
    def calculate_all_metrics(portfolio: PortfolioProtocol) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio: Portfolio object with stats() method and trades list

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
    def _calculate_profit_factor(portfolio: PortfolioProtocol) -> float:
        """
        Calculate profit factor (total wins / total losses).
        """
        try:
            pnl_values = PerformanceMetrics._get_pnl_values(portfolio)

            if len(pnl_values) == 0:
                return 0.0

            wins = sum(p for p in pnl_values if p > 0)
            losses = abs(sum(p for p in pnl_values if p < 0))

            if losses == 0:
                return float('inf') if wins > 0 else 0.0

            return float(wins / losses)
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_avg_win(portfolio: PortfolioProtocol) -> float:
        """
        Calculate average winning trade.
        """
        try:
            pnl_values = PerformanceMetrics._get_pnl_values(portfolio)
            wins = [p for p in pnl_values if p > 0]
            return float(np.mean(wins)) if len(wins) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_avg_loss(portfolio: PortfolioProtocol) -> float:
        """
        Calculate average losing trade.
        """
        try:
            pnl_values = PerformanceMetrics._get_pnl_values(portfolio)
            losses = [p for p in pnl_values if p < 0]
            return float(np.mean(losses)) if len(losses) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_largest_win(portfolio: PortfolioProtocol) -> float:
        """
        Calculate largest winning trade.
        """
        try:
            pnl_values = PerformanceMetrics._get_pnl_values(portfolio)
            wins = [p for p in pnl_values if p > 0]
            return float(max(wins)) if len(wins) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_largest_loss(portfolio: PortfolioProtocol) -> float:
        """
        Calculate largest losing trade (most negative).
        """
        try:
            pnl_values = PerformanceMetrics._get_pnl_values(portfolio)
            losses = [p for p in pnl_values if p < 0]
            return float(min(losses)) if len(losses) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def generate_report(
        portfolio: PortfolioProtocol,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate detailed performance report.

        Args:
            portfolio: Portfolio object
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
        portfolios: Dict[str, PortfolioProtocol]
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
    def plot_equity_curve(portfolio: PortfolioProtocol, title: Optional[str] = None):
        """
        Plot equity curve using matplotlib.

        Args:
            portfolio: Portfolio object with equity_curve attribute
            title: Optional chart title
        """
        import matplotlib.pyplot as plt

        equity = portfolio.equity_curve

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity.index, equity.values, linewidth=1.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title(title or 'Equity Curve')
        ax.grid(True, alpha=0.3)

        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_drawdown(portfolio: PortfolioProtocol, title: Optional[str] = None):
        """
        Plot drawdown chart using matplotlib.

        Args:
            portfolio: Portfolio object with equity_curve attribute
            title: Optional chart title
        """
        import matplotlib.pyplot as plt

        equity = portfolio.equity_curve

        # Calculate drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title or 'Drawdown')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_returns_distribution(portfolio: PortfolioProtocol, title: Optional[str] = None):
        """
        Plot distribution of trade returns.

        Args:
            portfolio: Portfolio object with trades list
            title: Optional chart title
        """
        import matplotlib.pyplot as plt

        pnl_values = PerformanceMetrics._get_pnl_values(portfolio)

        if len(pnl_values) == 0:
            logger.warning("No trades to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Color wins green, losses red
        colors = ['green' if p > 0 else 'red' for p in pnl_values]
        ax.bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('P&L ($)')
        ax.set_title(title or 'Trade Returns Distribution')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()
