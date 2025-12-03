"""
Standard Backtest Report Generator.

Generates monthly and overall performance metrics for any strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.utils.logger import logger
from src.settings import get_backtest_results_dir


class StandardReportGenerator:
    """
    Generate standardized backtest reports with monthly breakdowns.

    Outputs:
    - Console: Formatted table
    - Markdown: Timestamped report file
    - CSV: Raw metrics for analysis
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or get_backtest_results_dir() / "standard_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        equity_curve: pd.Series,
        strategy_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Generate full report with overall and monthly metrics.

        Returns dict with:
        - overall_metrics: Dict of overall performance
        - monthly_metrics: DataFrame of monthly breakdown
        """
        # Calculate overall metrics
        overall = self._calculate_overall_metrics(equity_curve, initial_capital)

        # Calculate monthly metrics
        monthly = self._calculate_monthly_metrics(equity_curve)

        # Generate outputs
        self._print_console_report(overall, monthly, strategy_name, symbols)
        md_path = self._save_markdown_report(overall, monthly, strategy_name, symbols, start_date, end_date)
        csv_path = self._save_csv_report(monthly, strategy_name)

        return {
            'overall_metrics': overall,
            'monthly_metrics': monthly,
            'markdown_path': md_path,
            'csv_path': csv_path
        }

    def _calculate_overall_metrics(
        self,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate overall Sharpe ratio and max drawdown."""
        # Daily returns
        returns = equity_curve.pct_change().dropna()

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min() * 100  # As percentage

        # Total return
        total_return = ((equity_curve.iloc[-1] - initial_capital) / initial_capital) * 100

        # CAGR
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        if years > 0:
            cagr = ((equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'trading_days': len(equity_curve)
        }

    def _calculate_monthly_metrics(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate Sharpe ratio and max drawdown for each month."""
        # Get daily returns
        daily_returns = equity_curve.pct_change().dropna()

        # Ensure index is datetime
        daily_returns.index = pd.to_datetime(daily_returns.index)
        equity_curve.index = pd.to_datetime(equity_curve.index)

        # Group by month
        monthly_groups = daily_returns.groupby(pd.Grouper(freq='ME'))

        records = []
        for month_end, month_returns in monthly_groups:
            if len(month_returns) == 0:
                continue

            # Monthly Sharpe (annualized)
            if month_returns.std() > 0:
                sharpe = (month_returns.mean() / month_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0

            # Monthly max drawdown (within that month)
            month_start = month_returns.index[0]
            month_last = month_returns.index[-1]
            month_equity = equity_curve.loc[month_start:month_last]
            if len(month_equity) > 0:
                cummax = month_equity.cummax()
                drawdown = (month_equity - cummax) / cummax
                max_dd = drawdown.min() * 100
            else:
                max_dd = 0.0

            # Monthly return
            month_return = month_returns.sum() * 100  # Simple sum for approximation

            records.append({
                'month': month_end.strftime('%Y-%m'),
                'sharpe_ratio': round(sharpe, 3),
                'max_drawdown_pct': round(max_dd, 2),
                'return_pct': round(month_return, 2),
                'trading_days': len(month_returns)
            })

        return pd.DataFrame(records)

    def _print_console_report(
        self,
        overall: Dict[str, float],
        monthly: pd.DataFrame,
        strategy_name: str,
        symbols: List[str]
    ):
        """Print formatted report to console."""
        logger.info("=" * 70)
        logger.info(f"STANDARD BACKTEST REPORT: {strategy_name}")
        logger.info("=" * 70)

        logger.info(f"\nSymbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        logger.info(f"Total symbols: {len(symbols)}")

        logger.info("\n--- OVERALL METRICS ---")
        logger.info(f"  Sharpe Ratio:    {overall['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown:    {overall['max_drawdown_pct']:.2f}%")
        logger.info(f"  Total Return:    {overall['total_return_pct']:.2f}%")
        logger.info(f"  CAGR:            {overall['cagr_pct']:.2f}%")
        logger.info(f"  Trading Days:    {overall['trading_days']}")

        logger.info("\n--- MONTHLY BREAKDOWN ---")
        logger.info(f"{'Month':<10} {'Sharpe':>10} {'Max DD':>10} {'Return':>10}")
        logger.info("-" * 42)

        for _, row in monthly.iterrows():
            logger.info(
                f"{row['month']:<10} "
                f"{row['sharpe_ratio']:>10.3f} "
                f"{row['max_drawdown_pct']:>9.2f}% "
                f"{row['return_pct']:>9.2f}%"
            )

        # Summary statistics
        logger.info("-" * 42)
        logger.info(
            f"{'Average':<10} "
            f"{monthly['sharpe_ratio'].mean():>10.3f} "
            f"{monthly['max_drawdown_pct'].mean():>9.2f}% "
            f"{monthly['return_pct'].mean():>9.2f}%"
        )

        logger.info("=" * 70)

    def _save_markdown_report(
        self,
        overall: Dict[str, float],
        monthly: pd.DataFrame,
        strategy_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Path:
        """Save report as timestamped markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{timestamp}_STANDARD_BACKTEST_{strategy_name.upper()}.md"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(f"# Standard Backtest Report: {strategy_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Period:** {start_date} to {end_date}\n\n")
            f.write(f"**Symbols ({len(symbols)}):** {', '.join(symbols)}\n\n")

            f.write("## Overall Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Sharpe Ratio | {overall['sharpe_ratio']:.3f} |\n")
            f.write(f"| Max Drawdown | {overall['max_drawdown_pct']:.2f}% |\n")
            f.write(f"| Total Return | {overall['total_return_pct']:.2f}% |\n")
            f.write(f"| CAGR | {overall['cagr_pct']:.2f}% |\n")
            f.write(f"| Trading Days | {overall['trading_days']} |\n\n")

            f.write("## Monthly Breakdown\n\n")
            f.write("| Month | Sharpe | Max DD | Return |\n")
            f.write("|-------|--------|--------|--------|\n")

            for _, row in monthly.iterrows():
                f.write(
                    f"| {row['month']} | {row['sharpe_ratio']:.3f} | "
                    f"{row['max_drawdown_pct']:.2f}% | {row['return_pct']:.2f}% |\n"
                )

            f.write(f"\n**Average Sharpe:** {monthly['sharpe_ratio'].mean():.3f}\n")
            f.write(f"**Average Monthly DD:** {monthly['max_drawdown_pct'].mean():.2f}%\n")
            f.write(f"**Average Monthly Return:** {monthly['return_pct'].mean():.2f}%\n")

        logger.info(f"Markdown report saved to: {filepath}")
        return filepath

    def _save_csv_report(self, monthly: pd.DataFrame, strategy_name: str) -> Path:
        """Save monthly metrics as CSV."""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{timestamp}_monthly_metrics_{strategy_name.lower()}.csv"
        filepath = self.output_dir / filename

        monthly.to_csv(filepath, index=False)
        logger.info(f"CSV report saved to: {filepath}")
        return filepath
