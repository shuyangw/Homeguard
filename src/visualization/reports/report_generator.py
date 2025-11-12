"""
Report generation for backtest results.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from src.utils import logger


class ReportGenerator:
    """
    Generates comprehensive backtest reports with performance metrics and trade summaries.
    """

    @staticmethod
    def generate_summary_report(
        strategy_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        performance_stats: Dict[str, Any],
        trade_summary: Dict[str, Any],
        output_path: Path
    ):
        """
        Generate a comprehensive summary report.

        Args:
            strategy_name: Name of the strategy
            symbols: List of symbols tested
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            fees: Transaction fees
            performance_stats: Performance metrics from backtest
            trade_summary: Trade summary statistics
            output_path: Path to save report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BACKTEST SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("STRATEGY CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Strategy:        {strategy_name}\n")
            f.write(f"Symbols:         {', '.join(symbols)}\n")
            f.write(f"Period:          {start_date} to {end_date}\n")
            f.write(f"Initial Capital: ${initial_capital:,.2f}\n")
            f.write(f"Transaction Fees: {fees:.4f} ({fees*100:.2f}%)\n")
            f.write("\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            if performance_stats:
                for key, value in performance_stats.items():
                    if isinstance(value, (int, float)):
                        if abs(value) >= 1000:
                            f.write(f"{key:30s}: {value:>20,.2f}\n")
                        else:
                            f.write(f"{key:30s}: {value:>20.4f}\n")
                    else:
                        f.write(f"{key:30s}: {value:>20}\n")
            else:
                f.write("No performance metrics available.\n")
            f.write("\n")

            f.write("TRADE SUMMARY\n")
            f.write("-" * 80 + "\n")
            if trade_summary:
                f.write(f"Total Trades:     {trade_summary.get('total_trades', 0):>10}\n")
                f.write(f"Buy Orders:       {trade_summary.get('buy_count', 0):>10}\n")
                f.write(f"Sell Orders:      {trade_summary.get('sell_count', 0):>10}\n")
                if 'total_volume' in trade_summary:
                    f.write(f"Total Volume:     {trade_summary['total_volume']:>10,.2f} shares\n")
                if 'avg_trade_size' in trade_summary:
                    f.write(f"Avg Trade Size:   {trade_summary['avg_trade_size']:>10,.2f} shares\n")
            else:
                f.write("No trade data available.\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        logger.success(f"Summary report saved to: {output_path}")

    @staticmethod
    def generate_json_report(
        strategy_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        performance_stats: Dict[str, Any],
        trade_summary: Dict[str, Any],
        output_path: Path
    ):
        """
        Generate a JSON report for programmatic access.

        Args:
            strategy_name: Name of the strategy
            symbols: List of symbols tested
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            fees: Transaction fees
            performance_stats: Performance metrics
            trade_summary: Trade summary
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'strategy_name': strategy_name,
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'transaction_fees': fees
            },
            'performance_metrics': performance_stats,
            'trade_summary': trade_summary
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.success(f"JSON report saved to: {output_path}")

    @staticmethod
    def generate_trade_log_summary(
        trades_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Generate a detailed trade log summary.

        Args:
            trades_df: DataFrame with trade information
            output_path: Path to save summary
        """
        # Defensive check: ensure trades_df is actually a DataFrame
        if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
            logger.warning("No trades to summarize.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRADE LOG SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Trades: {len(trades_df)}\n\n")

            f.write("TRADES BY ACTION\n")
            f.write("-" * 80 + "\n")
            action_counts = trades_df['action'].value_counts()
            for action, count in action_counts.items():
                f.write(f"{action:10s}: {count:>5d}\n")
            f.write("\n")

            f.write("TRADES BY SYMBOL\n")
            f.write("-" * 80 + "\n")
            symbol_counts = trades_df['symbol'].value_counts()
            for symbol, count in symbol_counts.items():
                f.write(f"{symbol:10s}: {count:>5d} trades\n")
            f.write("\n")

            f.write("TRADE SIZE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Trade Size:   {trades_df['size'].mean():>10,.2f} shares\n")
            f.write(f"Median Trade Size: {trades_df['size'].median():>10,.2f} shares\n")
            f.write(f"Max Trade Size:    {trades_df['size'].max():>10,.2f} shares\n")
            f.write(f"Min Trade Size:    {trades_df['size'].min():>10,.2f} shares\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        logger.success(f"Trade log summary saved to: {output_path}")
