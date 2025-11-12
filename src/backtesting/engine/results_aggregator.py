"""
Results aggregation for multi-symbol backtesting sweeps.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.utils import logger
from src.backtesting.engine.tearsheet_generator import TearsheetGenerator
from src.backtesting.engine.portfolio_aggregator import PortfolioAggregator
from src.backtesting.engine.benchmark_calculator import BenchmarkCalculator

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False


class ResultsAggregator:
    """
    Aggregates and analyzes results from multiple backtests.
    """

    @staticmethod
    def extract_stats(portfolio_stats: pd.Series, symbol: str) -> Dict[str, Any]:
        """
        Extract key statistics from portfolio stats.

        Args:
            portfolio_stats: Portfolio statistics from portfolio.stats()
            symbol: Symbol identifier

        Returns:
            Dictionary of extracted stats
        """
        return {
            'Symbol': symbol,
            'Total Return [%]': portfolio_stats.get('Total Return [%]', 0.0),
            'Sharpe Ratio': portfolio_stats.get('Sharpe Ratio', 0.0),
            'Max Drawdown [%]': portfolio_stats.get('Max Drawdown [%]', 0.0),
            'Win Rate [%]': portfolio_stats.get('Win Rate [%]', 0.0),
            'Total Trades': portfolio_stats.get('Total Trades', 0),
            'Profit Factor': portfolio_stats.get('Profit Factor', 0.0),
            'Start Value': portfolio_stats.get('Start Value', 0.0),
            'End Value': portfolio_stats.get('End Value', 0.0),
            'Duration': portfolio_stats.get('Period', ''),
        }

    @staticmethod
    def extract_advanced_metrics(portfolio, symbol: str) -> Dict[str, Any]:
        """
        Extract advanced metrics from portfolio object using QuantStats.

        Args:
            portfolio: Portfolio object with equity_curve attribute
            symbol: Symbol identifier

        Returns:
            Dictionary of advanced metrics (Sortino, Calmar, CAGR, Volatility, etc.)
        """
        if not HAS_QUANTSTATS:
            logger.warning("QuantStats not available - advanced metrics will be unavailable")
            return {}

        try:
            equity_curve = portfolio.equity_curve
            if equity_curve is None or len(equity_curve) == 0:
                return {}

            # Calculate daily returns
            daily_returns = equity_curve.pct_change().dropna()
            daily_returns = daily_returns.replace([float('inf'), float('-inf')], float('nan')).dropna()

            if len(daily_returns) == 0:
                return {}

            # Calculate advanced metrics using QuantStats
            metrics = {
                'Sortino Ratio': float(qs.stats.sortino(daily_returns)),  # type: ignore[arg-type]
                'Calmar Ratio': float(qs.stats.calmar(daily_returns)),  # type: ignore[arg-type]
                'CAGR [%]': float(qs.stats.cagr(daily_returns) * 100),  # type: ignore[arg-type]
                'Volatility [%]': float(qs.stats.volatility(daily_returns, annualize=True) * 100),  # type: ignore[arg-type]
                'Skewness': float(qs.stats.skew(daily_returns)),  # type: ignore[arg-type]
                'Kurtosis': float(qs.stats.kurtosis(daily_returns)),  # type: ignore[arg-type]
                'VaR (95%) [%]': float(qs.stats.value_at_risk(daily_returns) * 100),  # type: ignore[arg-type]
                'CVaR (95%) [%]': float(qs.stats.cvar(daily_returns) * 100),  # type: ignore[arg-type]
            }

            return metrics

        except Exception as e:
            logger.warning(f"Could not extract advanced metrics for {symbol}: {e}")
            return {}

    @classmethod
    def aggregate_results(
        cls,
        results: Dict[str, pd.Series],
        sort_by: str = 'Sharpe Ratio',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Aggregate multiple backtest results into summary DataFrame.

        Args:
            results: Dictionary of {symbol: portfolio_stats}
            sort_by: Column to sort by (default: 'Sharpe Ratio')
            ascending: Sort ascending if True (default: False for descending)

        Returns:
            DataFrame with aggregated results
        """
        if not results:
            logger.warning("No results to aggregate")
            return pd.DataFrame()

        rows = []
        for symbol, stats in results.items():
            if stats is None:
                logger.warning(f"Skipping {symbol}: No stats available")
                continue

            try:
                row = cls.extract_stats(stats, symbol)
                rows.append(row)
            except Exception as e:
                logger.warning(f"Error extracting stats for {symbol}: {e}")

        if not rows:
            logger.warning("No valid results to aggregate")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        return df

    @staticmethod
    def calculate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive summary statistics across all results.

        Args:
            df: DataFrame with aggregated results

        Returns:
            Dictionary of summary statistics with enhanced metrics
        """
        if df.empty:
            return {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n = len(df)

        # Basic counts
        profitable = len(df[df['Total Return [%]'] > 0])
        unprofitable = n - profitable

        summary = {
            # Overall Performance
            'Total Symbols': n,
            'Profitable Symbols': profitable,
            'Unprofitable Symbols': unprofitable,
            'Win Rate (Symbols)': (profitable / n * 100) if n > 0 else 0,
        }

        # Distribution statistics for each numeric column
        for col in numeric_cols:
            if col == 'Symbol':
                continue

            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            summary[f'{col} - Mean'] = col_data.mean()
            summary[f'{col} - Median'] = col_data.median()
            summary[f'{col} - Std Dev'] = col_data.std()
            summary[f'{col} - Min'] = col_data.min()
            summary[f'{col} - Max'] = col_data.max()
            summary[f'{col} - 25th Percentile'] = col_data.quantile(0.25)
            summary[f'{col} - 75th Percentile'] = col_data.quantile(0.75)

        # Enhanced Risk-Reward Metrics
        if 'Sharpe Ratio' in df.columns:
            sharpe_data = df['Sharpe Ratio'].dropna()
            summary['Sharpe Positive Count'] = len(sharpe_data[sharpe_data > 0])
            summary['Sharpe Excellent Count'] = len(sharpe_data[sharpe_data > 1.0])
            summary['Sharpe Good Count'] = len(sharpe_data[sharpe_data > 0.5])

        if 'Max Drawdown [%]' in df.columns:
            dd_data = df['Max Drawdown [%]'].dropna()
            summary['Low Risk Count'] = len(dd_data[dd_data > -15])
            summary['Acceptable Risk Count'] = len(dd_data[dd_data > -20])
            summary['High Risk Count'] = len(dd_data[dd_data <= -20])

        # Consistency Score (0-10 scale)
        if 'Total Return [%]' in df.columns:
            returns = df['Total Return [%]'].dropna()
            if len(returns) > 0 and returns.mean() != 0:
                cv = abs(returns.std() / returns.mean())  # Coefficient of variation
                consistency = max(0, min(10, 10 - (cv * 2)))  # Scale: lower CV = higher score
                summary['Consistency Score'] = round(consistency, 1)
            else:
                summary['Consistency Score'] = 0.0

        # Risk-Reward Ratio
        if 'Total Return [%]' in df.columns and 'Max Drawdown [%]' in df.columns:
            mean_return = df['Total Return [%]'].mean()
            mean_dd = abs(df['Max Drawdown [%]'].mean())
            if mean_dd != 0:
                summary['Risk-Reward Ratio'] = round(mean_return / mean_dd, 2)
            else:
                summary['Risk-Reward Ratio'] = 0.0

        # Quality Count (high Sharpe + acceptable drawdown)
        if 'Sharpe Ratio' in df.columns and 'Max Drawdown [%]' in df.columns:
            high_quality = df[
                (df['Sharpe Ratio'] > 1.0) &
                (df['Max Drawdown [%]'] > -15)
            ]
            summary['High Quality Count'] = len(high_quality)

        # Best/Worst Performers
        if 'Total Return [%]' in df.columns and 'Symbol' in df.columns:
            best_return_idx = df['Total Return [%]'].idxmax()
            worst_return_idx = df['Total Return [%]'].idxmin()
            summary['Best Symbol (Return)'] = df.loc[best_return_idx, 'Symbol']
            summary['Best Return Value'] = df.loc[best_return_idx, 'Total Return [%]']
            summary['Worst Symbol (Return)'] = df.loc[worst_return_idx, 'Symbol']
            summary['Worst Return Value'] = df.loc[worst_return_idx, 'Total Return [%]']

        if 'Sharpe Ratio' in df.columns and 'Symbol' in df.columns:
            best_sharpe_idx = df['Sharpe Ratio'].idxmax()
            summary['Best Symbol (Sharpe)'] = df.loc[best_sharpe_idx, 'Symbol']
            summary['Best Sharpe Value'] = df.loc[best_sharpe_idx, 'Sharpe Ratio']

        if 'Max Drawdown [%]' in df.columns and 'Symbol' in df.columns:
            safest_idx = df['Max Drawdown [%]'].idxmax()  # Closest to 0 (least negative)
            summary['Safest Symbol'] = df.loc[safest_idx, 'Symbol']
            summary['Safest Drawdown Value'] = df.loc[safest_idx, 'Max Drawdown [%]']

        # Best Overall (highest Sharpe with acceptable drawdown)
        if 'Sharpe Ratio' in df.columns and 'Max Drawdown [%]' in df.columns and 'Symbol' in df.columns:
            acceptable = df[df['Max Drawdown [%]'] > -25]  # Filter out extreme drawdowns
            if not acceptable.empty:
                best_overall_idx = acceptable['Sharpe Ratio'].idxmax()
                summary['Best Overall Symbol'] = acceptable.loc[best_overall_idx, 'Symbol']
                summary['Best Overall Sharpe'] = acceptable.loc[best_overall_idx, 'Sharpe Ratio']
                summary['Best Overall Return'] = acceptable.loc[best_overall_idx, 'Total Return [%]']
                summary['Best Overall Drawdown'] = acceptable.loc[best_overall_idx, 'Max Drawdown [%]']

        # Total Trading Activity
        if 'Total Trades' in df.columns:
            summary['Total Trades (All Symbols)'] = int(df['Total Trades'].sum())

        return summary

    @staticmethod
    def display_summary_stats(summary: Dict[str, Any]) -> None:
        """
        Display summary statistics in an intuitive, grouped format.

        Args:
            summary: Dictionary of summary statistics from calculate_summary_stats()
        """
        if not summary:
            logger.warning("No summary statistics to display")
            return

        logger.blank()
        logger.separator("=", 80)
        logger.header("COMPREHENSIVE SWEEP ANALYSIS")
        logger.separator("=", 80)

        # Overall Performance Section
        logger.blank()
        logger.header("📊 OVERALL PERFORMANCE")
        logger.metric(f"  Total Symbols Tested: {summary.get('Total Symbols', 0)}")

        profitable = summary.get('Profitable Symbols', 0)
        unprofitable = summary.get('Unprofitable Symbols', 0)
        win_rate = summary.get('Win Rate (Symbols)', 0)

        if profitable > unprofitable:
            logger.profit(f"  ✓ Profitable: {profitable} ({win_rate:.1f}%)")
            logger.loss(f"  ✗ Unprofitable: {unprofitable}")
        else:
            logger.loss(f"  ✓ Profitable: {profitable} ({win_rate:.1f}%)")
            logger.loss(f"  ✗ Unprofitable: {unprofitable}")

        # Consistency Score
        consistency = summary.get('Consistency Score', 0)
        stars = '⭐' * int(consistency)
        if consistency >= 7:
            logger.success(f"  Consistency: {consistency:.1f}/10 {stars} (Excellent)")
        elif consistency >= 5:
            logger.info(f"  Consistency: {consistency:.1f}/10 {stars} (Good)")
        else:
            logger.warning(f"  Consistency: {consistency:.1f}/10 {stars} (Needs Improvement)")

        # Returns Distribution Section
        logger.blank()
        logger.header("💰 RETURNS DISTRIBUTION")

        median_return = summary.get('Total Return [%] - Median', 0)
        mean_return = summary.get('Total Return [%] - Mean', 0)

        if median_return >= 0:
            logger.profit(f"  Median Return: {median_return:.2f}%")
        else:
            logger.loss(f"  Median Return: {median_return:.2f}%")

        logger.metric(f"  Mean Return: {mean_return:.2f}%")

        if 'Best Symbol (Return)' in summary:
            best_sym = summary['Best Symbol (Return)']
            best_val = summary.get('Best Return Value', 0)
            logger.success(f"  Best Performer: {best_sym} (+{best_val:.2f}%)")

        if 'Worst Symbol (Return)' in summary:
            worst_sym = summary['Worst Symbol (Return)']
            worst_val = summary.get('Worst Return Value', 0)
            if worst_val < 0:
                logger.loss(f"  Worst Performer: {worst_sym} ({worst_val:.2f}%)")
            else:
                logger.warning(f"  Worst Performer: {worst_sym} (+{worst_val:.2f}%)")

        q25 = summary.get('Total Return [%] - 25th Percentile', 0)
        q75 = summary.get('Total Return [%] - 75th Percentile', 0)
        logger.dim(f"  Interquartile Range: {q25:.2f}% to {q75:.2f}%")

        # Risk Metrics Section
        logger.blank()
        logger.header("⚠️  RISK METRICS")

        median_dd = summary.get('Max Drawdown [%] - Median', 0)
        mean_dd = summary.get('Max Drawdown [%] - Mean', 0)
        worst_dd = summary.get('Max Drawdown [%] - Min', 0)

        if median_dd > -10:
            logger.success(f"  Median Drawdown: {median_dd:.2f}% (Low Risk)")
        elif median_dd > -20:
            logger.warning(f"  Median Drawdown: {median_dd:.2f}% (Moderate Risk)")
        else:
            logger.error(f"  Median Drawdown: {median_dd:.2f}% (High Risk)")

        if worst_dd < -25:
            logger.error(f"  Worst Drawdown: {worst_dd:.2f}%")
        else:
            logger.warning(f"  Worst Drawdown: {worst_dd:.2f}%")

        low_risk = summary.get('Low Risk Count', 0)
        acceptable_risk = summary.get('Acceptable Risk Count', 0)
        high_risk = summary.get('High Risk Count', 0)
        total = summary.get('Total Symbols', 1)

        logger.metric(f"  Low Risk (<-15% DD): {low_risk} symbols ({low_risk/total*100:.1f}%)")
        logger.metric(f"  Acceptable Risk (<-20% DD): {acceptable_risk} symbols")
        if high_risk > 0:
            logger.warning(f"  High Risk (>-20% DD): {high_risk} symbols")

        # Sharpe Ratio Section
        logger.blank()
        logger.header("📈 SHARPE RATIO ANALYSIS")

        median_sharpe = summary.get('Sharpe Ratio - Median', 0)
        mean_sharpe = summary.get('Sharpe Ratio - Mean', 0)

        if median_sharpe >= 1.0:
            logger.success(f"  Median Sharpe: {median_sharpe:.2f} (Excellent)")
        elif median_sharpe >= 0.5:
            logger.info(f"  Median Sharpe: {median_sharpe:.2f} (Good)")
        elif median_sharpe >= 0:
            logger.warning(f"  Median Sharpe: {median_sharpe:.2f} (Marginal)")
        else:
            logger.error(f"  Median Sharpe: {median_sharpe:.2f} (Poor)")

        positive_sharpe = summary.get('Sharpe Positive Count', 0)
        excellent_sharpe = summary.get('Sharpe Excellent Count', 0)
        good_sharpe = summary.get('Sharpe Good Count', 0)

        logger.metric(f"  Positive Sharpe (>0): {positive_sharpe} symbols ({positive_sharpe/total*100:.1f}%)")
        logger.metric(f"  Excellent Sharpe (>1.0): {excellent_sharpe} symbols ({excellent_sharpe/total*100:.1f}%)")

        # Top Recommendations Section
        logger.blank()
        logger.header("🏆 TOP RECOMMENDATIONS")

        if 'Best Overall Symbol' in summary:
            best = summary['Best Overall Symbol']
            sharpe = summary.get('Best Overall Sharpe', 0)
            ret = summary.get('Best Overall Return', 0)
            dd = summary.get('Best Overall Drawdown', 0)
            logger.success(f"  Best Overall: {best}")
            logger.dim(f"    Sharpe: {sharpe:.2f} | Return: {ret:.2f}% | Drawdown: {dd:.2f}%")

        if 'Best Symbol (Return)' in summary and summary.get('Best Symbol (Return)') != summary.get('Best Overall Symbol'):
            best_ret_sym = summary['Best Symbol (Return)']
            best_ret_val = summary.get('Best Return Value', 0)
            logger.profit(f"  Highest Return: {best_ret_sym} (+{best_ret_val:.2f}%)")

        if 'Safest Symbol' in summary:
            safest = summary['Safest Symbol']
            safe_dd = summary.get('Safest Drawdown Value', 0)
            logger.info(f"  Safest: {safest} (Drawdown: {safe_dd:.2f}%)")

        # Strategy Assessment Section
        logger.blank()
        logger.header("✅ STRATEGY ASSESSMENT")

        high_quality = summary.get('High Quality Count', 0)
        risk_reward = summary.get('Risk-Reward Ratio', 0)

        # Consistency assessment
        if consistency >= 7:
            logger.success(f"  ✓ Consistency: Excellent ({consistency:.1f}/10)")
        elif consistency >= 5:
            logger.info(f"  ✓ Consistency: Good ({consistency:.1f}/10)")
        else:
            logger.warning(f"  ⚠ Consistency: Needs Improvement ({consistency:.1f}/10)")

        # Win rate assessment
        if win_rate >= 70:
            logger.success(f"  ✓ Win Rate: Excellent ({win_rate:.1f}% profitable)")
        elif win_rate >= 50:
            logger.info(f"  ✓ Win Rate: Good ({win_rate:.1f}% profitable)")
        else:
            logger.warning(f"  ⚠ Win Rate: Low ({win_rate:.1f}% profitable)")

        # Risk assessment
        if median_dd > -10:
            logger.success(f"  ✓ Risk: Low (median DD {median_dd:.2f}%)")
        elif median_dd > -20:
            logger.warning(f"  ⚠ Risk: Moderate (median DD {median_dd:.2f}%)")
        else:
            logger.error(f"  ✗ Risk: High (median DD {median_dd:.2f}%)")

        # Quality count
        logger.metric(f"  High Quality Trades: {high_quality} ({high_quality/total*100:.1f}%)")
        logger.dim(f"    (Sharpe > 1.0 AND Drawdown > -15%)")

        # Risk-reward ratio
        if risk_reward >= 2.0:
            logger.success(f"  ✓ Risk-Reward Ratio: {risk_reward:.2f} (Excellent)")
        elif risk_reward >= 1.0:
            logger.info(f"  ✓ Risk-Reward Ratio: {risk_reward:.2f} (Good)")
        else:
            logger.warning(f"  ⚠ Risk-Reward Ratio: {risk_reward:.2f} (Poor)")

        # Trading Activity
        total_trades = summary.get('Total Trades (All Symbols)', 0)
        if total_trades > 0:
            logger.blank()
            logger.metric(f"Total Trading Activity: {total_trades} trades across all symbols")

        logger.separator("=", 80)
        logger.blank()

    @classmethod
    def filter_results(
        cls,
        df: pd.DataFrame,
        min_sharpe: Optional[float] = None,
        min_return: Optional[float] = None,
        min_trades: Optional[int] = None,
        max_drawdown: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter results based on criteria.

        Args:
            df: DataFrame with aggregated results
            min_sharpe: Minimum Sharpe ratio
            min_return: Minimum total return percentage
            min_trades: Minimum number of trades
            max_drawdown: Maximum drawdown percentage (absolute value)

        Returns:
            Filtered DataFrame
        """
        filtered = df.copy()

        if min_sharpe is not None:
            filtered = filtered[filtered['Sharpe Ratio'] >= min_sharpe]

        if min_return is not None:
            filtered = filtered[filtered['Total Return [%]'] >= min_return]

        if min_trades is not None:
            filtered = filtered[filtered['Total Trades'] >= min_trades]

        if max_drawdown is not None:
            filtered = filtered[filtered['Max Drawdown [%]'].abs() <= abs(max_drawdown)]

        logger.info(f"Filtered {len(df)} -> {len(filtered)} results")

        return filtered

    @classmethod
    def get_top_performers(
        cls,
        df: pd.DataFrame,
        metric: str = 'Sharpe Ratio',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N performers by metric.

        Args:
            df: DataFrame with aggregated results
            metric: Metric to rank by (default: 'Sharpe Ratio')
            top_n: Number of top performers to return

        Returns:
            DataFrame with top N performers
        """
        if metric not in df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return pd.DataFrame()

        return df.nlargest(top_n, metric)

    @classmethod
    def get_bottom_performers(
        cls,
        df: pd.DataFrame,
        metric: str = 'Total Return [%]',
        bottom_n: int = 10
    ) -> pd.DataFrame:
        """
        Get bottom N performers by metric.

        Args:
            df: DataFrame with aggregated results
            metric: Metric to rank by (default: 'Total Return [%]')
            bottom_n: Number of bottom performers to return

        Returns:
            DataFrame with bottom N performers
        """
        if metric not in df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return pd.DataFrame()

        return df.nsmallest(bottom_n, metric)

    @classmethod
    def export_to_csv(
        cls,
        df: pd.DataFrame,
        output_path: Path | str,
        include_summary: bool = True
    ) -> None:
        """
        Export results to CSV file.

        Args:
            df: DataFrame with aggregated results
            output_path: Path to save CSV
            include_summary: Include summary statistics at the end (default: True)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if include_summary:
            summary = cls.calculate_summary_stats(df)

            with open(output_path, 'w') as f:
                df.to_csv(f, index=False)

                f.write('\n\n')
                f.write('SUMMARY STATISTICS\n')

                for key, value in summary.items():
                    if isinstance(value, float):
                        f.write(f'{key},{value:.4f}\n')
                    else:
                        f.write(f'{key},{value}\n')
        else:
            df.to_csv(output_path, index=False)

        logger.success(f"Results exported to: {output_path}")

    @classmethod
    def export_to_html(
        cls,
        df: pd.DataFrame,
        output_path: Path | str,
        title: str = "Backtest Sweep Results",
        portfolios: Optional[Dict] = None,
        data_loader = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_benchmarks: bool = True
    ) -> None:
        """
        Export results to comprehensive HTML file with dark mode, advanced charts, and quantstats-style tearsheet.

        Args:
            df: DataFrame with aggregated results
            output_path: Path to save HTML
            title: Title for HTML page
            portfolios: Optional dictionary of {symbol: Portfolio} for advanced metrics
            data_loader: Optional DataLoader instance for benchmark calculations
            start_date: Start date for benchmark calculations (YYYY-MM-DD)
            end_date: End date for benchmark calculations (YYYY-MM-DD)
            include_benchmarks: Whether to include benchmark comparisons (default True)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract timestamp from filename for CSV links
        # Filename format: YYYYMMDD_HHMMSS_strategy_sweep_results.html
        # CSV files format: YYYYMMDD_HHMMSS_SYMBOL_trades.csv
        filename_parts = output_path.stem.split('_')
        if len(filename_parts) >= 2:
            timestamp = f"{filename_parts[0]}_{filename_parts[1]}"  # YYYYMMDD_HHMMSS
        else:
            timestamp = output_path.stem  # Fallback to full stem

        # Calculate advanced metrics if portfolios provided
        if portfolios and HAS_QUANTSTATS:
            logger.info("Calculating advanced metrics from portfolios...")
            for symbol, portfolio in portfolios.items():
                if portfolio is None:
                    continue
                advanced = cls.extract_advanced_metrics(portfolio, symbol)
                if advanced and symbol in df['Symbol'].values:
                    idx = df[df['Symbol'] == symbol].index[0]
                    for key, value in advanced.items():
                        df.loc[idx, key] = value

        summary = cls.calculate_summary_stats(df)

        # Calculate aggregate portfolio metrics if portfolios available
        aggregate_metrics = {}
        aggregate_html = ""
        equity_chart_data = {}
        initial_capital = 100000  # Default, will be overridden if available

        if portfolios:
            # Try to get initial capital from first portfolio
            for portfolio in portfolios.values():
                if portfolio is not None:
                    try:
                        stats = portfolio.stats()
                        if stats is not None and 'Start Value' in stats:
                            initial_capital = float(stats['Start Value'])
                            break
                    except:
                        pass

            aggregate_metrics = PortfolioAggregator.calculate_aggregate_metrics(portfolios, initial_capital)
            aggregate_html = PortfolioAggregator.generate_aggregate_summary_html(portfolios, initial_capital)
            equity_chart_data = PortfolioAggregator.generate_portfolio_composition_chart_data(portfolios, initial_capital)

        # Calculate benchmark comparisons if enabled
        benchmark_data = {}
        benchmark_chart_data = {}
        spy_chart_data = {}
        has_benchmarks = False

        if include_benchmarks and portfolios and data_loader and start_date and end_date:
            try:
                logger.info("Calculating benchmark comparisons (buy-and-hold vs strategy)...")
                symbols = list(df['Symbol'].unique())

                # Calculate all benchmarks
                benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
                    portfolios=portfolios,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    data_loader=data_loader,
                    include_spy=True
                )

                if benchmark_data and benchmark_data.get('per_symbol'):
                    has_benchmarks = True

                    # Generate chart data for benchmark comparison
                    benchmark_chart_data = PortfolioAggregator.generate_benchmark_comparison_chart_data(
                        benchmark_data, initial_capital
                    )

                    # Generate SPY comparison chart if SPY data available
                    if benchmark_data.get('spy', {}).get('equity') is not None:
                        equity_df = PortfolioAggregator.combine_equity_curves(portfolios, initial_capital)
                        if not equity_df.empty and 'Total Portfolio' in equity_df.columns:
                            spy_chart_data = PortfolioAggregator.generate_spy_comparison_chart_data(
                                benchmark_data,
                                equity_df['Total Portfolio'],
                                initial_capital * len(symbols)
                            )

                    logger.success(f"Benchmark calculations complete: {len(benchmark_data.get('outperformers', []))} outperformers, {len(benchmark_data.get('underperformers', []))} underperformers")
                else:
                    logger.warning("No benchmark data calculated - insufficient data")

            except Exception as e:
                logger.warning(f"Could not calculate benchmarks: {e}")
                has_benchmarks = False

        # Prepare data for charts
        symbols_list = df['Symbol'].tolist()
        returns_list = df['Total Return [%]'].tolist()
        sharpe_list = df['Sharpe Ratio'].tolist()
        drawdown_list = df['Max Drawdown [%]'].tolist()
        win_rate_list = df['Win Rate [%]'].tolist() if 'Win Rate [%]' in df.columns else [0] * len(symbols_list)

        # Calculate performance assessment
        median_sharpe = summary.get('Sharpe Ratio - Median', 0)
        median_return = summary.get('Total Return [%] - Median', 0)
        mean_dd = summary.get('Max Drawdown [%] - Mean', 0)
        win_rate_pct = summary.get('Win Rate (Symbols)', 0)

        # Performance rating (similar to QuantStats)
        score = 0
        if median_sharpe >= 2.0:
            score += 30
        elif median_sharpe >= 1.5:
            score += 25
        elif median_sharpe >= 1.0:
            score += 20
        elif median_sharpe >= 0.5:
            score += 10

        if median_return >= 50:
            score += 30
        elif median_return >= 30:
            score += 25
        elif median_return >= 15:
            score += 20
        elif median_return >= 0:
            score += 10

        if mean_dd >= -10:
            score += 25
        elif mean_dd >= -15:
            score += 20
        elif mean_dd >= -20:
            score += 15
        else:
            score += 5

        if win_rate_pct >= 80:
            score += 15
        elif win_rate_pct >= 60:
            score += 10
        elif win_rate_pct >= 50:
            score += 5

        # Determine rating
        if score >= 85:
            performance_rating = "Excellent"
            performance_color = "#10b981"
        elif score >= 70:
            performance_rating = "Good"
            performance_color = "#3b82f6"
        elif score >= 55:
            performance_rating = "Acceptable"
            performance_color = "#f59e0b"
        elif score >= 40:
            performance_rating = "Below Average"
            performance_color = "#f97316"
        else:
            performance_rating = "Poor"
            performance_color = "#ef4444"

        # CSV data for download
        csv_data = df.to_csv(index=False)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --info: #3b82f6;
            --primary: #6366f1;
            --card-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }}

        [data-theme="dark"] {{
            --bg-primary: #1e1e1e;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3d3d3d;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --border-color: #404040;
            --card-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.4);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--bg-primary);
            border-radius: 12px;
            box-shadow: var(--card-shadow);
        }}

        h1 {{
            font-size: 2em;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
        }}

        .theme-toggle {{
            background: var(--bg-tertiary);
            border: none;
            border-radius: 50px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 1.2em;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }}

        .theme-toggle:hover {{
            transform: scale(1.05);
            background: var(--primary);
            color: white;
        }}

        h2 {{
            font-size: 1.5em;
            font-weight: 600;
            color: var(--text-primary);
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: var(--bg-primary);
            padding: 20px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}

        .metric-label {{
            font-size: 0.85em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .metric-value.positive {{
            color: var(--success);
        }}

        .metric-value.negative {{
            color: var(--danger);
        }}

        .metric-value.warning {{
            color: var(--warning);
        }}

        .metric-value.info {{
            color: var(--info);
        }}

        .chart-container {{
            background: var(--bg-primary);
            padding: 24px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            margin-bottom: 30px;
        }}

        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}

        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--bg-primary);
            box-shadow: var(--card-shadow);
            border-radius: 12px;
            overflow: hidden;
        }}

        thead {{
            background: var(--primary);
            color: white;
        }}

        th {{
            padding: 16px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
        }}

        tbody tr {{
            transition: background-color 0.2s ease;
        }}

        tbody tr:hover {{
            background: var(--bg-secondary);
        }}

        tbody tr:last-child td {{
            border-bottom: none;
        }}

        .positive-value {{
            color: var(--success);
            font-weight: 600;
        }}

        .negative-value {{
            color: var(--danger);
            font-weight: 600;
        }}

        .neutral-value {{
            color: var(--text-secondary);
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-success {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
        }}

        .badge-danger {{
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
        }}

        .badge-warning {{
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning);
        }}

        .badge-info {{
            background: rgba(59, 130, 246, 0.1);
            color: var(--info);
        }}

        .footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 40px;
            padding: 20px;
        }}

        /* Executive Summary Styles */
        .executive-summary {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--info) 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            color: white;
            box-shadow: var(--card-shadow);
        }}

        .summary-badges {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}

        .performance-badge {{
            background: white;
            color: var(--text-primary);
            padding: 15px 25px;
            border-radius: 12px;
            font-weight: 600;
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex: 1;
            min-width: 200px;
        }}

        .badge-label {{
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
        }}

        .badge-value {{
            font-size: 1.5em;
            font-weight: 700;
        }}

        /* Advanced Metrics Table */
        .advanced-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .metric-item {{
            background: var(--bg-tertiary);
            padding: 12px;
            border-radius: 8px;
        }}

        .metric-item-label {{
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}

        .metric-item-value {{
            font-size: 1.2em;
            font-weight: 600;
            color: var(--text-primary);
        }}

        /* Download Buttons */
        .download-section {{
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}

        .download-btn {{
            background: var(--primary);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }}

        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }}

        /* Expandable Sections */
        .expandable {{
            cursor: pointer;
            user-select: none;
        }}

        .expandable:hover {{
            background: var(--bg-tertiary);
        }}

        .expanded-content {{
            display: none;
            padding: 20px;
            background: var(--bg-secondary);
        }}

        .expanded-content.show {{
            display: block;
        }}

        .expand-icon {{
            transition: transform 0.3s ease;
        }}

        .expand-icon.rotated {{
            transform: rotate(90deg);
        }}

        h3 {{
            color: var(--text-primary);
            margin-bottom: 16px;
        }}

        @media (max-width: 768px) {{
            .header {{
                flex-direction: column;
                gap: 16px;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}

            .charts-row {{
                grid-template-columns: 1fr;
            }}

            .chart-wrapper {{
                height: 300px;
            }}

            .summary-badges {{
                flex-direction: column;
            }}

            .download-section {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> {title}</h1>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle Dark Mode">
                <i class="fas fa-moon"></i>
            </button>
        </div>

        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2 style="color: white; border: none; margin: 0 0 10px 0; padding: 0;"><i class="fas fa-star"></i> Executive Summary</h2>
            <p style="font-size: 1.1em; opacity: 0.95; margin-bottom: 20px;">
                Analyzed {summary.get('Total Symbols', 0)} symbols with {summary.get('Win Rate (Symbols)', 0):.1f}% profitability.
                Median return: {median_return:.2f}% | Median Sharpe: {median_sharpe:.2f}
            </p>
            <div class="summary-badges">
                <div class="performance-badge">
                    <div class="badge-label">Performance Rating</div>
                    <div class="badge-value" style="color: {performance_color};">
                        <i class="fas fa-award"></i> {performance_rating}
                    </div>
                </div>
                <div class="performance-badge">
                    <div class="badge-label">Score</div>
                    <div class="badge-value" style="color: {performance_color};">
                        {score}/100
                    </div>
                </div>
                <div class="performance-badge">
                    <div class="badge-label">High Quality Symbols</div>
                    <div class="badge-value" style="color: var(--info);">
                        {summary.get('High Quality Count', 0)} / {summary.get('Total Symbols', 0)}
                    </div>
                </div>
            </div>
        </div>

        <!-- Download Section -->
        <div class="download-section">
            <button class="download-btn" onclick="downloadCSV()">
                <i class="fas fa-download"></i> Download CSV
            </button>
            <button class="download-btn" onclick="window.print()">
                <i class="fas fa-print"></i> Print Report
            </button>
        </div>

        <!-- Combined Portfolio Section -->
        {aggregate_html}

        <!-- Trade Logs Section -->
        <div class="chart-container" style="margin-bottom: 30px;">
            <h3 style="margin-bottom: 16px; color: var(--text-primary);"><i class="fas fa-file-csv"></i> Detailed Trade Logs</h3>
            <p style="color: var(--text-secondary); margin-bottom: 20px;">
                Click below to download detailed CSV files for each symbol:
            </p>

            {''.join([f'''
            <div style="background: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6366f1;">
                <h4 style="color: var(--text-primary); margin: 0 0 12px 0; font-size: 1.1em;">
                    <i class="fas fa-chart-line" style="color: #6366f1;"></i> {symbol}
                </h4>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <a href="trades/{timestamp}_{symbol}_trades.csv"
                       download
                       style="display: inline-flex; align-items: center; gap: 6px; background: var(--primary); color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 0.9em; transition: all 0.2s;">
                        <i class="fas fa-exchange-alt"></i> Trades CSV
                    </a>
                    <a href="trades/{timestamp}_{symbol}_equity_curve.csv"
                       download
                       style="display: inline-flex; align-items: center; gap: 6px; background: #10b981; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 0.9em; transition: all 0.2s;">
                        <i class="fas fa-chart-area"></i> Equity Curve
                    </a>
                    <a href="trades/{timestamp}_{symbol}_portfolio_state.csv"
                       download
                       style="display: inline-flex; align-items: center; gap: 6px; background: #f59e0b; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 0.9em; transition: all 0.2s;">
                        <i class="fas fa-wallet"></i> Portfolio State
                    </a>
                </div>
            </div>
            ''' for symbol in df['Symbol'].unique()])}

            <div class="advanced-metrics" style="margin-top: 20px;">
                <div class="metric-item">
                    <div class="metric-item-label"><i class="fas fa-exchange-alt"></i> Trades CSV</div>
                    <div class="metric-item-value" style="font-size: 0.85em;">Entry/exit prices, P&L per trade</div>
                </div>
                <div class="metric-item">
                    <div class="metric-item-label"><i class="fas fa-chart-line"></i> Equity Curve CSV</div>
                    <div class="metric-item-value" style="font-size: 0.85em;">Bar-by-bar portfolio value</div>
                </div>
                <div class="metric-item">
                    <div class="metric-item-label"><i class="fas fa-wallet"></i> Portfolio State CSV</div>
                    <div class="metric-item-value" style="font-size: 0.85em;">Cash, positions, returns per bar</div>
                </div>
            </div>
        </div>

        <h2><i class="fas fa-chart-bar"></i> Performance Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-hashtag"></i> Total Symbols</div>
                <div class="metric-value">{summary.get('Total Symbols', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-check-circle"></i> Profitable</div>
                <div class="metric-value positive">{summary.get('Profitable Symbols', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-percentage"></i> Win Rate</div>
                <div class="metric-value {('positive' if summary.get('Win Rate (Symbols)', 0) >= 50 else 'negative')}">{summary.get('Win Rate (Symbols)', 0):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-chart-line"></i> Median Return</div>
                <div class="metric-value {('positive' if summary.get('Total Return [%] - Median', 0) >= 0 else 'negative')}">{summary.get('Total Return [%] - Median', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-chart-area"></i> Median Sharpe</div>
                <div class="metric-value {('positive' if summary.get('Sharpe Ratio - Median', 0) >= 1.0 else 'warning' if summary.get('Sharpe Ratio - Median', 0) >= 0.5 else 'negative')}">{summary.get('Sharpe Ratio - Median', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-arrow-down"></i> Mean Drawdown</div>
                <div class="metric-value negative">{summary.get('Max Drawdown [%] - Mean', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-star"></i> Consistency</div>
                <div class="metric-value {('positive' if summary.get('Consistency Score', 0) >= 7 else 'warning' if summary.get('Consistency Score', 0) >= 5 else 'negative')}">{summary.get('Consistency Score', 0):.1f}/10</div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-award"></i> High Quality</div>
                <div class="metric-value info">{summary.get('High Quality Count', 0)}</div>
            </div>
        </div>

        <h2><i class="fas fa-chart-pie"></i> Visualizations</h2>

        <!-- Individual Symbol Performance Comparison -->
        {"" if not equity_chart_data else '''
        <div class="chart-container" style="margin-bottom: 30px;">
            <h3 style="margin-bottom: 16px; color: var(--text-primary);"><i class="fas fa-chart-line"></i> Individual Symbol Performance Comparison</h3>
            <div class="chart-wrapper" style="height: 500px;">
                <canvas id="combinedEquityChart"></canvas>
            </div>
            <p style="color: var(--text-secondary); margin-top: 10px; font-size: 0.9em;">
                <i class="fas fa-info-circle"></i> Compares portfolio performance across symbols. Each symbol was allocated <strong>${initial_capital:,.0f}</strong> independently and trades on its own equity curve.
            </p>
        </div>
        '''}

        <!-- Benchmark Comparison Section -->
        {"" if not has_benchmarks else f'''
        <div class="chart-container" style="margin-bottom: 30px; border: 2px solid #10b981; border-radius: 12px;">
            <h3 style="margin-bottom: 16px; color: var(--text-primary);"><i class="fas fa-balance-scale"></i> Strategy vs Buy-and-Hold Comparison</h3>
            <p style="color: var(--text-secondary); margin-bottom: 20px; font-size: 0.95em;">
                <i class="fas fa-info-circle"></i> Compare managed trading strategy against passive buy-and-hold for each symbol.
                <strong>Solid lines</strong> show strategy performance, <strong>dashed gray lines</strong> show buy-and-hold benchmarks.
            </p>

            <!-- Benchmark Stats Cards -->
            <div class="metrics-grid" style="margin-bottom: 20px;">
                <div class="metric-card" style="border-left: 4px solid #10b981;">
                    <div class="metric-label"><i class="fas fa-trophy"></i> Outperformers</div>
                    <div class="metric-value positive">{len(benchmark_data.get('outperformers', []))}</div>
                    <div style="font-size: 0.85em; color: var(--text-secondary); margin-top: 4px;">Beat buy-and-hold</div>
                </div>
                <div class="metric-card" style="border-left: 4px solid #ef4444;">
                    <div class="metric-label"><i class="fas fa-arrow-down"></i> Underperformers</div>
                    <div class="metric-value negative">{len(benchmark_data.get('underperformers', []))}</div>
                    <div style="font-size: 0.85em; color: var(--text-secondary); margin-top: 4px;">Trailed buy-and-hold</div>
                </div>
                <div class="metric-card" style="border-left: 4px solid #6366f1;">
                    <div class="metric-label"><i class="fas fa-percent"></i> Success Rate</div>
                    <div class="metric-value info">{(len(benchmark_data.get('outperformers', [])) / max(1, len(benchmark_data.get('per_symbol', {})))) * 100:.1f}%</div>
                    <div style="font-size: 0.85em; color: var(--text-secondary); margin-top: 4px;">Symbols with alpha</div>
                </div>
            </div>

            <!-- Toggle Controls -->
            <div style="background: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: center;">
                    <button class="download-btn" onclick="toggleAllSymbols(true)" style="padding: 8px 16px; font-size: 0.9em;">
                        <i class="fas fa-eye"></i> Show All
                    </button>
                    <button class="download-btn" onclick="toggleAllSymbols(false)" style="padding: 8px 16px; font-size: 0.9em; background: var(--danger);">
                        <i class="fas fa-eye-slash"></i> Hide All
                    </button>
                    <button class="download-btn" onclick="showOnlyOutperformers()" style="padding: 8px 16px; font-size: 0.9em; background: var(--success);">
                        <i class="fas fa-star"></i> Only Outperformers
                    </button>
                    <label style="display: flex; align-items: center; gap: 8px; color: var(--text-primary); cursor: pointer; margin-left: 20px;">
                        <input type="checkbox" id="showBenchmarks" checked onchange="toggleBenchmarks()" style="width: 18px; height: 18px; cursor: pointer;">
                        <span><i class="fas fa-chart-line"></i> Show Buy-Hold Lines</span>
                    </label>
                </div>

                <!-- Symbol Toggle Checkboxes -->
                <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px; padding-top: 15px; border-top: 1px solid var(--border-color);">
                    <div style="color: var(--text-secondary); font-size: 0.9em; font-weight: 600; width: 100%;">Toggle Symbols:</div>
                    {' '.join([f"""
                    <label style="display: flex; align-items: center; gap: 6px; color: var(--text-primary); cursor: pointer;">
                        <input type="checkbox" class="symbol-toggle" data-symbol="{symbol}" checked onchange="toggleSymbol('{symbol}')" style="width: 16px; height: 16px; cursor: pointer;">
                        <span style="font-size: 0.9em;">{symbol}</span>
                    </label>
                    """ for symbol in benchmark_data.get('per_symbol', {}).keys()])}
                </div>
            </div>

            <!-- Benchmark Chart -->
            <div class="chart-wrapper" style="height: 500px;">
                <canvas id="benchmarkComparisonChart"></canvas>
            </div>
        </div>
        '''}

        <!-- SPY Comparison Section -->
        {"" if not spy_chart_data else f'''
        <div class="chart-container" style="margin-bottom: 30px; border: 2px solid #3b82f6; border-radius: 12px;">
            <h3 style="margin-bottom: 16px; color: var(--text-primary);"><i class="fas fa-chart-line"></i> Aggregate Portfolio vs S&P 500</h3>
            <p style="color: var(--text-secondary); margin-bottom: 15px; font-size: 0.95em;">
                <i class="fas fa-info-circle"></i> Compare combined strategy portfolio against the S&P 500 index (SPY) benchmark.
            </p>
            <div class="chart-wrapper" style="height: 450px;">
                <canvas id="spyComparisonChart"></canvas>
            </div>
        </div>
        '''}

        <div class="charts-row">
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Returns by Symbol</h3>
                <div class="chart-wrapper">
                    <canvas id="returnsChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Sharpe Ratio by Symbol</h3>
                <div class="chart-wrapper">
                    <canvas id="sharpeChart"></canvas>
                </div>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Drawdown Distribution</h3>
                <div class="chart-wrapper">
                    <canvas id="drawdownChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Performance Metrics</h3>
                <div class="chart-wrapper">
                    <canvas id="metricsRadar"></canvas>
                </div>
            </div>
        </div>

        <!-- Additional Visualizations -->
        <div class="charts-row">
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Returns Distribution</h3>
                <div class="chart-wrapper">
                    <canvas id="returnsDistChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Risk-Return Scatter</h3>
                <div class="chart-wrapper">
                    <canvas id="riskReturnScatter"></canvas>
                </div>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-container">
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Win Rate Distribution</h3>
                <div class="chart-wrapper">
                    <canvas id="winRateChart"></canvas>
                </div>
            </div>
        </div>

        <h2><i class="fas fa-table"></i> Detailed Results</h2>
"""

        # Determine visible columns (main table columns)
        main_cols = ['Symbol', 'Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]',
                     'Win Rate [%]', 'Total Trades', 'Profit Factor']
        visible_cols = [col for col in main_cols if col in df.columns]

        # Advanced metrics columns (shown in expandable section)
        advanced_cols = [col for col in df.columns if col not in visible_cols and col not in ['Start Value', 'End Value', 'Duration']]
        has_advanced = len(advanced_cols) > 0

        # Create styled table HTML
        html += '        <div style="overflow-x: auto;">\n'
        html += '            <table>\n'
        html += '                <thead>\n'
        html += '                    <tr>\n'
        if has_advanced:
            html += '                        <th style="width: 30px;"></th>\n'
        for col in visible_cols:
            html += f'                        <th>{col}</th>\n'
        html += '                    </tr>\n'
        html += '                </thead>\n'
        html += '                <tbody>\n'

        for idx, row in df.iterrows():
            symbol = row['Symbol']
            row_id = f"row_{idx}"

            # Main row
            html += f'                    <tr class="expandable" onclick="toggleRow(\'{row_id}\')">\n'

            # Expand icon
            if has_advanced:
                html += f'                        <td><i class="fas fa-chevron-right expand-icon" id="icon_{row_id}"></i></td>\n'

            for col in visible_cols:
                val = row[col]
                css_class = ''

                # Apply styling based on column and value
                if col == 'Total Return [%]':
                    css_class = 'positive-value' if val >= 0 else 'negative-value'
                    html += f'                        <td class="{css_class}">{val:.2f}%</td>\n'
                elif col == 'Sharpe Ratio':
                    css_class = 'positive-value' if val >= 1.0 else 'neutral-value' if val >= 0 else 'negative-value'
                    html += f'                        <td class="{css_class}">{val:.2f}</td>\n'
                elif col == 'Max Drawdown [%]':
                    css_class = 'positive-value' if val > -10 else 'negative-value'
                    html += f'                        <td class="{css_class}">{val:.2f}%</td>\n'
                elif col in ['Win Rate [%]', 'Profit Factor']:
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        html += f'                        <td>{val:.2f}</td>\n'
                    else:
                        html += f'                        <td>N/A</td>\n'
                elif col == 'Total Trades':
                    if not pd.isna(val):
                        html += f'                        <td>{int(val)}</td>\n'
                    else:
                        html += f'                        <td>0</td>\n'
                else:
                    html += f'                        <td>{val}</td>\n'
            html += '                    </tr>\n'

            # Expandable row with comprehensive tearsheet
            if has_advanced or portfolios:
                html += f'                    <tr id="{row_id}" class="expanded-content"><td colspan="{len(visible_cols) + 1}">\n'
                html += f'                        <h3 style="color: var(--text-primary); margin-bottom: 20px;"><i class="fas fa-file-alt"></i> Comprehensive Tearsheet for {symbol}</h3>\n'

                # Generate comprehensive tearsheet if portfolio available
                if portfolios and symbol in portfolios and portfolios[symbol] is not None:
                    portfolio = portfolios[symbol]

                    try:
                        # Generate returns
                        returns = TearsheetGenerator.generate_returns(portfolio)

                        # Section 1: All QuantStats Metrics
                        html += '<div style="margin-bottom: 30px;">\n'
                        html += '<h4 style="color: var(--text-primary); border-bottom: 2px solid var(--border-color); padding-bottom: 8px; margin-bottom: 15px;"><i class="fas fa-chart-line"></i> Performance Metrics</h4>\n'
                        qs_metrics = TearsheetGenerator.generate_all_quantstats_metrics(returns)
                        html += TearsheetGenerator.generate_metrics_table_html(qs_metrics)
                        html += '</div>\n'

                        # Section 2: Monthly Returns
                        html += '<div style="margin-bottom: 30px;">\n'
                        html += '<h4 style="color: var(--text-primary); border-bottom: 2px solid var(--border-color); padding-bottom: 8px; margin-bottom: 15px;"><i class="fas fa-calendar"></i> Monthly Returns (%)</h4>\n'
                        html += TearsheetGenerator.generate_monthly_returns_table(returns)
                        html += '</div>\n'

                        # Section 3: Yearly Returns
                        html += '<div style="margin-bottom: 30px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">\n'
                        html += '<div><h4 style="color: var(--text-primary); border-bottom: 2px solid var(--border-color); padding-bottom: 8px; margin-bottom: 15px;"><i class="fas fa-chart-bar"></i> Yearly Returns</h4>\n'
                        html += TearsheetGenerator.generate_yearly_returns_table(returns)
                        html += '</div>\n'

                        # Section 4: Drawdown Analysis
                        html += '<div><h4 style="color: var(--text-primary); border-bottom: 2px solid var(--border-color); padding-bottom: 8px; margin-bottom: 15px;"><i class="fas fa-arrow-down"></i> Worst Drawdowns</h4>\n'
                        html += TearsheetGenerator.generate_drawdown_periods_table(returns, top_n=5)
                        html += '</div></div>\n'

                        # Section 5: Best/Worst Days
                        html += '<div style="margin-bottom: 30px;">\n'
                        html += '<h4 style="color: var(--text-primary); border-bottom: 2px solid var(--border-color); padding-bottom: 8px; margin-bottom: 15px;"><i class="fas fa-trophy"></i> Best & Worst Performing Days</h4>\n'
                        best_worst = TearsheetGenerator.generate_best_worst_periods(returns)
                        html += best_worst.get('days', '')
                        html += '</div>\n'

                    except Exception as e:
                        html += f'<p style="color: var(--text-secondary);">Could not generate comprehensive tearsheet: {e}</p>\n'

                # Fallback: Show basic advanced metrics if no portfolio
                elif has_advanced:
                    html += '                        <div class="advanced-metrics">\n'
                    for adv_col in advanced_cols:
                        val = row[adv_col]
                        if pd.isna(val):
                            val_str = 'N/A'
                        elif isinstance(val, (int, float)):
                            if '[%]' in adv_col:
                                val_str = f'{val:.2f}%'
                            else:
                                val_str = f'{val:.4f}'
                        else:
                            val_str = str(val)

                        html += f'                            <div class="metric-item">\n'
                        html += f'                                <div class="metric-item-label">{adv_col}</div>\n'
                        html += f'                                <div class="metric-item-value">{val_str}</div>\n'
                        html += '                            </div>\n'
                    html += '                        </div>\n'

                html += '                    </td></tr>\n'

        html += '                </tbody>\n'
        html += '            </table>\n'
        html += '        </div>\n'

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html += f"""
        <div class="footer">
            <i class="fas fa-clock"></i> Generated: {timestamp} |
            <i class="fas fa-robot"></i> Powered by Homeguard Backtesting Engine
        </div>
    </div>

    <script>
        // Dark mode toggle
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            const icon = document.querySelector('.theme-toggle i');
            icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';

            updateChartsTheme(newTheme);
        }}

        // Load theme from localStorage
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        if (savedTheme === 'dark') {{
            document.querySelector('.theme-toggle i').className = 'fas fa-sun';
        }}

        // Get theme colors
        function getThemeColors() {{
            const theme = document.documentElement.getAttribute('data-theme');
            const isDark = theme === 'dark';

            return {{
                textColor: isDark ? '#e0e0e0' : '#212529',
                gridColor: isDark ? '#404040' : '#dee2e6',
                success: '#10b981',
                danger: '#ef4444',
                warning: '#f59e0b',
                info: '#3b82f6',
                primary: '#6366f1'
            }};
        }}

        // Chart data
        const symbols = {symbols_list};
        const returns = {returns_list};
        const sharpe = {sharpe_list};
        const drawdown = {drawdown_list};
        const winRates = {win_rate_list};

        // CSV data for download
        const csvData = `{csv_data.replace('`', '\\`')}`;

        // Chart options
        function getChartOptions(title) {{
            const colors = getThemeColors();
            return {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        ticks: {{
                            color: colors.textColor
                        }},
                        grid: {{
                            color: colors.gridColor
                        }}
                    }},
                    x: {{
                        ticks: {{
                            color: colors.textColor
                        }},
                        grid: {{
                            color: colors.gridColor
                        }}
                    }}
                }}
            }};
        }}

        // Returns Chart
        const returnsCtx = document.getElementById('returnsChart').getContext('2d');
        const returnsChart = new Chart(returnsCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Total Return (%)',
                    data: returns,
                    backgroundColor: returns.map(r => r >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: returns.map(r => r >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Returns by Symbol')
        }});

        // Sharpe Chart
        const sharpeCtx = document.getElementById('sharpeChart').getContext('2d');
        const sharpeChart = new Chart(sharpeCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Sharpe Ratio',
                    data: sharpe,
                    backgroundColor: sharpe.map(s => s >= 1.0 ? 'rgba(16, 185, 129, 0.6)' : s >= 0.5 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: sharpe.map(s => s >= 1.0 ? '#10b981' : s >= 0.5 ? '#f59e0b' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Sharpe Ratio by Symbol')
        }});

        // Drawdown Chart
        const drawdownCtx = document.getElementById('drawdownChart').getContext('2d');
        const drawdownChart = new Chart(drawdownCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Max Drawdown (%)',
                    data: drawdown,
                    backgroundColor: drawdown.map(d => d > -10 ? 'rgba(16, 185, 129, 0.6)' : d > -20 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: drawdown.map(d => d > -10 ? '#10b981' : d > -20 ? '#f59e0b' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Drawdown Distribution')
        }});

        // Radar Chart for overall metrics
        const radarCtx = document.getElementById('metricsRadar').getContext('2d');
        const radarChart = new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Win Rate', 'Median Return', 'Median Sharpe', 'Low Drawdown', 'Consistency'],
                datasets: [{{
                    label: 'Strategy Performance',
                    data: [
                        {summary.get('Win Rate (Symbols)', 0)},
                        Math.max(0, {summary.get('Total Return [%] - Median', 0)} * 2),
                        Math.max(0, {summary.get('Sharpe Ratio - Median', 0)} * 20),
                        Math.max(0, 100 + {summary.get('Max Drawdown [%] - Mean', 0)}),
                        {summary.get('Consistency Score', 0)} * 10
                    ],
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: '#6366f1',
                    borderWidth: 2,
                    pointBackgroundColor: '#6366f1',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#6366f1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        ticks: {{
                            color: getThemeColors().textColor
                        }},
                        grid: {{
                            color: getThemeColors().gridColor
                        }},
                        pointLabels: {{
                            color: getThemeColors().textColor
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});

        // Returns Distribution Histogram
        const returnsDistCtx = document.getElementById('returnsDistChart').getContext('2d');
        const returnsBins = createHistogramBins(returns, 8);
        const returnsDistChart = new Chart(returnsDistCtx, {{
            type: 'bar',
            data: {{
                labels: returnsBins.labels,
                datasets: [{{
                    label: 'Frequency',
                    data: returnsBins.data,
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: '#6366f1',
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Returns Distribution')
        }});

        // Risk-Return Scatter Plot
        const riskReturnCtx = document.getElementById('riskReturnScatter').getContext('2d');
        const scatterData = symbols.map((sym, i) => ({{
            x: Math.abs(drawdown[i]),
            y: returns[i],
            label: sym
        }}));
        const riskReturnChart = new Chart(riskReturnCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Symbols',
                    data: scatterData,
                    backgroundColor: returns.map(r => r >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: returns.map(r => r >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const point = context.raw;
                                return point.label + ': Return=' + point.y.toFixed(2) + '%, Risk=' + point.x.toFixed(2) + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Return (%)',
                            color: getThemeColors().textColor
                        }},
                        ticks: {{
                            color: getThemeColors().textColor
                        }},
                        grid: {{
                            color: getThemeColors().gridColor
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Max Drawdown (abs %)',
                            color: getThemeColors().textColor
                        }},
                        ticks: {{
                            color: getThemeColors().textColor
                        }},
                        grid: {{
                            color: getThemeColors().gridColor
                        }}
                    }}
                }}
            }}
        }});

        // Win Rate Distribution Chart
        const winRateCtx = document.getElementById('winRateChart').getContext('2d');
        const winRateChart = new Chart(winRateCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Win Rate (%)',
                    data: winRates,
                    backgroundColor: winRates.map(w => w >= 60 ? 'rgba(16, 185, 129, 0.6)' : w >= 50 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: winRates.map(w => w >= 60 ? '#10b981' : w >= 50 ? '#f59e0b' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Win Rate Distribution')
        }});

        // Combined Portfolio Equity Curve Chart
        let combinedEquityChart = null;
        if ({str(bool(equity_chart_data)).lower()}) {{
            const equityChartData = {json.dumps(equity_chart_data)};
            const equityCanvasElement = document.getElementById('combinedEquityChart');

            if (equityCanvasElement) {{
                const equityCtx = equityCanvasElement.getContext('2d');
                combinedEquityChart = new Chart(equityCtx, {{
                    type: 'line',
                    data: equityChartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    color: getThemeColors().textColor,
                                    padding: 15,
                                    font: {{
                                        size: 12
                                    }}
                                }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: '#6366f1',
                                borderWidth: 1,
                                callbacks: {{
                                    label: function(context) {{
                                        let label = context.dataset.label || '';
                                        if (label) {{
                                            label += ': ';
                                        }}
                                        label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                        return label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    maxRotation: 45,
                                    minRotation: 45,
                                    font: {{
                                        size: 12
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor,
                                    display: false
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Portfolio Value ($)',
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 15,
                                        weight: 'bold'
                                    }}
                                }},
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 13,
                                        weight: '500'
                                    }},
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString();
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        // Benchmark Comparison Chart
        let benchmarkChart = null;
        const benchmarkOutperformers = {json.dumps(benchmark_data.get('outperformers', []))};
        const benchmarkUnderperformers = {json.dumps(benchmark_data.get('underperformers', []))};

        if ({str(bool(benchmark_chart_data)).lower()}) {{
            const benchmarkChartData = {json.dumps(benchmark_chart_data)};
            const benchmarkCanvasElement = document.getElementById('benchmarkComparisonChart');

            if (benchmarkCanvasElement && benchmarkChartData.datasets) {{
                const benchmarkCtx = benchmarkCanvasElement.getContext('2d');
                benchmarkChart = new Chart(benchmarkCtx, {{
                    type: 'line',
                    data: benchmarkChartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    color: getThemeColors().textColor,
                                    padding: 12,
                                    font: {{
                                        size: 11
                                    }},
                                    boxWidth: 30,
                                    usePointStyle: false
                                }},
                                onClick: (e, legendItem, legend) => {{
                                    const index = legendItem.datasetIndex;
                                    const chart = legend.chart;
                                    const meta = chart.getDatasetMeta(index);
                                    meta.hidden = !meta.hidden;
                                    chart.update();
                                }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: '#6366f1',
                                borderWidth: 1,
                                callbacks: {{
                                    label: function(context) {{
                                        let label = context.dataset.label || '';
                                        if (label) {{
                                            label += ': ';
                                        }}
                                        label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                        return label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    maxRotation: 45,
                                    minRotation: 45,
                                    font: {{
                                        size: 11
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor,
                                    display: false
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Portfolio Value ($)',
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 14,
                                        weight: 'bold'
                                    }}
                                }},
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 12
                                    }},
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString();
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        // SPY Comparison Chart
        let spyChart = null;
        if ({str(bool(spy_chart_data)).lower()}) {{
            const spyChartData = {json.dumps(spy_chart_data)};
            const spyCanvasElement = document.getElementById('spyComparisonChart');

            if (spyCanvasElement && spyChartData.datasets) {{
                const spyCtx = spyCanvasElement.getContext('2d');
                spyChart = new Chart(spyCtx, {{
                    type: 'line',
                    data: spyChartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    color: getThemeColors().textColor,
                                    padding: 15,
                                    font: {{
                                        size: 13
                                    }}
                                }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: '#6366f1',
                                borderWidth: 1,
                                callbacks: {{
                                    label: function(context) {{
                                        let label = context.dataset.label || '';
                                        if (label) {{
                                            label += ': ';
                                        }}
                                        label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                        return label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    maxRotation: 45,
                                    minRotation: 45
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor,
                                    display: false
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Portfolio Value ($)',
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 15,
                                        weight: 'bold'
                                    }}
                                }},
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString();
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        // Toggle Functions for Benchmark Chart
        function toggleAllSymbols(show) {{
            if (!benchmarkChart) return;
            benchmarkChart.data.datasets.forEach(ds => {{
                ds.hidden = !show;
            }});
            benchmarkChart.update();
        }}

        function toggleBenchmarks() {{
            if (!benchmarkChart) return;
            const showBenchmarks = document.getElementById('showBenchmarks').checked;
            benchmarkChart.data.datasets.forEach(ds => {{
                if (ds.type === 'benchmark') {{
                    ds.hidden = !showBenchmarks;
                }}
            }});
            benchmarkChart.update();
        }}

        function toggleSymbol(symbol) {{
            if (!benchmarkChart) return;
            const checkbox = document.querySelector(`input[data-symbol="${{symbol}}"]`);
            const isChecked = checkbox ? checkbox.checked : false;

            benchmarkChart.data.datasets.forEach(ds => {{
                if (ds.symbol === symbol) {{
                    ds.hidden = !isChecked;
                }}
            }});
            benchmarkChart.update();
        }}

        function showOnlyOutperformers() {{
            if (!benchmarkChart) return;
            benchmarkChart.data.datasets.forEach(ds => {{
                if (ds.symbol) {{
                    ds.hidden = !benchmarkOutperformers.includes(ds.symbol);
                }}
            }});

            // Update checkboxes
            document.querySelectorAll('.symbol-toggle').forEach(checkbox => {{
                const symbol = checkbox.getAttribute('data-symbol');
                checkbox.checked = benchmarkOutperformers.includes(symbol);
            }});

            benchmarkChart.update();
        }}

        // Utility function to create histogram bins
        function createHistogramBins(data, numBins) {{
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binSize = (max - min) / numBins;
            const bins = new Array(numBins).fill(0);
            const labels = [];

            for (let i = 0; i < numBins; i++) {{
                const binStart = min + i * binSize;
                const binEnd = min + (i + 1) * binSize;
                labels.push(`${{binStart.toFixed(1)}} to ${{binEnd.toFixed(1)}}`);
            }}

            data.forEach(val => {{
                let binIndex = Math.floor((val - min) / binSize);
                if (binIndex >= numBins) binIndex = numBins - 1;
                if (binIndex < 0) binIndex = 0;
                bins[binIndex]++;
            }});

            return {{ labels, data: bins }};
        }}

        // Update charts when theme changes
        function updateChartsTheme(theme) {{
            const colors = getThemeColors();

            [returnsChart, sharpeChart, drawdownChart, returnsDistChart, winRateChart].forEach(chart => {{
                chart.options.scales.x.ticks.color = colors.textColor;
                chart.options.scales.y.ticks.color = colors.textColor;
                chart.options.scales.x.grid.color = colors.gridColor;
                chart.options.scales.y.grid.color = colors.gridColor;
                chart.update();
            }});

            radarChart.options.scales.r.ticks.color = colors.textColor;
            radarChart.options.scales.r.grid.color = colors.gridColor;
            radarChart.options.scales.r.pointLabels.color = colors.textColor;
            radarChart.update();

            riskReturnChart.options.scales.x.ticks.color = colors.textColor;
            riskReturnChart.options.scales.y.ticks.color = colors.textColor;
            riskReturnChart.options.scales.x.grid.color = colors.gridColor;
            riskReturnChart.options.scales.y.grid.color = colors.gridColor;
            riskReturnChart.options.scales.x.title.color = colors.textColor;
            riskReturnChart.options.scales.y.title.color = colors.textColor;
            riskReturnChart.update();

            // Update combined equity chart if it exists
            if (combinedEquityChart) {{
                combinedEquityChart.options.plugins.legend.labels.color = colors.textColor;
                combinedEquityChart.options.scales.x.ticks.color = colors.textColor;
                combinedEquityChart.options.scales.x.grid.color = colors.gridColor;
                combinedEquityChart.options.scales.y.ticks.color = colors.textColor;
                combinedEquityChart.options.scales.y.grid.color = colors.gridColor;
                combinedEquityChart.options.scales.y.title.color = colors.textColor;
                combinedEquityChart.update();
            }}

            // Update benchmark chart if it exists
            if (benchmarkChart) {{
                benchmarkChart.options.plugins.legend.labels.color = colors.textColor;
                benchmarkChart.options.scales.x.ticks.color = colors.textColor;
                benchmarkChart.options.scales.x.grid.color = colors.gridColor;
                benchmarkChart.options.scales.y.ticks.color = colors.textColor;
                benchmarkChart.options.scales.y.grid.color = colors.gridColor;
                benchmarkChart.options.scales.y.title.color = colors.textColor;
                benchmarkChart.update();
            }}

            // Update SPY chart if it exists
            if (spyChart) {{
                spyChart.options.plugins.legend.labels.color = colors.textColor;
                spyChart.options.scales.x.ticks.color = colors.textColor;
                spyChart.options.scales.x.grid.color = colors.gridColor;
                spyChart.options.scales.y.ticks.color = colors.textColor;
                spyChart.options.scales.y.grid.color = colors.gridColor;
                spyChart.options.scales.y.title.color = colors.textColor;
                spyChart.update();
            }}
        }}

        // Download CSV function
        function downloadCSV() {{
            const blob = new Blob([csvData], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'backtest_results.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }}

        // Toggle row expansion
        function toggleRow(rowId) {{
            const row = document.getElementById(rowId);
            const icon = document.getElementById('icon_' + rowId);
            if (row.classList.contains('show')) {{
                row.classList.remove('show');
                icon.classList.remove('rotated');
            }} else {{
                row.classList.add('show');
                icon.classList.add('rotated');
            }}
        }}

        // Initialize charts with correct theme
        updateChartsTheme(savedTheme);
    </script>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.success(f"HTML report exported to: {output_path}")
