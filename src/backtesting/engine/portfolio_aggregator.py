"""
Portfolio aggregation for multi-symbol backtests.

Combines individual symbol portfolios into aggregate metrics and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from utils import logger


class PortfolioAggregator:
    """
    Aggregates multiple symbol portfolios into combined portfolio metrics.

    Each symbol is allocated the full starting capital independently,
    and this class provides aggregate views across all symbols.
    """

    @staticmethod
    def combine_equity_curves(portfolios: Dict, initial_capital: float) -> pd.DataFrame:
        """
        Combine equity curves from multiple portfolios.

        Args:
            portfolios: Dictionary of {symbol: Portfolio}
            initial_capital: Starting capital per symbol

        Returns:
            DataFrame with combined equity curves
        """
        equity_data = {}

        for symbol, portfolio in portfolios.items():
            if portfolio is None:
                continue

            try:
                # Try different methods to get portfolio equity curve
                equity = None

                # Method 1: Custom Portfolio .equity_curve attribute
                if hasattr(portfolio, 'equity_curve'):
                    equity = portfolio.equity_curve

                # Method 2-3: Try VectorBT attributes
                if equity is None:
                    for attr in ['value', 'total_value']:
                        if hasattr(portfolio, attr):
                            try:
                                equity = getattr(portfolio, attr)
                                if callable(equity):
                                    equity = equity()
                                if equity is not None:
                                    break
                            except:
                                continue

                if equity is not None:
                    equity_data[symbol] = equity
                else:
                    logger.warning(f"Could not extract equity for {symbol}: no equity_curve or value attribute found")
            except Exception as e:
                logger.warning(f"Could not extract equity for {symbol}: {e}")

        if not equity_data:
            return pd.DataFrame()

        # Combine into single DataFrame
        df = pd.DataFrame(equity_data)

        # Fill any missing values (symbols may have different date ranges)
        df = df.ffill().bfill().fillna(initial_capital)

        # Calculate total portfolio value (sum across all symbols)
        df['Total Portfolio'] = df.sum(axis=1)

        # Calculate equal-weighted average
        df['Equal Weight Avg'] = df.drop('Total Portfolio', axis=1).mean(axis=1)

        return df

    @staticmethod
    def calculate_aggregate_metrics(portfolios: Dict, initial_capital: float) -> Dict[str, Any]:
        """
        Calculate aggregate performance metrics across all symbols.

        Uses equal-weighted average of returns calculated from RAW equity curves
        to avoid bias from ffill/bfill creating artificial flat segments.

        Args:
            portfolios: Dictionary of {symbol: Portfolio}
            initial_capital: Starting capital per symbol

        Returns:
            Dictionary of aggregate metrics
        """
        # Extract RAW equity curves directly from portfolios (before any ffill)
        raw_equity_data = {}
        for symbol, portfolio in portfolios.items():
            if portfolio is None:
                continue
            try:
                if hasattr(portfolio, 'equity_curve'):
                    equity = portfolio.equity_curve
                elif hasattr(portfolio, 'value'):
                    equity = getattr(portfolio, 'value')
                    if callable(equity):
                        equity = equity()
                else:
                    continue

                if equity is not None and len(equity) > 0:
                    raw_equity_data[symbol] = equity
            except Exception as e:
                logger.warning(f"Could not extract equity for {symbol}: {e}")

        if not raw_equity_data:
            return {}

        # Calculate returns from RAW equity (with NaN where data doesn't exist)
        symbol_returns = {}
        for symbol, equity in raw_equity_data.items():
            symbol_returns[symbol] = equity.pct_change()

        # Combine into DataFrame (NaN preserved where symbol has no data)
        returns_df = pd.DataFrame(symbol_returns)

        # Calculate equal-weighted average returns
        # skipna=True means we only average symbols that have data on each date
        # This is the correct way to handle different date ranges
        daily_returns = returns_df.mean(axis=1, skipna=True).dropna()

        # Calculate Sharpe on equal-weighted returns
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # For total return and drawdown, get the combined equity curve (with ffill for visualization)
        equity_df = PortfolioAggregator.combine_equity_curves(portfolios, initial_capital)
        total_portfolio = equity_df['Total Portfolio']
        total_initial = initial_capital * len([p for p in portfolios.values() if p is not None])
        total_return_pct = ((total_portfolio.iloc[-1] - total_initial) / total_initial) * 100

        # Calculate max drawdown
        cummax = total_portfolio.cummax()
        drawdown = (total_portfolio - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0

        # Volatility
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # CAGR
        days = (total_portfolio.index[-1] - total_portfolio.index[0]).days
        years = days / 365.25
        cagr = ((total_portfolio.iloc[-1] / total_initial) ** (1/years) - 1) * 100 if years > 0 else 0

        return {
            'Total Initial Capital': total_initial,
            'Total Final Value': total_portfolio.iloc[-1],
            'Total Return [%]': total_return_pct,
            'Total Profit/Loss': total_portfolio.iloc[-1] - total_initial,
            'CAGR [%]': cagr,
            'Sharpe Ratio': sharpe,
            'Max Drawdown [%]': max_drawdown,
            'Volatility [%]': volatility,
            'Win Rate [%]': win_rate,
            'Number of Symbols': len([p for p in portfolios.values() if p is not None]),
            'Trading Days': len(total_portfolio),
        }

    @staticmethod
    def calculate_symbol_contributions(portfolios: Dict, initial_capital: float) -> pd.DataFrame:
        """
        Calculate each symbol's contribution to total portfolio.

        Args:
            portfolios: Dictionary of {symbol: Portfolio}
            initial_capital: Starting capital per symbol

        Returns:
            DataFrame with symbol contributions
        """
        equity_df = PortfolioAggregator.combine_equity_curves(portfolios, initial_capital)

        if equity_df.empty:
            return pd.DataFrame()

        # Calculate final values
        contributions = []

        for symbol in equity_df.columns:
            if symbol in ['Total Portfolio', 'Equal Weight Avg']:
                continue

            final_value = equity_df[symbol].iloc[-1]
            pnl = final_value - initial_capital
            pnl_pct = (pnl / initial_capital) * 100

            # Calculate allocation percentage
            total_final = equity_df['Total Portfolio'].iloc[-1]
            allocation_pct = (final_value / total_final) * 100

            contributions.append({
                'Symbol': symbol,
                'Initial Capital': initial_capital,
                'Final Value': final_value,
                'P&L': pnl,
                'Return [%]': pnl_pct,
                'Portfolio Allocation [%]': allocation_pct
            })

        return pd.DataFrame(contributions).sort_values('P&L', ascending=False)

    @staticmethod
    def generate_portfolio_composition_chart_data(portfolios: Dict, initial_capital: float) -> Dict[str, Any]:
        """
        Generate data for overlaid equity curves chart.

        Each symbol's equity curve is shown independently (not stacked),
        allowing visual comparison of performance.

        Args:
            portfolios: Dictionary of {symbol: Portfolio}
            initial_capital: Starting capital per symbol

        Returns:
            Dictionary with chart data
        """
        equity_df = PortfolioAggregator.combine_equity_curves(portfolios, initial_capital)

        if equity_df.empty:
            return {}

        # Sample data points (don't overwhelm the chart)
        sample_freq = max(1, len(equity_df) // 100)  # Max 100 data points
        sampled = equity_df.iloc[::sample_freq]

        # Prepare data for overlaid line chart (NOT stacked)
        symbols = [col for col in sampled.columns if col not in ['Total Portfolio', 'Equal Weight Avg']]

        chart_data = {
            'labels': [str(dt.date()) for dt in sampled.index],
            'datasets': []
        }

        # Color palette for lines
        colors = [
            'rgba(99, 102, 241, 1.0)',   # Indigo
            'rgba(16, 185, 129, 1.0)',   # Green
            'rgba(245, 158, 11, 1.0)',   # Orange
            'rgba(239, 68, 68, 1.0)',    # Red
            'rgba(59, 130, 246, 1.0)',   # Blue
            'rgba(236, 72, 153, 1.0)',   # Pink
            'rgba(168, 85, 247, 1.0)',   # Purple
            'rgba(34, 197, 94, 1.0)',    # Emerald
        ]

        # Add individual symbol equity curves (overlaid, not stacked)
        for idx, symbol in enumerate(symbols):
            color = colors[idx % len(colors)]
            chart_data['datasets'].append({
                'label': symbol,
                'data': sampled[symbol].tolist(),
                'backgroundColor': 'transparent',  # No fill for line chart
                'borderColor': color,
                'borderWidth': 2,
                'pointRadius': 0,  # No points, just lines
                'pointHoverRadius': 5,
                'tension': 0.1  # Slight curve smoothing
            })

        return chart_data

    @staticmethod
    def generate_aggregate_summary_html(portfolios: Dict, initial_capital: float) -> str:
        """
        Generate HTML summary of aggregate portfolio.

        Args:
            portfolios: Dictionary of {symbol: Portfolio}
            initial_capital: Starting capital per symbol

        Returns:
            HTML string
        """
        metrics = PortfolioAggregator.calculate_aggregate_metrics(portfolios, initial_capital)
        contributions = PortfolioAggregator.calculate_symbol_contributions(portfolios, initial_capital)

        html = '<div style="background: var(--bg-primary); padding: 24px; border-radius: 12px; margin-bottom: 30px; box-shadow: var(--card-shadow);">\n'
        html += '<h3 style="color: var(--text-primary); margin-bottom: 20px; border-bottom: 2px solid var(--border-color); padding-bottom: 10px;">'
        html += '<i class="fas fa-briefcase"></i> Combined Portfolio Performance</h3>\n'

        # Key metrics grid
        html += '<div class="metrics-grid" style="margin-bottom: 20px;">\n'

        # Total capital
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-dollar-sign"></i> Total Capital Deployed</div>'
        html += f'<div class="metric-value info">${metrics.get("Total Initial Capital", 0):,.0f}</div>'
        html += '<div style="font-size: 0.85em; color: var(--text-secondary); margin-top: 5px;">'
        html += f'${initial_capital:,.0f} per symbol × {metrics.get("Number of Symbols", 0)} symbols</div>'
        html += '</div>\n'

        # Final value
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-wallet"></i> Final Portfolio Value</div>'
        html += f'<div class="metric-value info">${metrics.get("Total Final Value", 0):,.0f}</div>'
        html += '</div>\n'

        # Total return
        total_return = metrics.get('Total Return [%]', 0)
        return_class = 'positive' if total_return >= 0 else 'negative'
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-chart-line"></i> Total Return</div>'
        html += f'<div class="metric-value {return_class}">{total_return:.2f}%</div>'
        html += '</div>\n'

        # P&L
        pnl = metrics.get('Total Profit/Loss', 0)
        pnl_class = 'positive' if pnl >= 0 else 'negative'
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-money-bill-wave"></i> Total P&L</div>'
        html += f'<div class="metric-value {pnl_class}">${pnl:,.0f}</div>'
        html += '</div>\n'

        # CAGR
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-chart-area"></i> CAGR</div>'
        html += f'<div class="metric-value">{metrics.get("CAGR [%]", 0):.2f}%</div>'
        html += '</div>\n'

        # Sharpe
        sharpe = metrics.get('Sharpe Ratio', 0)
        sharpe_class = 'positive' if sharpe >= 1.0 else 'warning' if sharpe >= 0.5 else 'negative'
        # Format Sharpe with 3 decimal places if value is very small, otherwise 2
        if abs(sharpe) < 0.01 and sharpe != 0:
            sharpe_str = f'{sharpe:.3f}'
        else:
            sharpe_str = f'{sharpe:.2f}'
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-balance-scale"></i> Sharpe Ratio</div>'
        html += f'<div class="metric-value {sharpe_class}">{sharpe_str}</div>'
        html += '</div>\n'

        # Max DD
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-arrow-down"></i> Max Drawdown</div>'
        html += f'<div class="metric-value negative">{metrics.get("Max Drawdown [%]", 0):.2f}%</div>'
        html += '</div>\n'

        # Volatility
        html += '<div class="metric-card">'
        html += '<div class="metric-label"><i class="fas fa-wave-square"></i> Volatility (Annual)</div>'
        html += f'<div class="metric-value">{metrics.get("Volatility [%]", 0):.2f}%</div>'
        html += '</div>\n'

        html += '</div>\n'

        # Symbol contributions table
        if not contributions.empty:
            html += '<h4 style="color: var(--text-primary); margin: 30px 0 15px 0;"><i class="fas fa-chart-pie"></i> Symbol Contributions</h4>\n'
            html += '<table style="width: 100%; border-collapse: collapse;">\n'
            html += '<thead><tr style="background: var(--primary); color: white;">'
            html += '<th style="padding: 12px; text-align: left;">Symbol</th>'
            html += '<th style="padding: 12px; text-align: right;">Initial</th>'
            html += '<th style="padding: 12px; text-align: right;">Final Value</th>'
            html += '<th style="padding: 12px; text-align: right;">P&L</th>'
            html += '<th style="padding: 12px; text-align: right;">Return</th>'
            html += '<th style="padding: 12px; text-align: right;">Portfolio %</th>'
            html += '</tr></thead>\n<tbody>\n'

            for _, row in contributions.iterrows():
                pnl_class = 'positive-value' if row['P&L'] >= 0 else 'negative-value'
                html += '<tr style="border-bottom: 1px solid var(--border-color);">'
                html += f'<td style="padding: 10px; font-weight: 600;">{row["Symbol"]}</td>'
                html += f'<td style="padding: 10px; text-align: right;">${row["Initial Capital"]:,.0f}</td>'
                html += f'<td style="padding: 10px; text-align: right;">${row["Final Value"]:,.0f}</td>'
                html += f'<td style="padding: 10px; text-align: right;" class="{pnl_class}">${row["P&L"]:,.0f}</td>'
                html += f'<td style="padding: 10px; text-align: right;" class="{pnl_class}">{row["Return [%]"]:.2f}%</td>'
                html += f'<td style="padding: 10px; text-align: right;">{row["Portfolio Allocation [%]"]:.1f}%</td>'
                html += '</tr>\n'

            html += '</tbody></table>\n'

        html += '</div>\n'

        return html

    @staticmethod
    def generate_benchmark_comparison_chart_data(
        benchmark_data: Dict,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Generate chart data for strategy vs buy-and-hold comparison.

        Creates datasets for Chart.js showing strategy equity curves
        overlaid with buy-and-hold benchmarks.

        Args:
            benchmark_data: Output from BenchmarkCalculator.calculate_all_benchmarks()
            initial_capital: Starting capital per symbol

        Returns:
            Dictionary with chart data including:
            - labels: Timestamp labels
            - datasets: Array of strategy + benchmark datasets
            - outperformers: List of symbols that beat buy-and-hold
        """
        if not benchmark_data or not benchmark_data.get('per_symbol'):
            return {}

        # Collect all equity series
        all_series = {}
        for symbol, data in benchmark_data['per_symbol'].items():
            if 'strategy_equity' in data:
                all_series[f'{symbol}_strategy'] = data['strategy_equity']
            if 'buy_hold_equity' in data:
                all_series[f'{symbol}_buyhold'] = data['buy_hold_equity']

        if not all_series:
            return {}

        # Combine all series into DataFrame
        df = pd.DataFrame(all_series)

        # Forward-fill missing values (different date ranges)
        df = df.ffill().bfill()

        # Sample data points (max 150 points for performance)
        sample_freq = max(1, len(df) // 150)
        sampled = df.iloc[::sample_freq]

        # Prepare chart data
        chart_data = {
            'labels': [str(dt.date()) for dt in sampled.index],
            'datasets': [],
            'outperformers': benchmark_data.get('outperformers', []),
            'underperformers': benchmark_data.get('underperformers', [])
        }

        # Color palette for strategies
        colors = [
            '#6366f1',  # Indigo
            '#10b981',  # Green
            '#f59e0b',  # Orange
            '#3b82f6',  # Blue
            '#ec4899',  # Pink
            '#a855f7',  # Purple
            '#22c55e',  # Emerald
            '#ef4444',  # Red
        ]

        symbols = list(benchmark_data['per_symbol'].keys())

        # Add datasets (strategy + benchmark for each symbol)
        for idx, symbol in enumerate(symbols):
            color = colors[idx % len(colors)]

            # Strategy line (solid, colored)
            strategy_col = f'{symbol}_strategy'
            if strategy_col in sampled.columns:
                chart_data['datasets'].append({
                    'label': f'{symbol} Strategy',
                    'data': sampled[strategy_col].tolist(),
                    'backgroundColor': 'transparent',
                    'borderColor': color,
                    'borderWidth': 2.5,
                    'pointRadius': 0,
                    'pointHoverRadius': 6,
                    'tension': 0.1,
                    'hidden': False,
                    'symbol': symbol,
                    'type': 'strategy'
                })

            # Buy-and-hold line (dashed, gray)
            buyhold_col = f'{symbol}_buyhold'
            if buyhold_col in sampled.columns:
                chart_data['datasets'].append({
                    'label': f'{symbol} Buy-Hold',
                    'data': sampled[buyhold_col].tolist(),
                    'backgroundColor': 'transparent',
                    'borderColor': '#9ca3af',  # Gray
                    'borderWidth': 1.5,
                    'borderDash': [5, 5],
                    'pointRadius': 0,
                    'pointHoverRadius': 4,
                    'tension': 0.1,
                    'hidden': False,
                    'symbol': symbol,
                    'type': 'benchmark'
                })

        return chart_data

    @staticmethod
    def generate_spy_comparison_chart_data(
        benchmark_data: Dict,
        aggregate_equity: pd.Series,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Generate chart data for aggregate portfolio vs SPY comparison.

        Args:
            benchmark_data: Output from BenchmarkCalculator.calculate_all_benchmarks()
            aggregate_equity: Combined portfolio equity curve
            initial_capital: Total initial capital

        Returns:
            Dictionary with chart data for SPY comparison
        """
        if not benchmark_data or benchmark_data.get('spy', {}).get('equity') is None:
            return {}

        spy_equity = benchmark_data['spy']['equity']

        if spy_equity is None or spy_equity.empty or aggregate_equity.empty:
            return {}

        # Combine into DataFrame
        df = pd.DataFrame({
            'Strategy': aggregate_equity,
            'SPY': spy_equity
        })

        # Forward-fill missing values
        df = df.ffill().bfill()

        # Sample data points
        sample_freq = max(1, len(df) // 150)
        sampled = df.iloc[::sample_freq]

        chart_data = {
            'labels': [str(dt.date()) for dt in sampled.index],
            'datasets': [
                {
                    'label': 'Strategy (Aggregate)',
                    'data': sampled['Strategy'].tolist(),
                    'backgroundColor': 'transparent',
                    'borderColor': '#6366f1',  # Indigo
                    'borderWidth': 3,
                    'pointRadius': 0,
                    'pointHoverRadius': 6,
                    'tension': 0.1
                },
                {
                    'label': 'S&P 500 (SPY)',
                    'data': sampled['SPY'].tolist(),
                    'backgroundColor': 'transparent',
                    'borderColor': '#9ca3af',  # Gray
                    'borderWidth': 2,
                    'borderDash': [5, 5],
                    'pointRadius': 0,
                    'pointHoverRadius': 5,
                    'tension': 0.1
                }
            ]
        }

        return chart_data
