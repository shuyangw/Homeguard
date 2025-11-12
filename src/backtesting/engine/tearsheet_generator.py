"""
Generate comprehensive financial tearsheets using QuantStats.

Creates professional tearsheet visualizations and metrics for backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

from src.utils import logger


class TearsheetGenerator:
    """
    Generates comprehensive financial tearsheets with QuantStats integration.
    """

    @staticmethod
    def generate_returns(portfolio) -> pd.Series:
        """
        Extract returns series from portfolio.

        Args:
            portfolio: VectorBT Portfolio object

        Returns:
            Daily returns as pandas Series
        """
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

        if equity is None:
            # Return empty Series if we can't get equity
            return pd.Series(dtype=float)

        returns = equity.pct_change().fillna(0)
        return returns

    @staticmethod
    def plot_to_base64(fig) -> str:
        """
        Convert matplotlib figure to base64 encoded string.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64 encoded PNG image string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    @staticmethod
    def generate_monthly_returns_table(returns: pd.Series) -> str:
        """
        Generate HTML table of monthly returns.

        Args:
            returns: Daily returns series

        Returns:
            HTML table string
        """
        if not HAS_QUANTSTATS:
            return "<p>QuantStats not available</p>"

        try:
            # Get monthly returns
            monthly = qs.stats.monthly_returns(returns)

            if monthly.empty:
                return "<p>Insufficient data for monthly returns</p>"

            # Create HTML table with color coding
            html = '<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">\n'
            html += '<thead><tr><th style="border: 1px solid var(--border-color); padding: 8px;">Year</th>'

            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month in months:
                html += f'<th style="border: 1px solid var(--border-color); padding: 8px;">{month}</th>'
            html += '</tr></thead>\n<tbody>\n'

            for year in monthly.index:
                html += f'<tr><td style="border: 1px solid var(--border-color); padding: 8px; font-weight: bold;">{year}</td>'
                for month_num in range(1, 13):
                    if month_num in monthly.columns:
                        val = monthly.loc[year, month_num]
                        if pd.isna(val):
                            html += '<td style="border: 1px solid var(--border-color); padding: 8px; text-align: center;">-</td>'
                        else:
                            color = '#10b981' if val > 0 else '#ef4444' if val < 0 else 'var(--text-primary)'
                            html += f'<td style="border: 1px solid var(--border-color); padding: 8px; text-align: center; color: {color}; font-weight: 600;">{val:.2f}%</td>'
                    else:
                        html += '<td style="border: 1px solid var(--border-color); padding: 8px; text-align: center;">-</td>'
                html += '</tr>\n'

            html += '</tbody></table>'
            return html

        except Exception as e:
            logger.warning(f"Could not generate monthly returns table: {e}")
            return f"<p>Error generating monthly returns: {e}</p>"

    @staticmethod
    def generate_yearly_returns_table(returns: pd.Series) -> str:
        """
        Generate HTML table of yearly returns.

        Args:
            returns: Daily returns series

        Returns:
            HTML table string
        """
        if not HAS_QUANTSTATS:
            return "<p>QuantStats not available</p>"

        try:
            yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100

            html = '<table style="width: 100%; border-collapse: collapse;">\n'
            html += '<thead><tr><th style="border: 1px solid var(--border-color); padding: 12px;">Year</th>'
            html += '<th style="border: 1px solid var(--border-color); padding: 12px;">Return</th></tr></thead>\n<tbody>\n'

            for year, ret in yearly.items():
                color = '#10b981' if ret > 0 else '#ef4444' if ret < 0 else 'var(--text-primary)'
                html += f'<tr><td style="border: 1px solid var(--border-color); padding: 10px; font-weight: bold;">{year.year}</td>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 10px; text-align: center; color: {color}; font-weight: 600; font-size: 1.1em;">{ret:.2f}%</td></tr>\n'

            html += '</tbody></table>'
            return html

        except Exception as e:
            logger.warning(f"Could not generate yearly returns table: {e}")
            return f"<p>Error generating yearly returns: {e}</p>"

    @staticmethod
    def generate_drawdown_periods_table(returns: pd.Series, top_n: int = 5) -> str:
        """
        Generate HTML table of worst drawdown periods.

        Args:
            returns: Daily returns series
            top_n: Number of worst drawdowns to show

        Returns:
            HTML table string
        """
        if not HAS_QUANTSTATS:
            return "<p>QuantStats not available</p>"

        try:
            # Calculate cumulative returns and drawdowns
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max

            # Find drawdown periods
            is_drawdown = drawdown < 0
            drawdown_periods = []

            in_drawdown = False
            start_date = None

            for date, is_dd in is_drawdown.items():
                if is_dd and not in_drawdown:
                    start_date = date
                    in_drawdown = True
                elif not is_dd and in_drawdown:
                    end_date = date
                    dd_values = drawdown[start_date:end_date]
                    worst_dd = dd_values.min()
                    drawdown_periods.append({
                        'Start': start_date,
                        'End': end_date,
                        'Drawdown': worst_dd * 100,
                        'Duration': (end_date - start_date).days
                    })
                    in_drawdown = False

            # Sort by worst drawdown
            drawdown_periods.sort(key=lambda x: x['Drawdown'])
            drawdown_periods = drawdown_periods[:top_n]

            html = '<table style="width: 100%; border-collapse: collapse;">\n'
            html += '<thead><tr>'
            html += '<th style="border: 1px solid var(--border-color); padding: 12px;">Start</th>'
            html += '<th style="border: 1px solid var(--border-color); padding: 12px;">End</th>'
            html += '<th style="border: 1px solid var(--border-color); padding: 12px;">Drawdown</th>'
            html += '<th style="border: 1px solid var(--border-color); padding: 12px;">Duration (days)</th>'
            html += '</tr></thead>\n<tbody>\n'

            for period in drawdown_periods:
                html += '<tr>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 10px;">{period["Start"].strftime("%Y-%m-%d")}</td>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 10px;">{period["End"].strftime("%Y-%m-%d")}</td>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 10px; color: #ef4444; font-weight: 600;">{period["Drawdown"]:.2f}%</td>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 10px; text-align: center;">{period["Duration"]}</td>'
                html += '</tr>\n'

            html += '</tbody></table>'
            return html

        except Exception as e:
            logger.warning(f"Could not generate drawdown periods table: {e}")
            return f"<p>Error generating drawdown periods: {e}</p>"

    @staticmethod
    def generate_best_worst_periods(returns: pd.Series) -> Dict[str, str]:
        """
        Generate HTML tables for best/worst days, weeks, months.

        Args:
            returns: Daily returns series

        Returns:
            Dictionary with 'best_days', 'worst_days', etc.
        """
        try:
            result = {}

            # Best/Worst Days
            best_days = returns.nlargest(5) * 100
            worst_days = returns.nsmallest(5) * 100

            html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">'

            # Best days
            html += '<div><h4 style="color: var(--text-primary); margin-bottom: 10px;">📈 Best Days</h4>'
            html += '<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">'
            html += '<tr><th style="border: 1px solid var(--border-color); padding: 8px;">Date</th><th style="border: 1px solid var(--border-color); padding: 8px;">Return</th></tr>'
            for date, ret in best_days.items():
                html += f'<tr><td style="border: 1px solid var(--border-color); padding: 6px;">{date.strftime("%Y-%m-%d")}</td>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 6px; color: #10b981; font-weight: 600;">{ret:.2f}%</td></tr>'
            html += '</table></div>'

            # Worst days
            html += '<div><h4 style="color: var(--text-primary); margin-bottom: 10px;">📉 Worst Days</h4>'
            html += '<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">'
            html += '<tr><th style="border: 1px solid var(--border-color); padding: 8px;">Date</th><th style="border: 1px solid var(--border-color); padding: 8px;">Return</th></tr>'
            for date, ret in worst_days.items():
                html += f'<tr><td style="border: 1px solid var(--border-color); padding: 6px;">{date.strftime("%Y-%m-%d")}</td>'
                html += f'<td style="border: 1px solid var(--border-color); padding: 6px; color: #ef4444; font-weight: 600;">{ret:.2f}%</td></tr>'
            html += '</table></div>'

            html += '</div>'
            result['days'] = html

            return result

        except Exception as e:
            logger.warning(f"Could not generate best/worst periods: {e}")
            return {'days': f'<p>Error: {e}</p>'}

    @staticmethod
    def generate_all_quantstats_metrics(returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate all QuantStats metrics.

        Args:
            returns: Daily returns series

        Returns:
            Dictionary of all metrics
        """
        if not HAS_QUANTSTATS:
            return {}

        try:
            metrics = {
                'CAGR': qs.stats.cagr(returns),
                'Sharpe': qs.stats.sharpe(returns),
                'Sortino': qs.stats.sortino(returns),
                'Max Drawdown': qs.stats.max_drawdown(returns),
                'Calmar': qs.stats.calmar(returns),
                'Volatility (annual)': qs.stats.volatility(returns, annualize=True),
                'VaR (95%)': qs.stats.value_at_risk(returns),
                'CVaR (95%)': qs.stats.cvar(returns),
                'Skewness': qs.stats.skew(returns),
                'Kurtosis': qs.stats.kurtosis(returns),
                'Win Rate': qs.stats.win_rate(returns),
                'Avg Win': qs.stats.avg_win(returns),
                'Avg Loss': qs.stats.avg_loss(returns),
                'Best Day': returns.max(),
                'Worst Day': returns.min(),
                'Winning Days %': (returns > 0).sum() / len(returns) * 100,
                'Losing Days %': (returns < 0).sum() / len(returns) * 100,
            }

            # Convert to percentages where appropriate
            for key in ['CAGR', 'Max Drawdown', 'Volatility (annual)', 'VaR (95%)', 'CVaR (95%)',
                        'Win Rate', 'Avg Win', 'Avg Loss', 'Best Day', 'Worst Day']:
                if key in metrics:
                    metrics[key] = metrics[key] * 100

            return metrics

        except Exception as e:
            logger.warning(f"Could not calculate QuantStats metrics: {e}")
            return {}

    @staticmethod
    def generate_metrics_table_html(metrics: Dict[str, Any]) -> str:
        """
        Generate HTML table from metrics dictionary.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            HTML table string
        """
        html = '<div class="advanced-metrics" style="grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">\n'

        for name, value in metrics.items():
            # Determine if value is positive/negative for coloring
            color_class = ''
            if isinstance(value, (int, float)):
                if 'Drawdown' in name or 'Loss' in name or 'Worst' in name or 'VaR' in name or 'CVaR' in name:
                    color_class = 'negative-value' if value < 0 else 'positive-value'
                else:
                    color_class = 'positive-value' if value > 0 else 'negative-value'

                # Format value
                if abs(value) >= 1:
                    val_str = f'{value:.2f}%' if any(x in name for x in ['%', 'CAGR', 'Drawdown', 'Volatility', 'VaR', 'CVaR', 'Rate', 'Win', 'Loss', 'Day']) else f'{value:.4f}'
                else:
                    val_str = f'{value:.4f}'
            else:
                val_str = str(value)
                color_class = ''

            html += f'''
            <div class="metric-item">
                <div class="metric-item-label">{name}</div>
                <div class="metric-item-value {color_class}">{val_str}</div>
            </div>
            '''

        html += '</div>'
        return html
