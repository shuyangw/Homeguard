"""
Multi-symbol portfolio chart data generator.

Generates Chart.js-compatible data structures for visualizing multi-asset portfolio dynamics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

# Import logger for error reporting
try:
    from utils.logger import log_error, log_warning
except ImportError:
    # Fallback if logger not available
    def log_error(msg): print(f"ERROR: {msg}")
    def log_warning(msg): print(f"WARNING: {msg}")


class MultiSymbolChartGenerator:
    """
    Generate Chart.js-compatible chart data for multi-symbol portfolios.

    Creates interactive visualizations showing portfolio composition, symbol attribution,
    diversification, and other multi-asset specific insights.
    """

    # Color palette for charts (professional, accessible colors)
    COLORS = [
        'rgba(54, 162, 235, 0.8)',   # Blue
        'rgba(255, 99, 132, 0.8)',   # Red
        'rgba(75, 192, 192, 0.8)',   # Teal
        'rgba(255, 206, 86, 0.8)',   # Yellow
        'rgba(153, 102, 255, 0.8)',  # Purple
        'rgba(255, 159, 64, 0.8)',   # Orange
        'rgba(201, 203, 207, 0.8)',  # Grey
        'rgba(100, 181, 246, 0.8)',  # Light Blue
        'rgba(129, 199, 132, 0.8)',  # Green
        'rgba(255, 138, 128, 0.8)',  # Salmon
    ]

    @staticmethod
    def _get_color(index: int, alpha: float = 0.8) -> str:
        """Get color from palette with specified alpha."""
        base_color = MultiSymbolChartGenerator.COLORS[index % len(MultiSymbolChartGenerator.COLORS)]
        # Replace alpha value
        return base_color.replace('0.8', str(alpha))

    @staticmethod
    def _downsample_timeseries(timestamps: List[pd.Timestamp], values: List[Any], max_points: int = 250) -> tuple:
        """
        Downsample time series data to reduce chart rendering time.
        Uses intelligent time-based resampling (daily or weekly).

        Args:
            timestamps: List of timestamps
            values: List of values (can be dict for multi-series)
            max_points: Maximum number of points to keep (default: 250 for ~1 year daily)

        Returns:
            Tuple of (downsampled_timestamps, downsampled_values)
        """
        if len(timestamps) <= max_points:
            return timestamps, values

        # Use time-based resampling for better results
        # Create pandas Series for intelligent resampling
        series = pd.Series(values, index=timestamps)

        # Determine appropriate frequency based on time range
        time_range = timestamps[-1] - timestamps[0]
        days = time_range.days

        if days <= 90:
            # 3 months or less: daily
            resampled = series.resample('1D').last()
        elif days <= 365:
            # 1 year or less: daily (but drop weekends/gaps)
            resampled = series.resample('1D').last().dropna()
        elif days <= 1095:
            # 1-3 years: weekly
            resampled = series.resample('1W').last()
        else:
            # >3 years: monthly
            resampled = series.resample('1M').last()

        # Still apply max_points limit if needed
        if len(resampled) > max_points:
            step = len(resampled) // max_points
            resampled = resampled.iloc[::step]

        return list(resampled.index), list(resampled.values)

    @staticmethod
    def generate_portfolio_composition_chart(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Generate stacked area chart showing portfolio composition over time.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Chart.js data structure for stacked area chart
        """
        if not portfolio.symbol_weights_history:
            return {}

        # Extract unique timestamps and symbols
        raw_timestamps = [ts for ts, _ in portfolio.symbol_weights_history]
        raw_weights = [weights for _, weights in portfolio.symbol_weights_history]

        # Simple downsampling approach (faster, more reliable)
        # Limit to ~500 points maximum for clean charts
        max_points = 500
        if len(raw_timestamps) > max_points:
            step = len(raw_timestamps) // max_points
            sampled_timestamps = raw_timestamps[::step]
            sampled_weights = raw_weights[::step]
        else:
            sampled_timestamps = raw_timestamps
            sampled_weights = raw_weights

        # Extract timestamps and weights
        timestamps = [ts.strftime('%Y-%m-%d') for ts in sampled_timestamps]
        weights_by_symbol = {symbol: [] for symbol in portfolio.symbols}

        for weights_dict in sampled_weights:
            for symbol in portfolio.symbols:
                weight = weights_dict.get(symbol, 0) * 100  # Convert to percentage
                weights_by_symbol[symbol].append(weight)

        # Create datasets for each symbol
        datasets = []
        for idx, symbol in enumerate(portfolio.symbols):
            datasets.append({
                'label': symbol,
                'data': weights_by_symbol[symbol],
                'backgroundColor': MultiSymbolChartGenerator._get_color(idx, 0.6),
                'borderColor': MultiSymbolChartGenerator._get_color(idx, 1.0),
                'borderWidth': 1,
                'fill': True,
            })

        return {
            'labels': timestamps,
            'datasets': datasets,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'x': {'stacked': True, 'title': {'display': True, 'text': 'Time'}},
                    'y': {'stacked': True, 'title': {'display': True, 'text': 'Portfolio Weight (%)'}, 'max': 100}
                },
                'plugins': {
                    'title': {'display': True, 'text': 'Portfolio Composition Over Time'},
                    'legend': {'position': 'bottom'},
                    'tooltip': {'mode': 'index', 'intersect': False}
                }
            }
        }

    @staticmethod
    def generate_pnl_contribution_pie_chart(
        portfolio: 'MultiAssetPortfolio',
        symbol_attribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate pie chart showing P&L contribution by symbol.

        Args:
            portfolio: MultiAssetPortfolio instance
            symbol_attribution: Attribution metrics from MultiSymbolMetrics

        Returns:
            Chart.js data structure for pie chart
        """
        per_symbol = symbol_attribution.get('per_symbol', {})
        if not per_symbol:
            return {}

        labels = []
        data = []
        colors = []

        for idx, (symbol, stats) in enumerate(per_symbol.items()):
            labels.append(symbol)
            data.append(stats['Total P&L'])
            colors.append(MultiSymbolChartGenerator._get_color(idx))

        return {
            'labels': labels,
            'datasets': [{
                'label': 'P&L Contribution',
                'data': data,
                'backgroundColor': colors,
                'borderColor': 'rgba(255, 255, 255, 0.8)',
                'borderWidth': 2,
            }],
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {'display': True, 'text': 'P&L Contribution by Symbol'},
                    'legend': {'position': 'right'},
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return context.label + ": $" + context.parsed.toFixed(2); }'
                        }
                    }
                }
            }
        }

    @staticmethod
    def generate_per_symbol_equity_chart(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Generate overlaid line chart showing per-symbol equity curves.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Chart.js data structure for line chart
        """
        # PERFORMANCE FIX: For very large datasets (>100K bars), skip expensive per-symbol calculation
        # Reconstructing symbol equity from trades is O(n*m) complexity - skip for minute-level data
        if len(portfolio.equity_curve) > 100000:
            # Just show portfolio total equity curve (fast)
            max_points = 500
            step = max(1, len(portfolio.equity_curve) // max_points)
            timestamps = [ts.strftime('%Y-%m-%d') for ts in portfolio.equity_timestamps[::step]]

            return {
                'labels': timestamps,
                'datasets': [{
                    'label': 'Total Portfolio',
                    'data': portfolio.equity_curve[::step],
                    'borderColor': 'rgba(54, 162, 235, 1.0)',
                    'backgroundColor': 'transparent',
                    'borderWidth': 2,
                    'pointRadius': 0,
                    'tension': 0.1,
                }],
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'x': {'title': {'display': True, 'text': 'Time'}},
                        'y': {'title': {'display': True, 'text': 'Equity ($)'}}
                    },
                    'plugins': {
                        'title': {'display': True, 'text': 'Portfolio Equity (Per-Symbol View Disabled for Performance)'},
                        'legend': {'position': 'bottom'},
                        'tooltip': {
                            'callbacks': {
                                'label': 'function(context) { return "$" + context.parsed.y.toLocaleString(); }'
                            }
                        }
                    }
                }
            }

        # For smaller datasets, use the full per-symbol breakdown
        symbol_equity = portfolio.get_symbol_equity_curves(resample='1H')
        if not symbol_equity:
            return {}

        # Downsample to ~500 points for clean charts
        first_symbol_equity = next(iter(symbol_equity.values()))
        max_points = 500
        if len(first_symbol_equity) > max_points:
            step = len(first_symbol_equity) // max_points
            # Resample each symbol's equity
            sampled_equity = {}
            for symbol, equity in symbol_equity.items():
                sampled_equity[symbol] = equity.iloc[::step]
            symbol_equity = sampled_equity
            first_symbol_equity = next(iter(symbol_equity.values()))

        # Get timestamps
        timestamps = [ts.strftime('%Y-%m-%d') for ts in first_symbol_equity.index]

        # Create datasets for each symbol
        datasets = []
        for idx, (symbol, equity) in enumerate(symbol_equity.items()):
            datasets.append({
                'label': symbol,
                'data': equity.tolist(),
                'borderColor': MultiSymbolChartGenerator._get_color(idx, 1.0),
                'backgroundColor': 'transparent',
                'borderWidth': 2,
                'pointRadius': 0,  # Hide points for cleaner look
                'tension': 0.1,
            })

        # PERFORMANCE FIX: Downsample total portfolio equity before creating Series
        if len(portfolio.equity_curve) > max_points:
            downsampled_equity = pd.Series(
                portfolio.equity_curve[::step],
                index=portfolio.equity_timestamps[::step]
            )
        else:
            downsampled_equity = pd.Series(portfolio.equity_curve, index=portfolio.equity_timestamps)

        # Add aggregate portfolio line
        datasets.append({
            'label': 'Total Portfolio',
            'data': downsampled_equity.tolist(),
            'borderColor': 'rgba(0, 0, 0, 0.9)',
            'backgroundColor': 'transparent',
            'borderWidth': 3,
            'pointRadius': 0,
            'tension': 0.1,
            'borderDash': [5, 5],  # Dashed line to distinguish
        })

        return {
            'labels': timestamps,
            'datasets': datasets,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'x': {'title': {'display': True, 'text': 'Time'}},
                    'y': {'title': {'display': True, 'text': 'Equity ($)'}}
                },
                'plugins': {
                    'title': {'display': True, 'text': 'Per-Symbol Equity Curves'},
                    'legend': {'position': 'bottom'},
                    'tooltip': {'mode': 'index', 'intersect': False}
                }
            }
        }

    @staticmethod
    def generate_correlation_matrix_chart(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Generate heatmap showing symbol correlation matrix.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Data structure for heatmap visualization
        """
        # Use simple downsampling for correlation (faster)
        symbol_equity = portfolio.get_symbol_equity_curves(resample=None)
        if not symbol_equity:
            return {}

        # Downsample if needed
        first_symbol_equity = next(iter(symbol_equity.values()))
        max_points = 500
        if len(first_symbol_equity) > max_points:
            step = len(first_symbol_equity) // max_points
            sampled_equity = {}
            for symbol, equity in symbol_equity.items():
                sampled_equity[symbol] = equity.iloc[::step]
            symbol_equity = sampled_equity

        # Calculate returns for each symbol
        returns_data = {}
        for symbol, equity in symbol_equity.items():
            returns_data[symbol] = equity.pct_change().dropna()

        # Create returns DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()

        if corr_matrix.empty:
            return {}

        # Convert to list of lists for heatmap
        symbols = corr_matrix.columns.tolist()
        data = []

        for i, row_symbol in enumerate(symbols):
            for j, col_symbol in enumerate(symbols):
                data.append({
                    'x': col_symbol,
                    'y': row_symbol,
                    'value': round(corr_matrix.iloc[i, j], 3)
                })

        return {
            'data': data,
            'symbols': symbols,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {'display': True, 'text': 'Symbol Correlation Matrix'},
                    'legend': {'display': False},
                    'tooltip': {
                        'callbacks': {
                            'title': 'function(context) { return context[0].raw.x + " vs " + context[0].raw.y; }',
                            'label': 'function(context) { return "Correlation: " + context.raw.value.toFixed(3); }'
                        }
                    }
                }
            }
        }

    @staticmethod
    def generate_drawdown_timeline_chart(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Generate area chart showing drawdown from peak over time.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Chart.js data structure for area chart
        """
        # PERFORMANCE FIX: Downsample before creating Series
        max_points = 500
        if len(portfolio.equity_curve) > max_points:
            step = len(portfolio.equity_curve) // max_points
            equity = pd.Series(
                portfolio.equity_curve[::step],
                index=portfolio.equity_timestamps[::step]
            )
        else:
            equity = pd.Series(portfolio.equity_curve, index=portfolio.equity_timestamps)

        cummax = equity.cummax()
        drawdown = ((equity - cummax) / cummax) * 100  # Percentage

        timestamps = [ts.strftime('%Y-%m-%d') for ts in equity.index]

        return {
            'labels': timestamps,
            'datasets': [{
                'label': 'Drawdown (%)',
                'data': drawdown.tolist(),
                'backgroundColor': 'rgba(255, 99, 132, 0.3)',
                'borderColor': 'rgba(255, 99, 132, 1.0)',
                'borderWidth': 2,
                'fill': True,
                'tension': 0.1,
            }],
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'x': {'title': {'display': True, 'text': 'Time'}},
                    'y': {'title': {'display': True, 'text': 'Drawdown (%)'},  'max': 0}
                },
                'plugins': {
                    'title': {'display': True, 'text': 'Portfolio Drawdown Timeline'},
                    'legend': {'display': False},
                    'tooltip': {'mode': 'index', 'intersect': False}
                }
            }
        }

    @staticmethod
    def generate_monthly_returns_heatmap(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Generate calendar heatmap showing monthly returns.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Data structure for monthly returns heatmap
        """
        # PERFORMANCE FIX: For monthly returns, we only need daily/hourly data
        # Downsample before creating Series for large datasets
        if len(portfolio.equity_curve) > 10000:
            equity = pd.Series(portfolio.equity_curve, index=portfolio.equity_timestamps)
            equity = equity.resample('1H').last()
        else:
            equity = pd.Series(portfolio.equity_curve, index=portfolio.equity_timestamps)

        monthly_returns = equity.resample('M').last().pct_change().dropna() * 100

        if len(monthly_returns) == 0:
            return {}

        # Organize by year and month
        data = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for timestamp, ret in monthly_returns.items():
            data.append({
                'year': timestamp.year,
                'month': months[timestamp.month - 1],
                'value': round(ret, 2)
            })

        return {
            'data': data,
            'months': months,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {'display': True, 'text': 'Monthly Returns Heatmap'},
                    'legend': {'display': False},
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return context.raw.value.toFixed(2) + "%"; }'
                        }
                    }
                }
            }
        }

    @staticmethod
    def generate_position_count_timeline_chart(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Generate line chart showing number of positions over time.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Chart.js data structure for line chart
        """
        if not portfolio.position_count_history:
            return {}

        # Simple downsampling
        raw_timestamps = [ts for ts, _ in portfolio.position_count_history]
        raw_counts = [count for _, count in portfolio.position_count_history]

        max_points = 500
        if len(raw_timestamps) > max_points:
            step = len(raw_timestamps) // max_points
            timestamps = [raw_timestamps[i].strftime('%Y-%m-%d') for i in range(0, len(raw_timestamps), step)]
            counts = [raw_counts[i] for i in range(0, len(raw_counts), step)]
        else:
            timestamps = [ts.strftime('%Y-%m-%d') for ts in raw_timestamps]
            counts = raw_counts

        return {
            'labels': timestamps,
            'datasets': [{
                'label': 'Position Count',
                'data': counts,
                'borderColor': 'rgba(75, 192, 192, 1.0)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'borderWidth': 2,
                'fill': True,
                'stepped': True,  # Use step line for discrete counts
            }],
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'x': {'title': {'display': True, 'text': 'Time'}},
                    'y': {'title': {'display': True, 'text': 'Number of Positions'}, 'ticks': {'stepSize': 1}}
                },
                'plugins': {
                    'title': {'display': True, 'text': 'Portfolio Diversification Over Time'},
                    'legend': {'display': False},
                    'tooltip': {'mode': 'index', 'intersect': False}
                }
            }
        }

    @staticmethod
    def generate_rolling_sharpe_chart(portfolio: 'MultiAssetPortfolio', window: int = 30) -> Dict[str, Any]:
        """
        Generate line chart showing rolling Sharpe ratio.

        Args:
            portfolio: MultiAssetPortfolio instance
            window: Rolling window size in days (default: 30)

        Returns:
            Chart.js data structure for line chart
        """
        # PERFORMANCE FIX: Get hourly returns for large datasets
        returns = portfolio.returns(freq='1H' if len(portfolio.equity_curve) > 10000 else None)
        if len(returns) < window:
            return {}

        # Downsample returns for performance
        max_points = 500
        if len(returns) > max_points:
            step = len(returns) // max_points
            downsampled_returns = returns.iloc[::step]
        else:
            downsampled_returns = returns

        rolling_mean = downsampled_returns.rolling(window=window).mean()
        rolling_std = downsampled_returns.rolling(window=window).std()

        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()

        timestamps = [ts.strftime('%Y-%m-%d') for ts in rolling_sharpe.index]

        return {
            'labels': timestamps,
            'datasets': [{
                'label': f'{window}-Day Rolling Sharpe',
                'data': rolling_sharpe.tolist(),
                'borderColor': 'rgba(153, 102, 255, 1.0)',
                'backgroundColor': 'transparent',
                'borderWidth': 2,
                'tension': 0.2,
            }],
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'x': {'title': {'display': True, 'text': 'Time'}},
                    'y': {'title': {'display': True, 'text': 'Sharpe Ratio'}}
                },
                'plugins': {
                    'title': {'display': True, 'text': f'{window}-Day Rolling Sharpe Ratio'},
                    'legend': {'display': False},
                    'tooltip': {'mode': 'index', 'intersect': False}
                }
            }
        }

    @staticmethod
    def generate_symbol_performance_heatmap(
        portfolio: 'MultiAssetPortfolio',
        symbol_attribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate heatmap showing multiple metrics per symbol.

        Args:
            portfolio: MultiAssetPortfolio instance
            symbol_attribution: Attribution metrics from MultiSymbolMetrics

        Returns:
            Data structure for performance heatmap
        """
        per_symbol = symbol_attribution.get('per_symbol', {})
        if not per_symbol:
            return {}

        metrics = ['Total Return [%]', 'Sharpe Ratio', 'Win Rate [%]']
        symbols = list(per_symbol.keys())

        data = []
        for symbol in symbols:
            stats = per_symbol[symbol]
            for metric in metrics:
                value = stats.get(metric, 0)
                data.append({
                    'symbol': symbol,
                    'metric': metric,
                    'value': round(value, 2)
                })

        return {
            'data': data,
            'symbols': symbols,
            'metrics': metrics,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {'display': True, 'text': 'Symbol Performance Comparison'},
                    'legend': {'display': False}
                }
            }
        }

    @staticmethod
    def generate_all_charts(
        portfolio: 'MultiAssetPortfolio',
        metrics: Dict[str, Any],
        parallel: bool = True,
        max_workers: int = 9
    ) -> Dict[str, Any]:
        """
        Generate all chart data for multi-symbol portfolio.

        Args:
            portfolio: MultiAssetPortfolio instance
            metrics: Pre-calculated metrics from MultiSymbolMetrics
            parallel: If True, generate charts in parallel (default: True)
            max_workers: Maximum parallel workers (default: 9, one per chart)

        Returns:
            Dictionary with all chart data organized by chart type
        """
        attribution = metrics.get('attribution', {})

        if not parallel:
            # Sequential generation (original behavior)
            return {
                'portfolio_composition': MultiSymbolChartGenerator.generate_portfolio_composition_chart(portfolio),
                'pnl_contribution_pie': MultiSymbolChartGenerator.generate_pnl_contribution_pie_chart(portfolio, attribution),
                'per_symbol_equity': MultiSymbolChartGenerator.generate_per_symbol_equity_chart(portfolio),
                'correlation_matrix': MultiSymbolChartGenerator.generate_correlation_matrix_chart(portfolio),
                'drawdown_timeline': MultiSymbolChartGenerator.generate_drawdown_timeline_chart(portfolio),
                'monthly_returns_heatmap': MultiSymbolChartGenerator.generate_monthly_returns_heatmap(portfolio),
                'position_count_timeline': MultiSymbolChartGenerator.generate_position_count_timeline_chart(portfolio),
                'rolling_sharpe': MultiSymbolChartGenerator.generate_rolling_sharpe_chart(portfolio, window=30),
                'symbol_performance_heatmap': MultiSymbolChartGenerator.generate_symbol_performance_heatmap(portfolio, attribution),
            }

        # ============================================================
        # PARALLEL CHART GENERATION
        # ============================================================
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Define chart generation tasks
        chart_tasks = {
            'portfolio_composition': (
                MultiSymbolChartGenerator.generate_portfolio_composition_chart,
                (portfolio,)
            ),
            'pnl_contribution_pie': (
                MultiSymbolChartGenerator.generate_pnl_contribution_pie_chart,
                (portfolio, attribution)
            ),
            'per_symbol_equity': (
                MultiSymbolChartGenerator.generate_per_symbol_equity_chart,
                (portfolio,)
            ),
            'correlation_matrix': (
                MultiSymbolChartGenerator.generate_correlation_matrix_chart,
                (portfolio,)
            ),
            'drawdown_timeline': (
                MultiSymbolChartGenerator.generate_drawdown_timeline_chart,
                (portfolio,)
            ),
            'monthly_returns_heatmap': (
                MultiSymbolChartGenerator.generate_monthly_returns_heatmap,
                (portfolio,)
            ),
            'position_count_timeline': (
                MultiSymbolChartGenerator.generate_position_count_timeline_chart,
                (portfolio,)
            ),
            'rolling_sharpe': (
                MultiSymbolChartGenerator.generate_rolling_sharpe_chart,
                (portfolio, 30)  # window=30
            ),
            'symbol_performance_heatmap': (
                MultiSymbolChartGenerator.generate_symbol_performance_heatmap,
                (portfolio, attribution)
            ),
        }

        # Execute all charts in parallel with timing
        import time
        results = {}
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ChartGen") as executor:
            # Submit all tasks
            future_to_chart = {
                executor.submit(func, *args): chart_name
                for chart_name, (func, args) in chart_tasks.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_chart):
                chart_name = future_to_chart[future]
                chart_start = time.time()
                try:
                    chart_data = future.result()
                    results[chart_name] = chart_data
                    chart_time = time.time() - chart_start
                    if chart_time > 1.0:  # Log slow charts
                        try:
                            from utils.logger import log_warning
                            log_warning(f"Chart '{chart_name}' took {chart_time:.1f}s to generate")
                        except:
                            pass
                except Exception as exc:
                    # Log error but don't fail entire chart generation
                    log_error(f"Chart generation failed for '{chart_name}': {exc}")
                    results[chart_name] = {}  # Empty chart data

        # Log summary with total time
        successful_charts = sum(1 for v in results.values() if v)
        total_charts = len(results)
        total_time = time.time() - start_time

        try:
            from utils.logger import log_info
            log_info(f"Generated {successful_charts}/{total_charts} charts in {total_time:.1f}s (parallel mode, {max_workers} workers)")
        except:
            pass

        if successful_charts < total_charts:
            log_warning(f"{total_charts - successful_charts} chart(s) failed to generate")

        return results
