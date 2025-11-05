"""
mplfinance chart generator for static backtesting charts.
Provides memory-efficient static chart generation for large datasets.
"""

# Configure matplotlib backend BEFORE importing matplotlib/mplfinance
# This prevents GUI backend conflicts on macOS
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import mplfinance as mpf
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from utils import logger


class MplfinanceChart:
    """
    Generate static candlestick charts using mplfinance.
    Memory-efficient solution for large datasets.
    """

    # Available styles in mplfinance
    STYLES = ['binance', 'blueskies', 'brasil', 'charles', 'checkers',
              'classic', 'default', 'ibd', 'kenan', 'mike', 'nightclouds',
              'sas', 'starsandstripes', 'yahoo']

    @staticmethod
    def create_chart(
        data: pd.DataFrame,
        symbol: str,
        trades_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        style: str = 'yahoo',
        show_volume: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        max_data_points: int = 5000
    ) -> plt.Figure:
        """
        Create static candlestick chart using mplfinance.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            trades_df: Optional DataFrame with trades
            title: Chart title
            style: mplfinance style (default: 'yahoo')
            show_volume: Whether to show volume bars
            figsize: Figure size (width, height)
            max_data_points: Maximum data points for performance

        Returns:
            matplotlib Figure object
        """
        if title is None:
            title = f"{symbol} Backtest Results"

        # Prepare data
        chart_data = MplfinanceChart._prepare_data(data, max_data_points)

        # Prepare trade markers
        trade_plots = []
        if trades_df is not None and not trades_df.empty:
            trade_plots = MplfinanceChart._prepare_trade_markers(trades_df, chart_data)

        # Create kwargs for mplfinance
        kwargs = {
            'type': 'candle',
            'style': style if style in MplfinanceChart.STYLES else 'yahoo',
            'title': title,
            'ylabel': 'Price ($)',
            'volume': show_volume and 'volume' in chart_data.columns,
            'figsize': figsize,
            'tight_layout': True,
            'returnfig': True,  # Important: return figure for saving
            'warn_too_much_data': max_data_points + 1000  # Suppress warning since we handle aggregation
        }

        # Add trade markers if available
        if trade_plots:
            kwargs['addplot'] = trade_plots

        # Create the plot
        fig, axes = mpf.plot(chart_data, **kwargs)

        return fig

    @staticmethod
    def _prepare_data(data: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """
        Prepare data for mplfinance (requires specific column names).
        """
        # Copy data
        df = data.copy()

        # mplfinance requires lowercase OHLC column names
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }

        # Rename columns if necessary
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must contain columns: {required}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
            else:
                df.index = pd.to_datetime(df.index)

        # Data aggregation for large datasets (OHLC resampling)
        if len(df) > max_points:
            logger.info(f"Aggregating {len(df)} points to {max_points} for performance", to_file=False)

            # Calculate resample frequency
            total_duration = df.index[-1] - df.index[0]
            target_periods = max_points

            # Determine appropriate resampling rule
            days_per_point = (total_duration.days / target_periods)

            if days_per_point > 30:
                rule = f'{int(days_per_point/30)}M'  # Monthly
            elif days_per_point > 7:
                rule = f'{int(days_per_point/7)}W'  # Weekly
            elif days_per_point > 1:
                rule = f'{int(days_per_point)}D'  # Daily
            elif days_per_point > 0.04:  # More than 1 hour
                rule = f'{int(days_per_point*24)}h'  # Hourly (lowercase 'h' for pandas 2.0+)
            else:
                rule = f'{int(days_per_point*24*60)}min'  # Minutes (use 'min' instead of 'T')

            # Resample OHLC data
            resampled = pd.DataFrame()
            resampled['open'] = df['open'].resample(rule).first()
            resampled['high'] = df['high'].resample(rule).max()
            resampled['low'] = df['low'].resample(rule).min()
            resampled['close'] = df['close'].resample(rule).last()

            if 'volume' in df.columns:
                resampled['volume'] = df['volume'].resample(rule).sum()

            # Drop NaN values
            df = resampled.dropna()

        return df

    @staticmethod
    def _prepare_trade_markers(trades_df: pd.DataFrame, chart_data: pd.DataFrame) -> List:
        """
        Prepare trade markers as additional plots for mplfinance.
        """
        plots = []

        # Separate buy and sell trades
        buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
        sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()

        # Convert timestamps to datetime if needed
        for df in [buy_trades, sell_trades]:
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create buy markers
        if not buy_trades.empty:
            buy_signals = pd.Series(index=chart_data.index, dtype=float)
            for _, trade in buy_trades.iterrows():
                # Find closest timestamp in chart_data
                timestamp = pd.to_datetime(trade['timestamp'])
                if timestamp in chart_data.index:
                    buy_signals[timestamp] = trade['price']
                else:
                    # Find nearest timestamp
                    idx = chart_data.index.get_indexer([timestamp], method='nearest')[0]
                    if idx >= 0 and idx < len(chart_data.index):
                        buy_signals[chart_data.index[idx]] = trade['price']

            plots.append(mpf.make_addplot(
                buy_signals,
                type='scatter',
                markersize=100,
                marker='^',
                color='green',
                label='Buy'
            ))

        # Create sell markers
        if not sell_trades.empty:
            sell_signals = pd.Series(index=chart_data.index, dtype=float)
            for _, trade in sell_trades.iterrows():
                timestamp = pd.to_datetime(trade['timestamp'])
                if timestamp in chart_data.index:
                    sell_signals[timestamp] = trade['price']
                else:
                    # Find nearest timestamp
                    idx = chart_data.index.get_indexer([timestamp], method='nearest')[0]
                    if idx >= 0 and idx < len(chart_data.index):
                        sell_signals[chart_data.index[idx]] = trade['price']

            plots.append(mpf.make_addplot(
                sell_signals,
                type='scatter',
                markersize=100,
                marker='v',
                color='red',
                label='Sell'
            ))

        return plots

    @staticmethod
    def save_chart(
        data: pd.DataFrame,
        filepath: Path,
        symbol: str,
        trades_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        style: str = 'yahoo',
        show_volume: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        max_data_points: int = 5000,
        dpi: int = 100
    ) -> bool:
        """
        Create and save static chart to file.

        Args:
            data: OHLCV DataFrame
            filepath: Output file path (.png, .jpg, .svg, .pdf)
            symbol: Stock symbol
            trades_df: Optional trades DataFrame
            title: Chart title
            style: mplfinance style
            show_volume: Show volume bars
            figsize: Figure size
            max_data_points: Max points for performance
            dpi: Resolution for raster formats

        Returns:
            True if successful
        """
        try:
            # Create the chart
            fig = MplfinanceChart.create_chart(
                data=data,
                symbol=symbol,
                trades_df=trades_df,
                title=title,
                style=style,
                show_volume=show_volume,
                figsize=figsize,
                max_data_points=max_data_points
            )

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save the figure
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

            # Clear memory (important for large datasets)
            plt.close(fig)

            logger.success(f"Static chart saved to: {filepath}")

            # Log performance info
            if len(data) > max_data_points:
                logger.info(f"Data aggregated to {max_data_points} points for performance")

            return True

        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
            return False

    @staticmethod
    def create_multi_panel_chart(
        data: pd.DataFrame,
        symbol: str,
        indicators: Optional[dict] = None,
        trades_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        style: str = 'yahoo',
        figsize: Tuple[int, int] = (14, 10),
        max_data_points: int = 5000
    ) -> plt.Figure:
        """
        Create multi-panel chart with indicators.

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            indicators: Dict of indicators to plot {'RSI': rsi_series, 'MACD': macd_df}
            trades_df: Optional trades DataFrame
            title: Chart title
            style: mplfinance style
            figsize: Figure size
            max_data_points: Max points

        Returns:
            matplotlib Figure
        """
        if title is None:
            title = f"{symbol} Analysis"

        # Prepare data
        chart_data = MplfinanceChart._prepare_data(data, max_data_points)

        # Base kwargs
        kwargs = {
            'type': 'candle',
            'style': style if style in MplfinanceChart.STYLES else 'yahoo',
            'title': title,
            'volume': True,
            'figsize': figsize,
            'returnfig': True,
            'warn_too_much_data': max_data_points + 1000  # Suppress warning since we handle aggregation
        }

        # Add panels for indicators
        addplots = []

        if indicators:
            # Example: RSI in separate panel
            if 'RSI' in indicators and indicators['RSI'] is not None:
                addplots.append(mpf.make_addplot(
                    indicators['RSI'],
                    panel=2,  # Separate panel
                    ylabel='RSI',
                    color='purple',
                    secondary_y=False
                ))

                # Add RSI levels
                addplots.append(mpf.make_addplot(
                    [70] * len(chart_data),
                    panel=2,
                    color='r',
                    linestyle='--',
                    linewidth=0.5
                ))
                addplots.append(mpf.make_addplot(
                    [30] * len(chart_data),
                    panel=2,
                    color='g',
                    linestyle='--',
                    linewidth=0.5
                ))

        # Add trade markers
        if trades_df is not None and not trades_df.empty:
            trade_plots = MplfinanceChart._prepare_trade_markers(trades_df, chart_data)
            addplots.extend(trade_plots)

        if addplots:
            kwargs['addplot'] = addplots

        # Create the plot
        fig, axes = mpf.plot(chart_data, **kwargs)

        return fig