"""
Candlestick chart generation using mplfinance for static images.
For comprehensive reports with interactive charts, use QuantStats instead.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Literal
from src.utils import logger

from .mplfinance_chart import MplfinanceChart


class CandlestickChart:
    """
    Creates candlestick charts using mplfinance for static images.

    Note: For interactive reports with comprehensive metrics, use QuantStats:
        engine.run_and_report(...) or use --quantstats flag
    """

    @staticmethod
    def create_chart_static(
        data: pd.DataFrame,
        symbol: str,
        title: Optional[str] = None,
        show_volume: bool = True,
        trades_df: Optional[pd.DataFrame] = None,
        style: str = 'yahoo',
        figsize: tuple = (12, 8),
        max_data_points: int = 5000
    ):
        """
        Create static chart using mplfinance.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            title: Chart title
            show_volume: Whether to show volume subplot
            trades_df: Optional DataFrame with trade markers
            style: mplfinance style ('yahoo', 'charles', 'mike', etc.)
            figsize: Figure size as (width, height)
            max_data_points: Maximum data points to plot

        Returns:
            matplotlib Figure object
        """
        return MplfinanceChart.create_chart(
            data=data,
            symbol=symbol,
            trades_df=trades_df,
            title=title,
            style=style,
            show_volume=show_volume,
            figsize=figsize,
            max_data_points=max_data_points
        )

    @staticmethod
    def save_chart_scalable(
        data: pd.DataFrame,
        filepath: Path,
        symbol: str,
        trades_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        chart_type: Literal['html', 'static', 'auto'] = 'auto',
        show_volume: bool = True,
        max_data_points: int = 10000,
        **kwargs
    ) -> bool:
        """
        Save chart using mplfinance (static images).

        Args:
            data: OHLCV DataFrame
            filepath: Output file path
            symbol: Stock symbol
            trades_df: Optional trades DataFrame
            title: Chart title
            chart_type: 'static' for image (html is no longer supported)
            show_volume: Show volume bars
            max_data_points: Maximum data points (for performance)
            **kwargs: Additional arguments for mplfinance

        Returns:
            True if successful
        """
        # Determine output format from file extension if auto
        if chart_type == 'auto':
            suffix = filepath.suffix.lower()
            if suffix in ['.html', '.htm']:
                logger.warning("HTML charts deprecated. Use --quantstats for interactive reports.")
                # Change to PNG format
                filepath = filepath.with_suffix('.png')
            elif suffix not in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                # Default to PNG
                filepath = filepath.with_suffix('.png')

        if chart_type == 'html':
            logger.warning("HTML charts have been replaced by QuantStats.")
            logger.info("Use --quantstats flag for comprehensive interactive reports.")
            # Fall back to PNG
            filepath = filepath.with_suffix('.png')

        # Use mplfinance for static images
        return MplfinanceChart.save_chart(
            data=data,
            filepath=filepath,
            symbol=symbol,
            trades_df=trades_df,
            title=title,
            show_volume=show_volume,
            max_data_points=max_data_points,
            **kwargs
        )

    @staticmethod
    def save_all_formats(
        data: pd.DataFrame,
        output_dir: Path,
        symbol: str,
        trades_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        show_volume: bool = True,
        max_data_points_html: int = 10000,
        max_data_points_static: int = 5000,
        style: str = 'yahoo',
        figsize: tuple = (12, 8),
        dpi: int = 100
    ) -> dict:
        """
        Save chart in static formats (PNG, SVG) to specified directory.

        Args:
            data: OHLCV DataFrame
            output_dir: Directory to save all charts
            symbol: Stock symbol
            trades_df: Optional trades DataFrame
            title: Chart title
            show_volume: Show volume bars
            max_data_points_html: Ignored (kept for compatibility)
            max_data_points_static: Maximum data points for static charts
            style: mplfinance style
            figsize: Figure size
            dpi: DPI for PNG output

        Returns:
            Dictionary with format as key and filepath as value
            Example: {'png': Path, 'svg': Path}
        """
        if title is None:
            title = f"{symbol} Backtest Results"

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        base_filename = f"{symbol}_chart"

        # 1. Save PNG chart (static raster)
        png_path = output_dir / f"{base_filename}.png"
        try:
            success = MplfinanceChart.save_chart(
                data=data,
                filepath=png_path,
                symbol=symbol,
                trades_df=trades_df,
                title=title,
                style=style,
                show_volume=show_volume,
                figsize=figsize,
                max_data_points=max_data_points_static,
                dpi=dpi
            )
            results['png'] = png_path if success else None
            if success:
                logger.success(f"PNG chart saved: {png_path}")
        except Exception as e:
            logger.error(f"Failed to save PNG chart: {e}")
            results['png'] = None

        # 2. Save SVG chart (static vector)
        svg_path = output_dir / f"{base_filename}.svg"
        try:
            success = MplfinanceChart.save_chart(
                data=data,
                filepath=svg_path,
                symbol=symbol,
                trades_df=trades_df,
                title=title,
                style=style,
                show_volume=show_volume,
                figsize=figsize,
                max_data_points=max_data_points_static
            )
            results['svg'] = svg_path if success else None
            if success:
                logger.success(f"SVG chart saved: {svg_path}")
        except Exception as e:
            logger.error(f"Failed to save SVG chart: {e}")
            results['svg'] = None

        # Log summary
        successful_formats = [fmt for fmt, path in results.items() if path is not None]
        if successful_formats:
            logger.info(f"Charts saved in formats: {', '.join(successful_formats)}")
        else:
            logger.error(f"Failed to save any chart formats for {symbol}")

        return results
