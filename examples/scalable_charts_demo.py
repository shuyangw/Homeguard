"""
Demonstration of scalable chart generation for backtesting results.

This example shows how to generate both interactive HTML charts and static image charts
using the new scalable implementations that don't require a GUI window.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import get_output_dir
from src.visualization.charts.candlestick import CandlestickChart
from src.visualization.charts.lightweight_html import LightweightChartsHTML
from src.visualization.charts.mplfinance_chart import MplfinanceChart
from src.utils import logger


def generate_sample_data(num_days: int = 100, symbol: str = 'AAPL') -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration."""
    # Use historical dates (ending 30 days ago) to avoid future date issues
    end_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(end=end_date, periods=num_days, freq='D')

    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(num_days) * 2)

    data = {
        'open': close_prices + np.random.randn(num_days) * 0.5,
        'high': close_prices + abs(np.random.randn(num_days)) * 2,
        'low': close_prices - abs(np.random.randn(num_days)) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, num_days)
    }

    df = pd.DataFrame(data, index=dates)

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df


def generate_sample_trades(price_data: pd.DataFrame, num_trades: int = 10) -> pd.DataFrame:
    """Generate sample trade data."""
    trade_indices = np.random.choice(len(price_data), num_trades, replace=False)
    trade_indices.sort()

    trades = []
    for i, idx in enumerate(trade_indices):
        action = 'BUY' if i % 2 == 0 else 'SELL'
        trades.append({
            'timestamp': price_data.index[idx],
            'action': action,
            'price': price_data.iloc[idx]['close'],
            'symbol': 'AAPL'
        })

    return pd.DataFrame(trades)


def demo_lightweight_html_charts():
    """Demonstrate HTML chart generation with lightweight-charts JS."""
    logger.header("=== Lightweight HTML Charts Demo ===")

    # Generate sample data
    logger.info("Generating sample data...")
    price_data = generate_sample_data(num_days=500)  # 500 days of data
    trades_df = generate_sample_trades(price_data, num_trades=20)

    # Create output directory
    output_dir = get_output_dir() / 'examples' / 'charts_demo'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard HTML chart
    logger.info("Creating standard HTML chart...")
    success = CandlestickChart.save_chart_scalable(
        data=price_data,
        filepath=output_dir / 'standard_chart.html',
        symbol='AAPL',
        trades_df=trades_df,
        title='AAPL Backtest Results - Standard',
        chart_type='html',
        show_volume=True,
        max_data_points=10000
    )

    if success:
        logger.success(f"Chart saved to: {output_dir / 'standard_chart.html'}")

    # 2. Large dataset with aggregation
    logger.info("Creating chart with large dataset (5000 days)...")
    large_data = generate_sample_data(num_days=5000)

    success = LightweightChartsHTML.save_chart(
        data=large_data,
        filepath=output_dir / 'large_dataset_chart.html',
        symbol='AAPL',
        title='AAPL - Large Dataset (Auto-Aggregated)',
        max_data_points=1000  # Will aggregate 5000 points to 1000
    )

    if success:
        logger.success(f"Large dataset chart saved with aggregation")

    # 3. Chart without trades
    logger.info("Creating chart without trade markers...")
    success = LightweightChartsHTML.save_chart(
        data=price_data,
        filepath=output_dir / 'no_trades_chart.html',
        symbol='AAPL',
        title='AAPL - Price Only',
        trades_df=None,
        show_volume=False
    )

    if success:
        logger.success(f"Price-only chart saved")


def demo_mplfinance_static_charts():
    """Demonstrate static chart generation with mplfinance."""
    logger.header("=== Mplfinance Static Charts Demo ===")

    # Generate sample data
    logger.info("Generating sample data...")
    price_data = generate_sample_data(num_days=250)
    trades_df = generate_sample_trades(price_data, num_trades=15)

    # Create output directory
    output_dir = get_output_dir() / 'examples' / 'charts_demo'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. PNG chart with default style
    logger.info("Creating PNG chart with yahoo style...")
    success = CandlestickChart.save_chart_scalable(
        data=price_data,
        filepath=output_dir / 'static_chart_yahoo.png',
        symbol='AAPL',
        trades_df=trades_df,
        title='AAPL - Yahoo Style',
        chart_type='static',
        style='yahoo',
        dpi=150
    )

    if success:
        logger.success(f"PNG chart saved")

    # 2. Different styles
    styles_to_test = ['binance', 'charles', 'nightclouds']

    for style in styles_to_test:
        logger.info(f"Creating chart with {style} style...")
        success = MplfinanceChart.save_chart(
            data=price_data,
            filepath=output_dir / f'static_chart_{style}.png',
            symbol='AAPL',
            trades_df=trades_df,
            title=f'AAPL - {style.capitalize()} Style',
            style=style,
            figsize=(14, 8)
        )

        if success:
            logger.success(f"{style} style chart saved")

    # 3. SVG vector format (scalable)
    logger.info("Creating SVG vector chart...")
    success = MplfinanceChart.save_chart(
        data=price_data,
        filepath=output_dir / 'static_chart_vector.svg',
        symbol='AAPL',
        trades_df=trades_df,
        title='AAPL - Vector Format',
        style='classic'
    )

    if success:
        logger.success(f"SVG vector chart saved")

    # 4. Large dataset with auto-aggregation
    logger.info("Creating static chart with large dataset...")
    large_data = generate_sample_data(num_days=2000)

    success = MplfinanceChart.save_chart(
        data=large_data,
        filepath=output_dir / 'large_static_chart.png',
        symbol='AAPL',
        title='AAPL - 2000 Days (Aggregated)',
        max_data_points=500,  # Aggregate to 500 points
        figsize=(16, 10)
    )

    if success:
        logger.success(f"Large dataset static chart saved")


def demo_performance_comparison():
    """Compare performance with different data sizes."""
    logger.header("=== Performance Comparison ===")

    import time

    output_dir = get_output_dir() / 'examples' / 'charts_demo' / 'performance'
    output_dir.mkdir(parents=True, exist_ok=True)

    data_sizes = [100, 500, 1000, 5000, 10000]

    for size in data_sizes:
        logger.info(f"\nTesting with {size} data points...")
        data = generate_sample_data(num_days=size)

        # HTML chart (with aggregation if needed)
        start = time.time()
        LightweightChartsHTML.save_chart(
            data=data,
            filepath=output_dir / f'perf_html_{size}.html',
            symbol='TEST',
            title=f'Performance Test - {size} points',
            max_data_points=2000  # Aggregate if > 2000
        )
        html_time = time.time() - start

        # Static chart (with aggregation if needed)
        start = time.time()
        MplfinanceChart.save_chart(
            data=data,
            filepath=output_dir / f'perf_static_{size}.png',
            symbol='TEST',
            title=f'Performance Test - {size} points',
            max_data_points=1000,  # Aggregate if > 1000
            figsize=(10, 6)
        )
        static_time = time.time() - start

        logger.metric(f"  HTML generation: {html_time:.2f}s")
        logger.metric(f"  Static generation: {static_time:.2f}s")

        # Memory estimate
        memory_mb = (data.memory_usage(deep=True).sum() / 1024 / 1024)
        logger.metric(f"  Data memory usage: {memory_mb:.2f} MB")


def demo_auto_format_detection():
    """Demonstrate automatic format detection based on file extension."""
    logger.header("=== Auto Format Detection Demo ===")

    price_data = generate_sample_data(num_days=100)
    output_dir = get_output_dir() / 'examples' / 'charts_demo'

    # Test different extensions
    test_files = [
        ('auto_detect.html', 'HTML'),
        ('auto_detect.png', 'PNG'),
        ('auto_detect.svg', 'SVG'),
        ('auto_detect.pdf', 'PDF'),
    ]

    for filename, format_name in test_files:
        logger.info(f"Saving as {format_name} (auto-detected)...")
        success = CandlestickChart.save_chart_scalable(
            data=price_data,
            filepath=output_dir / filename,
            symbol='AAPL',
            title=f'Auto-detected {format_name} Format',
            chart_type='auto'  # Auto-detect from extension
        )

        if success:
            logger.success(f"{format_name} saved successfully")


def main():
    """Run all demonstrations."""
    logger.blank()
    logger.separator('=', 80)
    logger.header("SCALABLE CHARTS DEMONSTRATION")
    logger.separator('=', 80)
    logger.blank()

    # Run demos
    demo_lightweight_html_charts()
    logger.blank()

    demo_mplfinance_static_charts()
    logger.blank()

    demo_performance_comparison()
    logger.blank()

    demo_auto_format_detection()

    logger.blank()
    logger.separator('=', 80)
    logger.success("All demonstrations completed!")
    logger.info(f"Check the '{get_output_dir() / 'examples' / 'charts_demo'}' directory for generated charts")
    logger.separator('=', 80)


if __name__ == '__main__':
    main()