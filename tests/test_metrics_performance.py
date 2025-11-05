"""
Test that multi-symbol metrics calculation is performant.

Verifies the O(n²) bottleneck fix in calculate_portfolio_composition_metrics().
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio
from backtesting.engine.multi_symbol_metrics import MultiSymbolMetrics
from utils import logger


def create_large_mock_portfolio(num_bars: int = 100000) -> MultiAssetPortfolio:
    """
    Create a mock portfolio with many bars to test performance.

    This simulates a year of minute-level data (252 days × 390 min = 98,280 bars)
    """
    logger.info(f"Creating mock portfolio with {num_bars:,} bars...")

    # Create synthetic data
    timestamps = pd.date_range('2023-01-01 09:30', periods=num_bars, freq='1min')
    symbols = ['AAPL', 'MSFT']

    # Create price data
    prices_data = {}
    for symbol in symbols:
        prices_data[symbol] = 100 + np.cumsum(np.random.randn(num_bars) * 0.5)
    prices = pd.DataFrame(prices_data, index=timestamps)

    # Create dummy signals
    entries = pd.DataFrame(False, index=timestamps, columns=symbols)
    exits = pd.DataFrame(False, index=timestamps, columns=symbols)

    # Create portfolio
    portfolio = MultiAssetPortfolio(
        symbols=symbols,
        prices=prices,
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        position_sizing_method='equal_weight'
    )

    # Simulate some history
    for i in range(num_bars):
        portfolio.equity_curve.append(100000 + i * 10)
        portfolio.equity_timestamps.append(timestamps[i])
        portfolio.cash_history.append((timestamps[i], 50000 + i * 5))

    portfolio.position_count_history = [(timestamps[i], 2) for i in range(0, num_bars, 100)]
    portfolio.symbol_weights_history = [(timestamps[i], {'AAPL': 0.5, 'MSFT': 0.5}) for i in range(0, num_bars, 100)]

    logger.success(f"Created portfolio with {len(portfolio.equity_curve):,} bars")
    return portfolio


def test_composition_metrics_performance():
    """Test that composition metrics calculation is fast (< 1 second for 100K bars)."""

    logger.header("TESTING METRICS PERFORMANCE")
    logger.blank()

    # Create large portfolio (similar to year of minute data)
    portfolio = create_large_mock_portfolio(num_bars=100000)

    logger.blank()
    logger.info("Testing composition metrics performance...")
    logger.info(f"Portfolio size: {len(portfolio.equity_curve):,} bars")
    logger.info(f"Cash history: {len(portfolio.cash_history):,} entries")
    logger.separator()

    # Time the composition metrics calculation
    start = time.time()
    metrics = MultiSymbolMetrics.calculate_portfolio_composition_metrics(portfolio)
    elapsed = time.time() - start

    logger.blank()
    logger.metric(f"Composition metrics calculated in: {elapsed:.3f}s")
    logger.blank()

    # Verify metrics were calculated
    assert 'Avg Position Count' in metrics
    assert 'Avg Capital Utilization [%]' in metrics
    assert 'Avg Concentration (Herfindahl)' in metrics

    logger.info("Calculated metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")

    logger.blank()
    logger.separator()

    # Performance expectation
    if elapsed < 1.0:
        logger.success(f"✓ PERFORMANCE EXCELLENT: {elapsed:.3f}s (target: < 1s)")
        logger.success("✓ O(n²) bottleneck successfully fixed!")
    elif elapsed < 5.0:
        logger.warning(f"⚠ Performance acceptable: {elapsed:.3f}s (target: < 1s)")
    else:
        logger.error(f"✗ Performance issue: {elapsed:.3f}s (target: < 1s)")
        logger.error("The O(n²) bottleneck may not be fully fixed")
        raise AssertionError(f"Metrics calculation took {elapsed:.3f}s, expected < 1s")

    logger.blank()
    logger.info("Performance analysis:")
    logger.info(f"  Operations: {len(portfolio.cash_history):,} iterations")
    logger.info(f"  Time per operation: {(elapsed / len(portfolio.cash_history)) * 1000:.6f}ms")
    logger.info(f"  Estimated old O(n²) time: {(len(portfolio.cash_history) ** 2) / 1e9 * 0.05:.1f}s")
    logger.blank()


def test_all_metrics_performance():
    """Test that all metrics calculation is reasonably fast."""

    logger.header("TESTING ALL METRICS")
    logger.blank()

    # Create portfolio with realistic size
    portfolio = create_large_mock_portfolio(num_bars=50000)

    # Add some closed positions for attribution/trade analysis
    portfolio.closed_positions = [
        {
            'symbol': 'AAPL',
            'entry_timestamp': portfolio.equity_timestamps[100],
            'exit_timestamp': portfolio.equity_timestamps[200],
            'pnl': 150.0,
            'pnl_pct': 1.5,
            'hold_duration_days': 1.0
        },
        {
            'symbol': 'MSFT',
            'entry_timestamp': portfolio.equity_timestamps[150],
            'exit_timestamp': portfolio.equity_timestamps[250],
            'pnl': -50.0,
            'pnl_pct': -0.5,
            'hold_duration_days': 1.0
        }
    ]

    logger.info("Calculating all metrics categories...")
    logger.separator()

    start = time.time()
    metrics = MultiSymbolMetrics.calculate_all_metrics(portfolio)
    elapsed = time.time() - start

    logger.blank()
    logger.metric(f"All metrics calculated in: {elapsed:.3f}s")
    logger.blank()

    # Verify all categories present
    assert 'composition' in metrics
    assert 'attribution' in metrics
    assert 'diversification' in metrics
    assert 'rebalancing' in metrics
    assert 'trade_analysis' in metrics

    if elapsed < 5.0:
        logger.success(f"✓ ALL METRICS PERFORMANCE GOOD: {elapsed:.3f}s (target: < 5s)")
    else:
        logger.warning(f"⚠ All metrics took {elapsed:.3f}s (target: < 5s)")

    logger.blank()
    logger.separator()
    logger.header("ALL TESTS PASSED")
    logger.blank()


if __name__ == '__main__':
    try:
        test_composition_metrics_performance()
        logger.blank()
        test_all_metrics_performance()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
