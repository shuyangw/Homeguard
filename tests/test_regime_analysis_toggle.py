"""
Test script for toggleable regime analysis in BacktestEngine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger

def test_regime_analysis_disabled():
    """Test with regime analysis DISABLED (default behavior)."""
    logger.blank()
    logger.separator()
    logger.header("TEST 1: Regime Analysis DISABLED (Default)")
    logger.separator()
    logger.blank()

    # Create engine WITHOUT regime analysis
    engine = BacktestEngine(
        initial_capital=10000,
        fees=0.001,
        enable_regime_analysis=False  # Disabled (this is also the default)
    )

    # Create strategy
    strategy = MovingAverageCrossover(fast_window=20, slow_window=100)

    # Run backtest
    portfolio = engine.run(
        strategy=strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        price_type='close'
    )

    logger.info("Backtest completed WITHOUT regime analysis")
    logger.blank()


def test_regime_analysis_enabled():
    """Test with regime analysis ENABLED."""
    logger.blank()
    logger.separator()
    logger.header("TEST 2: Regime Analysis ENABLED")
    logger.separator()
    logger.blank()

    # Create engine WITH regime analysis
    engine = BacktestEngine(
        initial_capital=10000,
        fees=0.001,
        enable_regime_analysis=True  # ENABLED!
    )

    # Create strategy
    strategy = MovingAverageCrossover(fast_window=20, slow_window=100)

    # Run backtest
    portfolio = engine.run(
        strategy=strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        price_type='close'
    )

    logger.info("Backtest completed WITH regime analysis")
    logger.blank()


def main():
    """Run tests."""
    logger.blank()
    logger.separator()
    logger.header("TESTING TOGGLEABLE REGIME ANALYSIS")
    logger.separator()
    logger.info("This script tests the enable_regime_analysis parameter")
    logger.separator()
    logger.blank()

    try:
        # Test 1: Disabled (default behavior, backward compatible)
        test_regime_analysis_disabled()

        # Test 2: Enabled (new feature)
        test_regime_analysis_enabled()

        logger.blank()
        logger.separator()
        logger.success("All tests completed successfully!")
        logger.separator()
        logger.blank()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
