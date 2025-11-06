"""
Example: Toggleable Regime Analysis

Demonstrates how to enable/disable automatic regime analysis in backtests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def example_without_regime_analysis():
    """
    Example 1: Standard Backtest (Default)

    Regime analysis is DISABLED by default for backward compatibility.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 1: Standard Backtest (Regime Analysis Disabled)")
    logger.separator()
    logger.blank()

    # Create engine (regime analysis disabled by default)
    engine = BacktestEngine(
        initial_capital=10000,
        fees=0.001
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

    logger.info("✓ Backtest completed - standard output only")
    logger.blank()


def example_with_regime_analysis():
    """
    Example 2: Backtest with Regime Analysis

    Enable regime analysis with one parameter to get detailed insights
    into how the strategy performs across different market conditions.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 2: Backtest with Regime Analysis Enabled")
    logger.separator()
    logger.blank()

    # Create engine with regime analysis ENABLED
    engine = BacktestEngine(
        initial_capital=10000,
        fees=0.001,
        enable_regime_analysis=True  # ← Enable regime analysis
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

    logger.info("✓ Backtest completed - standard output + regime analysis")
    logger.blank()


def main():
    """Run examples."""
    logger.blank()
    logger.separator()
    logger.header("TOGGLEABLE REGIME ANALYSIS EXAMPLES")
    logger.separator()
    logger.info("This script demonstrates:")
    logger.info("  1. Standard backtest (default behavior)")
    logger.info("  2. Backtest with regime analysis (enhanced insights)")
    logger.separator()
    logger.blank()

    try:
        # Example 1: Without regime analysis (default)
        example_without_regime_analysis()

        # Example 2: With regime analysis (enabled)
        example_with_regime_analysis()

        logger.blank()
        logger.separator()
        logger.success("Examples completed successfully!")
        logger.separator()
        logger.info("Key Takeaway:")
        logger.info("  - Use enable_regime_analysis=True for strategy validation")
        logger.info("  - Get instant insights into regime-specific performance")
        logger.info("  - Check robustness score before production deployment")
        logger.separator()
        logger.blank()

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
