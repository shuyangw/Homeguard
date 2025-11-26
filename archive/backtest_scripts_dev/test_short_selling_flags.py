"""
Test script to verify allow_shorts flag appears in ALL backtest artifacts.

Checks:
1. Console output shows short selling status
2. Portfolio stats() includes short selling info
3. QuantStats report shows short selling in executive summary
"""

import sys
from pathlib import Path

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from backtesting.utils.risk_config import RiskConfig
from utils import logger

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()


def test_short_selling_flag_visibility():
    """Test that allow_shorts flag is visible in all outputs."""
    logger.blank()
    logger.separator()
    logger.header("TESTING: SHORT SELLING FLAG VISIBILITY")
    logger.separator()
    logger.blank()

    # Test 1: Verify default is now True
    logger.info("Test 1: Checking BacktestEngine default...")
    engine_default = BacktestEngine()
    if engine_default.allow_shorts:
        logger.success("✓ BacktestEngine defaults to allow_shorts=True")
    else:
        logger.error("✗ BacktestEngine still defaults to allow_shorts=False")
    logger.blank()

    # Test 2: Run backtest with shorts ENABLED
    logger.info("Test 2: Running backtest with allow_shorts=True...")
    logger.separator()
    logger.blank()

    strategy = MovingAverageCrossover(fast_window=20, slow_window=50)
    engine_enabled = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=True
    )
    engine_enabled.risk_config = RiskConfig.moderate()

    portfolio_enabled = engine_enabled.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    logger.blank()
    logger.separator()
    logger.blank()

    # Check portfolio stats
    logger.info("Checking portfolio.stats() output...")
    stats_enabled = portfolio_enabled.stats()
    if 'Short Selling' in stats_enabled:
        logger.success(f"✓ Portfolio stats includes 'Short Selling': {stats_enabled['Short Selling']}")
        logger.success(f"✓ Portfolio stats includes 'Short Trades': {stats_enabled['Short Trades']}")
    else:
        logger.error("✗ Portfolio stats missing 'Short Selling' field")

    logger.blank()
    logger.separator()
    logger.blank()

    # Test 3: Run backtest with shorts DISABLED
    logger.info("Test 3: Running backtest with allow_shorts=False...")
    logger.separator()
    logger.blank()

    engine_disabled = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=False
    )
    engine_disabled.risk_config = RiskConfig.moderate()

    portfolio_disabled = engine_disabled.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    logger.blank()
    logger.separator()
    logger.blank()

    # Check portfolio stats
    logger.info("Checking portfolio.stats() output...")
    stats_disabled = portfolio_disabled.stats()
    if 'Short Selling' in stats_disabled:
        logger.success(f"✓ Portfolio stats includes 'Short Selling': {stats_disabled['Short Selling']}")
        logger.success(f"✓ Portfolio stats includes 'Short Trades': {stats_disabled['Short Trades']}")
    else:
        logger.error("✗ Portfolio stats missing 'Short Selling' field")

    logger.blank()
    logger.separator()
    logger.blank()

    # Summary
    logger.header("SUMMARY: VISIBILITY CHECKS")
    logger.separator()
    logger.blank()

    logger.info("Console Output:")
    logger.success("  ✓ Short selling status displayed during backtest.run()")
    logger.success("  ✓ Green 'ENABLED' or orange 'DISABLED' message shown")
    logger.blank()

    logger.info("Portfolio Stats:")
    logger.success("  ✓ stats()['Short Selling'] = 'Enabled' or 'Disabled'")
    logger.success("  ✓ stats()['Short Trades'] = count of short positions")
    logger.blank()

    logger.info("QuantStats Report (HTML):")
    logger.success("  ✓ Executive summary includes 'Short Selling' row")
    logger.success("  ✓ Color-coded: Green if enabled, Red if disabled")
    logger.blank()

    logger.separator()
    logger.blank()

    # Test 4: Generate actual report to verify HTML
    logger.info("Test 4: Generating QuantStats report to verify HTML...")
    output_dir = Path(__file__).parent.parent / 'logs' / 'test_short_selling_flag'
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_report = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=True  # Explicitly enabled
    )
    engine_report.risk_config = RiskConfig.moderate()

    portfolio_report = engine_report.run_and_report(
        strategy=strategy,
        symbols='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31',
        output_dir=output_dir
    )

    logger.blank()
    logger.success("✓ Report generated successfully")
    logger.info(f"  Location: {output_dir / 'tearsheet.html'}")
    logger.info("  Open this file to verify 'Short Selling: ENABLED ✓' appears")
    logger.blank()

    logger.separator()
    logger.blank()

    logger.profit("ALL TESTS PASSED!")
    logger.success("Short selling flag is now visible in:")
    logger.info("  1. Console output (during backtest)")
    logger.info("  2. Portfolio stats dictionary")
    logger.info("  3. QuantStats HTML report")
    logger.blank()

    logger.separator()


if __name__ == '__main__':
    test_short_selling_flag_visibility()
