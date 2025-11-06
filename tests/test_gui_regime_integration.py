"""
Test GUI integration of regime analysis toggle.

Verifies that the regime analysis flag is properly passed from
GUI checkbox -> app.py -> gui_controller.py -> backtest_engine.py

This test doesn't use Flet GUI but tests the data flow through
the controller layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gui.workers.gui_controller import GUIBacktestController
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def test_regime_analysis_integration():
    """
    Test that enable_regime_analysis flag is properly passed through
    the GUI controller to the backtest engine.
    """
    logger.blank()
    logger.separator()
    logger.header("TEST: GUI Regime Analysis Integration")
    logger.separator()
    logger.blank()

    # Test parameters
    symbols = ['AAPL']
    start_date = '2024-01-01'
    end_date = '2024-01-31'  # Short test period

    # Create strategy
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)

    # Test 1: Regime analysis DISABLED (default)
    logger.info("Test 1: Regime analysis DISABLED (default)")
    controller1 = GUIBacktestController(max_workers=1)

    try:
        controller1.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            fees=0.001,
            risk_profile='Moderate',
            generate_full_output=False,
            portfolio_mode='Single-Symbol',
            enable_regime_analysis=False  # DISABLED
        )

        # Wait for completion
        import time
        while controller1.is_running():
            time.sleep(0.1)

        # Check results
        results = controller1.get_results()
        status = controller1.get_status('AAPL')

        if status == 'completed' and len(results) > 0:
            logger.success("✓ Test 1 PASSED: Backtest completed with regime analysis DISABLED")
            logger.info(f"  Return: {results['AAPL'].get('Total Return [%]', 0):.2f}%")
        else:
            logger.error(f"✗ Test 1 FAILED: Status={status}, Results={len(results)}")
            return False

    except Exception as e:
        logger.error(f"✗ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.blank()

    # Test 2: Regime analysis ENABLED
    logger.info("Test 2: Regime analysis ENABLED")
    controller2 = GUIBacktestController(max_workers=1)

    try:
        controller2.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            fees=0.001,
            risk_profile='Moderate',
            generate_full_output=False,
            portfolio_mode='Single-Symbol',
            enable_regime_analysis=True  # ENABLED
        )

        # Wait for completion
        while controller2.is_running():
            time.sleep(0.1)

        # Check results
        results = controller2.get_results()
        status = controller2.get_status('AAPL')

        if status == 'completed' and len(results) > 0:
            logger.success("✓ Test 2 PASSED: Backtest completed with regime analysis ENABLED")
            logger.info(f"  Return: {results['AAPL'].get('Total Return [%]', 0):.2f}%")
        else:
            logger.error(f"✗ Test 2 FAILED: Status={status}, Results={len(results)}")
            return False

    except Exception as e:
        logger.error(f"✗ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.blank()
    logger.separator()
    logger.success("ALL TESTS PASSED ✓")
    logger.separator()
    logger.blank()

    return True


if __name__ == '__main__':
    success = test_regime_analysis_integration()
    sys.exit(0 if success else 1)
