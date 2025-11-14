"""
Test Paper Trading Integration Off-Hours.

Demonstrates how to test paper trading adapters when markets are closed.
Uses monkey patching to override market hours check.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import MACrossoverLiveAdapter, OMRLiveAdapter
from src.strategies.universe import ETFUniverse, EquityUniverse
from src.utils.logger import logger


def test_ma_crossover_off_hours():
    """
    Test MA Crossover adapter off-hours.

    Bypasses market hours check to test during non-trading hours.
    """
    logger.info("=" * 80)
    logger.info("MA CROSSOVER OFF-HOURS TEST")
    logger.info("=" * 80)

    try:
        # Load API keys from environment (try both naming conventions)
        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

        if not api_key or not secret_key:
            logger.error("Alpaca API credentials not found in environment variables")
            logger.info("Please set ALPACA_API_KEY/ALPACA_PAPER_KEY_ID and ALPACA_SECRET_KEY/ALPACA_PAPER_SECRET_KEY in .env file")
            return False

        # 1. Initialize broker
        logger.info("Initializing Alpaca broker (paper trading)...")
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)

        # Check account
        account = broker.get_account()
        if not account:
            logger.error("Failed to connect to Alpaca")
            return False

        logger.success(f"Connected to Alpaca Paper Trading")
        logger.info(f"  Account: {account['account_id']}")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")

        # 2. Check market status
        market_open = broker.is_market_open()
        logger.info(f"Market status: {'OPEN' if market_open else 'CLOSED'}")

        # 3. Create MA Crossover adapter
        logger.info("Creating MA Crossover adapter...")
        adapter = MACrossoverLiveAdapter(
            broker=broker,
            symbols=EquityUniverse.FAANG[:3],  # Just test with 3 symbols
            fast_period=50,
            slow_period=200,
            ma_type='sma',
            position_size=0.05,  # Small position for testing
            max_positions=2
        )

        logger.success("MA Crossover adapter created")

        # 4. Override is_market_open to return True
        logger.info("")
        logger.info("=" * 80)
        logger.info("RUNNING ADAPTER (MARKET HOURS CHECK BYPASSED)")
        logger.info("=" * 80)

        with patch.object(broker, 'is_market_open', return_value=True):
            logger.info("Market hours check overridden to OPEN")

            # Run the adapter
            adapter.run_once()

        logger.info("")
        logger.success("MA Crossover off-hours test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error in MA Crossover test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_omr_off_hours():
    """
    Test OMR adapter off-hours.

    Bypasses market hours check and time check to test OMR strategy.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("OMR OFF-HOURS TEST")
    logger.info("=" * 80)

    try:
        # Load API keys from environment (try both naming conventions)
        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

        if not api_key or not secret_key:
            logger.error("Alpaca API credentials not found in environment variables")
            logger.info("Please set ALPACA_API_KEY/ALPACA_PAPER_KEY_ID and ALPACA_SECRET_KEY/ALPACA_PAPER_SECRET_KEY in .env file")
            return False

        # 1. Initialize broker
        logger.info("Initializing Alpaca broker (paper trading)...")
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)

        # Check account
        account = broker.get_account()
        if not account:
            logger.error("Failed to connect to Alpaca")
            return False

        logger.success(f"Connected to Alpaca Paper Trading")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")

        # 2. Create OMR adapter (use fewer symbols for testing)
        logger.info("Creating OMR adapter...")
        test_symbols = ETFUniverse.LEVERAGED_3X[:5]  # Just 5 symbols for testing

        adapter = OMRLiveAdapter(
            broker=broker,
            symbols=test_symbols,
            min_probability=0.50,  # Lower threshold for testing
            min_expected_return=0.001,
            max_positions=2,
            position_size=0.05  # Small position for testing
        )

        logger.success("OMR adapter created")

        # 3. Override market hours and time checks
        logger.info("")
        logger.info("=" * 80)
        logger.info("RUNNING ADAPTER (MARKET HOURS & TIME CHECKS BYPASSED)")
        logger.info("=" * 80)

        with patch.object(broker, 'is_market_open', return_value=True):
            with patch.object(adapter, 'should_run_now', return_value=True):
                logger.info("Market hours check overridden to OPEN")
                logger.info("Time check overridden to allow execution")

                # Run the adapter
                adapter.run_once()

        logger.info("")
        logger.success("OMR off-hours test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error in OMR test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter_workflow_with_mock_data():
    """
    Test adapter workflow with completely mocked data.

    This tests the adapter logic without actually connecting to Alpaca.
    Useful for testing when you don't have API credentials.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("ADAPTER WORKFLOW TEST (FULLY MOCKED)")
    logger.info("=" * 80)

    try:
        # Create mock broker
        mock_broker = MagicMock()

        # Mock account
        mock_account = MagicMock()
        mock_account.id = "test_account_123"
        mock_account.buying_power = "50000.00"
        mock_account.portfolio_value = "100000.00"
        mock_broker.get_account.return_value = mock_account

        # Mock market status
        mock_broker.is_market_open.return_value = True

        # Mock historical data
        import pandas as pd
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')

        def mock_get_historical_bars(symbol, start, end, timeframe):
            """Return mock historical data with a simple uptrend."""
            data = {
                'open': range(100, 100 + len(dates)),
                'high': range(101, 101 + len(dates)),
                'low': range(99, 99 + len(dates)),
                'close': range(100, 100 + len(dates)),
                'volume': [1000000] * len(dates)
            }
            return pd.DataFrame(data, index=dates)

        mock_broker.get_historical_bars.side_effect = mock_get_historical_bars

        # Mock positions
        mock_broker.get_positions.return_value = []

        # Create adapter with mock broker
        logger.info("Creating MA Crossover adapter with mock broker...")
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            fast_period=10,  # Shorter periods for testing
            slow_period=20,
            ma_type='sma',
            position_size=0.10,
            max_positions=3
        )

        logger.success("Adapter created with mock broker")

        # Mock execution engine to prevent actual order placement
        mock_order = MagicMock()
        mock_order.id = "test_order_123"
        adapter.execution_engine.place_market_order = MagicMock(return_value=mock_order)

        # Run adapter
        logger.info("")
        logger.info("Running adapter with mock data...")
        adapter.run_once()

        # Check if orders were attempted
        if adapter.execution_engine.place_market_order.called:
            logger.success(f"Adapter attempted to place {adapter.execution_engine.place_market_order.call_count} orders")
        else:
            logger.info("No signals generated (expected with mock data)")

        logger.info("")
        logger.success("Mock workflow test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error in mock workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all off-hours tests."""
    logger.info("=" * 80)
    logger.info("PAPER TRADING OFF-HOURS TEST SUITE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This test suite demonstrates how to test paper trading")
    logger.info("adapters when markets are closed.")
    logger.info("")

    results = {}

    # Test 1: MA Crossover with real broker (market hours bypassed)
    logger.info("Test 1/3: MA Crossover with Alpaca (market hours bypassed)")
    results['ma_crossover'] = test_ma_crossover_off_hours()
    time.sleep(2)

    # Test 2: OMR with real broker (market hours bypassed)
    logger.info("Test 2/3: OMR with Alpaca (market hours bypassed)")
    results['omr'] = test_omr_off_hours()
    time.sleep(2)

    # Test 3: Fully mocked workflow
    logger.info("Test 3/3: Adapter workflow (fully mocked)")
    results['mock_workflow'] = test_adapter_workflow_with_mock_data()

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    logger.info("")
    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.success("All tests passed!")
    else:
        logger.error(f"{total_tests - passed_tests} test(s) failed")

    return passed_tests == total_tests


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()

        if test_name == 'ma':
            # Run only MA Crossover test
            success = test_ma_crossover_off_hours()
        elif test_name == 'omr':
            # Run only OMR test
            success = test_omr_off_hours()
        elif test_name == 'mock':
            # Run only mock workflow test
            success = test_adapter_workflow_with_mock_data()
        else:
            logger.error(f"Unknown test: {test_name}")
            logger.info("Usage: python test_paper_trading_off_hours.py [ma|omr|mock]")
            sys.exit(1)

        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
