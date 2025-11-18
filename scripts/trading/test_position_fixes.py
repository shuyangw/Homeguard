"""
Test script to validate position handling fixes.

This script tests that the adapter code correctly handles positions
returned as dictionaries from the Alpaca broker.

Run this to validate bug fixes before deploying to EC2.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.trading.adapters.omr_live_adapter import OMRLiveAdapter
from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.utils.logger import logger


class MockBroker:
    """Mock broker that returns positions as dicts (like real Alpaca API)."""

    def get_positions(self) -> List[Dict]:
        """Return positions as list of dicts (mimics Alpaca API)."""
        return [
            {
                'symbol': 'SPY',
                'quantity': 3,
                'avg_entry_price': 669.28,
                'current_price': 661.85,
                'market_value': 1985.55,
                'unrealized_pnl': -22.29,
                'unrealized_pnl_pct': -0.0111,
                'side': 'long'
            },
            {
                'symbol': 'QQQ',
                'quantity': 5,
                'avg_entry_price': 450.00,
                'current_price': 455.00,
                'market_value': 2275.00,
                'unrealized_pnl': 25.00,
                'unrealized_pnl_pct': 0.0111,
                'side': 'long'
            }
        ]

    def get_account(self):
        """Return mock account."""
        mock_account = Mock()
        mock_account.buying_power = "100000.00"
        mock_account.cash = "95000.00"
        mock_account.portfolio_value = "100000.00"
        return mock_account

    def is_market_open(self) -> bool:
        """Mock market as open."""
        return True


class MockExecutionEngine:
    """Mock execution engine."""

    def __init__(self, broker):
        self.broker = broker

    def place_market_order(self, symbol: str, qty: int, side: str):
        """Mock order placement."""
        mock_order = Mock()
        mock_order.id = f"ORDER_{symbol}_{datetime.now().timestamp()}"
        mock_order.symbol = symbol
        mock_order.qty = qty
        mock_order.side = side
        logger.info(f"[MOCK] Placed order: {side.upper()} {qty} {symbol}")
        return mock_order


class MockPositionManager:
    """Mock position manager."""

    def __init__(self, config):
        self.config = config

    def get_open_positions(self):
        """Return empty positions for testing."""
        return []


def test_close_overnight_positions():
    """Test OMRLiveAdapter.close_overnight_positions with dict positions."""
    logger.info("=" * 70)
    logger.info("TEST 1: close_overnight_positions() with dict positions")
    logger.info("=" * 70)

    try:
        # Create mock broker
        broker = MockBroker()

        # Create OMR adapter (need to mock the strategy and other components)
        with patch('src.trading.adapters.omr_live_adapter.OvernightReversionSignals'), \
             patch('src.trading.adapters.omr_live_adapter.MarketRegimeDetector'), \
             patch('src.trading.adapters.omr_live_adapter.BayesianReversionModel'):

            adapter = OMRLiveAdapter(
                broker=broker,
                symbols=['SPY', 'QQQ', 'TQQQ']
            )

            # Replace execution engine with mock
            adapter.execution_engine = MockExecutionEngine(broker)

            # Call close_overnight_positions
            logger.info("\nCalling close_overnight_positions()...")
            adapter.close_overnight_positions()

            logger.success("\n[PASS] TEST PASSED: close_overnight_positions() completed without errors")
            return True

    except Exception as e:
        logger.error(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_update_positions():
    """Test StrategyAdapter.update_positions with dict positions."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: update_positions() with dict positions")
    logger.info("=" * 70)

    try:
        # Create mock broker
        broker = MockBroker()

        # Create a simple mock strategy
        mock_strategy = Mock()
        mock_strategy.__class__.__name__ = "MockStrategy"

        # Create concrete adapter class for testing (StrategyAdapter is abstract)
        class TestAdapter(StrategyAdapter):
            def get_schedule(self):
                return {'interval': '5min', 'market_hours_only': True}

        # Create strategy adapter
        adapter = TestAdapter(
            strategy=mock_strategy,
            broker=broker,
            symbols=['SPY', 'QQQ'],
            position_size=0.1,
            max_positions=5
        )

        # Replace position manager with mock
        adapter.position_manager = MockPositionManager({})

        # Call update_positions
        logger.info("\nCalling update_positions()...")
        adapter.update_positions()

        logger.success("\n[PASS] TEST PASSED: update_positions() completed without errors")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_position_dict_access():
    """Test direct position dict access patterns."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Position dict access patterns")
    logger.info("=" * 70)

    try:
        # Create sample position dict (as returned by broker)
        position = {
            'symbol': 'AAPL',
            'quantity': 10,
            'avg_entry_price': 150.00,
            'current_price': 155.00,
            'market_value': 1550.00,
            'unrealized_pnl': 50.00,
            'unrealized_pnl_pct': 0.0333,
            'side': 'long'
        }

        # Test all required access patterns
        logger.info("\nTesting dict access patterns...")

        symbol = position['symbol']
        logger.info(f"  symbol: {symbol}")

        qty = int(position['quantity'])
        logger.info(f"  quantity: {qty}")

        entry = float(position['avg_entry_price'])
        logger.info(f"  avg_entry_price: ${entry:.2f}")

        current = float(position['current_price'])
        logger.info(f"  current_price: ${current:.2f}")

        pnl = (current - entry) * qty
        pnl_pct = (current - entry) / entry * 100
        logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

        logger.success("\n[PASS] TEST PASSED: All dict access patterns work correctly")
        return True

    except Exception as e:
        logger.error(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("     POSITION HANDLING FIX VALIDATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing bug fixes for position dict access...")
    logger.info("This validates the fix for:")
    logger.info("  1. Error: 'dict' object has no attribute 'symbol'")
    logger.info("  2. Error: 'dict' object has no attribute 'current_price'")
    logger.info("")

    # Run all tests
    results = []
    results.append(("Position dict access patterns", test_position_dict_access()))
    results.append(("close_overnight_positions()", test_close_overnight_positions()))
    results.append(("update_positions()", test_update_positions()))

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.success("=" * 70)
        logger.success("         ALL TESTS PASSED!")
        logger.success("       Ready to deploy to EC2 instance")
        logger.success("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error("           SOME TESTS FAILED")
        logger.error("        Fix errors before deploying")
        logger.error("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
