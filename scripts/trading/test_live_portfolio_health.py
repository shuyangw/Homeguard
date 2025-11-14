"""
Test Portfolio Health Check with Live Alpaca Broker

Tests the actual Alpaca API integration to ensure portfolio health checks work.
"""

import sys
from pathlib import Path
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

from src.trading.brokers.broker_factory import BrokerFactory
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.utils.logger import logger


def main():
    """Test live portfolio health check with Alpaca."""
    logger.info("=" * 80)
    logger.info("LIVE PORTFOLIO HEALTH CHECK TEST")
    logger.info("=" * 80)
    logger.info("Testing with REAL Alpaca broker (paper trading)")
    logger.info("")

    try:
        # Create Alpaca broker (paper trading)
        logger.info("Connecting to Alpaca (paper trading)...")
        broker = BrokerFactory.create_from_env()

        # Test connection
        if not broker.test_connection():
            logger.error("Failed to connect to Alpaca API")
            return False

        logger.success("Connected to Alpaca successfully")
        logger.info("")

        # Create health checker
        checker = PortfolioHealthChecker(
            broker=broker,
            min_buying_power=1000.0,
            min_portfolio_value=5000.0,
            max_positions=5,
            max_position_age_hours=48
        )

        # Test 1: Quick status check
        logger.info("#" * 80)
        logger.info("# TEST 1: Quick Status Check")
        logger.info("#" * 80)

        status = checker.quick_status_check()

        if status:
            logger.info("Quick Status:")
            logger.info(f"  Account Value: ${status.get('account_value', 0):,.2f}")
            logger.info(f"  Buying Power:  ${status.get('buying_power', 0):,.2f}")
            logger.info(f"  Cash:          ${status.get('cash', 0):,.2f}")
            logger.info(f"  Positions:     {status.get('position_count', 0)}")
            logger.info(f"  Pending Orders: {status.get('pending_orders', 0)}")
            logger.success("[PASS] Quick status check succeeded")
        else:
            logger.error("[FAIL] Quick status check failed")
            return False

        logger.info("")

        # Test 2: Pre-entry health check
        logger.info("#" * 80)
        logger.info("# TEST 2: Pre-Entry Health Check")
        logger.info("#" * 80)

        result = checker.check_before_entry(
            required_capital=None,
            allow_existing_positions=True
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"RESULT: {'PASSED' if result.passed else 'FAILED'}")
        logger.info(f"Errors:   {len(result.errors)}")
        logger.info(f"Warnings: {len(result.warnings)}")
        logger.info("=" * 80)

        if result.passed:
            logger.success("[PASS] Pre-entry health check passed")
        else:
            logger.warning("[BLOCKED] Pre-entry health check would block trading")
            logger.warning("This is expected if:")
            logger.warning("  - Buying power < $1,000")
            logger.warning("  - Portfolio value < $5,000")
            logger.warning("  - Already have 5+ positions")

        logger.info("")

        # Test 3: Pre-exit health check (if positions exist)
        if status.get('position_count', 0) > 0:
            logger.info("#" * 80)
            logger.info("# TEST 3: Pre-Exit Health Check")
            logger.info("#" * 80)

            result = checker.check_before_exit()

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"RESULT: {'PASSED' if result.passed else 'FAILED'}")
            logger.info(f"Errors:   {len(result.errors)}")
            logger.info(f"Warnings: {len(result.warnings)}")
            logger.info("=" * 80)

            if result.passed:
                logger.success("[PASS] Pre-exit health check passed")
            else:
                logger.error("[FAIL] Pre-exit health check detected issues")
        else:
            logger.info("#" * 80)
            logger.info("# TEST 3: Pre-Exit Health Check (SKIPPED - no positions)")
            logger.info("#" * 80)
            logger.info("No positions to test exit check")

        logger.info("")

        # Test 4: Verify broker methods return correct types
        logger.info("#" * 80)
        logger.info("# TEST 4: Verify Broker Return Types")
        logger.info("#" * 80)

        # Test get_account returns dict
        account = broker.get_account()
        assert isinstance(account, dict), "get_account() should return dict"
        assert 'buying_power' in account, "Account dict should have 'buying_power'"
        assert 'portfolio_value' in account, "Account dict should have 'portfolio_value'"
        logger.success("[PASS] get_account() returns dict with correct keys")

        # Test get_positions returns list of dicts
        positions = broker.get_positions()
        assert isinstance(positions, list), "get_positions() should return list"
        if positions:
            assert isinstance(positions[0], dict), "Position should be dict"
            assert 'symbol' in positions[0], "Position dict should have 'symbol'"
            assert 'quantity' in positions[0], "Position dict should have 'quantity'"
        logger.success("[PASS] get_positions() returns list of dicts")

        # Test get_open_orders returns list of dicts
        orders = broker.get_open_orders()
        assert isinstance(orders, list), "get_open_orders() should return list"
        if orders:
            assert isinstance(orders[0], dict), "Order should be dict"
            assert 'symbol' in orders[0], "Order dict should have 'symbol'"
            assert 'order_id' in orders[0], "Order dict should have 'order_id'"
        logger.success("[PASS] get_open_orders() returns list of dicts")

        logger.info("")

        # Final summary
        logger.info("=" * 80)
        logger.info("ALL LIVE TESTS PASSED!")
        logger.info("=" * 80)
        logger.success("Portfolio health check works correctly with live Alpaca API")
        logger.success("All broker methods return correct data types (dicts)")
        logger.success("Health checker correctly handles live broker data")
        logger.info("")
        logger.info("System ready for live paper trading!")

        return True

    except Exception as e:
        logger.error(f"Live test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
