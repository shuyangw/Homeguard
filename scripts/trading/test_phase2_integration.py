"""
Phase 2 Integration Test

Tests the complete paper trading system with Alpaca paper account.
Validates all Phase 2 components working together.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.trading.brokers.broker_factory import BrokerFactory
from src.trading.brokers.broker_interface import OrderSide, OrderType

# Import directly from modules to avoid __init__.py import chains
import importlib.util

# Load execution_engine module directly
spec = importlib.util.spec_from_file_location(
    "execution_engine",
    project_root / "src" / "trading" / "core" / "execution_engine.py"
)
execution_engine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution_engine_module)
ExecutionEngine = execution_engine_module.ExecutionEngine

# Load position_manager module directly
spec = importlib.util.spec_from_file_location(
    "position_manager",
    project_root / "src" / "trading" / "core" / "position_manager.py"
)
position_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(position_manager_module)
PositionManager = position_manager_module.PositionManager

from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_broker_connection():
    """Test 1: Broker connection."""
    logger.header("Test 1: Broker Connection")

    try:
        config_path = project_root / "config" / "trading" / "broker_alpaca.yaml"
        broker = BrokerFactory.create_from_yaml(str(config_path))

        if broker.test_connection():
            logger.success("[PASS] Broker connection successful")

            # Get account info
            account = broker.get_account()
            logger.info(f"  Account ID: {account['account_id']}")
            logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
            return True, broker
        else:
            logger.error("[FAIL] Broker connection failed")
            return False, None

    except Exception as e:
        logger.error(f"[FAIL] Broker connection error: {e}")
        return False, None


def test_execution_engine(broker):
    """Test 2: Execution engine."""
    logger.header("\nTest 2: Execution Engine")

    try:
        engine = ExecutionEngine(
            broker=broker,
            max_retries=3,
            retry_delay=1.0,
            fill_timeout=30.0
        )

        logger.success("[PASS] Execution engine initialized")

        # Test metrics
        metrics = engine.get_execution_metrics()
        logger.info(f"  Total orders: {metrics['total_orders']}")
        logger.info(f"  Success rate: {metrics['success_rate']:.1%}")

        return True, engine

    except Exception as e:
        logger.error(f"[FAIL] Execution engine error: {e}")
        return False, None


def test_position_manager():
    """Test 3: Position manager."""
    logger.header("\nTest 3: Position Manager")

    try:
        config = {
            'max_position_size_pct': 0.15,
            'max_concurrent_positions': 3,
            'max_total_exposure_pct': 0.45,
            'stop_loss_pct': -0.02,
        }

        manager = PositionManager(config)
        logger.success("[PASS] Position manager initialized")

        # Test risk limits
        is_valid, reason = manager.check_risk_limits()
        logger.info(f"  Risk limits: {reason}")

        return True, manager

    except Exception as e:
        logger.error(f"[FAIL] Position manager error: {e}")
        return False, None


def test_quote_fetching(broker):
    """Test 4: Market data fetching."""
    logger.header("\nTest 4: Market Data Fetching")

    try:
        # Test fetching quotes for multiple symbols
        symbols = ['SPY', 'QQQ', 'TQQQ']

        for symbol in symbols:
            quote = broker.get_latest_quote(symbol)
            logger.info(f"  {symbol}: Bid=${quote['bid']:.2f}, Ask=${quote['ask']:.2f}")

        logger.success("[PASS] Market data fetching successful")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Market data error: {e}")
        return False


def test_order_execution_dry_run(broker, engine):
    """Test 5: Order execution (dry run - no actual orders)."""
    logger.header("\nTest 5: Order Execution (Dry Run)")

    try:
        # Check current positions
        positions = broker.get_positions()
        logger.info(f"  Current positions: {len(positions)}")

        # Check account info
        account = broker.get_account()
        logger.info(f"  Buying power: ${account['buying_power']:,.2f}")

        # Verify execution engine is ready
        metrics = engine.get_execution_metrics()
        logger.success(f"[PASS] Execution engine ready (processed {metrics['total_orders']} orders)")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Order execution check error: {e}")
        return False


def test_market_status(broker):
    """Test 6: Market status check."""
    logger.header("\nTest 6: Market Status")

    try:
        is_open = broker.is_market_open()
        logger.info(f"  Market open: {is_open}")

        hours = broker.get_market_hours(date=None)
        open_time, close_time = hours
        logger.info(f"  Market hours: Open={open_time}, Close={close_time}")

        logger.success("[PASS] Market status check successful")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Market status error: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.header("=" * 70)
    logger.header("Phase 2 Integration Test")
    logger.header("=" * 70)
    logger.blank()

    results = []

    # Test 1: Broker connection
    passed, broker = test_broker_connection()
    results.append(("Broker Connection", passed))
    if not passed:
        logger.error("\nCritical test failed. Stopping.")
        return False

    # Test 2: Execution engine
    passed, engine = test_execution_engine(broker)
    results.append(("Execution Engine", passed))
    if not passed:
        logger.error("\nCritical test failed. Stopping.")
        return False

    # Test 3: Position manager
    passed, manager = test_position_manager()
    results.append(("Position Manager", passed))

    # Test 4: Market data
    passed = test_quote_fetching(broker)
    results.append(("Market Data", passed))

    # Test 5: Order execution check
    passed = test_order_execution_dry_run(broker, engine)
    results.append(("Order Execution", passed))

    # Test 6: Market status
    passed = test_market_status(broker)
    results.append(("Market Status", passed))

    # Summary
    logger.blank()
    logger.header("=" * 70)
    logger.header("Test Results Summary")
    logger.header("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"  {status} {test_name}")

    logger.blank()
    logger.header(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.success("\nALL TESTS PASSED!")
        logger.success("Phase 2 integration validated successfully")
        return True
    else:
        logger.error(f"\n{total_count - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
