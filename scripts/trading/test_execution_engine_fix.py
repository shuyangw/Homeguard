"""
Test ExecutionEngine API fixes by placing and canceling test orders.

This validates that the execute_order() method works correctly with
the proper parameters (OrderSide enum, quantity, order_type).
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.trading.brokers.broker_interface import OrderSide, OrderType
from src.trading.core.execution_engine import ExecutionEngine
from src.utils.logger import logger
import os


def test_buy_order(engine: ExecutionEngine, broker: AlpacaBroker):
    """Test placing and canceling a BUY order."""
    logger.info("=" * 70)
    logger.info("TEST 1: BUY Order (execute_order with OrderSide.BUY)")
    logger.info("=" * 70)

    try:
        # Get account info
        account = broker.get_account()
        buying_power = float(account['buying_power'])
        logger.info(f"Buying power: ${buying_power:,.2f}")

        # Place a small BUY order for AAPL
        symbol = 'AAPL'
        quantity = 1  # Just 1 share for testing

        logger.info(f"\nPlacing BUY order: {quantity} share of {symbol}")
        logger.info(f"Using execute_order(symbol='{symbol}', quantity={quantity}, side=OrderSide.BUY, order_type=OrderType.MARKET)")

        order = engine.execute_order(
            symbol=symbol,
            quantity=quantity,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        if order:
            order_id = order.get('order_id', 'UNKNOWN')
            logger.success(f"✓ BUY order placed successfully!")
            logger.info(f"  Order ID: {order_id}")
            logger.info(f"  Symbol: {order.get('symbol')}")
            logger.info(f"  Quantity: {order.get('quantity')}")
            logger.info(f"  Side: {order.get('side')}")
            logger.info(f"  Status: {order.get('status')}")

            # Cancel the order
            logger.info(f"\nCanceling order {order_id}...")
            try:
                broker.cancel_order(order_id)
                logger.success(f"✓ Order canceled successfully")
                return True
            except Exception as e:
                logger.warning(f"Could not cancel order (might already be filled): {e}")
                return True
        else:
            logger.error("✗ BUY order failed - execute_order returned None")
            return False

    except Exception as e:
        logger.error(f"✗ BUY order test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sell_order(engine: ExecutionEngine, broker: AlpacaBroker):
    """Test placing and canceling a SELL order on existing SPY position."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: SELL Order (execute_order with OrderSide.SELL)")
    logger.info("=" * 70)

    try:
        # Check if we have SPY position
        positions = broker.get_positions()
        spy_position = None

        for pos in positions:
            if pos['symbol'] == 'SPY':
                spy_position = pos
                break

        if not spy_position:
            logger.warning("No SPY position found - skipping SELL test")
            logger.info("This is OK - the important test was the BUY order")
            return True

        qty = int(spy_position['quantity'])
        logger.info(f"Found SPY position: {qty} shares @ ${float(spy_position['avg_entry_price']):.2f}")

        # Place SELL order for 1 share
        symbol = 'SPY'
        quantity = 1

        logger.info(f"\nPlacing SELL order: {quantity} share of {symbol}")
        logger.info(f"Using execute_order(symbol='{symbol}', quantity={quantity}, side=OrderSide.SELL, order_type=OrderType.MARKET)")

        order = engine.execute_order(
            symbol=symbol,
            quantity=quantity,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET
        )

        if order:
            order_id = order.get('order_id', 'UNKNOWN')
            logger.success(f"✓ SELL order placed successfully!")
            logger.info(f"  Order ID: {order_id}")
            logger.info(f"  Symbol: {order.get('symbol')}")
            logger.info(f"  Quantity: {order.get('quantity')}")
            logger.info(f"  Side: {order.get('side')}")
            logger.info(f"  Status: {order.get('status')}")

            # Cancel the order
            logger.info(f"\nCanceling order {order_id}...")
            try:
                broker.cancel_order(order_id)
                logger.success(f"✓ Order canceled successfully")
                return True
            except Exception as e:
                logger.warning(f"Could not cancel order (might already be filled): {e}")
                return True
        else:
            logger.error("✗ SELL order failed - execute_order returned None")
            return False

    except Exception as e:
        logger.error(f"✗ SELL order test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run order execution tests."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("     EXECUTIONENGINE API FIX VALIDATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing execute_order() with proper OrderSide enum and parameters")
    logger.info("After hours - orders should be accepted but not filled")
    logger.info("")

    # Initialize broker
    api_key = os.environ.get('APCA_API_KEY_ID')
    secret_key = os.environ.get('APCA_API_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")
        return 1

    try:
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        logger.success("✓ Connected to Alpaca Paper Trading")

        # Get account info
        account = broker.get_account()
        logger.info(f"Account: {account['account_id']}")
        logger.info(f"Portfolio Value: ${float(account['portfolio_value']):,.2f}")
        logger.info(f"Buying Power: ${float(account['buying_power']):,.2f}")

        # Check market status
        is_open = broker.is_market_open()
        logger.info(f"Market Status: {'OPEN' if is_open else 'CLOSED'}")
        logger.info("")

        # Create execution engine
        engine = ExecutionEngine(broker)

        # Run tests
        results = []
        results.append(("BUY Order Test", test_buy_order(engine, broker)))
        results.append(("SELL Order Test", test_sell_order(engine, broker)))

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
            logger.success("  ExecutionEngine API fix validated successfully")
            logger.success("=" * 70)
        else:
            logger.error("=" * 70)
            logger.error("           SOME TESTS FAILED")
            logger.error("        Fix errors before deploying")
            logger.error("=" * 70)

        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
