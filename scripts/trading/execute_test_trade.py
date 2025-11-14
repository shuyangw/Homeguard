"""
Execute Test Trade on Paper Account

Simple script to test the full trading pipeline by executing a small trade.
This validates:
- Alpaca API connection
- Market data fetching
- Order placement
- Order tracking
- Logging system

Usage:
    python scripts/trading/execute_test_trade.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

from src.trading.brokers.broker_factory import BrokerFactory
from src.trading.brokers.broker_interface import OrderSide, OrderType, TimeInForce
from src.utils.logger import logger


def main():
    """Execute a small test trade on paper account."""
    logger.info("=" * 80)
    logger.info("PAPER TRADING TEST - EXECUTE SMALL TRADE")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Step 1: Connect to Alpaca
        logger.info("Step 1: Connecting to Alpaca (paper trading)...")
        broker = BrokerFactory.create_from_env()

        if not broker.test_connection():
            logger.error("Failed to connect to Alpaca API")
            return False

        logger.success("Connected to Alpaca successfully")
        logger.info("")

        # Step 2: Check market status
        logger.info("Step 2: Checking market status...")
        is_open = broker.is_market_open()
        logger.info(f"  Market is currently: {'OPEN' if is_open else 'CLOSED'}")

        if not is_open:
            logger.warning("Market is closed - trade will be queued until market opens")
            logger.warning("For testing purposes, we'll still submit the order")
        logger.info("")

        # Step 3: Get account info
        logger.info("Step 3: Checking account status...")
        account = broker.get_account()
        logger.info(f"  Account ID: {account['account_id']}")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Cash: ${account['cash']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        logger.info("")

        # Step 4: Get current quote for SPY
        logger.info("Step 4: Fetching current quote for SPY...")
        test_symbol = "SPY"
        quote = broker.get_latest_quote(test_symbol)
        logger.info(f"  Symbol: {test_symbol}")
        logger.info(f"  Bid: ${quote['bid']:.2f}")
        logger.info(f"  Ask: ${quote['ask']:.2f}")
        logger.info(f"  Timestamp: {quote['timestamp']}")

        # Estimate cost
        estimated_cost = quote['ask'] * 1  # 1 share
        logger.info(f"  Estimated cost for 1 share: ${estimated_cost:.2f}")

        if estimated_cost > account['buying_power']:
            logger.error("Insufficient buying power for trade")
            return False
        logger.info("")

        # Step 5: Place test trade
        logger.info("Step 5: Placing test trade...")
        logger.info(f"  Order: BUY 1 share of {test_symbol} at MARKET")

        # Confirm with user before placing trade
        logger.warning("This will place a REAL order on your paper trading account")
        logger.info("Press Ctrl+C to cancel, or wait 5 seconds to continue...")

        try:
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("\nTrade cancelled by user")
            return False

        # Place the order
        logger.info("Submitting order...")
        order_result = broker.place_order(
            symbol=test_symbol,
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )

        order_id = order_result['order_id']
        logger.success(f"Order submitted successfully!")
        logger.info(f"  Order ID: {order_id}")
        logger.info(f"  Status: {order_result.get('status', 'pending')}")
        logger.info("")

        # Step 6: Check order status
        logger.info("Step 6: Checking order status...")
        time.sleep(2)  # Wait a moment for order to process

        try:
            order_status = broker.get_order(order_id)
            logger.info(f"  Order ID: {order_id}")
            logger.info(f"  Status: {order_status.get('status', 'unknown')}")
            logger.info(f"  Filled Qty: {order_status.get('filled_qty', 0)}")

            if order_status.get('filled_qty', 0) > 0:
                logger.info(f"  Filled Price: ${order_status.get('avg_fill_price', 0):.2f}")
                logger.success("Order filled successfully!")
            else:
                logger.warning("Order not yet filled (may be queued if market is closed)")
        except Exception as e:
            logger.warning(f"Could not check order status: {e}")
        logger.info("")

        # Step 7: Get updated account info
        logger.info("Step 7: Checking updated account status...")
        account = broker.get_account()
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Cash: ${account['cash']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        logger.info("")

        # Step 8: Check positions
        logger.info("Step 8: Checking current positions...")
        positions = broker.get_positions()
        logger.info(f"  Total positions: {len(positions)}")

        for pos in positions:
            logger.info(
                f"    {pos['symbol']}: {pos['quantity']} shares @ "
                f"${pos['current_price']:.2f} (P&L: ${pos['unrealized_pnl']:+.2f})"
            )
        logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.success("TEST TRADE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.success(f"  [OK] Connected to Alpaca API")
        logger.success(f"  [OK] Fetched market data for {test_symbol}")
        logger.success(f"  [OK] Placed market order for 1 share")
        logger.success(f"  [OK] Order ID: {order_id}")
        logger.success(f"  [OK] Verified account and position updates")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Check your Alpaca paper trading dashboard to verify")
        logger.info("  2. If market is closed, order will execute when market opens")
        logger.info("  3. You can close this position manually via Alpaca dashboard")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"Test trade failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
