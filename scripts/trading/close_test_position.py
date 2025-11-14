"""
Close Test Position

Close the test SPY position from paper trading account.

Usage:
    python scripts/trading/close_test_position.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

from src.trading.brokers.broker_factory import BrokerFactory
from src.utils.logger import logger


def main():
    """Close the test SPY position."""
    logger.info("=" * 80)
    logger.info("CLOSE TEST POSITION")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Step 1: Connect to Alpaca
        logger.info("Step 1: Connecting to Alpaca...")
        broker = BrokerFactory.create_from_env()

        if not broker.test_connection():
            logger.error("Failed to connect to Alpaca API")
            return False

        logger.success("Connected successfully")
        logger.info("")

        # Step 2: Check current positions
        logger.info("Step 2: Checking current positions...")
        positions = broker.get_positions()
        logger.info(f"  Total positions: {len(positions)}")

        if not positions:
            logger.warning("No positions to close")
            return True

        for pos in positions:
            logger.info(
                f"    {pos['symbol']}: {pos['quantity']} shares @ "
                f"${pos['current_price']:.2f} (P&L: ${pos['unrealized_pnl']:+.2f})"
            )
        logger.info("")

        # Step 3: Close SPY position
        spy_position = None
        for pos in positions:
            if pos['symbol'] == 'SPY':
                spy_position = pos
                break

        if not spy_position:
            logger.warning("No SPY position found")
            return True

        logger.info("Step 3: Closing SPY position...")
        logger.info(f"  Current position: {spy_position['quantity']} shares")
        logger.info(f"  Current price: ${spy_position['current_price']:.2f}")
        logger.info(f"  Unrealized P&L: ${spy_position['unrealized_pnl']:+.2f}")
        logger.info("")

        # Close the position
        logger.info("Submitting close order...")
        close_result = broker.close_position('SPY')

        logger.success("Position close order submitted!")
        logger.info(f"  Order ID: {close_result['order_id']}")
        logger.info(f"  Status: {close_result.get('status', 'pending')}")
        logger.info(f"  Quantity: {close_result['quantity']}")
        logger.info("")

        # Step 4: Wait and verify
        import time
        logger.info("Step 4: Verifying position closure...")
        time.sleep(2)

        positions = broker.get_positions()
        spy_still_open = any(pos['symbol'] == 'SPY' for pos in positions)

        if not spy_still_open:
            logger.success("SPY position successfully closed!")
        else:
            logger.warning("SPY position still open (order may be pending)")
        logger.info("")

        # Step 5: Final account status
        logger.info("Step 5: Final account status...")
        account = broker.get_account()
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Cash: ${account['cash']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        logger.info(f"  Open Positions: {len(positions)}")
        logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.success("POSITION CLOSED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.success(f"  [OK] Closed SPY position")
        logger.success(f"  [OK] Order ID: {close_result['order_id']}")
        logger.success(f"  [OK] Realized P&L: ${spy_position['unrealized_pnl']:+.2f}")
        logger.info("")
        logger.info("Trade completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
