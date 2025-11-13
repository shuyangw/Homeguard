"""
Test Alpaca Connection

Simple script to test connectivity with Alpaca paper trading account.
Verifies that credentials are set up correctly and API is accessible.

Usage:
    python scripts/trading/test_alpaca_connection.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

from src.trading.brokers.broker_factory import BrokerFactory
from src.trading.brokers.broker_interface import OrderSide, OrderType
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_connection():
    """Test Alpaca API connection."""
    logger.info("=" * 60)
    logger.info("Testing Alpaca Paper Trading Connection")
    logger.info("=" * 60)

    try:
        # Create broker from YAML config
        config_path = project_root / "config" / "trading" / "broker_alpaca.yaml"
        logger.info(f"Loading config from: {config_path}")

        broker = BrokerFactory.create_from_yaml(str(config_path))
        logger.success("[OK] Broker created successfully")

        # Test connection
        logger.info("\n1. Testing connection...")
        if broker.test_connection():
            logger.success("[OK] Connection successful")
        else:
            logger.error("[FAILED] Connection failed")
            return False

        # Get account info
        logger.info("\n2. Getting account information...")
        account = broker.get_account()
        logger.success(f"[OK] Account ID: {account['account_id']}")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Cash: ${account['cash']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")

        # Check market status
        logger.info("\n3. Checking market status...")
        is_open = broker.is_market_open()
        logger.info(f"  Market Open: {is_open}")

        # Get current positions
        logger.info("\n4. Getting current positions...")
        positions = broker.get_positions()
        logger.info(f"  Open Positions: {len(positions)}")
        for pos in positions:
            logger.info(
                f"    {pos['symbol']}: {pos['quantity']} shares @ "
                f"${pos['current_price']:.2f} (P&L: ${pos['unrealized_pnl']:+.2f})"
            )

        # Test quote fetching
        logger.info("\n5. Testing quote fetching...")
        test_symbol = "SPY"
        quote = broker.get_latest_quote(test_symbol)
        logger.success(f"[OK] Got quote for {test_symbol}")
        logger.info(f"  Bid: ${quote['bid']:.2f}")
        logger.info(f"  Ask: ${quote['ask']:.2f}")
        logger.info(f"  Timestamp: {quote['timestamp']}")

        logger.info("\n" + "=" * 60)
        logger.success("All tests passed!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
