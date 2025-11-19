"""Cancel all open orders on paper trading account."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.utils.logger import logger
import os


def main():
    """Cancel all open orders."""
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

    broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)

    # Get all open orders
    orders = broker.trading_client.get_orders()

    logger.info(f"Found {len(orders)} open orders")

    for order in orders:
        logger.info(f"Canceling order {order.id}: {order.side} {order.qty} {order.symbol}")
        broker.cancel_order(order.id)
        logger.success(f"  Canceled {order.id}")

    logger.success(f"All {len(orders)} test orders canceled!")


if __name__ == "__main__":
    main()
