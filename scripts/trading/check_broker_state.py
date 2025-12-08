"""Check broker state - open orders and positions."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.trading.brokers import AlpacaBroker
from src.utils.logger import logger

api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)

logger.info("=" * 50)
logger.info("BROKER STATE")
logger.info("=" * 50)

# Check open orders
logger.info("\nOPEN ORDERS:")
orders = broker.get_open_orders()
if orders:
    for o in orders:
        logger.info(f"  {o.get('symbol')}: {o.get('side')} {o.get('quantity')} @ {o.get('type')} - Status: {o.get('status')}")
else:
    logger.info("  No open orders")

# Check positions
logger.info("\nCURRENT POSITIONS:")
positions = broker.get_positions()
if positions:
    for p in positions:
        logger.info(f"  {p['symbol']}: {p['quantity']} shares @ ${float(p.get('current_price', 0)):.2f}")
else:
    logger.info("  No positions")

# Check account
account = broker.get_account()
logger.info(f"\nACCOUNT:")
logger.info(f"  Portfolio value: ${float(account.get('portfolio_value', 0)):.2f}")
logger.info(f"  Buying power: ${float(account.get('buying_power', 0)):.2f}")
logger.info(f"  Cash: ${float(account.get('cash', 0)):.2f}")
