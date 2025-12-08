"""Clear MP positions from strategy state after orders are queued."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.state import StrategyStateManager
from src.utils.logger import logger

manager = StrategyStateManager()
positions = manager.get_positions("mp")

logger.info(f"Clearing {len(positions)} MP positions from state...")
for symbol in list(positions.keys()):
    manager.remove_position("mp", symbol)
    logger.info(f"  Removed: {symbol}")

logger.success("Done.")
logger.info(f"Remaining MP positions: {manager.get_positions('mp')}")
