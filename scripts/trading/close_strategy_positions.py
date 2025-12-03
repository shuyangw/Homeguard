"""
Close Strategy Positions.

Closes all positions owned by a specific strategy using the broker.
Updates state after each successful close.

Usage:
    python scripts/trading/close_strategy_positions.py --strategy mp
    python scripts/trading/close_strategy_positions.py --strategy omr --dry-run
"""

import sys
import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.brokers import AlpacaBroker
from src.trading.state import StrategyStateManager
from src.utils.logger import logger
from src.utils.timezone import tz


def close_strategy_positions(
    strategy: str,
    dry_run: bool = False,
    timeout: int = 30
) -> dict:
    """
    Close all positions for a strategy.

    Args:
        strategy: Strategy name (omr, mp)
        dry_run: If True, don't actually close, just show what would be closed
        timeout: Timeout in seconds for each order

    Returns:
        Dict with results: {closed: [], failed: [], partial: []}
    """
    # Load environment
    load_dotenv()

    # Initialize state manager
    manager = StrategyStateManager()

    # Get positions to close
    positions = manager.get_positions(strategy)

    if not positions:
        logger.info(f"No positions to close for {strategy}")
        return {'closed': [], 'failed': [], 'partial': []}

    logger.info(f"Positions to close for {strategy}:")
    for symbol, data in positions.items():
        logger.info(f"  {symbol}: {data['qty']} shares @ ${data['entry_price']:.2f}")

    if dry_run:
        logger.warning("DRY RUN - no orders will be placed")
        return {'closed': [], 'failed': [], 'partial': [], 'dry_run': list(positions.keys())}

    # Initialize broker
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca API credentials not found")
        return {'closed': [], 'failed': list(positions.keys()), 'partial': []}

    broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)

    # Verify connection
    account = broker.get_account()
    if not account:
        logger.error("Failed to connect to broker")
        return {'closed': [], 'failed': list(positions.keys()), 'partial': []}

    logger.success("Connected to Alpaca")

    # Acquire execution lock
    if not manager.acquire_execution_lock(strategy):
        logger.error("Could not acquire execution lock")
        return {'closed': [], 'failed': list(positions.keys()), 'partial': []}

    results = {'closed': [], 'failed': [], 'partial': []}

    try:
        for symbol, data in list(positions.items()):
            qty = data['qty']
            logger.info(f"Closing {symbol}: {qty} shares...")

            try:
                # Check if we actually have this position at broker
                broker_positions = broker.get_positions()
                broker_qty = 0
                for pos in broker_positions:
                    if pos['symbol'] == symbol:
                        broker_qty = int(pos['quantity'])
                        break

                if broker_qty == 0:
                    # Position already closed at broker
                    logger.warning(f"  {symbol}: No position at broker, removing from state")
                    manager.remove_position(strategy, symbol)
                    results['closed'].append(symbol)
                    continue

                # Use actual broker quantity (may differ from state)
                qty_to_close = min(qty, broker_qty)

                # Submit sell order
                from src.trading.brokers.broker_interface import OrderSide, OrderType

                order = broker.submit_order(
                    symbol=symbol,
                    qty=qty_to_close,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET
                )

                if not order:
                    logger.error(f"  {symbol}: Order submission failed")
                    results['failed'].append(symbol)
                    continue

                order_id = order.get('id')
                logger.info(f"  {symbol}: Order submitted (ID: {order_id})")

                # Wait for fill
                start_time = time.time()
                filled_qty = 0

                while time.time() - start_time < timeout:
                    order_status = broker.get_order(order_id)
                    if order_status:
                        status = order_status.get('status')
                        filled_qty = int(order_status.get('filled_qty', 0))

                        if status == 'filled':
                            logger.success(f"  {symbol}: Filled {filled_qty} shares")
                            break
                        elif status in ('cancelled', 'rejected', 'expired'):
                            logger.error(f"  {symbol}: Order {status}")
                            break

                    time.sleep(1)

                # Update state based on fill
                if filled_qty >= qty:
                    # Complete close
                    manager.remove_position(strategy, symbol)
                    results['closed'].append(symbol)
                elif filled_qty > 0:
                    # Partial close
                    remaining = qty - filled_qty
                    manager.update_position_qty(strategy, symbol, remaining)
                    results['partial'].append(f"{symbol}:{remaining}")
                    logger.warning(f"  {symbol}: Partial fill, {remaining} shares remaining")
                else:
                    # No fill
                    results['failed'].append(symbol)
                    logger.error(f"  {symbol}: No fill within timeout")

            except Exception as e:
                logger.error(f"  {symbol}: Error closing - {e}")
                results['failed'].append(symbol)

    finally:
        # Release execution lock
        manager.release_execution_lock(strategy)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("CLOSE POSITIONS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Closed: {len(results['closed'])} positions")
    if results['closed']:
        logger.info(f"  {', '.join(results['closed'])}")

    if results['partial']:
        logger.warning(f"Partial: {len(results['partial'])} positions")
        for p in results['partial']:
            logger.warning(f"  {p}")

    if results['failed']:
        logger.error(f"Failed: {len(results['failed'])} positions")
        logger.error(f"  {', '.join(results['failed'])}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Close strategy positions')
    parser.add_argument('--strategy', required=True, choices=['omr', 'mp'],
                       help='Strategy to close positions for')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be closed without closing')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout per order in seconds (default: 30)')

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("CLOSE STRATEGY POSITIONS")
    logger.info("=" * 50)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Time: {tz.timestamp()}")
    logger.info("")

    results = close_strategy_positions(
        strategy=args.strategy,
        dry_run=args.dry_run,
        timeout=args.timeout
    )

    # Exit with error code if any failures
    if results['failed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
