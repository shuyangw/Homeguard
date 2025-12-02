#!/usr/bin/env python3
"""
OMR Strategy Cycle Test Script

Condenses a full trading day cycle (close->open->close) into a configurable
short timeframe for rapid testing.

Normal OMR schedule:
- 9:31 AM: Close overnight positions (exit)
- 3:50 PM: Open new positions (entry)
- Next day 9:31 AM: Close positions (exit)

This script compresses that into minutes for testing.

Usage:
    python scripts/trading/test_omr_cycle.py --start-delay 30
    python scripts/trading/test_omr_cycle.py --exit-interval 60 --entry-interval 120
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
import os

from src.utils.logger import logger
from src.utils import timezone as tz


def cancel_all_orders(broker):
    """Cancel all open orders."""
    try:
        orders = broker.get_open_orders()
        if not orders:
            logger.info("No open orders to cancel")
            return 0

        cancelled = 0
        for order in orders:
            try:
                broker.cancel_order(order.id)
                logger.info(f"Cancelled order: {order.id} ({order.symbol})")
                cancelled += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {order.id}: {e}")

        logger.success(f"Cancelled {cancelled}/{len(orders)} orders")
        return cancelled
    except Exception as e:
        logger.error(f"Error getting open orders: {e}")
        return 0


def close_all_positions(broker):
    """Close all open positions."""
    try:
        positions = broker.get_positions()
        if not positions:
            logger.info("No open positions to close")
            return 0

        closed = 0
        for pos in positions:
            try:
                # Submit market sell order
                from src.trading.brokers.broker_interface import OrderSide, OrderType
                order = broker.submit_order(
                    symbol=pos.symbol,
                    qty=abs(float(pos.qty)),
                    side=OrderSide.SELL if float(pos.qty) > 0 else OrderSide.BUY,
                    order_type=OrderType.MARKET
                )
                logger.info(f"Closed position: {pos.symbol} ({pos.qty} shares)")
                closed += 1
            except Exception as e:
                logger.error(f"Failed to close position {pos.symbol}: {e}")

        logger.success(f"Closed {closed}/{len(positions)} positions")
        return closed
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return 0


def run_exit_logic(adapter):
    """Run the exit logic (simulating 9:31 AM close)."""
    logger.info("")
    logger.header("=" * 60)
    logger.info("SIMULATED 9:31 AM - RUNNING EXIT LOGIC")
    logger.header("=" * 60)

    try:
        # Run the adapter's exit logic
        adapter.run_once(action='exit')
        logger.success("Exit logic completed")
    except Exception as e:
        logger.error(f"Exit logic failed: {e}")


def run_entry_logic(adapter):
    """Run the entry logic (simulating 3:50 PM open)."""
    logger.info("")
    logger.header("=" * 60)
    logger.info("SIMULATED 3:50 PM - RUNNING ENTRY LOGIC")
    logger.header("=" * 60)

    try:
        # Pre-fetch intraday data first
        if hasattr(adapter, 'prefetch_intraday_data'):
            logger.info("Pre-fetching intraday data...")
            adapter.prefetch_intraday_data()

        # Run the adapter's entry logic
        adapter.run_once(action='entry')
        logger.success("Entry logic completed")
    except Exception as e:
        logger.error(f"Entry logic failed: {e}")


def show_account_status(broker):
    """Display current account and position status."""
    logger.info("")
    logger.header("-" * 40)
    logger.info("ACCOUNT STATUS")
    logger.header("-" * 40)

    try:
        account = broker.get_account()
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")

        positions = broker.get_positions()
        if positions:
            logger.info(f"\nOpen Positions ({len(positions)}):")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                logger.info(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
                           f"(P&L: ${pnl:+.2f} / {pnl_pct:+.2f}%)")
        else:
            logger.info("\nNo open positions")

        orders = broker.get_open_orders()
        if orders:
            logger.info(f"\nOpen Orders ({len(orders)}):")
            for order in orders:
                logger.info(f"  {order.symbol}: {order.side} {order.qty} @ {order.type}")
        else:
            logger.info("\nNo open orders")

    except Exception as e:
        logger.error(f"Error getting account status: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Test OMR strategy cycle in compressed timeframe'
    )
    parser.add_argument(
        '--start-delay',
        type=int,
        default=10,
        help='Seconds to wait before starting the test (default: 10)'
    )
    parser.add_argument(
        '--exit-interval',
        type=int,
        default=60,
        help='Seconds between start and first exit (simulated 9:31 AM) (default: 60)'
    )
    parser.add_argument(
        '--entry-interval',
        type=int,
        default=120,
        help='Seconds between first exit and entry (simulated 3:50 PM) (default: 120)'
    )
    parser.add_argument(
        '--final-exit-interval',
        type=int,
        default=60,
        help='Seconds between entry and final exit (next day 9:31 AM) (default: 60)'
    )
    parser.add_argument(
        '--skip-cleanup',
        action='store_true',
        help='Skip cancelling orders and closing positions at the end'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show timing without executing trades'
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Calculate schedule
    now = datetime.now()
    start_time = now + timedelta(seconds=args.start_delay)
    exit1_time = start_time + timedelta(seconds=args.exit_interval)
    entry_time = exit1_time + timedelta(seconds=args.entry_interval)
    exit2_time = entry_time + timedelta(seconds=args.final_exit_interval)

    total_duration = args.start_delay + args.exit_interval + args.entry_interval + args.final_exit_interval

    logger.info("")
    logger.header("=" * 70)
    logger.info("OMR STRATEGY CYCLE TEST")
    logger.header("=" * 70)
    logger.info(f"Current time: {now.strftime('%H:%M:%S')}")
    logger.info(f"Total test duration: {total_duration} seconds ({total_duration/60:.1f} minutes)")
    logger.info("")
    logger.info("Schedule:")
    logger.info(f"  Start:           {start_time.strftime('%H:%M:%S')} (in {args.start_delay}s)")
    logger.info(f"  Exit #1 (9:31):  {exit1_time.strftime('%H:%M:%S')} (T+{args.exit_interval}s)")
    logger.info(f"  Entry (3:50):    {entry_time.strftime('%H:%M:%S')} (T+{args.exit_interval + args.entry_interval}s)")
    logger.info(f"  Exit #2 (9:31):  {exit2_time.strftime('%H:%M:%S')} (T+{total_duration - args.start_delay}s)")
    logger.header("=" * 70)

    if args.dry_run:
        logger.warning("DRY RUN - No trades will be executed")
        logger.info("Remove --dry-run to execute actual trades")
        return 0

    # Check API credentials
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca API credentials not found")
        return 1

    # Initialize broker and adapter
    logger.info("")
    logger.info("Initializing broker and adapter...")

    try:
        from src.trading.brokers.alpaca_broker import AlpacaBroker
        from src.trading.adapters.omr_adapter import OMRLiveAdapter

        broker = AlpacaBroker(paper=True)
        logger.success("Connected to Alpaca Paper Trading")

        # Load OMR config
        from src.trading.adapters.omr_adapter import load_omr_config
        config = load_omr_config()

        # Create adapter
        adapter = OMRLiveAdapter(
            broker=broker,
            symbols=config.symbols,
            position_size=config.position_size,
            max_positions=config.max_positions
        )
        logger.success("OMR adapter initialized")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show initial status
    show_account_status(broker)

    # Wait for start
    logger.info("")
    logger.info(f"Starting test in {args.start_delay} seconds...")
    time.sleep(args.start_delay)

    try:
        # Phase 1: Exit (simulated 9:31 AM)
        logger.info("")
        logger.warning(f"[Phase 1] Waiting {args.exit_interval}s until first exit...")
        time.sleep(args.exit_interval)
        run_exit_logic(adapter)
        show_account_status(broker)

        # Phase 2: Entry (simulated 3:50 PM)
        logger.info("")
        logger.warning(f"[Phase 2] Waiting {args.entry_interval}s until entry...")
        time.sleep(args.entry_interval)
        run_entry_logic(adapter)
        show_account_status(broker)

        # Phase 3: Final Exit (simulated next day 9:31 AM)
        logger.info("")
        logger.warning(f"[Phase 3] Waiting {args.final_exit_interval}s until final exit...")
        time.sleep(args.final_exit_interval)
        run_exit_logic(adapter)
        show_account_status(broker)

    except KeyboardInterrupt:
        logger.warning("\nTest interrupted by user")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if not args.skip_cleanup:
            logger.info("")
            logger.header("=" * 60)
            logger.info("CLEANUP - Cancelling orders and closing positions")
            logger.header("=" * 60)

            cancel_all_orders(broker)
            time.sleep(1)  # Wait for cancellations to process
            close_all_positions(broker)
            time.sleep(2)  # Wait for closes to process

            # Final status
            show_account_status(broker)
        else:
            logger.warning("Skipping cleanup (--skip-cleanup specified)")

    logger.info("")
    logger.header("=" * 60)
    logger.success("OMR CYCLE TEST COMPLETE")
    logger.header("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
