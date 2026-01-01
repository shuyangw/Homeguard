#!/usr/bin/env python3
"""
CSCM Paper Trading Runner.

Main entry point for running CSCM paper trading on EC2.

Features:
- Binance data for momentum signals (with Alpaca fallback)
- Alpaca paper trading for order execution
- Monday rebalancing
- Portfolio tracking during equity hours
- Automatic restart on errors

Usage:
    # Normal operation (continuous loop)
    python scripts/trading/run_cscm_paper.py

    # Run once and exit (for testing)
    python scripts/trading/run_cscm_paper.py --once

    # Custom check interval
    python scripts/trading/run_cscm_paper.py --check-interval 30

    # Show status and exit
    python scripts/trading/run_cscm_paper.py --status
"""

import sys
import time
import argparse
import signal
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
from src.trading.portfolio.tracker import PortfolioTracker
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

# Global for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info("[CSCM Paper] Shutdown signal received")
    _shutdown_requested = True


def print_status(adapter: CSCMPaperAdapter, tracker: PortfolioTracker) -> None:
    """Print current status."""
    print("=" * 60)
    print("CSCM Paper Trading Status")
    print("=" * 60)
    print()

    status = tracker.get_status()
    print("--- Portfolio ---")
    print(f"Value:          ${status['current_value']:,.2f}")
    print(f"Cash:           ${status['cash']:,.2f}")
    print(f"Positions:      {status['positions_count']}")
    print()

    print("--- Strategy ---")
    print(f"Regime:         {status['regime'].upper()}")
    print(f"Last Update:    {status['last_update'] or 'Never'}")
    print(f"Equity Hours:   {'Yes' if status['is_equity_hours'] else 'No'}")
    print()

    adapter_status = adapter.get_status()
    print("--- Trailing Stop ---")
    print(f"Triggered:      {adapter_status.get('trailing_stop_triggered', False)}")
    print(f"Current DD:     {adapter_status.get('current_drawdown', 0):.1%}")
    print(f"Peak Value:     ${adapter_status.get('peak_value', 0):,.2f}")
    print()

    if status['positions']:
        print("--- Current Positions ---")
        for symbol, qty in status['positions'].items():
            print(f"  {symbol}: {qty:.6f}")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Run CSCM paper trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cscm_paper.py              # Normal operation
  python run_cscm_paper.py --once       # Single iteration
  python run_cscm_paper.py --status     # Show status and exit
        """
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help='Minutes between checks (default: 60)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show status and exit'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top coins to hold (default: 5)'
    )
    parser.add_argument(
        '--rebalance-day',
        type=str,
        default='monday',
        choices=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        help='Day of week to rebalance (default: monday)'
    )

    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 80)
    logger.info("CSCM PAPER TRADING")
    logger.info("=" * 80)
    logger.info(f"[CSCM Paper] Starting at {tz.now()}")
    logger.info(f"[CSCM Paper] Check interval: {args.check_interval} minutes")
    logger.info(f"[CSCM Paper] Top N: {args.top_n}")
    logger.info(f"[CSCM Paper] Rebalance day: {args.rebalance_day}")

    # Create adapter and tracker
    try:
        adapter = CSCMPaperAdapter(
            top_n=args.top_n,
            rebalance_day=args.rebalance_day
        )
        tracker = PortfolioTracker(adapter)
    except Exception as e:
        logger.error(f"[CSCM Paper] Failed to initialize: {e}")
        return 1

    # Handle status command
    if args.status:
        print_status(adapter, tracker)
        return 0

    # Handle single run
    if args.once:
        logger.info("[CSCM Paper] Running single iteration...")
        adapter.run_once()
        tracker.update_value(force=True)
        print_status(adapter, tracker)
        return 0

    # Main loop
    logger.info("[CSCM Paper] Starting main loop...")
    portfolio_update_interval = timedelta(minutes=15)
    last_portfolio_update = datetime.min

    while not _shutdown_requested:
        try:
            now = tz.now()

            # Check for rebalance
            if adapter._should_rebalance():
                logger.info("[CSCM Paper] Rebalance day - executing rebalance...")
                adapter.rebalance()
            else:
                # Run strategy check (trailing stop, etc.)
                adapter.run_once()

            # Update portfolio value during equity hours
            if tracker.is_equity_hours():
                if now - last_portfolio_update > portfolio_update_interval:
                    tracker.update_value()
                    last_portfolio_update = now
            else:
                logger.debug("[CSCM Paper] Outside equity hours")

            # Sleep until next check
            sleep_seconds = args.check_interval * 60
            logger.debug(f"[CSCM Paper] Sleeping for {args.check_interval} minutes...")

            # Use smaller sleep intervals to check for shutdown
            for _ in range(sleep_seconds):
                if _shutdown_requested:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("[CSCM Paper] Keyboard interrupt received")
            break

        except Exception as e:
            logger.error(f"[CSCM Paper] Error in main loop: {e}")
            # Wait before retrying
            time.sleep(60)

    logger.info("[CSCM Paper] Shutting down...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
