"""
Demo: MA Crossover Paper Trading.

Demonstrates running the MA Crossover strategy in Alpaca paper trading.
"""

import time
from datetime import datetime

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import MACrossoverLiveAdapter
from src.strategies.universe import EquityUniverse
from src.utils.logger import logger


def run_ma_paper_trading():
    """
    Run MA Crossover strategy in paper trading mode.

    Strategy Configuration:
    - 50/200 SMA golden cross
    - Run every 5 minutes during market hours
    - Trade FAANG stocks
    - 10% position size per trade
    - Max 5 concurrent positions
    """
    logger.info("=" * 80)
    logger.info("MA CROSSOVER PAPER TRADING DEMO")
    logger.info("=" * 80)

    # 1. Initialize broker (paper trading)
    logger.info("Initializing Alpaca broker (paper trading)...")
    broker = AlpacaBroker(mode='paper')

    # Check account
    account = broker.get_account()
    if account:
        logger.success(f"Connected to Alpaca Paper Trading")
        logger.info(f"  Account: {account.id}")
        logger.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    else:
        logger.error("Failed to connect to Alpaca")
        return

    # 2. Define universe (FAANG stocks)
    symbols = EquityUniverse.FAANG
    logger.info(f"Trading universe: {symbols}")

    # 3. Create MA Crossover adapter
    logger.info("Creating MA Crossover adapter...")
    adapter = MACrossoverLiveAdapter(
        broker=broker,
        symbols=symbols,
        fast_period=50,
        slow_period=200,
        ma_type='sma',
        min_confidence=0.7,
        position_size=0.10,  # 10% per position
        max_positions=5
    )

    logger.success("MA Crossover adapter created")
    logger.info("  Strategy: 50/200 SMA Golden Cross")
    logger.info("  Schedule: Every 5 minutes during market hours")
    logger.info("  Min confidence: 70%")

    # 4. Run strategy once (demo)
    logger.info("")
    logger.info("=" * 80)
    logger.info("RUNNING STRATEGY (ONE ITERATION)")
    logger.info("=" * 80)

    adapter.run_once()

    logger.info("")
    logger.info("=" * 80)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info("To run continuously, use the scheduler loop:")
    logger.info("")
    logger.info("  while True:")
    logger.info("      if adapter.should_run_now():")
    logger.info("          adapter.run_once()")
    logger.info("      time.sleep(60)  # Check every minute")


def run_ma_continuous():
    """
    Run MA Crossover strategy continuously (scheduled).

    Checks every minute if it's time to run the strategy.
    Runs every 5 minutes during market hours.
    """
    logger.info("=" * 80)
    logger.info("MA CROSSOVER CONTINUOUS PAPER TRADING")
    logger.info("=" * 80)

    # Initialize
    broker = AlpacaBroker(mode='paper')
    symbols = EquityUniverse.FAANG

    adapter = MACrossoverLiveAdapter(
        broker=broker,
        symbols=symbols,
        fast_period=50,
        slow_period=200,
        position_size=0.10,
        max_positions=5
    )

    logger.info("Starting continuous trading...")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    try:
        while True:
            # Check if should run now
            if adapter.should_run_now():
                logger.info(f"Running strategy at {datetime.now()}")
                adapter.run_once()
            else:
                # Not time to run yet
                now = datetime.now()
                market_open = broker.is_market_open()
                status = "Market OPEN" if market_open else "Market CLOSED"
                logger.info(f"{now.strftime('%H:%M:%S')} - {status} - Waiting...")

            # Check every minute
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Error in continuous trading: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Run continuously
        run_ma_continuous()
    else:
        # Run once (demo)
        run_ma_paper_trading()
