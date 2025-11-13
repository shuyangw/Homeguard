"""
Demo: OMR (Overnight Mean Reversion) Paper Trading.

Demonstrates running the OMR strategy in Alpaca paper trading.
"""

import time
from datetime import datetime, time as dt_time

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import OMRLiveAdapter
from src.strategies.universe import ETFUniverse
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.utils.logger import logger


def run_omr_paper_trading():
    """
    Run OMR strategy in paper trading mode.

    Strategy Configuration:
    - Overnight mean reversion on leveraged 3x ETFs
    - Signal time: 3:50 PM EST
    - Entry: 3:50 PM | Exit: Next day 9:31 AM
    - Max 5 concurrent positions
    - 10% position size per trade
    """
    logger.info("=" * 80)
    logger.info("OMR (OVERNIGHT MEAN REVERSION) PAPER TRADING DEMO")
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

    # 2. Define universe (leveraged 3x ETFs)
    symbols = ETFUniverse.LEVERAGED_3X
    logger.info(f"Trading universe: {len(symbols)} leveraged 3x ETFs")
    logger.info(f"  Bull ETFs: {ETFUniverse.LEVERAGED_3X[:3]}...")
    logger.info(f"  Bear ETFs: {ETFUniverse.LEVERAGED_3X[1::2][:3]}...")

    # 3. Initialize models (optional - can provide pre-trained models)
    logger.info("Initializing regime detector and Bayesian model...")
    logger.info("  Note: Using untrained models for demo")
    logger.info("  In production, load pre-trained models from disk")

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()

    # 4. Create OMR adapter
    logger.info("Creating OMR adapter...")
    adapter = OMRLiveAdapter(
        broker=broker,
        symbols=symbols,
        min_probability=0.55,
        min_expected_return=0.002,
        max_positions=5,
        position_size=0.10,
        regime_detector=regime_detector,
        bayesian_model=bayesian_model
    )

    logger.success("OMR adapter created")
    logger.info("  Strategy: Overnight Mean Reversion")
    logger.info("  Signal time: 3:50 PM EST")
    logger.info("  Entry: 3:50 PM | Exit: Next day 9:31 AM")
    logger.info("  Min probability: 55%")
    logger.info("  Min expected return: 0.2%")

    # 5. Run strategy once (demo)
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
    logger.info("      # Generate signals at 3:50 PM")
    logger.info("      if adapter.should_run_now():")
    logger.info("          adapter.run_once()")
    logger.info("")
    logger.info("      # Close positions at 9:31 AM")
    logger.info("      now = datetime.now()")
    logger.info("      if now.time() >= time(9, 30) and now.time() <= time(9, 35):")
    logger.info("          adapter.close_overnight_positions()")
    logger.info("")
    logger.info("      time.sleep(60)  # Check every minute")


def run_omr_continuous():
    """
    Run OMR strategy continuously (scheduled).

    Generates signals at 3:50 PM EST.
    Closes positions at 9:31 AM EST.
    """
    logger.info("=" * 80)
    logger.info("OMR CONTINUOUS PAPER TRADING")
    logger.info("=" * 80)

    # Initialize
    broker = AlpacaBroker(mode='paper')
    symbols = ETFUniverse.LEVERAGED_3X

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()

    adapter = OMRLiveAdapter(
        broker=broker,
        symbols=symbols,
        min_probability=0.55,
        min_expected_return=0.002,
        max_positions=5,
        position_size=0.10,
        regime_detector=regime_detector,
        bayesian_model=bayesian_model
    )

    logger.info("Starting continuous OMR trading...")
    logger.info("  Signal time: 3:50 PM EST")
    logger.info("  Exit time: 9:31 AM EST (next day)")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    # Track if we've closed positions today
    positions_closed_today = False
    last_signal_date = None

    try:
        while True:
            now = datetime.now()
            current_time = now.time()
            market_open = broker.is_market_open()

            # Reset position closing flag at midnight
            if now.date() != (last_signal_date if last_signal_date else now.date()):
                positions_closed_today = False

            # Close overnight positions at 9:31 AM
            if (current_time >= dt_time(9, 30) and
                current_time <= dt_time(9, 35) and
                market_open and
                not positions_closed_today):

                logger.info(f"Time to close overnight positions: {now.strftime('%H:%M:%S')}")
                adapter.close_overnight_positions()
                positions_closed_today = True

            # Generate signals at 3:50 PM
            elif adapter.should_run_now():
                logger.info(f"Time to generate signals: {now.strftime('%H:%M:%S')}")
                adapter.run_once()
                last_signal_date = now.date()

            else:
                # Not time to run yet
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
        run_omr_continuous()
    else:
        # Run once (demo)
        run_omr_paper_trading()
