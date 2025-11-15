"""
Demo: OMR (Overnight Mean Reversion) Paper Trading.

Demonstrates running the OMR strategy in Alpaca paper trading.
"""

import time
from datetime import datetime, time as dt_time

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import OMRLiveAdapter
from src.trading.config import load_omr_config
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

    # 2. Load production config (AUTHORITATIVE)
    logger.info("Loading production OMR configuration...")
    omr_config = load_omr_config()
    symbols = omr_config.symbols
    logger.info(f"Trading universe: {len(symbols)} validated ETFs (from production config)")
    logger.info(f"  First 3: {symbols[:3]}...")
    logger.info(f"  Last 3: {symbols[-3:]}...")

    # 3. Initialize models (optional - can provide pre-trained models)
    logger.info("Initializing regime detector and Bayesian model...")
    logger.info("  Note: Using untrained models for demo")
    logger.info("  In production, load pre-trained models from disk")

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()

    # 4. Create OMR adapter from production config
    logger.info("Creating OMR adapter from production config...")
    adapter_params = omr_config.to_adapter_params()
    adapter = OMRLiveAdapter(
        broker=broker,
        regime_detector=regime_detector,
        bayesian_model=bayesian_model,
        **adapter_params  # Use config parameters
    )

    logger.success("OMR adapter created from production config")
    logger.info("  Strategy: Overnight Mean Reversion")
    logger.info(f"  Signal time: {omr_config.entry_time}")
    logger.info(f"  Entry: {omr_config.entry_time} | Exit: {omr_config.exit_time}")
    logger.info(f"  Min probability: {omr_config.min_win_rate:.1%}")
    logger.info(f"  Min expected return: {omr_config.min_expected_return:.2%}")
    logger.info(f"  Position size: {omr_config.position_size_pct:.1%}")
    logger.info(f"  Max positions: {omr_config.max_concurrent_positions}")

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

    # Initialize with production config
    broker = AlpacaBroker(mode='paper')

    # Load production config (AUTHORITATIVE)
    logger.info("Loading production OMR configuration...")
    omr_config = load_omr_config()

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()

    adapter_params = omr_config.to_adapter_params()
    adapter = OMRLiveAdapter(
        broker=broker,
        regime_detector=regime_detector,
        bayesian_model=bayesian_model,
        **adapter_params  # Use config parameters
    )

    logger.info("Starting continuous OMR trading...")
    logger.info(f"  Symbols: {len(omr_config.symbols)} ETFs (production config)")
    logger.info(f"  Signal time: {omr_config.entry_time}")
    logger.info(f"  Exit time: {omr_config.exit_time} (next day)")
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
