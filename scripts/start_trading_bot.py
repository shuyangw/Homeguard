#!/usr/bin/env python3
"""
Start the live trading bot with production logging.

Usage:
    python scripts/start_trading_bot.py

This script:
1. Sets up production logging with rotation
2. Initializes the trading bot
3. Trains the strategy
4. Starts continuous trading

Press Ctrl+C to stop gracefully.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from trading.core.paper_trading_bot import PaperTradingBot
from utils.trading_logger import setup_trading_logs

# Configuration paths
BROKER_CONFIG = project_root / "config/trading/broker_alpaca.yaml"
STRATEGY_CONFIG = project_root / "config/trading/omr_trading_config.yaml"


def main():
    """Main entry point for trading bot."""

    # ===== SETUP LOGGING =====
    # This creates rotating logs in /home/ec2-user/logs (or current user's home)
    # Adjust log_dir if you want a different location
    import os
    log_dir = Path.home() / "logs"

    # Use DEBUG for troubleshooting, INFO for production
    logger, exec_logger = setup_trading_logs(
        log_dir=str(log_dir),
        log_level="INFO"  # Change to "DEBUG" for verbose output
    )

    logger.info("=" * 80)
    logger.info("STARTING HOMEGUARD TRADING BOT")
    logger.info("=" * 80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Broker config: {BROKER_CONFIG}")
    logger.info(f"Strategy config: {STRATEGY_CONFIG}")

    try:
        # ===== INITIALIZE BOT =====
        logger.info("\nInitializing trading bot...")
        bot = PaperTradingBot(
            broker_config_path=str(BROKER_CONFIG),
            strategy_config_path=str(STRATEGY_CONFIG)
        )

        bot.initialize()

        # ===== TRAIN STRATEGY =====
        # Note: In production, you might want to train offline and load models
        # For now, we'll skip training or implement fetching historical data
        logger.warning("\nStrategy training not implemented in this script")
        logger.warning("Bot will attempt to trade but may fail without trained models")
        logger.warning("Implement bot.train_strategy(historical_data) before production use")

        # If you have historical data ready:
        # logger.info("\nTraining strategy...")
        # historical_data = fetch_historical_data()  # Your implementation
        # bot.train_strategy(historical_data)

        # ===== START TRADING =====
        logger.info("\nStarting continuous trading...")
        logger.info("Press Ctrl+C to stop\n")

        # Run continuously (checks signals every 10 seconds)
        bot.start_trading(run_once=False)

    except KeyboardInterrupt:
        logger.info("\n\nReceived stop signal (Ctrl+C)")
        logger.info("Shutting down gracefully...")

    except FileNotFoundError as e:
        logger.error(f"\n\nConfiguration file not found: {e}")
        logger.error("Make sure broker_alpaca.yaml and omr_trading_config.yaml exist")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("\n" + "=" * 80)
        logger.info("Trading bot stopped")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
