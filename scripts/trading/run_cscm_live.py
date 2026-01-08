"""
CSCM (Cross-Sectional Crypto Momentum) Live Trading Runner.

Runs the CSCM strategy continuously with Coinbase (primary) or Alpaca (secondary).
Features:
- 24/7 crypto trading (no market hours)
- Weekly rebalancing (Sunday 0:00 UTC)
- BTC regime filter
- Automatic broker failover

Usage:
    python scripts/trading/run_cscm_live.py
    python scripts/trading/run_cscm_live.py --paper
    python scripts/trading/run_cscm_live.py --broker alpaca
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import argparse
from dotenv import load_dotenv

# Set process title for easy identification in ps/htop
try:
    import setproctitle
    setproctitle.setproctitle("homeguard-cscm")
except ImportError:
    pass

from src.trading.adapters.cscm_live_adapter import CSCMLiveAdapter
from src.utils.logger import logger


def load_config(config_path: str = None) -> dict:
    """Load CSCM configuration from YAML file."""
    import yaml

    if config_path is None:
        config_path = project_root / 'config' / 'trading' / 'cscm_live.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run CSCM live crypto trading')

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config/trading/cscm_live.yaml)'
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Use paper trading (Alpaca paper mode)'
    )

    parser.add_argument(
        '--broker',
        type=str,
        default='auto',
        choices=['auto', 'coinbase', 'alpaca'],
        help='Broker to use (default: auto with failover)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=1,
        help='Hours between rebalance checks (default: 1)'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (for testing)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current strategy status and exit'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    config = load_config(args.config)

    logger.info("=" * 80)
    logger.info("CSCM LIVE CRYPTO TRADING")
    logger.info("=" * 80)
    logger.info(f"Paper mode: {args.paper}")
    logger.info(f"Broker: {args.broker}")
    logger.info(f"Check interval: {args.check_interval} hours")
    logger.info("=" * 80)

    try:
        # Extract parameters from config
        universe = config.get('universe', CSCMLiveAdapter.DEFAULT_UNIVERSE)
        top_n = config.get('top_n', CSCMLiveAdapter.DEFAULT_TOP_N)
        momentum_period = config.get('momentum_period', CSCMLiveAdapter.DEFAULT_MOMENTUM_PERIOD)
        btc_sma_period = config.get('btc_sma_period', CSCMLiveAdapter.DEFAULT_BTC_SMA_PERIOD)
        rebalance_day = config.get('rebalance_day', 'sunday')
        go_to_cash_in_bear = config.get('go_to_cash_in_bear', True)

        # Setup broker based on selection
        broker = None
        if args.broker != 'auto':
            if args.broker == 'coinbase':
                from src.trading.brokers.coinbase_broker import CoinbaseBroker
                broker_instance = CoinbaseBroker()
            else:  # alpaca
                from src.trading.brokers.alpaca_crypto_broker import AlpacaCryptoBroker
                broker_instance = AlpacaCryptoBroker(paper=args.paper)

            # Wrap in router with single broker
            from src.trading.brokers.crypto_router import CryptoBrokerRouter
            broker = CryptoBrokerRouter(
                primary=broker_instance if args.broker == 'coinbase' else None,
                secondary=broker_instance if args.broker == 'alpaca' else None,
                auto_failover=False
            )

        # Create adapter
        adapter = CSCMLiveAdapter(
            universe=universe,
            top_n=top_n,
            momentum_period=momentum_period,
            btc_sma_period=btc_sma_period,
            rebalance_day=rebalance_day,
            go_to_cash_in_bear=go_to_cash_in_bear,
            broker=broker,
            paper=args.paper
        )

        logger.info(f"Universe: {len(universe)} symbols")
        logger.info(f"Top N: {top_n}")
        logger.info(f"Momentum Period: {momentum_period} days")
        logger.info(f"BTC SMA Period: {btc_sma_period} days")
        logger.info(f"Rebalance Day: {rebalance_day}")
        logger.info(f"Go to Cash in Bear: {go_to_cash_in_bear}")
        logger.info("")

        if args.status:
            # Show status and exit
            status = adapter.get_status()
            logger.info("Current Strategy Status:")
            logger.info("-" * 40)
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
            return 0

        if args.once:
            # Run once and exit
            logger.info("Running single iteration...")
            adapter.run_once()
            logger.success("Done")
        else:
            # Run continuous loop
            logger.info("Starting continuous trading loop...")
            adapter.run(check_interval_hours=args.check_interval)

        return 0

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
