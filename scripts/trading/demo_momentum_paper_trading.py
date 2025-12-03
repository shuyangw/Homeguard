#!/usr/bin/env python
"""
Demo script for Momentum Protection paper trading.

Shows how to use the MomentumLiveAdapter for paper trading with Alpaca.

Usage:
    python scripts/trading/demo_momentum_paper_trading.py [--show-signals] [--run-once]

Options:
    --show-signals   Show current momentum signals and risk status
    --run-once       Run one rebalance cycle and exit
    --dry-run        Don't submit orders, just show what would happen
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import os

from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.trading.adapters.momentum_live_adapter import MomentumLiveAdapter
from src.utils.logger import logger
from src.utils.timezone import tz

# Load environment variables
load_dotenv()


def load_config():
    """Load momentum trading configuration."""
    import yaml

    config_path = PROJECT_ROOT / 'config' / 'trading' / 'momentum_trading_config.yaml'

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        return None


def main():
    parser = argparse.ArgumentParser(description='Momentum Protection Paper Trading Demo')
    parser.add_argument('--show-signals', action='store_true', help='Show current signals')
    parser.add_argument('--run-once', action='store_true', help='Run one rebalance cycle')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without orders')
    parser.add_argument('--top-n', type=int, default=10, help='Number of stocks to hold')
    parser.add_argument('--universe', choices=['sp500', 'top100', 'top250'], default='sp500',
                       help='Stock universe to trade')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MOMENTUM PROTECTION PAPER TRADING DEMO")
    logger.info("=" * 70)
    logger.info(f"Time: {tz.now()}")
    logger.info("")

    # Load configuration
    config = load_config()

    # Initialize broker (paper trading mode)
    logger.info("Initializing Alpaca broker (PAPER mode)...")
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca API keys not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        return

    broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)

    # Get account info
    account = broker.get_account()
    logger.info(f"Account status: {account.get('status', 'unknown')}")
    logger.info(f"Portfolio value: ${float(account.get('portfolio_value', 0)):,.2f}")
    logger.info(f"Buying power: ${float(account.get('buying_power', 0)):,.2f}")

    # Load symbols based on universe
    if args.universe == 'top100':
        symbols_path = PROJECT_ROOT / 'backtest_lists' / 'sp500_top100-2025.csv'
    elif args.universe == 'top250':
        symbols_path = PROJECT_ROOT / 'backtest_lists' / 'sp500_top250-2025.csv'
    else:
        symbols_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'

    import pandas as pd
    symbols_df = pd.read_csv(symbols_path)
    symbols = symbols_df['Symbol'].tolist()
    logger.info(f"Loaded {len(symbols)} symbols from {symbols_path.name}")

    # Get strategy parameters from config or use defaults
    strategy_config = config.get('strategy', {}) if config else {}

    # Initialize adapter
    logger.info("Initializing Momentum Live Adapter...")
    adapter = MomentumLiveAdapter(
        broker=broker,
        symbols=symbols,
        top_n=args.top_n,
        position_size=strategy_config.get('position_size_pct', 0.10),
        reduced_exposure=strategy_config.get('reduced_exposure', 0.50),
        vix_threshold=strategy_config.get('vix_threshold', 25.0),
        vix_spike_threshold=strategy_config.get('vix_spike_threshold', 0.20),
        spy_dd_threshold=strategy_config.get('spy_dd_threshold', -0.05),
        mom_vol_percentile=strategy_config.get('mom_vol_percentile', 0.90),
        slippage_per_share=strategy_config.get('slippage_per_share', 0.01)
    )

    # Pre-load historical data
    logger.info("\nPre-loading historical data...")
    adapter.preload_historical_data()

    if args.show_signals:
        # Show current signals
        logger.info("\n")
        adapter.show_current_signals()
        return

    if args.run_once:
        if args.dry_run:
            logger.warning("DRY RUN MODE - No orders will be submitted")
            # Just show signals without executing
            adapter.show_current_signals()
        else:
            # Run one rebalance cycle
            logger.info("\nRunning rebalance cycle...")
            adapter.run_once()

        # Show current positions
        positions = broker.get_positions()
        if positions:
            logger.info("\nCurrent Positions:")
            for pos in positions:
                symbol = pos.get('symbol')
                qty = pos.get('quantity')
                value = float(pos.get('market_value', 0))
                pnl = float(pos.get('unrealized_pl', 0))
                logger.info(f"  {symbol}: {qty} shares, ${value:,.2f} (P&L: ${pnl:+,.2f})")
        else:
            logger.info("\nNo current positions")

        return

    # Interactive mode
    logger.info("\nMomentum Protection Strategy Ready")
    logger.info("=" * 70)
    logger.info("Schedule:")
    schedule = adapter.get_schedule()
    for exec_time in schedule.get('execution_times', []):
        logger.info(f"  {exec_time['time']} EST - {exec_time['action'].upper()}")
    logger.info("")
    logger.info("Commands:")
    logger.info("  --show-signals   Show current signals")
    logger.info("  --run-once       Execute one rebalance")
    logger.info("  --dry-run        Preview without trading")
    logger.info("")
    logger.info("To run continuously, integrate with scheduler or cron job")


if __name__ == "__main__":
    main()
