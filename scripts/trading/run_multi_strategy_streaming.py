"""
Multi-Strategy Live Trading Runner with Shared Streaming.

Runs OMR and RAMP strategies in a single process, sharing one LiveDataProvider
WebSocket connection. This allows both strategies to use streaming with the free
IEX feed (which allows only 1 connection per account).

Schedule:
- OMR: Entry 3:50 PM, Exit 9:31 AM (next day)
- RAMP: Rebalance 3:55 PM

Both strategies share the same LiveDataProvider instance, which maintains a single
WebSocket connection with ~515 symbols (15 OMR + 500 RAMP).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import time
import signal
from datetime import datetime, time as dt_time
from dotenv import load_dotenv

# Set process title
try:
    import setproctitle
    setproctitle.setproctitle("homeguard-multi")
except ImportError:
    pass

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import OMRLiveAdapter, RAMPLiveAdapter
from src.trading.config import load_omr_config
from src.streaming import LiveDataProvider
from src.utils.logger import logger
from src.utils.timezone import tz
from src.utils.trading_logger import cleanup_old_logs

# Import RAMP universe
import pandas as pd


class MultiStrategyRunner:
    """
    Runs multiple strategies in a single process with shared streaming.

    Schedules:
    - OMR: 3:50 PM (entry), 9:31 AM (exit)
    - RAMP: 3:55 PM (rebalance)
    """

    def __init__(
        self,
        omr_adapter: OMRLiveAdapter,
        ramp_adapter: RAMPLiveAdapter,
        check_interval: int = 15
    ):
        """
        Initialize multi-strategy runner.

        Args:
            omr_adapter: OMR strategy adapter
            ramp_adapter: RAMP strategy adapter
            check_interval: Seconds between schedule checks
        """
        self.omr_adapter = omr_adapter
        self.ramp_adapter = ramp_adapter
        self.check_interval = check_interval

        # Execution tracking
        self.omr_last_entry = None
        self.omr_last_exit = None
        self.ramp_last_execution = None

        self.running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Received shutdown signal")
        self.running = False

    def should_run_omr_entry(self) -> bool:
        """Check if OMR entry should run (3:50 PM ET)."""
        now = tz.now()
        current_time = now.time()

        # OMR entry window: 3:50-3:51 PM
        entry_time = dt_time(15, 50)
        entry_end = dt_time(15, 51)

        is_entry_time = entry_time <= current_time < entry_end

        if is_entry_time:
            # Check if already ran today
            if self.omr_last_entry is not None:
                if self.omr_last_entry.date() == now.date():
                    return False  # Already ran entry today
            return True

        return False

    def should_run_omr_exit(self) -> bool:
        """Check if OMR exit should run (9:31 AM ET next day)."""
        now = tz.now()
        current_time = now.time()

        # OMR exit window: 9:31-9:32 AM
        exit_time = dt_time(9, 31)
        exit_end = dt_time(9, 32)

        is_exit_time = exit_time <= current_time < exit_end

        if is_exit_time:
            # Check if already ran today
            if self.omr_last_exit is not None:
                if self.omr_last_exit.date() == now.date():
                    return False  # Already ran exit today
            return True

        return False

    def should_run_ramp(self) -> bool:
        """Check if RAMP should run (3:55 PM ET)."""
        now = tz.now()
        current_time = now.time()

        # RAMP execution window: 3:55-3:56 PM
        exec_time = dt_time(15, 55)
        exec_end = dt_time(15, 56)

        is_exec_time = exec_time <= current_time < exec_end

        if is_exec_time:
            # Check if already ran today
            if self.ramp_last_execution is not None:
                if self.ramp_last_execution.date() == now.date():
                    return False  # Already ran today
            return True

        return False

    def run_omr_entry(self):
        """Execute OMR entry signals (3:50 PM)."""
        logger.info("=" * 80)
        logger.info(f"[{tz.timestamp()}] OMR ENTRY EXECUTION")
        logger.info("=" * 80)

        try:
            self.omr_adapter.run_once()
            self.omr_last_entry = tz.now()
            logger.success("OMR entry execution complete")
        except Exception as e:
            logger.error(f"OMR entry execution failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("")

    def run_omr_exit(self):
        """Execute OMR exit signals (9:31 AM)."""
        logger.info("=" * 80)
        logger.info(f"[{tz.timestamp()}] OMR EXIT EXECUTION")
        logger.info("=" * 80)

        try:
            # OMR exit logic is handled automatically by the adapter's
            # check_exit_conditions() which runs on every execution
            # We just need to trigger run_once() which will check positions
            self.omr_adapter.run_once()
            self.omr_last_exit = tz.now()
            logger.success("OMR exit execution complete")
        except Exception as e:
            logger.error(f"OMR exit execution failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("")

    def run_ramp(self):
        """Execute RAMP rebalance (3:55 PM)."""
        logger.info("=" * 80)
        logger.info(f"[{tz.timestamp()}] RAMP EXECUTION")
        logger.info("=" * 80)

        try:
            self.ramp_adapter.run_once()
            self.ramp_last_execution = tz.now()
            logger.success("RAMP execution complete")
        except Exception as e:
            logger.error(f"RAMP execution failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("")

    def run_continuous(self):
        """Run strategies continuously with schedule checks."""
        logger.info("=" * 80)
        logger.info("MULTI-STRATEGY RUNNER STARTED")
        logger.info("=" * 80)
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info("Schedules:")
        logger.info("  OMR Entry:  3:50 PM ET")
        logger.info("  RAMP:       3:55 PM ET")
        logger.info("  OMR Exit:   9:31 AM ET (next day)")
        logger.info("=" * 80)
        logger.info("")

        while self.running:
            now = tz.now()

            # Check OMR entry schedule
            if self.should_run_omr_entry():
                self.run_omr_entry()

            # Check RAMP schedule
            if self.should_run_ramp():
                self.run_ramp()

            # Check OMR exit schedule
            if self.should_run_omr_exit():
                self.run_omr_exit()

            # Sleep until next check
            time.sleep(self.check_interval)

        logger.info("Multi-strategy runner stopped")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Clean up old logs
    try:
        cleanup_old_logs(log_dir="/home/ec2-user/logs", keep_days=30)
    except Exception as e:
        logger.warning(f"Failed to cleanup old logs (non-critical): {e}")

    # Check for API credentials
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca API credentials not found in environment variables")
        logger.info("Please set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY in .env file")
        return 1

    # Check streaming configuration
    use_streaming = os.getenv('USE_STREAMING', 'false').lower() in ('true', '1', 'yes')
    streaming_feed = os.getenv('STREAMING_FEED', 'iex')

    if not use_streaming:
        logger.error("Streaming must be enabled for multi-strategy mode")
        logger.info("Set USE_STREAMING=true in .env file")
        return 1

    logger.info("=" * 80)
    logger.info("MULTI-STRATEGY LIVE TRADING SETUP")
    logger.info("=" * 80)
    logger.info("Strategies: OMR + RAMP")
    logger.info("Data: Shared streaming (single WebSocket)")
    logger.info(f"Feed: {streaming_feed.upper()}")
    logger.info("=" * 80)

    try:
        # Initialize broker
        logger.info("Initializing Alpaca broker...")
        broker = AlpacaBroker(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )

        # Verify connection
        account = broker.get_account()
        if not account:
            logger.error("Failed to connect to Alpaca")
            return 1

        logger.success("Connected to Alpaca Paper Trading")
        logger.info(f"  Account: {account['account_id']}")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")

        # Check market status
        market_open = broker.is_market_open()
        logger.info(f"  Market: {'OPEN' if market_open else 'CLOSED'}")
        logger.info("")

        # Load strategy configurations
        logger.info("Loading strategy configurations...")

        # OMR configuration
        omr_config = load_omr_config()
        omr_symbols = omr_config.symbols
        logger.info(f"  OMR: {len(omr_symbols)} symbols")

        # RAMP configuration
        ramp_symbols = pd.read_csv('backtest_lists/sp500-2025.csv')['Symbol'].tolist()
        logger.info(f"  RAMP: {len(ramp_symbols)} symbols")

        # Combine all symbols for streaming
        all_symbols = list(set(omr_symbols + ramp_symbols + ['SPY']))
        logger.info(f"  Total unique: {len(all_symbols)} symbols")
        logger.info("")

        # Create shared LiveDataProvider
        logger.info("=" * 80)
        logger.info("CREATING SHARED STREAMING PROVIDER")
        logger.info("=" * 80)
        logger.info(f"Feed: {streaming_feed.upper()}")
        logger.info(f"Symbols: {len(all_symbols)}")
        logger.info("Starting WebSocket connection...")

        data_provider = LiveDataProvider(
            api_key=api_key,
            secret_key=secret_key,
            feed=streaming_feed,
            max_bars_per_symbol=500,
            fallback_enabled=True
        )

        # Start streaming for all symbols
        data_provider.start(all_symbols)

        logger.success(f"Streaming enabled: {len(all_symbols)} symbols")
        logger.info(f"Feed: {streaming_feed.upper()}")
        logger.info(f"Smart fallback: ENABLED (90% threshold)")
        logger.info("=" * 80)
        logger.info("")

        # Create OMR adapter
        logger.info("Creating OMR adapter...")
        omr_adapter = OMRLiveAdapter(
            broker=broker,
            data_provider=data_provider,  # Shared streaming provider
            **omr_config.to_adapter_params()  # Use config's adapter params
        )
        logger.success("OMR adapter created")

        # Create RAMP adapter
        logger.info("Creating RAMP adapter...")
        ramp_adapter = RAMPLiveAdapter(
            broker=broker,
            symbols=ramp_symbols,
            data_provider=data_provider  # Shared streaming provider
        )
        logger.success("RAMP adapter created")

        # Pre-load RAMP historical data at startup (uses disk cache if available)
        # This prevents the 5-minute delay at 3:55 PM execution time
        logger.info("")
        logger.info("=" * 60)
        logger.info("PRE-LOADING RAMP HISTORICAL DATA")
        logger.info("=" * 60)
        ramp_adapter.preload_historical_data()
        logger.info("")

        # Create multi-strategy runner
        runner = MultiStrategyRunner(
            omr_adapter=omr_adapter,
            ramp_adapter=ramp_adapter,
            check_interval=15  # Check every 15 seconds
        )

        # Run continuously
        runner.run_continuous()

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
