#!/usr/bin/env python3
"""
CSCM Demo Trading Service - Binance streaming with simulated execution.

Runs the Cross-Sectional Crypto Momentum strategy using the demo broker
with real-time Binance data and simulated execution.

Features:
- Real-time Binance WebSocket streaming
- Automatic trading decisions based on momentum
- BTC regime filter (go to cash in bear markets)
- Weekly rebalancing with trailing stop protection
- Self-contained paper trading (no external platform)

Usage:
    python scripts/trading/run_cscm_demo.py

    # With custom initial cash
    python scripts/trading/run_cscm_demo.py --cash 50000

    # Reset portfolio
    python scripts/trading/run_cscm_demo.py --reset

    # Market hours only mode
    python scripts/trading/run_cscm_demo.py --market-hours-only

Environment:
    Requires websocket-client: pip install websocket-client

Service Installation (Linux):
    sudo cp scripts/ec2/services/homeguard-cscm-demo.service /etc/systemd/system/
    sudo systemctl enable homeguard-cscm-demo
    sudo systemctl start homeguard-cscm-demo
"""

import argparse
import signal
import sys
import time

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rstrip("/\\"))

from src.trading.adapters.cscm_demo_adapter import CSCMDemoAdapter
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

# Demo universe - 16 symbols including MKR (not available on Alpaca)
DEMO_UNIVERSE = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "AVAX/USD",
    "LINK/USD",
    "DOGE/USD",
    "DOT/USD",
    "LTC/USD",
    "BCH/USD",
    "UNI/USD",
    "AAVE/USD",
    "SUSHI/USD",
    "XRP/USD",
    "CRV/USD",
    "GRT/USD",
    "MKR/USD",  # Available on Binance, not Alpaca
]


class CSCMDemoService:
    """
    CSCM Demo Trading Service.

    Runs the CSCM strategy with demo broker using Binance streaming.
    Designed for 24/7 operation with graceful shutdown support.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        slippage_bps: float = 5.0,
        fee_bps: float = 10.0,
        top_n: int = 7,
        rebalance_day: str = 'sunday',
        market_hours_only: bool = False,
    ):
        """
        Initialize demo service.

        Args:
            initial_cash: Starting cash (used if no saved state)
            slippage_bps: Slippage in basis points
            fee_bps: Fee in basis points
            top_n: Number of positions to hold
            rebalance_day: Day to rebalance
            market_hours_only: Restrict to equity market hours
        """
        # Create the adapter with strategy logic
        self.adapter = CSCMDemoAdapter(
            universe=DEMO_UNIVERSE,
            top_n=top_n,
            rebalance_day=rebalance_day,
            initial_cash=initial_cash,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            market_hours_only=market_hours_only,
        )

        self._running = False
        self._shutdown_requested = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"[CSCMDemo] Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def start(self) -> None:
        """Start the demo trading service."""
        logger.info("[CSCMDemo] Starting CSCM Demo Trading Service...")
        logger.info(f"[CSCMDemo] Universe: {len(DEMO_UNIVERSE)} symbols")
        logger.info(f"[CSCMDemo] Top N: {self.adapter.top_n}")
        logger.info(f"[CSCMDemo] Rebalance Day: {self.adapter.rebalance_day}")
        logger.info(f"[CSCMDemo] Market Hours Only: {self.adapter.market_hours_only}")

        # Start streaming
        self.adapter.start_streaming()
        self._running = True

        # Wait for initial data
        logger.info("[CSCMDemo] Waiting for initial market data...")
        self._wait_for_data(timeout=60)

        # Log initial state
        self._log_portfolio_status()

        # Main loop
        logger.info("[CSCMDemo] Entering main loop. Press Ctrl+C to stop.")
        self._main_loop()

    def _wait_for_data(self, timeout: int = 60) -> None:
        """Wait for initial market data from stream."""
        start = time.time()
        while time.time() - start < timeout:
            if self._shutdown_requested:
                return

            # Check how many symbols have data
            symbols_with_data = len(
                [s for s in DEMO_UNIVERSE if self.adapter.broker._bar_buffer.has_symbol(s)]
            )

            if symbols_with_data >= len(DEMO_UNIVERSE) * 0.8:  # 80% of symbols
                logger.info(
                    f"[CSCMDemo] Received data for {symbols_with_data}/{len(DEMO_UNIVERSE)} symbols"
                )
                return

            time.sleep(1)

        # Log warning if not enough data
        symbols_with_data = len(
            [s for s in DEMO_UNIVERSE if self.adapter.broker._bar_buffer.has_symbol(s)]
        )
        logger.warning(
            f"[CSCMDemo] Timeout waiting for data. "
            f"Only {symbols_with_data}/{len(DEMO_UNIVERSE)} symbols available."
        )

    def _main_loop(self) -> None:
        """Main service loop with trading decisions."""
        last_status_log = 0
        STATUS_LOG_INTERVAL = 300  # 5 minutes
        last_strategy_check = 0
        STRATEGY_CHECK_INTERVAL = 60  # 1 minute
        error_count = 0
        MAX_BACKOFF = 300  # 5 minute max backoff

        while not self._shutdown_requested:
            try:
                current_time = time.time()

                # Log status periodically
                if current_time - last_status_log >= STATUS_LOG_INTERVAL:
                    self._log_portfolio_status()
                    last_status_log = current_time

                # Strategy check
                if current_time - last_strategy_check >= STRATEGY_CHECK_INTERVAL:
                    if self.adapter._should_rebalance():
                        logger.info("[CSCMDemo] Rebalance day - executing...")
                        self.adapter.signals.reset_trailing_stop()
                        self.adapter.rebalance()
                    else:
                        self.adapter.run_once()
                    last_strategy_check = current_time
                    error_count = 0  # Reset on success

                time.sleep(1)

            except Exception as e:
                logger.error(f"[CSCMDemo] Error in main loop: {e}")
                error_count += 1
                backoff = min(30 * (2 ** error_count), MAX_BACKOFF)
                logger.info(f"[CSCMDemo] Backing off for {backoff} seconds...")
                time.sleep(backoff)

        # Graceful shutdown
        self._shutdown()

    def _log_portfolio_status(self) -> None:
        """Log current portfolio status."""
        try:
            summary = self.adapter.broker.get_account_summary()

            logger.info(
                f"[CSCMDemo] Portfolio Status:\n"
                f"  Total Value: ${summary['total_value']:.2f}\n"
                f"  Cash: ${summary['cash']:.2f}\n"
                f"  Positions: {len(summary['positions'])}\n"
                f"  Unrealized P&L: ${summary['unrealized_pnl']:.2f}\n"
                f"  Realized P&L: ${summary['realized_pnl']:.2f}\n"
                f"  Streaming: {summary['streaming']}\n"
                f"  Bars Buffered: {summary['bars_buffered']}"
            )

            # Log position details if any
            if summary["positions"]:
                logger.info("[CSCMDemo] Current Positions:")
                for symbol, pos in summary["positions"].items():
                    logger.info(
                        f"  {symbol}: {pos['quantity']:.6f} @ ${pos['avg_entry_price']:.2f} "
                        f"(P&L: ${pos['unrealized_pnl']:.2f} / {pos['unrealized_pnl_pct']:.1f}%)"
                    )

            # Log strategy status
            try:
                status = self.adapter.get_status()
                if 'regime' in status:
                    logger.info(
                        f"[CSCMDemo] Strategy Status:\n"
                        f"  Regime: {status.get('regime', 'unknown')}\n"
                        f"  Last Rebalance: {status.get('last_rebalance', 'never')}\n"
                        f"  Trailing Stop: {'TRIGGERED' if status.get('trailing_stop_triggered') else 'OK'}"
                    )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[CSCMDemo] Error logging status: {e}")

    def _shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("[CSCMDemo] Shutting down...")

        # Stop streaming
        self.adapter.stop_streaming()

        # Log final status
        logger.info("[CSCMDemo] Final portfolio status:")
        self._log_portfolio_status()

        # Portfolio state is auto-saved
        logger.info("[CSCMDemo] Portfolio state saved. Shutdown complete.")
        self._running = False

    def reset_portfolio(self, initial_cash: float = 10000.0) -> None:
        """Reset portfolio to initial state."""
        self.adapter.reset(initial_cash=initial_cash)
        logger.info(f"[CSCMDemo] Portfolio reset with ${initial_cash:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CSCM Demo Trading Service with Binance streaming"
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10000.0,
        help="Initial cash balance (default: 10000)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5)",
    )
    parser.add_argument(
        "--fees",
        type=float,
        default=10.0,
        help="Fees in basis points (default: 10)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=7,
        help="Number of positions to hold (default: 7)",
    )
    parser.add_argument(
        "--rebalance-day",
        type=str,
        default="sunday",
        choices=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        help="Day to rebalance (default: sunday)",
    )
    parser.add_argument(
        "--market-hours-only",
        action="store_true",
        help="Only trade during US equity market hours",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset portfolio before starting",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CSCM Demo Trading Service")
    logger.info(f"Started: {tz.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info("=" * 60)

    try:
        service = CSCMDemoService(
            initial_cash=args.cash,
            slippage_bps=args.slippage,
            fee_bps=args.fees,
            top_n=args.top_n,
            rebalance_day=args.rebalance_day,
            market_hours_only=args.market_hours_only,
        )

        if args.reset:
            service.reset_portfolio(initial_cash=args.cash)

        service.start()

    except KeyboardInterrupt:
        logger.info("[CSCMDemo] Interrupted by user")
    except Exception as e:
        logger.error(f"[CSCMDemo] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
