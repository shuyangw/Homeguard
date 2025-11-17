"""
Live Paper Trading Runner with Comprehensive Logging.

Runs a strategy continuously with Alpaca paper trading.
Features:
- File and console logging
- Minute-by-minute progress updates
- End-of-day summary reports
- Trading session tracking
- Performance metrics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import time
import signal
import argparse
import json
from datetime import datetime, time as dt_time
from typing import Optional, Dict, List
from dotenv import load_dotenv
import pytz

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import (
    MACrossoverLiveAdapter,
    TripleMACrossoverLiveAdapter,
    OMRLiveAdapter
)
from src.trading.config import load_omr_config
from src.strategies.universe import EquityUniverse, ETFUniverse
from src.utils.logger import logger, get_trading_logger, TradingLogger


class TradingSessionTracker:
    """Tracks trading session metrics and generates reports."""

    def __init__(self, log_dir: Path, strategy_name: str):
        """
        Initialize session tracker.

        Args:
            log_dir: Directory for log files
            strategy_name: Name of the strategy
        """
        self.log_dir = log_dir
        self.strategy_name = strategy_name
        eastern = pytz.timezone('US/Eastern')
        self.session_start = datetime.now(pytz.UTC).astimezone(eastern)
        self.session_date = self.session_start.strftime('%Y%m%d')

        # Session metrics
        self.total_checks = 0
        self.total_runs = 0
        self.total_signals = 0
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.orders_log: List[Dict] = []
        self.signals_log: List[Dict] = []
        self.minute_progress: List[Dict] = []

        # Create session log files
        self.session_log_file = log_dir / f"{self.session_date}_{strategy_name}_session.json"
        self.summary_file = log_dir / f"{self.session_date}_{strategy_name}_summary.md"
        self.trades_log_file = log_dir / f"{self.session_date}_{strategy_name}_trades.csv"
        self.market_checks_log_file = log_dir / f"{self.session_date}_{strategy_name}_market_checks.csv"

        # Create trading logger with CSV logging
        self.trading_logger = get_trading_logger(strategy_name, log_dir)

        # Add CSV loggers for trades and market checks
        self.trading_logger.add_csv_logger(
            'trades',
            self.trades_log_file,
            ['timestamp', 'symbol', 'side', 'qty', 'price',
             'order_type', 'status', 'order_id', 'error']
        )

        self.trading_logger.add_csv_logger(
            'market_checks',
            self.market_checks_log_file,
            ['timestamp', 'market_open', 'check_number']
        )

    def log_check(self, market_open: bool):
        """Log a schedule check to memory and CSV file."""
        self.total_checks += 1
        timestamp = datetime.now().isoformat()

        self.minute_progress.append({
            'timestamp': timestamp,
            'type': 'check',
            'market_open': market_open,
            'total_checks': self.total_checks
        })

        # Log to CSV using trading logger
        self.trading_logger.log_market_check(market_open, self.total_checks)

    def log_run(self, signals_count: int):
        """Log a strategy run."""
        self.total_runs += 1
        self.total_signals += signals_count
        self.minute_progress.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'run',
            'signals': signals_count,
            'total_runs': self.total_runs
        })

    def log_signal(self, symbol: str, direction: str, price: float, confidence: float):
        """Log a trading signal."""
        self.signals_log.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'price': price,
            'confidence': confidence
        })

    def log_order(self, symbol: str, side: str, qty: int, price: float, success: bool, order_id: str = None, error: str = None, order_type: str = 'market'):
        """Log an order attempt to memory and CSV file."""
        self.total_orders += 1
        if success:
            self.successful_orders += 1
        else:
            self.failed_orders += 1

        timestamp = datetime.now().isoformat()

        self.orders_log.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'success': success,
            'order_id': order_id,
            'error': error
        })

        self.minute_progress.append({
            'timestamp': timestamp,
            'type': 'order',
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'success': success
        })

        # Log to CSV and console using trading logger
        self.trading_logger.log_trade(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            status='SUCCESS' if success else 'FAILED',
            order_type=order_type,
            order_id=order_id,
            error=error
        )

    def save_progress(self):
        """Save current progress to file."""
        progress_data = {
            'session_start': self.session_start.isoformat(),
            'strategy_name': self.strategy_name,
            'total_checks': self.total_checks,
            'total_runs': self.total_runs,
            'total_signals': self.total_signals,
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'orders': self.orders_log,
            'signals': self.signals_log,
            'minute_progress': self.minute_progress[-100:]  # Last 100 entries
        }

        with open(self.session_log_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def generate_end_of_day_report(self, broker: AlpacaBroker):
        """Generate end-of-day summary report."""
        eastern = pytz.timezone('US/Eastern')
        session_end = datetime.now(pytz.UTC).astimezone(eastern)
        session_duration = (session_end - self.session_start).total_seconds() / 3600  # Hours

        # Get account metrics
        account = broker.get_account()
        positions = broker.get_positions()

        # Generate markdown report
        report = []
        report.append("# Live Paper Trading Session Summary")
        report.append("")
        report.append(f"**Date:** {self.session_date}")
        report.append(f"**Strategy:** {self.strategy_name}")
        report.append(f"**Session Duration:** {session_duration:.2f} hours")
        report.append(f"**Session Start:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Session End:** {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Activity Summary
        report.append("## Activity Summary")
        report.append("")
        report.append(f"- **Total Schedule Checks:** {self.total_checks}")
        report.append(f"- **Strategy Executions:** {self.total_runs}")
        report.append(f"- **Signals Generated:** {self.total_signals}")
        report.append(f"- **Orders Placed:** {self.total_orders}")
        report.append(f"  - Successful: {self.successful_orders}")
        report.append(f"  - Failed: {self.failed_orders}")
        report.append(f"  - Success Rate: {(self.successful_orders / self.total_orders * 100) if self.total_orders > 0 else 0:.1f}%")
        report.append("")

        # Account Status
        report.append("## Account Status")
        report.append("")
        if account:
            report.append(f"- **Portfolio Value:** ${account['portfolio_value']:,.2f}")
            report.append(f"- **Buying Power:** ${account['buying_power']:,.2f}")
            report.append(f"- **Cash:** ${account['cash']:,.2f}")
            report.append("")

        # Current Positions
        report.append("## Current Positions")
        report.append("")
        if positions:
            report.append(f"**Total Positions:** {len(positions)}")
            report.append("")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                report.append(f"### {pos.symbol}")
                report.append(f"- **Quantity:** {pos.qty}")
                report.append(f"- **Entry Price:** ${float(pos.avg_entry_price):.2f}")
                report.append(f"- **Current Price:** ${float(pos.current_price):.2f}")
                report.append(f"- **Unrealized P&L:** ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                report.append(f"- **Market Value:** ${float(pos.market_value):,.2f}")
                report.append("")
        else:
            report.append("No open positions")
            report.append("")

        # Orders Summary
        if self.orders_log:
            report.append("## Orders Placed")
            report.append("")
            report.append("| Time | Symbol | Side | Qty | Price | Status |")
            report.append("|------|--------|------|-----|-------|--------|")
            for order in self.orders_log:
                timestamp = datetime.fromisoformat(order['timestamp']).strftime('%H:%M:%S')
                status = "SUCCESS" if order['success'] else "FAILED"
                report.append(f"| {timestamp} | {order['symbol']} | {order['side']} | {order['qty']} | ${order['price']:.2f} | {status} |")
            report.append("")

        # Signals Summary
        if self.signals_log:
            report.append("## Signals Generated")
            report.append("")
            report.append("| Time | Symbol | Direction | Price | Confidence |")
            report.append("|------|--------|-----------|-------|------------|")
            for signal in self.signals_log:
                timestamp = datetime.fromisoformat(signal['timestamp']).strftime('%H:%M:%S')
                report.append(f"| {timestamp} | {signal['symbol']} | {signal['direction']} | ${signal['price']:.2f} | {signal['confidence']:.1%} |")
            report.append("")

        # Performance Metrics
        report.append("## Performance Metrics")
        report.append("")
        report.append(f"- **Average Signals per Run:** {(self.total_signals / self.total_runs) if self.total_runs > 0 else 0:.2f}")
        report.append(f"- **Checks per Hour:** {(self.total_checks / session_duration) if session_duration > 0 else 0:.2f}")
        report.append(f"- **Runs per Hour:** {(self.total_runs / session_duration) if session_duration > 0 else 0:.2f}")
        report.append("")

        # Footer
        report.append("---")
        report.append("")
        report.append("*Generated by Homeguard Live Paper Trading System*")
        report.append("")

        # Write report to file
        with open(self.summary_file, 'w') as f:
            f.write('\n'.join(report))

        logger.success(f"End-of-day report saved: {self.summary_file}")
        return self.summary_file


class LiveTradingRunner:
    """Manages continuous live paper trading execution with logging."""

    def __init__(self, adapter, check_interval: int = 15, log_dir: Path = None, enable_intraday_prefetch: bool = True):
        """
        Initialize live trading runner.

        Args:
            adapter: Strategy adapter to run
            check_interval: Seconds between schedule checks (default: 15)
            log_dir: Directory for log files (default: logs/live_trading)
            enable_intraday_prefetch: Enable 3:45 PM intraday data pre-fetching (default: True)
                                     If False, fetches all data at execution time (3:50 PM)
        """
        self.adapter = adapter
        self.check_interval = check_interval
        self.running = False
        self.last_run_time: Optional[datetime] = None
        self.last_exit_time: Optional[datetime] = None  # Track last exit execution
        self.last_progress_log: Optional[datetime] = None
        self.data_preloaded_today: bool = False  # Track if data pre-loaded today
        self.intraday_prefetched_today: bool = False  # Track if intraday data pre-fetched today
        self.enable_intraday_prefetch: bool = enable_intraday_prefetch  # Toggle for intraday pre-fetching

        # Setup logging directory
        if log_dir is None:
            # Use configured output directory from settings.ini
            from src.config import get_live_trading_dir
            log_dir = get_live_trading_dir(mode='paper')

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Setup session tracker
        strategy_name = adapter.__class__.__name__.replace('LiveAdapter', '')
        self.session_tracker = TradingSessionTracker(log_dir, strategy_name)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Note: File logging is now handled by TradingLogger in TradingSessionTracker
        logger.info(f"File logging enabled: {log_dir}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nReceived shutdown signal, stopping...")
        self.running = False

    def should_run_now(self) -> Optional[str]:
        """
        Check if strategy should run now based on its schedule.

        Returns:
            'entry' if should run entry logic
            'exit' if should run exit logic
            None if should not run
        """
        schedule = self.adapter.get_schedule()

        # Check market hours if required
        if schedule.get('market_hours_only', True):
            if not self.adapter.broker.is_market_open():
                return None

        # Check for execution_times (new format for overnight strategies)
        execution_times = schedule.get('execution_times')
        if execution_times:
            # Convert current UTC time to EST for comparison with schedule times
            eastern = pytz.timezone('US/Eastern')
            now_utc = datetime.now(pytz.UTC)
            now_est = now_utc.astimezone(eastern)
            current_time = now_est.time()

            for exec_config in execution_times:
                target_time = datetime.strptime(exec_config['time'], '%H:%M').time()
                action = exec_config['action']

                # Run if within 1 minute of target time
                time_diff = abs((datetime.combine(now_est.date(), current_time) -
                               datetime.combine(now_est.date(), target_time)).total_seconds())

                if time_diff < 60:
                    # Check if we already ran this action within last minute
                    if action == 'entry' and self.last_run_time:
                        seconds_since_last = (now_utc.replace(tzinfo=None) - self.last_run_time).total_seconds()
                        if seconds_since_last < 60:
                            continue
                    elif action == 'exit' and self.last_exit_time:
                        seconds_since_last = (now_utc.replace(tzinfo=None) - self.last_exit_time).total_seconds()
                        if seconds_since_last < 60:
                            continue

                    return action

            return None

        # Check specific time (legacy single-time format)
        specific_time = schedule.get('specific_time')
        if specific_time:
            # Convert current UTC time to EST for comparison with schedule times
            eastern = pytz.timezone('US/Eastern')
            now_utc = datetime.now(pytz.UTC)
            now_est = now_utc.astimezone(eastern)
            target_time = datetime.strptime(specific_time, '%H:%M').time()
            current_time = now_est.time()

            # Run if within 1 minute of target time
            time_diff = abs((datetime.combine(now_est.date(), current_time) -
                           datetime.combine(now_est.date(), target_time)).total_seconds())

            if time_diff < 60:
                # Check if we already ran within last minute
                if self.last_run_time:
                    seconds_since_last = (now_utc.replace(tzinfo=None) - self.last_run_time).total_seconds()
                    if seconds_since_last < 60:
                        return None
                return 'entry'  # Default to entry action
            return None

        # For interval-based strategies, check minimum time between runs
        interval = schedule.get('interval', '5min')
        if self.last_run_time:
            now = datetime.now()
            seconds_since_last = (now - self.last_run_time).total_seconds()

            # Parse interval
            if 'min' in interval:
                min_interval = int(interval.replace('min', '')) * 60
            elif 'h' in interval or 'hour' in interval:
                min_interval = int(interval.replace('h', '').replace('hour', '')) * 3600
            elif 'd' in interval or 'day' in interval:
                min_interval = int(interval.replace('d', '').replace('day', '')) * 86400
            else:
                min_interval = 300  # Default 5 minutes

            if seconds_since_last < min_interval:
                return None

        return 'entry'  # Default action for interval-based strategies

    def _log_minute_progress(self, force: bool = False):
        """Log minute-by-minute progress with comprehensive market status."""
        now = datetime.now()

        # Log every 15 seconds or if forced
        log_interval = self.check_interval if self.check_interval >= 15 else 15
        if force or self.last_progress_log is None or (now - self.last_progress_log).total_seconds() >= log_interval:
            # Query market status with error handling
            try:
                market_open = self.adapter.broker.is_market_open()
                market_status_str = 'OPEN' if market_open else 'CLOSED'

                # Log market check to session tracker
                self.session_tracker.log_check(market_open)

                # Convert to EST for display
                eastern = pytz.timezone('US/Eastern')
                now_est = datetime.now(pytz.UTC).astimezone(eastern)

                # Log comprehensive status
                logger.info(
                    f"[{now_est.strftime('%H:%M:%S')}] "
                    f"Market: {market_status_str} | "
                    f"Checks: {self.session_tracker.total_checks} | "
                    f"Runs: {self.session_tracker.total_runs} | "
                    f"Signals: {self.session_tracker.total_signals} | "
                    f"Orders: {self.session_tracker.successful_orders}/{self.session_tracker.total_orders}"
                )

            except Exception as e:
                eastern = pytz.timezone('US/Eastern')
                now_est = datetime.now(pytz.UTC).astimezone(eastern)
                logger.error(f"MARKET CHECK FAILED: {e}")
                logger.error(f"API call to is_market_open() failed at {now_est.strftime('%H:%M:%S')}")
                # Log as failed check
                self.session_tracker.log_check(False)

            self.last_progress_log = now

            # Save progress every 5 checks (every ~75 seconds with 15s interval)
            if self.session_tracker.total_checks % 5 == 0:
                self.session_tracker.save_progress()

    def run_once(self, action: str = 'entry'):
        """
        Execute one iteration of the strategy.

        Args:
            action: 'entry' to generate signals and enter positions,
                   'exit' to close overnight positions
        """
        try:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"EXECUTING STRATEGY ({action.upper()}): {datetime.now()}")
            logger.info("=" * 80)

            if action == 'exit':
                # Close overnight positions (for OMR)
                if hasattr(self.adapter, 'close_overnight_positions'):
                    logger.info("Closing overnight positions...")
                    self.adapter.close_overnight_positions()
                    self.last_exit_time = datetime.now()
                    logger.success("Overnight positions closed")
                else:
                    logger.warning("Adapter does not support position closing")
            else:
                # Normal entry logic - run strategy
                self.adapter.run_once()
                self.session_tracker.log_run(0)  # Will need to get actual signal count
                self.last_run_time = datetime.now()

            logger.info(f"Next check in {self.check_interval} seconds")
            self._log_minute_progress(force=True)

        except Exception as e:
            logger.error(f"Error running strategy ({action}): {e}")
            import traceback
            traceback.print_exc()

    def _check_for_end_of_day(self) -> bool:
        """Check if it's end of trading day (4:00 PM EST)."""
        now = datetime.now()
        # Market close is 4:00 PM EST
        market_close = dt_time(16, 0)

        # If past market close and we haven't generated report today
        if now.time() >= market_close:
            report_file = self.log_dir / f"{now.strftime('%Y%m%d')}_{self.session_tracker.strategy_name}_summary.md"
            if not report_file.exists():
                return True
        return False

    def run_continuous(self):
        """Run strategy continuously based on schedule."""
        self.running = True
        logger.info("=" * 80)
        logger.info("LIVE PAPER TRADING - CONTINUOUS MODE")
        logger.info("=" * 80)
        logger.info(f"Strategy: {self.adapter.__class__.__name__}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info(f"Intraday pre-fetch: {'ENABLED (3:45 PM)' if self.enable_intraday_prefetch else 'DISABLED (3:50 PM only)'}")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)

        while self.running:
            try:
                # Log minute progress
                self._log_minute_progress()

                # Check for market open to pre-load historical data (once per day)
                now = datetime.now()
                if (not self.data_preloaded_today and
                    now.time() >= dt_time(9, 30) and
                    now.time() <= dt_time(9, 35) and
                    self.adapter.broker.is_market_open()):

                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("MARKET OPEN - PRE-LOADING HISTORICAL DATA")
                    logger.info("=" * 80)

                    if hasattr(self.adapter, 'preload_historical_data'):
                        self.adapter.preload_historical_data()
                        self.data_preloaded_today = True
                    else:
                        logger.warning("Adapter does not support data pre-loading")

                # Reset flag at midnight for new trading day
                if now.time() < dt_time(0, 5):  # Reset between midnight and 12:05 AM
                    if self.data_preloaded_today:
                        self.data_preloaded_today = False
                        logger.info("Reset data pre-load flag for new trading day")
                    if self.intraday_prefetched_today:
                        self.intraday_prefetched_today = False
                        logger.info("Reset intraday pre-fetch flag for new trading day")

                # Check for 3:45 PM to pre-fetch intraday data (once per day)
                # Only if intraday pre-fetching is enabled
                # Pre-fetch 5 minutes before execution to have fresh data with network buffer
                if (self.enable_intraday_prefetch and
                    not self.intraday_prefetched_today and
                    now.time() >= dt_time(15, 45) and
                    now.time() <= dt_time(15, 48) and
                    self.adapter.broker.is_market_open()):

                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("3:45 PM - PRE-FETCHING TODAY'S INTRADAY DATA")
                    logger.info("=" * 80)

                    if hasattr(self.adapter, 'prefetch_intraday_data'):
                        self.adapter.prefetch_intraday_data()
                        self.intraday_prefetched_today = True
                    else:
                        logger.warning("Adapter does not support intraday data pre-fetching")

                # Check for periodic flush (for multi-day sessions)
                if self.session_tracker.trading_logger.should_periodic_flush():
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("PERIODIC LOG FLUSH")
                    logger.info("=" * 80)
                    self.session_tracker.trading_logger.flush_to_disk(reason="Periodic flush (multi-day session)")

                # Check for end of day
                if self._check_for_end_of_day():
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("END OF TRADING DAY - GENERATING REPORT")
                    logger.info("=" * 80)

                    # Flush buffered logs to disk
                    self.session_tracker.trading_logger.flush_to_disk(reason="Market closed (4:00 PM ET)")

                    # Generate end-of-day report
                    self.session_tracker.generate_end_of_day_report(self.adapter.broker)

                # Check if should run strategy
                action = self.should_run_now()
                if action:
                    self.run_once(action=action)

                # Sleep until next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("\nKeyboard interrupt received, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.check_interval)

        # Save final progress and generate report
        logger.info("")
        logger.info("=" * 80)
        logger.info("GENERATING FINAL REPORT")
        logger.info("=" * 80)

        # Flush buffered logs to disk before stopping
        self.session_tracker.trading_logger.flush_to_disk(reason="Trading stopped")

        self.session_tracker.save_progress()
        self.session_tracker.generate_end_of_day_report(self.adapter.broker)

        logger.info("")
        logger.info("=" * 80)
        logger.info("LIVE PAPER TRADING STOPPED")
        logger.info("=" * 80)


def create_ma_crossover_adapter(broker, symbols, fast=50, slow=200, position_size=0.05, max_positions=3):
    """Create MA Crossover adapter."""
    return MACrossoverLiveAdapter(
        broker=broker,
        symbols=symbols,
        fast_period=fast,
        slow_period=slow,
        ma_type='sma',
        position_size=position_size,
        max_positions=max_positions
    )


def create_triple_ma_adapter(broker, symbols, fast=20, medium=50, slow=200, position_size=0.05, max_positions=3):
    """Create Triple MA Crossover adapter."""
    return TripleMACrossoverLiveAdapter(
        broker=broker,
        symbols=symbols,
        fast_period=fast,
        medium_period=medium,
        slow_period=slow,
        position_size=position_size,
        max_positions=max_positions
    )


def create_omr_adapter(broker, symbols=None, min_probability=None, min_return=None, position_size=None, max_positions=None, omr_config=None):
    """
    Create Overnight Mean Reversion adapter.

    IMPORTANT: For production, pass omr_config from load_omr_config().
    Individual parameters are only for testing/overrides.
    """
    # If config provided, use it (RECOMMENDED)
    if omr_config is not None:
        logger.info("Creating OMR adapter from production config")
        adapter_params = omr_config.to_adapter_params()
        return OMRLiveAdapter(broker=broker, **adapter_params)

    # Fallback to individual parameters (for testing)
    logger.warning("Creating OMR adapter with individual parameters (NOT using production config)")
    return OMRLiveAdapter(
        broker=broker,
        symbols=symbols if symbols is not None else [],
        min_probability=min_probability if min_probability is not None else 0.60,
        min_expected_return=min_return if min_return is not None else 0.002,
        max_positions=max_positions if max_positions is not None else 3,
        position_size=position_size if position_size is not None else 0.05
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run live paper trading')

    # Strategy selection
    parser.add_argument(
        '--strategy',
        type=str,
        default='ma',
        choices=['ma', 'triple-ma', 'omr'],
        help='Strategy to run (default: ma)'
    )

    # Symbol universe
    parser.add_argument(
        '--universe',
        type=str,
        default='faang',
        choices=['faang', 'tech', 'leveraged'],
        help='Symbol universe to trade (default: faang)'
    )

    # Risk parameters
    parser.add_argument(
        '--position-size',
        type=float,
        default=0.05,
        help='Position size as fraction of capital (default: 0.05 = 5%%)'
    )

    parser.add_argument(
        '--max-positions',
        type=int,
        default=3,
        help='Maximum concurrent positions (default: 3)'
    )

    # Execution mode
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (default: continuous)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=15,
        help='Seconds between schedule checks (default: 15)'
    )

    # Data pre-fetching
    parser.add_argument(
        '--no-intraday-prefetch',
        action='store_true',
        help='Disable intraday data pre-fetching (fetch all data at 3:50 PM instead of 3:45 PM)'
    )

    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory for log files (default: logs/live_trading)'
    )

    # Strategy-specific parameters
    parser.add_argument('--fast', type=int, default=50, help='Fast MA period (MA strategies)')
    parser.add_argument('--medium', type=int, default=50, help='Medium MA period (Triple MA)')
    parser.add_argument('--slow', type=int, default=200, help='Slow MA period (MA strategies)')
    parser.add_argument('--min-probability', type=float, default=0.60, help='Min probability (OMR)')
    parser.add_argument('--min-return', type=float, default=0.002, help='Min expected return (OMR)')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check for API credentials
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca API credentials not found in environment variables")
        logger.info("Please set ALPACA_API_KEY/ALPACA_PAPER_KEY_ID and ALPACA_SECRET_KEY/ALPACA_PAPER_SECRET_KEY in .env file")
        return 1

    # Select symbol universe
    # IMPORTANT: For OMR strategy, ALWAYS use production config
    omr_config = None  # Initialize
    if args.strategy == 'omr':
        logger.info("OMR strategy selected - loading production config...")
        omr_config = load_omr_config()
        symbols = omr_config.symbols
        logger.info(f"Loaded {len(symbols)} symbols from production config")
    elif args.universe == 'faang':
        symbols = EquityUniverse.FAANG
    elif args.universe == 'tech':
        symbols = EquityUniverse.TECH_GIANTS
    elif args.universe == 'leveraged':
        # For non-OMR strategies, use first 10 for testing
        symbols = ETFUniverse.LEVERAGED_3X[:10]
        logger.warning("Using first 10 LEVERAGED_3X symbols (TEST MODE)")
    else:
        symbols = EquityUniverse.FAANG

    logger.info("=" * 80)
    logger.info("LIVE PAPER TRADING SETUP")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Universe: {args.universe} ({len(symbols)} symbols)")
    logger.info(f"Position size: {args.position_size:.1%}")
    logger.info(f"Max positions: {args.max_positions}")
    logger.info(f"Mode: {'Run once' if args.once else 'Continuous'}")
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

        # Create strategy adapter
        logger.info("")
        logger.info(f"Creating {args.strategy} adapter...")

        if args.strategy == 'ma':
            adapter = create_ma_crossover_adapter(
                broker, symbols,
                fast=args.fast,
                slow=args.slow,
                position_size=args.position_size,
                max_positions=args.max_positions
            )
        elif args.strategy == 'triple-ma':
            adapter = create_triple_ma_adapter(
                broker, symbols,
                fast=args.fast,
                medium=args.medium,
                slow=args.slow,
                position_size=args.position_size,
                max_positions=args.max_positions
            )
        elif args.strategy == 'omr':
            # For OMR, use production config (already loaded above)
            adapter = create_omr_adapter(
                broker,
                omr_config=omr_config  # Use production config
            )
        else:
            logger.error(f"Unknown strategy: {args.strategy}")
            return 1

        logger.success("Adapter created successfully")
        logger.info("")

        # Setup log directory
        log_dir = Path(args.log_dir) if args.log_dir else None

        # Run strategy
        if args.once:
            # Single execution
            logger.info("Running strategy once...")
            adapter.run_once()
            logger.success("Strategy execution complete")
        else:
            # Continuous execution
            enable_prefetch = not args.no_intraday_prefetch  # Inverted: --no-intraday-prefetch disables
            runner = LiveTradingRunner(
                adapter,
                check_interval=args.check_interval,
                log_dir=log_dir,
                enable_intraday_prefetch=enable_prefetch
            )
            runner.run_continuous()

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
