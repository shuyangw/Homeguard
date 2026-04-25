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

# Set process title for easy identification in ps/htop
try:
    import setproctitle
    HAS_SETPROCTITLE = True
except ImportError:
    HAS_SETPROCTITLE = False


def set_process_name(strategy: str) -> None:
    """Set process name to identify strategy in ps/htop."""
    if HAS_SETPROCTITLE:
        proc_name = f"homeguard-{strategy}"
        setproctitle.setproctitle(proc_name)
        # Also set thread name for visibility
        setproctitle.setthreadtitle(proc_name)

from src.trading.brokers import AlpacaBroker
from src.trading.adapters import (
    MACrossoverLiveAdapter,
    TripleMACrossoverLiveAdapter,
    OMRLiveAdapter,
    MomentumLiveAdapter,
    RAMPLiveAdapter
)
from src.trading.config import load_omr_config
from src.trading.config.broker_routing import load_broker_routing
from src.data.providers import create_data_provider
from src.trading.state import StrategyStateManager
from src.strategies.universe import EquityUniverse, ETFUniverse
from src.utils.logger import logger, get_trading_logger, TradingLogger, console
from src.utils.timezone import tz
from src.utils.trading_logger import cleanup_old_logs


class TradingSessionTracker:
    """Tracks trading session metrics and generates reports."""

    def __init__(self, log_dir: Path, strategy_name: str):
        """
        Initialize session tracker.

        Args:
            log_dir: Directory for log files
            strategy_name: Name of the strategy
        """
        self.strategy_name = strategy_name
        self.session_start = tz.now()
        self.session_date = tz.date_str()
        # Include start time to avoid overwriting logs on multiple runs per day
        self.session_datetime = tz.datetime_str()

        # Create date-based subdirectory to organize logs by day
        # e.g., logs/live_trading/paper/20251117/
        date_subdir = log_dir / self.session_date
        date_subdir.mkdir(parents=True, exist_ok=True)
        self.log_dir = date_subdir

        # Session metrics
        self.total_checks = 0
        self.total_runs = 0
        self.minute_progress: List[Dict] = []

        # Create session log files with date and time to prevent overwriting
        self.session_log_file = date_subdir / f"{self.session_datetime}_{strategy_name}_session.json"
        self.summary_file = date_subdir / f"{self.session_datetime}_{strategy_name}_summary.md"
        self.trades_log_file = date_subdir / f"{self.session_datetime}_{strategy_name}_trades.csv"
        self.market_checks_log_file = date_subdir / f"{self.session_datetime}_{strategy_name}_market_checks.csv"

        # Create trading logger with CSV logging and 5-minute auto-flush
        # flush_interval_hours: 5 minutes = 5/60 hours = 0.0833 hours
        self.trading_logger = get_trading_logger(
            strategy_name,
            date_subdir,
            flush_interval_hours=5/60  # Auto-flush every 5 minutes
        )

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
        timestamp = tz.iso_timestamp()

        self.minute_progress.append({
            'timestamp': timestamp,
            'type': 'check',
            'market_open': market_open,
            'total_checks': self.total_checks
        })

        # Log to CSV using trading logger
        self.trading_logger.log_market_check(market_open, self.total_checks)

    def log_run(self, signals_count: int = 0):
        """Log a strategy run."""
        self.total_runs += 1
        self.minute_progress.append({
            'timestamp': tz.iso_timestamp(),
            'type': 'run',
            'total_runs': self.total_runs
        })

    def save_progress(self):
        """Save current progress to file."""
        progress_data = {
            'session_start': self.session_start.isoformat(),
            'strategy_name': self.strategy_name,
            'total_checks': self.total_checks,
            'total_runs': self.total_runs,
            'minute_progress': self.minute_progress[-100:]  # Last 100 entries
        }

        with open(self.session_log_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def save_journalctl_logs(self):
        """
        Export journalctl entries for this session to a log file.

        This captures all systemd journal logs for the homeguard-trading service
        since the session started, providing a complete record of console output.
        """
        import subprocess
        import platform

        # Only run on Linux (EC2) - skip on Windows/macOS during development
        if platform.system() != 'Linux':
            return  # Silently skip journalctl export on non-Linux

        try:
            # Format session start time for journalctl --since parameter
            # journalctl expects format like "2025-11-18 18:41:43"
            since_time = self.session_start.strftime('%Y-%m-%d %H:%M:%S')

            # Export journalctl entries for homeguard-trading service since session start
            journalctl_file = self.log_dir / f"{self.session_datetime}_{self.strategy_name}_journalctl.log"

            cmd = [
                'journalctl',
                '-u', 'homeguard-trading',
                '--since', since_time,
                '--no-pager',
                '--output', 'short-iso'  # ISO timestamps for consistency
            ]

            logger.info(f"Exporting journalctl logs since {since_time}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Write journalctl output to file
                with open(journalctl_file, 'w') as f:
                    f.write(result.stdout)

                # Count lines for feedback
                line_count = len(result.stdout.splitlines())
                logger.success(f"Saved {line_count} journalctl entries to {journalctl_file.name}")
            else:
                logger.error(f"Failed to export journalctl: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Timeout while exporting journalctl logs")
        except Exception as e:
            logger.error(f"Error exporting journalctl logs: {e}")

    def generate_end_of_day_report(self, broker) -> None:
        """Write daily summary, sourcing counts from the decision log."""
        from datetime import date as _date
        from src.trading.decision_log.reader import for_date as dl_for_date

        records = dl_for_date(self.strategy_name.lower(), _date.today())
        triggers = len(records)
        clean = sum(
            1 for r in records
            if r.preconditions.all_passed and r.error is None
        )
        blocked = sum(1 for r in records if not r.preconditions.all_passed)
        errored = sum(1 for r in records if r.error is not None)
        orders_total = sum(len(r.executions) for r in records)
        orders_filled = sum(
            1 for r in records for e in r.executions if e.status == "filled"
        )

        try:
            account = broker.get_account()
        except Exception:
            account = {}

        session_duration = (tz.now() - self.session_start).total_seconds() / 3600

        lines = [
            "# Live Paper Trading Session Summary",
            "",
            f"**Date:** {self.session_date}",
            f"**Strategy:** {self.strategy_name}",
            f"**Session Duration:** {session_duration:.2f} hours",
            f"**Session Start:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Session End:** {tz.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Activity Summary (sourced from decision log)",
            "",
            f"- **Total Schedule Checks:** {self.total_checks}",
            f"- **Strategy Executions:** {self.total_runs}",
            f"- **Decision Records:** {triggers}",
            f"  - Clean: {clean}",
            f"  - Blocked: {blocked}",
            f"  - Errored: {errored}",
            f"- **Orders:** {orders_filled}/{orders_total} filled",
            "",
            "## Account Status",
            "",
            f"- **Portfolio Value:** ${float(account.get('portfolio_value', 0) or 0):,.2f}",
            f"- **Buying Power:** ${float(account.get('buying_power', 0) or 0):,.2f}",
            f"- **Cash:** ${float(account.get('cash', 0) or 0):,.2f}",
            "",
            "## Performance Metrics",
            "",
            f"- **Checks per Hour:** {(self.total_checks / session_duration) if session_duration > 0 else 0:.2f}",
            f"- **Runs per Hour:** {(self.total_runs / session_duration) if session_duration > 0 else 0:.2f}",
            "",
            "---",
            "",
            "*Source of truth: decision log (`data/trading/decisions/<strategy>_<date>.jsonl`).*",
            "",
        ]
        self.summary_file.write_text("\n".join(lines), encoding="utf-8")

        logger.success(f"End-of-day report saved: {self.summary_file}")
        return self.summary_file


class LiveTradingRunner:
    """Manages continuous live paper trading execution with logging."""

    def __init__(self, adapter, check_interval: int = 15, log_dir: Path = None,
                 enable_intraday_prefetch: bool = True,
                 assume_market_open: bool = False,
                 metrics_registry=None):
        """
        Initialize live trading runner.

        Args:
            adapter: Strategy adapter to run
            check_interval: Seconds between schedule checks (default: 15)
            log_dir: Directory for log files (default: logs/live_trading)
            enable_intraday_prefetch: Deprecated - strategies now fetch data at execution time
            assume_market_open: Bypass market open check for testing (default: False)
            metrics_registry: Optional MetricsRegistry for monitoring (default: None)
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
        self.assume_market_open: bool = assume_market_open  # Bypass market open check for testing
        self.metrics_registry = metrics_registry  # Optional MetricsRegistry for monitoring
        self._metrics_tick_counter = 0  # Throttles broker.get_account() calls
        self._peak_equity: float = 0.0  # Rolling peak since process start (for drawdown)
        self._session_open_equity: Optional[float] = None  # Set on first tick after market open

        # Setup logging directory
        if log_dir is None:
            # Use configured output directory from settings.ini
            from src.settings import get_live_trading_dir
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

    def _is_market_open(self) -> bool:
        """Check if market is open, with bypass support for testing."""
        if self.assume_market_open:
            return True
        return self.adapter.broker.is_market_open()

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
            if not self._is_market_open():
                return None

        # Check for execution_times (new format for overnight strategies)
        execution_times = schedule.get('execution_times')
        if execution_times:
            # Get current time in EST for comparison with schedule times
            now_est = tz.now()
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
                        seconds_since_last = tz.seconds_since(self.last_run_time)
                        if seconds_since_last < 60:
                            continue
                    elif action == 'exit' and self.last_exit_time:
                        seconds_since_last = tz.seconds_since(self.last_exit_time)
                        if seconds_since_last < 60:
                            continue

                    return action

            return None

        # Check specific time (legacy single-time format)
        specific_time = schedule.get('specific_time')
        if specific_time:
            # Get current time in EST for comparison with schedule times
            now_est = tz.now()
            target_time = datetime.strptime(specific_time, '%H:%M').time()
            current_time = now_est.time()

            # Run if within 1 minute of target time
            time_diff = abs((datetime.combine(now_est.date(), current_time) -
                           datetime.combine(now_est.date(), target_time)).total_seconds())

            if time_diff < 60:
                # Check if we already ran within last minute
                if self.last_run_time:
                    seconds_since_last = tz.seconds_since(self.last_run_time)
                    if seconds_since_last < 60:
                        return None
                return 'entry'  # Default to entry action
            return None

        # For interval-based strategies, check minimum time between runs
        interval = schedule.get('interval', '5min')
        if self.last_run_time:
            seconds_since_last = tz.seconds_since(self.last_run_time)

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
        now = tz.now()

        # Log every 15 seconds or if forced
        log_interval = self.check_interval if self.check_interval >= 15 else 15
        if force or self.last_progress_log is None or tz.seconds_since(self.last_progress_log) >= log_interval:
            # Query market status with error handling
            try:
                market_open = self._is_market_open()
                market_status_str = 'OPEN' if market_open else 'CLOSED'
                if self.assume_market_open and not self.adapter.broker.is_market_open():
                    market_status_str = 'ASSUMED OPEN (testing)'

                # Log market check to session tracker
                self.session_tracker.log_check(market_open)

                # Use EST time for display
                now_est = tz.now()

                # Log comprehensive status (direct console output for real-time visibility)
                status_msg = (
                    f"[{now_est.strftime('%H:%M:%S')}] "
                    f"Market: {market_status_str} | "
                    f"Checks: {self.session_tracker.total_checks} | "
                    f"Runs: {self.session_tracker.total_runs}"
                )
                # Use Rich console directly for colored output that bypasses logger buffering
                console.print(f" [info]{status_msg}[/info]")

            except Exception as e:
                now_est = tz.now()
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
            logger.info(f"EXECUTING STRATEGY ({action.upper()}): {tz.timestamp()}")
            logger.info("=" * 80)

            # Safety check: verify market is open before executing
            if not self._is_market_open():
                logger.warning("Market is CLOSED - skipping strategy execution")
                logger.warning("This should not happen if schedule is configured correctly")
                return

            if action == 'exit':
                # Close overnight positions (for OMR)
                if hasattr(self.adapter, 'close_overnight_positions'):
                    logger.info("Closing overnight positions...")
                    self.adapter.close_overnight_positions()
                    self.last_exit_time = tz.now()
                    logger.success("Overnight positions closed")
                else:
                    logger.warning("Adapter does not support position closing")
            else:
                # Normal entry logic - run strategy
                self.adapter.run_once()
                self.session_tracker.log_run(0)  # Will need to get actual signal count
                self.last_run_time = tz.now()

            logger.info(f"Next check in {self.check_interval} seconds")
            self._log_minute_progress(force=True)

        except Exception as e:
            logger.error(f"Error running strategy ({action}): {e}")
            import traceback
            traceback.print_exc()

    def _emit_derived_portfolio_metrics(self, account: dict) -> None:
        """Emit drawdown and day-P&L gauges from a freshly fetched account dict."""
        equity = float(account.get('portfolio_value', 0))
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown_pct = 0.0 if self._peak_equity <= 0 else (
            (equity - self._peak_equity) / self._peak_equity * 100.0
        )
        self.metrics_registry.update_drawdown(drawdown_pct)

        # Alpaca account exposes `last_equity` (prior session close). Fall back
        # to an in-process open-of-session snapshot if that field isn't present.
        last_equity = float(account.get('last_equity') or 0)
        if last_equity > 0:
            day_pnl = equity - last_equity
        else:
            if self._session_open_equity is None and self._is_market_open():
                self._session_open_equity = equity
            day_pnl = equity - (self._session_open_equity or equity)
        self.metrics_registry.update_day_pnl(day_pnl)

    def _compute_strategy_equity(self, initial_capital: float, account: dict) -> float:
        """Compute equity attributed to this strategy, in USD.

        Formula: initial_capital + sum(unrealized_pnl) for strategy-tagged positions.

        Rationale: strategies share a broker account (e.g. OMR + RAMP both on IBKR
        paper). Reporting the full broker portfolio_value overstates each strategy's
        footprint -- the dashboard would show $1M for every strategy sharing a $1M
        account. Instead, anchor to the strategy's own capital budget and add its
        attributed P&L.

        With zero positions this returns initial_capital unchanged; as positions
        accumulate P&L the line diverges accordingly.
        """
        strategy = self.metrics_registry.strategy
        try:
            all_positions = self.adapter.broker.get_positions() or []
        except Exception as e:
            logger.error(f"Metrics: broker.get_positions for equity failed: {e}")
            return initial_capital

        owned_symbols: set = set()
        try:
            sm = getattr(self.adapter, 'state_manager', None)
            if sm is not None:
                owned_symbols = set(sm.get_positions(strategy).keys())
        except Exception as e:
            logger.error(f"Metrics: state_manager.get_positions for equity failed: {e}")

        if not owned_symbols:
            # No tracked positions yet -- all capital sits as cash budget.
            return initial_capital

        tagged = [p for p in all_positions if p.get('symbol') in owned_symbols]
        pnl = sum(float(p.get('unrealized_pnl', 0) or 0) for p in tagged)
        return initial_capital + pnl

    def _emit_position_and_strategy_metrics(self, update_position_metrics, update_strategy_metrics) -> None:
        """Emit per-position and strategy-aggregate gauges from a broker positions fetch.

        Filters broker positions by `state_manager.get_positions(strategy)` so OMR and
        RAMP (which share an Alpaca account) each report only their own attributed
        unrealized PnL and capital.
        """
        try:
            all_positions = self.adapter.broker.get_positions() or []
        except Exception as e:
            logger.error(f"Metrics: broker.get_positions failed: {e}")
            all_positions = []

        strategy = self.metrics_registry.strategy
        owned_symbols: set = set()
        try:
            sm = getattr(self.adapter, 'state_manager', None)
            if sm is not None:
                owned_symbols = set(sm.get_positions(strategy).keys())
        except Exception as e:
            logger.error(f"Metrics: state_manager.get_positions failed: {e}")

        if owned_symbols:
            positions = [p for p in all_positions if p.get('symbol') in owned_symbols]
        else:
            # Fallback preserves pre-attribution behavior if state manager is unavailable.
            positions = all_positions

        update_position_metrics(self.metrics_registry, positions)
        unrealized = sum(float(p.get('unrealized_pnl', 0)) for p in positions)
        capital = sum(abs(float(p.get('market_value', 0))) for p in positions)
        realized = self._compute_today_realized_pnl(strategy)
        update_strategy_metrics(
            self.metrics_registry,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            positions=len(positions),
            capital_allocated=capital,
        )

    def _compute_today_realized_pnl(self, strategy: str) -> float:
        """Sum today's realized PnL for `strategy` by scanning the trade-log JSONL."""
        try:
            from src.utils.trading_logger import get_trade_log_writer
            writer = get_trade_log_writer()
            log_file = writer._get_log_file()
            if not log_file.exists():
                return 0.0
            total = 0.0
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if (row.get('strategy') == strategy
                            and row.get('trade_type') == 'exit'
                            and row.get('pnl_dollars') is not None):
                        total += float(row['pnl_dollars'])
            return total
        except Exception as e:
            logger.error(f"Metrics: realized PnL aggregation failed: {e}")
            return 0.0

    def _emit_strategy_specific_metrics(self) -> None:
        """Emit regime + RAMP cache-age gauges. No-op for non-RAMP adapters."""
        if self.adapter.__class__.__name__ != 'RAMPLiveAdapter':
            return
        # Regime state code -- RAMP stores it on the inner RAMPSignals instance
        # exposed as adapter._ramp_signals, not on the wrapper at adapter.strategy.
        ramp_signals = getattr(self.adapter, '_ramp_signals', None)
        regime_name = getattr(ramp_signals, '_current_regime', None) if ramp_signals else None
        if regime_name:
            try:
                from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
                regime_code = MarketRegimeDetector.REGIMES.get(regime_name, 0)
                self.metrics_registry.update_regime(
                    state_code=regime_code,
                    sma_20=0.0, sma_50=0.0, sma_200=0.0,
                    time_in_state_seconds=0.0,
                )
            except Exception as e:
                logger.error(f"Metrics: regime emit failed: {e}")
        # RAMP cache age
        try:
            from src.trading.adapters.ramp_live_adapter import CACHE_DIR, CACHE_FILE_PREFIX
            if CACHE_DIR.exists():
                cache_files = sorted(
                    CACHE_DIR.glob(f'{CACHE_FILE_PREFIX}_*.pkl'), reverse=True
                )
                if cache_files:
                    age_sec = time.time() - cache_files[0].stat().st_mtime
                    self.metrics_registry.update_ramp_cache(age_sec, hit=False)
        except Exception as e:
            logger.error(f"Metrics: RAMP cache emit failed: {e}")

    def _check_for_end_of_day(self) -> bool:
        """Check if it's end of trading day (4:00 PM EST)."""
        # Get current time in EST
        now_est = tz.now()

        # Market close is 4:00 PM EST
        market_close = dt_time(16, 0)

        # If past market close and we haven't generated report for this session yet
        if now_est.time() >= market_close:
            # Check if this session's summary file exists (uses session timestamp)
            if not self.session_tracker.summary_file.exists():
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
        logger.info(f"Intraday pre-fetch: Strategies fetch at execution time")
        if self.assume_market_open:
            logger.warning(f"ASSUME MARKET OPEN: market-hours checks bypassed")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)

        while self.running:
            try:
                # Log minute progress
                self._log_minute_progress()

                # Update metrics
                if self.metrics_registry is not None:
                    from src.monitoring.hooks import (
                        update_portfolio_metrics,
                        update_market_status,
                        update_process_metrics,
                        update_websocket_metrics,
                        update_position_metrics,
                        update_strategy_metrics,
                        update_strategy_equity,
                    )
                    try:
                        # Portfolio gauges need a REST call per update -- throttle to
                        # every 4 ticks (~60s at check_interval=15s) so ENABLE_METRICS
                        # doesn't 4x the broker API load.
                        self._metrics_tick_counter += 1
                        if self._metrics_tick_counter % 4 == 1:
                            account = self.adapter.broker.get_account()
                            if account:
                                broker_name = getattr(self.adapter, 'broker_name', 'unknown')
                                update_portfolio_metrics(self.metrics_registry, account, broker_name)

                                # Strategy equity should reflect the strategy's own
                                # capital budget + P&L of its tagged positions, not
                                # the full broker account value (which is shared
                                # across all strategies on that broker). Prefer
                                # adapter.initial_capital; fall back to broker
                                # portfolio value only if the adapter doesn't
                                # expose a capital cap.
                                strategy_initial_capital = getattr(
                                    self.adapter, 'initial_capital', None
                                )
                                if strategy_initial_capital is None:
                                    strategy_equity = float(account.get('portfolio_value', 0) or 0)
                                else:
                                    strategy_equity = self._compute_strategy_equity(
                                        initial_capital=float(strategy_initial_capital),
                                        account=account,
                                    )
                                update_strategy_equity(
                                    self.metrics_registry,
                                    strategy_equity,
                                )
                                self._emit_derived_portfolio_metrics(account)
                                self._emit_position_and_strategy_metrics(
                                    update_position_metrics, update_strategy_metrics
                                )
                        update_market_status(self.metrics_registry, self._is_market_open())
                        update_process_metrics(self.metrics_registry)
                        self._emit_strategy_specific_metrics()

                        # WebSocket status if streaming
                        dp = getattr(self.adapter, '_data_provider', None) or getattr(self.adapter, 'data_provider', None)
                        if dp is not None:
                            if hasattr(dp, 'is_connected'):
                                feed = os.getenv('STREAMING_FEED', 'iex')
                                connected = dp.is_connected()
                                symbols = getattr(dp, '_subscribed_count', 0)
                                update_websocket_metrics(
                                    self.metrics_registry, feed, connected, symbols
                                )
                    except Exception as e:
                        logger.error(f"Metrics update failed: {e}")

                # Use EST for all time comparisons
                now_est = tz.now()

                # Check for market open to pre-load historical data (once per day)
                if (not self.data_preloaded_today and
                    now_est.time() >= dt_time(9, 30) and
                    now_est.time() <= dt_time(9, 35) and
                    self._is_market_open()):

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
                if now_est.time() < dt_time(0, 5):  # Reset between midnight and 12:05 AM EST
                    if self.data_preloaded_today:
                        self.data_preloaded_today = False
                        logger.info("Reset data pre-load flag for new trading day")
                    if self.intraday_prefetched_today:
                        self.intraday_prefetched_today = False
                        logger.info("Reset intraday pre-fetch flag for new trading day")

                # NOTE: Strategies fetch fresh data in their run_once() methods:
                # - OMR: prefetch_intraday_data() at 3:50 PM (full intraday bars)
                # - MP: fetch_todays_closes() at 3:55 PM (today's closes only, lightweight)

                # Check for periodic flush (for multi-day sessions)
                if self.session_tracker.trading_logger.should_periodic_flush():
                    self.session_tracker.trading_logger.flush_to_disk(reason="Periodic flush (multi-day session)")

                # Check for end of day
                if self._check_for_end_of_day():
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("END OF TRADING DAY - GENERATING REPORT")
                    logger.info("=" * 80)

                    # Flush buffered logs to disk
                    self.session_tracker.trading_logger.flush_to_disk(reason="Market closed (4:00 PM ET)")

                    # Export journalctl logs for this session
                    self.session_tracker.save_journalctl_logs()

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

        # Export journalctl logs for this session
        self.session_tracker.save_journalctl_logs()

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


def create_omr_adapter(broker, symbols=None, min_probability=None, min_return=None, position_size=None, max_positions=None, omr_config=None, data_provider=None, max_capital_usd=None, *, broker_name: str):
    """
    Create Overnight Mean Reversion adapter.

    IMPORTANT: For production, pass omr_config from load_omr_config().
    Individual parameters are only for testing/overrides.

    Args:
        broker: Broker interface
        data_provider: Optional data provider with fallback chain (Alpaca -> yfinance)
        omr_config: Production config from load_omr_config()
        ... other params for testing overrides
    """
    # If config provided, use it (RECOMMENDED)
    if omr_config is not None:
        logger.info("Creating OMR adapter from production config")
        adapter_params = omr_config.to_adapter_params()
        if data_provider is not None:
            adapter_params['data_provider'] = data_provider
            logger.info(f"  Using data provider: {data_provider.name}")
        if max_capital_usd is not None:
            adapter_params['max_capital_usd'] = max_capital_usd
        return OMRLiveAdapter(broker=broker, broker_name=broker_name, **adapter_params)

    # Fallback to individual parameters (for testing)
    logger.warning("Creating OMR adapter with individual parameters (NOT using production config)")
    return OMRLiveAdapter(
        broker=broker,
        broker_name=broker_name,
        symbols=symbols if symbols is not None else [],
        min_probability=min_probability if min_probability is not None else 0.60,
        min_expected_return=min_return if min_return is not None else 0.002,
        max_positions=max_positions if max_positions is not None else 3,
        position_size=position_size if position_size is not None else 0.05,
        data_provider=data_provider,
        max_capital_usd=max_capital_usd,
    )


def create_mp_adapter(broker, position_size=0.065, top_n=10, data_provider=None, *, broker_name: str):
    """
    Create Momentum Protection adapter.

    Args:
        broker: Broker interface
        position_size: Position size per stock (default: 6.5%)
        top_n: Number of top momentum stocks to hold (default: 10)
        data_provider: Optional data provider with fallback chain (Alpaca -> yfinance)

    Returns:
        MomentumLiveAdapter instance
    """
    logger.info("Creating MP adapter with crash protection")
    if data_provider is not None:
        logger.info(f"  Using data provider: {data_provider.name}")
    return MomentumLiveAdapter(
        broker=broker,
        broker_name=broker_name,
        symbols=None,  # Uses S&P 500 by default
        top_n=top_n,
        position_size=position_size,
        reduced_exposure=0.5,  # 50% exposure when risk triggers
        vix_threshold=25.0,
        vix_spike_threshold=0.20,
        spy_dd_threshold=-0.05,
        mom_vol_percentile=0.90,
        data_provider=data_provider
    )


def create_ramp_adapter(broker, data_provider=None, initial_capital=None, metrics_registry=None, *, broker_name: str):
    """
    Create Regime-Aware Momentum Protection adapter.

    Args:
        broker: Broker interface
        data_provider: Optional data provider with fallback chain (Alpaca -> yfinance)
        initial_capital: Optional cap on strategy allocation (USD). If None, uses full account equity.
        metrics_registry: Optional MetricsRegistry -- when provided, the adapter increments
                          hg_strategy_rebalance_errors_total on individual order failures so
                          silent multi-symbol rebalance breakage shows up on dashboards.

    Returns:
        RAMPLiveAdapter instance
    """
    logger.info("Creating RAMP adapter with regime detection")
    if data_provider is not None:
        logger.info(f"  Using data provider: {data_provider.name}")
    if initial_capital is not None:
        logger.info(f"  Capital cap: ${initial_capital:,.0f}")
    return RAMPLiveAdapter(
        broker=broker,
        broker_name=broker_name,
        symbols=None,  # Uses S&P 500 by default
        data_provider=data_provider,
        initial_capital=initial_capital,
        metrics_registry=metrics_registry,
    )


def preflight_reconcile(
    strategy: str,
    broker,
    broker_name: str,
    state_manager,
    force_start: bool,
) -> int:
    """
    Pre-flight reconciliation check.

    Returns 0 if safe to proceed, 1 if the runner should exit.
    Never mutates position data; may trigger a one-shot v1->v2 schema
    migration on first load. On mismatch + force_start, logs WARNING
    and returns 0.
    """
    state_positions = state_manager.get_positions(strategy)

    if not state_positions:
        logger.info(f"[Reconcile] {strategy} has no tracked positions. Clear to proceed.")
        logger.success(f"[Reconcile] Pre-flight check passed for {strategy}")
        return 0

    logger.info(
        f"[Reconcile] {strategy} has {len(state_positions)} tracked positions"
    )

    try:
        broker_positions = {
            p['symbol']: p['quantity']
            for p in broker.get_stock_positions()
        }
    except Exception as e:
        logger.error(f"[Reconcile] Cannot query broker positions: {e}")
        if not force_start:
            logger.error("Cannot verify positions. Use --force-start to skip.")
            return 1
        logger.warning("--force-start: proceeding despite broker error")
        return 0

    mismatches = []
    for symbol, pos_info in state_positions.items():
        state_qty = pos_info.get('qty', 0)
        pos_broker = pos_info.get('broker')
        if pos_broker is None:
            mismatches.append(
                f"{symbol}: state entry missing 'broker' tag"
            )
            continue
        broker_qty = broker_positions.get(symbol, 0)

        if pos_broker != broker_name and state_qty > 0:
            mismatches.append(
                f"{symbol}: {state_qty} shares tagged {pos_broker}, "
                f"runner on {broker_name}"
            )
        elif broker_qty == 0 and state_qty > 0 and pos_broker == broker_name:
            mismatches.append(
                f"{symbol}: state says {state_qty} on {broker_name}, "
                f"broker reports 0"
            )

    if mismatches:
        logger.error("=" * 60)
        logger.error("POSITION MISMATCH DETECTED")
        logger.error("=" * 60)
        for m in mismatches:
            logger.error(f"  {m}")
        logger.error("")
        logger.error("Options:")
        logger.error("  1. Close positions on the old broker first")
        logger.error("  2. Switch broker_routing.yaml back to the old broker")
        logger.error("  3. Use --force-start to start anyway (state untouched)")
        logger.error("=" * 60)
        if not force_start:
            return 1
        logger.warning(
            "--force-start: proceeding despite mismatches, state unchanged"
        )
        return 0

    logger.success(f"[Reconcile] Pre-flight check passed for {strategy}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run live paper trading')

    # Strategy selection
    parser.add_argument(
        '--strategy',
        type=str,
        default='ma',
        choices=['ma', 'triple-ma', 'omr', 'mp', 'ramp', 'multi'],
        help='Strategy to run: ma, triple-ma, omr, mp, ramp, or multi (runs all enabled strategies)'
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

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=None,
        help='Cap strategy allocation at this USD amount (default: use full account equity)'
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

    # Testing/Debug options
    parser.add_argument(
        '--assume-market-open',
        action='store_true',
        help='Assume market is open, skipping market-hours checks (for testing only)'
    )

    parser.add_argument(
        '--force-start',
        action='store_true',
        help='Start despite position mismatches. Does NOT modify state. (DANGEROUS)',
    )

    # Strategy-specific parameters
    parser.add_argument('--fast', type=int, default=50, help='Fast MA period (MA strategies)')
    parser.add_argument('--medium', type=int, default=50, help='Medium MA period (Triple MA)')
    parser.add_argument('--slow', type=int, default=200, help='Slow MA period (MA strategies)')
    parser.add_argument('--min-probability', type=float, default=0.60, help='Min probability (OMR)')
    parser.add_argument('--min-return', type=float, default=0.002, help='Min expected return (OMR)')

    args = parser.parse_args()

    # Set process name for easy identification in ps/htop
    set_process_name(args.strategy)

    # Load environment variables
    load_dotenv()

    # Clean up old trade logs (keep 30 days)
    # This runs on startup to manage disk space on EC2
    try:
        cleanup_old_logs(log_dir="/home/ec2-user/logs", keep_days=30)
    except Exception as e:
        logger.warning(f"Failed to cleanup old logs (non-critical): {e}")

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
    logger.info(f"CLI universe arg: {args.universe} ({len(symbols)} symbols)")
    logger.info(f"CLI position size: {args.position_size:.1%}")
    logger.info(f"CLI max positions: {args.max_positions}")
    logger.info("(Strategy runtime may override these -- see regime logs during rebalance)")
    logger.info(f"Mode: {'Run once' if args.once else 'Continuous'}")
    logger.info("=" * 80)

    # Initialize metrics registry if enabled
    metrics_registry = None
    if os.getenv('ENABLE_METRICS', 'false').lower() in ('true', '1', 'yes'):
        from src.monitoring import MetricsRegistry, start_metrics_server
        from src.monitoring.snapshot import SnapshotWriter

        strategy_ports = {'omr': 8081, 'ramp': 8082, 'mp': 8083, 'cscm': 8084}
        metrics_port_env = os.getenv('METRICS_PORT')
        if metrics_port_env:
            metrics_port = int(metrics_port_env)
        elif args.strategy in strategy_ports:
            metrics_port = strategy_ports[args.strategy]
        else:
            raise ValueError(
                f"Strategy '{args.strategy}' has no default metrics port. "
                f"Set METRICS_PORT env var or add to strategy_ports table."
            )

        metrics_registry = MetricsRegistry(strategy=args.strategy)
        start_metrics_server(metrics_registry, port=metrics_port)

        # Snapshot writer -- separate dir from trading state JSONs to avoid
        # filename collisions or accidental cleanup of durable state.
        from src.settings import get_local_storage_dir
        snapshot_dir = os.path.join(get_local_storage_dir(), 'metrics_snapshots')
        SnapshotWriter(metrics_registry, snapshot_dir=snapshot_dir).start_background()

        # Emit static starting-capital gauge so the Strategy Capital dashboard
        # panel can render the dashed reference line. CSCM does this in its
        # own sidecar at run_cscm_live.py:202 -- mirror the pattern here for
        # the stock strategies (RAMP/OMR/MP) running through this runner.
        if args.initial_capital:
            from src.monitoring.hooks import update_strategy_initial_capital
            update_strategy_initial_capital(metrics_registry, float(args.initial_capital))

        logger.info(f"Metrics enabled on port {metrics_port}")

    try:
        # Initialize broker via routing config
        logger.info("Loading broker routing...")
        try:
            routing = load_broker_routing()
            broker = routing[args.strategy]
            broker_name = routing.get_broker_name(args.strategy)
        except Exception as e:
            logger.error(f"[Routing] Failed to load broker routing: {e}")
            return 1

        logger.info(f"[Routing] {args.strategy} execution broker: {broker_name}")

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

        # Pre-flight reconciliation for state-tracked strategies
        if args.strategy in ('omr', 'ramp', 'mp'):
            preflight_strategies = [args.strategy]
        elif args.strategy == 'multi':
            preflight_strategies = ['omr', 'ramp', 'mp']
        else:
            preflight_strategies = []

        if preflight_strategies:
            preflight_state_manager = StrategyStateManager()
            for _strategy in preflight_strategies:
                preflight_rc = preflight_reconcile(
                    strategy=_strategy,
                    broker=broker,
                    broker_name=broker_name,
                    state_manager=preflight_state_manager,
                    force_start=args.force_start,
                )
                if preflight_rc != 0:
                    return preflight_rc
        else:
            logger.info(
                f"[Reconcile] Pre-flight skipped for strategy={args.strategy} "
                "(no state-tracked positions to check)"
            )

        # Check if streaming is enabled via environment variable
        use_streaming = os.getenv('USE_STREAMING', 'false').lower() in ('true', '1', 'yes')
        streaming_feed = os.getenv('STREAMING_FEED', 'iex')

        logger.info("")
        if use_streaming:
            # Create AlpacaMarketData (unified Alpaca historical + streaming).
            # Architectural invariant: data comes from Alpaca, execution from the
            # routed broker (e.g. IBKR). See src/streaming/alpaca_market_data.py.
            logger.info("=" * 80)
            logger.info("STREAMING DATA ENABLED (Alpaca historical + Alpaca streaming)")
            logger.info("=" * 80)
            logger.info(f"Creating AlpacaMarketData with {streaming_feed.upper()} feed...")

            from src.streaming.alpaca_market_data import AlpacaMarketData

            # Collect all symbols from enabled strategies
            all_symbols = []

            if args.strategy == 'omr' or args.strategy == 'multi':
                omr_symbols = omr_config.symbols if omr_config else ETFUniverse.LEVERAGED_3X
                all_symbols.extend(omr_symbols)
                logger.info(f"  OMR symbols: {len(omr_symbols)}")

            if args.strategy == 'ramp' or args.strategy == 'multi':
                import pandas as pd
                ramp_symbols = pd.read_csv('config/universes/sp500-2025.csv')['Symbol'].tolist()
                all_symbols.extend(ramp_symbols)
                logger.info(f"  RAMP symbols: {len(ramp_symbols)}")

            if args.strategy in ['ma', 'triple-ma', 'mp']:
                all_symbols.extend(symbols)
                logger.info(f"  Strategy symbols: {len(symbols)}")

            # Always include SPY (needed for regime detection)
            all_symbols.append('SPY')

            # Deduplicate symbols
            all_symbols = list(set(all_symbols))

            logger.info(f"Total unique symbols: {len(all_symbols)}")
            logger.info("Starting WebSocket connection...")

            # Create unified Alpaca market data (historical REST + streaming WebSocket)
            data_provider = AlpacaMarketData(
                api_key=api_key,
                secret_key=secret_key,
                feed=streaming_feed,
            )

            # Start WebSocket and subscribe to all symbols
            data_provider.start(all_symbols)

            logger.success(f"Streaming enabled: {len(all_symbols)} symbols")
            logger.info(f"Feed: {streaming_feed.upper()}")
            logger.info("Historical data source: Alpaca REST (via AlpacaMarketData)")
            logger.info("=" * 80)
            logger.info("")
        else:
            # Create data provider with Alpaca -> yfinance fallback (polling)
            logger.info("Creating data provider (Alpaca -> yfinance fallback)...")
            logger.info("Note: Streaming is disabled. Set USE_STREAMING=true in .env to enable.")
            data_provider = create_data_provider(broker=broker)
            logger.success(f"Data provider ready: {data_provider.name}")

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
                omr_config=omr_config,  # Use production config
                data_provider=data_provider,  # Alpaca -> yfinance fallback
                max_capital_usd=args.initial_capital,
                broker_name=broker_name
            )
        elif args.strategy == 'mp':
            # Momentum Protection strategy
            adapter = create_mp_adapter(
                broker,
                position_size=args.position_size if args.position_size != 0.05 else 0.065,
                top_n=args.max_positions if args.max_positions != 3 else 10,
                data_provider=data_provider,  # Alpaca -> yfinance fallback
                broker_name=broker_name
            )
        elif args.strategy == 'ramp':
            # Regime-Aware Momentum Protection strategy
            adapter = create_ramp_adapter(
                broker,
                data_provider=data_provider,  # Alpaca -> yfinance fallback
                initial_capital=args.initial_capital,
                metrics_registry=metrics_registry,
                broker_name=broker_name
            )
        elif args.strategy == 'multi':
            # Multi-strategy mode - run all enabled strategies
            logger.info("Multi-strategy mode - checking enabled strategies...")
            state_manager = StrategyStateManager()
            enabled = state_manager.get_enabled_strategies()

            if not enabled:
                logger.error("No strategies enabled. Use toggle_strategy.sh to enable strategies.")
                return 1

            logger.info(f"Enabled strategies: {', '.join(enabled)}")

            # For now, create OMR adapter if enabled (primary strategy)
            # TODO: Implement full multi-adapter runner
            if 'omr' in enabled:
                adapter = create_omr_adapter(broker, omr_config=omr_config, data_provider=data_provider, broker_name=broker_name)
            elif 'mp' in enabled:
                adapter = create_mp_adapter(broker, data_provider=data_provider, broker_name=broker_name)
            elif 'ramp' in enabled:
                adapter = create_ramp_adapter(broker, data_provider=data_provider, metrics_registry=metrics_registry, broker_name=broker_name)
            else:
                logger.error("No supported strategy enabled")
                return 1

            logger.warning("Note: Full multi-strategy support coming soon. Currently runs primary enabled strategy.")
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

            # Warn if assuming market open
            if args.assume_market_open:
                logger.warning("=" * 60)
                logger.warning("ASSUME MARKET OPEN: market-hours checks bypassed")
                logger.warning("For testing only -- strategy signals may not fire outside market hours")
                logger.warning("=" * 60)

            runner = LiveTradingRunner(
                adapter,
                check_interval=args.check_interval,
                log_dir=log_dir,
                enable_intraday_prefetch=enable_prefetch,
                assume_market_open=args.assume_market_open,
                metrics_registry=metrics_registry,
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
