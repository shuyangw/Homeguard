"""
GUI wrapper around SweepRunner for real-time backtest monitoring.

Provides queue-based communication between worker threads and UI thread.
"""

import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from backtesting.base.strategy import BaseStrategy
from backtesting.engine.backtest_engine import BacktestEngine, PortfolioType
from backtesting.engine.sweep_runner import SweepRunner
from backtesting.engine.portfolio_simulator import Portfolio
from backtesting.utils.risk_config import RiskConfig
import pandas as pd

from gui.utils.error_logger import log_error, log_info, log_exception
from config import get_log_output_dir, get_tearsheet_frequency
from backtesting.engine.results_aggregator import ResultsAggregator
from backtesting.engine.trade_logger import TradeLogger
from backtesting.engine.multi_symbol_metrics import MultiSymbolMetrics
from backtesting.engine.multi_symbol_charts import MultiSymbolChartGenerator
from backtesting.engine.multi_symbol_html_viewer import MultiSymbolHTMLViewer

if TYPE_CHECKING:
    from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio


@dataclass
class ProgressUpdate:
    """Progress update from worker thread"""
    symbol: str
    message: str
    progress: float
    timestamp: datetime


@dataclass
class LogMessage:
    """Log message from worker thread"""
    symbol: str
    message: str
    level: str
    timestamp: datetime


@dataclass
class WorkerStatusUpdate:
    """Status update for a specific worker thread (Phase 3)"""
    worker_id: int          # 0 to max_workers-1
    symbol: Optional[str]   # Symbol being processed, None if idle
    status: str             # "started"|"idle"
    timestamp: datetime


@dataclass
class WorkerLogMessage:
    """Log message from a specific worker thread (Phase 3)"""
    worker_id: int
    message: str
    level: str  # "info"|"success"|"warning"|"error"
    timestamp: datetime


class GUIBacktestController:
    """
    Thin wrapper around SweepRunner for GUI integration.

    Provides queue-based interface for real-time progress updates and log streaming.

    Thread Safety:
    - Main thread creates controller
    - Background thread runs SweepRunner
    - Callbacks push to queues (thread-safe)
    - Main thread polls queues via get_updates()

    Example:
        controller = GUIBacktestController(max_workers=8)

        controller.start_backtests(
            strategy=MovingAverageCrossover(fast=10, slow=50),
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_capital=100000.0,
            fees=0.001
        )

        while controller.is_running():
            updates = controller.get_updates()
            # Update UI with progress, logs, etc.
            time.sleep(0.1)

        results = controller.get_results()
        portfolios = controller.get_portfolios()
    """

    def __init__(self, max_workers: int = 8):
        """
        Initialize GUI backtest controller.

        Args:
            max_workers: Number of parallel workers (1-16, default: 8)
        """
        self.max_workers = max_workers

        # Queues (per symbol)
        self.progress_queues: Dict[str, Queue] = {}
        self.log_queues: Dict[str, Queue] = {}

        # State tracking
        self.status: Dict[str, str] = {}  # "pending"|"running"|"completed"|"failed"
        self.portfolios: Dict[str, PortfolioType] = {}
        self.stats: Dict[str, pd.Series] = {}
        self.errors: Dict[str, str] = {}

        # Background thread
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # SweepRunner instance (created when starting)
        self._runner: Optional[SweepRunner] = None

        # Phase 3: Worker tracking
        self.worker_log_queues: Dict[int, Queue] = {}          # worker_id -> log queue
        self.worker_assignments: Dict[int, Optional[str]] = {} # worker_id -> current symbol (None if idle)
        self.available_worker_ids: Queue = Queue()             # Pool of available IDs
        self.worker_status_queue: Queue = Queue()              # Status change updates
        self._worker_lock = threading.Lock()                   # Thread-safe ID assignment

    def start_backtests(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        fees: float = 0.001,
        risk_profile: str = "Moderate",
        generate_full_output: bool = True,
        portfolio_mode: str = "Single-Symbol",
        position_sizing_method: str = "equal_weight",
        rebalancing_frequency: str = "never",
        rebalancing_threshold_pct: float = 0.05
    ):
        """
        Start backtests in background thread.

        Args:
            strategy: Strategy instance to test
            symbols: List of symbols
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            initial_capital: Starting capital (default: 100000)
            fees: Trading fees as decimal (default: 0.001 = 0.1%)
            risk_profile: Risk management profile (default: "Moderate")
            generate_full_output: Generate tearsheets, charts, logs (default: True)
            portfolio_mode: "Single-Symbol" or "Multi-Symbol Portfolio" (default: "Single-Symbol")
            position_sizing_method: Position sizing for multi-symbol portfolio (default: "equal_weight")
            rebalancing_frequency: Rebalancing frequency for portfolio (default: "never")
            rebalancing_threshold_pct: Drift threshold for rebalancing (default: 0.05)
        """
        try:
            log_info(f"GUIController: Starting backtests for {len(symbols)} symbols")
            log_info(f"GUIController: Workers={self.max_workers}, Capital=${initial_capital:,.0f}, Fees={fees}, Risk={risk_profile}")

            # Reset state
            self.progress_queues.clear()
            self.log_queues.clear()
            self.status.clear()
            self.portfolios.clear()
            self.stats.clear()
            self.errors.clear()

            # Phase 3: Initialize worker tracking
            self.worker_log_queues.clear()
            self.worker_assignments.clear()
            while not self.available_worker_ids.empty():
                self.available_worker_ids.get()

            for worker_id in range(self.max_workers):
                self.worker_log_queues[worker_id] = Queue()
                self.worker_assignments[worker_id] = None
                self.available_worker_ids.put(worker_id)

            log_info(f"GUIController: Initialized {self.max_workers} worker log streams")

            # Store parameters for export
            self._generate_full_output = generate_full_output
            self._strategy_name = strategy.__class__.__name__
            self._start_date = start_date
            self._end_date = end_date
            self._initial_capital = initial_capital
            self._fees = fees
            self._portfolio_mode = portfolio_mode

            # Create queues for each symbol
            # For Multi-Symbol Portfolio mode, create single "Portfolio" entry
            # For Single-Symbol mode, create entry per symbol
            if portfolio_mode == "Multi-Symbol Portfolio":
                # Single portfolio entry
                portfolio_key = "Portfolio"
                self.progress_queues[portfolio_key] = Queue()
                self.log_queues[portfolio_key] = Queue()
                self.status[portfolio_key] = "pending"
                # Store symbols list for later use
                self._portfolio_symbols = symbols
            else:
                # Individual symbol entries
                for symbol in symbols:
                    self.progress_queues[symbol] = Queue()
                    self.log_queues[symbol] = Queue()
                    self.status[symbol] = "pending"

            # Map risk profile to RiskConfig
            risk_config_map = {
                "Conservative": RiskConfig.conservative(),
                "Moderate": RiskConfig.moderate(),
                "Aggressive": RiskConfig.aggressive(),
                "Disabled": RiskConfig.disabled()
            }
            risk_config = risk_config_map.get(risk_profile, RiskConfig.moderate())

            # Add portfolio settings to risk_config
            risk_config.portfolio_sizing_method = position_sizing_method
            risk_config.rebalancing_frequency = rebalancing_frequency
            risk_config.rebalancing_threshold_pct = rebalancing_threshold_pct

            # Create BacktestEngine
            engine = BacktestEngine(
                initial_capital=initial_capital,
                fees=fees,
                risk_config=risk_config
            )

            # Create SweepRunner with callbacks
            self._runner = SweepRunner(
                engine=engine,
                max_workers=self.max_workers,
                show_progress=False,  # We handle progress via callbacks
                on_symbol_start=self._on_symbol_start,
                on_symbol_progress=self._on_symbol_progress,
                on_symbol_complete=self._on_symbol_complete,
                on_symbol_error=self._on_symbol_error
            )

            # Run in background thread
            self._running = True
            self._thread = threading.Thread(
                target=self._run_backtests,
                args=(strategy, symbols, start_date, end_date),
                daemon=True
            )
            self._thread.start()

        except Exception as e:
            log_exception(e, "GUIController: Failed to start backtests")
            self._running = False
            raise

    def _run_backtests(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str
    ):
        """
        Run backtest in background thread.

        For Single-Symbol mode: Uses SweepRunner to test each symbol independently.
        For Multi-Symbol Portfolio mode: Calls engine.run() once with portfolio_mode='multi'.

        Executed in worker thread - DO NOT access UI from here.
        """
        try:
            # Route based on portfolio mode
            if self._portfolio_mode == "Multi-Symbol Portfolio":
                log_info("GUIController: Running multi-symbol portfolio backtest")
                self._run_multi_symbol_portfolio(strategy, symbols, start_date, end_date)
            else:
                log_info("GUIController: Background thread starting sweep")
                # Run sweep (blocks until complete)
                results = self._runner.run_sweep(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    parallel=True  # Always use parallel for GUI
                )

                # Store results
                self.stats = results

                # Get portfolios from runner (merge with any already stored by callbacks)
                runner_portfolios = self._runner.get_portfolios()
                log_info(f"GUIController: Runner has {len(runner_portfolios)} portfolios")
                log_info(f"GUIController: Callbacks stored {len(self.portfolios)} portfolios")

                # Use runner portfolios as they should be complete
                if runner_portfolios:
                    self.portfolios = runner_portfolios
                    log_info(f"GUIController: Using runner portfolios")
                else:
                    log_info(f"GUIController: Runner portfolios empty, keeping callback portfolios")

                log_info(f"GUIController: Sweep completed - {len(results)} results, {len(self.portfolios)} portfolios stored")

            # Generate full output if enabled
            if self._generate_full_output:
                try:
                    # Create output directory with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    symbols_str = '_'.join(symbols[:3])  # First 3 symbols

                    if self._portfolio_mode == "Multi-Symbol Portfolio":
                        run_name = f"{timestamp}_{self._strategy_name}_PORTFOLIO_{symbols_str}_GUI"
                    else:
                        run_name = f"{timestamp}_{self._strategy_name}_{symbols_str}_GUI"

                    output_dir = get_log_output_dir() / run_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    log_info(f"GUIController: Generating full output to {output_dir}")

                    # Route output generation based on mode
                    if self._portfolio_mode == "Multi-Symbol Portfolio":
                        # Multi-Symbol Portfolio Mode: Export single portfolio
                        self._export_portfolio_results(output_dir, timestamp, symbols)

                        # Mark portfolio as completed AFTER exports are done (single entry)
                        portfolio_key = "Portfolio"
                        symbols_display = f"Portfolio ({', '.join(symbols)})"
                        self.status[portfolio_key] = "completed"
                        self.progress_queues[portfolio_key].put(ProgressUpdate(
                            symbol=symbols_display,
                            message="Complete",
                            progress=1.0,
                            timestamp=datetime.now()
                        ))
                        self.log_queues[portfolio_key].put(LogMessage(
                            symbol=symbols_display,
                            message="All reports generated successfully",
                            level="success",
                            timestamp=datetime.now()
                        ))
                    else:
                        # Single-Symbol Mode: Export sweep results
                        self._export_sweep_results(output_dir, timestamp, symbols)

                    log_info(f"GUIController: Full output generated at {output_dir}")
                except Exception as export_error:
                    log_exception(export_error, "GUIController: Error generating full output")
                    # Don't fail the backtest if export fails
                    log_error("Continuing despite export error")
                    # Still mark as completed even if export fails
                    if self._portfolio_mode == "Multi-Symbol Portfolio":
                        portfolio_key = "Portfolio"
                        symbols_display = f"Portfolio ({', '.join(symbols)})"
                        if self.status.get(portfolio_key) != "completed":
                            self.status[portfolio_key] = "completed"
                            self.progress_queues[portfolio_key].put(ProgressUpdate(
                                symbol=symbols_display,
                                message="Complete (export errors)",
                                progress=1.0,
                                timestamp=datetime.now()
                            ))
            else:
                # No full output - mark as completed immediately for portfolio mode
                if self._portfolio_mode == "Multi-Symbol Portfolio":
                    portfolio_key = "Portfolio"
                    symbols_display = f"Portfolio ({', '.join(symbols)})"
                    self.status[portfolio_key] = "completed"
                    self.progress_queues[portfolio_key].put(ProgressUpdate(
                        symbol=symbols_display,
                        message="Complete",
                        progress=1.0,
                        timestamp=datetime.now()
                    ))

        except Exception as e:
            log_exception(e, "GUIController: Error in background thread")
            # Mark all non-completed items as failed
            if self._portfolio_mode == "Multi-Symbol Portfolio":
                # Portfolio mode: single entry
                portfolio_key = "Portfolio"
                if self.status.get(portfolio_key) not in ["completed", "failed"]:
                    self.status[portfolio_key] = "failed"
                    self.errors[portfolio_key] = str(e)
            else:
                # Single-symbol mode: loop over symbols
                for symbol in symbols:
                    if self.status.get(symbol) not in ["completed", "failed"]:
                        self.status[symbol] = "failed"
                        self.errors[symbol] = str(e)

        finally:
            self._running = False
            log_info("GUIController: Background thread finished")

    def _run_multi_symbol_portfolio(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str
    ):
        """
        Run multi-symbol portfolio backtest (single portfolio holding all symbols).

        Args:
            strategy: Strategy instance
            symbols: List of symbols to hold in portfolio
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
        """
        worker_id = None
        try:
            # Claim a worker for portfolio mode logging
            worker_id = self._claim_worker_id()
            # Store for export phase
            self._portfolio_worker_id = worker_id

            # Log to worker panel - Portfolio Mode indicator
            self._worker_log(worker_id, "╔════════════════════════════════════╗", "info")
            self._worker_log(worker_id, "║   PORTFOLIO MODE - MULTI-ASSET     ║", "info")
            self._worker_log(worker_id, "╚════════════════════════════════════╝", "info")
            self._worker_log(worker_id, "", "info")
            self._worker_log(worker_id, f"Symbols: {', '.join(symbols)}", "info")
            self._worker_log(worker_id, f"Period: {start_date} to {end_date}", "info")
            self._worker_log(worker_id, f"Capital: ${self._initial_capital:,.2f} | Fees: {self._fees*100:.2f}%", "info")
            self._worker_log(worker_id, "", "info")

            # Mark portfolio as running (single entry)
            portfolio_key = "Portfolio"
            symbols_display = f"Portfolio ({', '.join(symbols)})"

            self.status[portfolio_key] = "running"
            self.progress_queues[portfolio_key].put(ProgressUpdate(
                symbol=symbols_display,
                message="Portfolio Mode",
                progress=0.0,
                timestamp=datetime.now()
            ))

            # Step 1: Load data
            self._worker_log(worker_id, "Step 1/4: Loading data...", "info")
            self.progress_queues[portfolio_key].put(ProgressUpdate(
                symbol=symbols_display,
                message="Loading data...",
                progress=0.1,
                timestamp=datetime.now()
            ))

            # Step 2: Generate signals
            self._worker_log(worker_id, "Step 2/4: Generating signals...", "info")
            self.progress_queues[portfolio_key].put(ProgressUpdate(
                symbol=symbols_display,
                message="Generating signals...",
                progress=0.3,
                timestamp=datetime.now()
            ))

            # Step 3: Run portfolio simulation
            self._worker_log(worker_id, "Step 3/4: Running portfolio simulation...", "info")
            self.progress_queues[portfolio_key].put(ProgressUpdate(
                symbol=symbols_display,
                message="Simulating portfolio...",
                progress=0.5,
                timestamp=datetime.now()
            ))

            # Run backtest with portfolio_mode='multi'
            portfolio = self._runner.engine.run(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                portfolio_mode='multi'
            )

            if portfolio is None:
                raise ValueError("Portfolio backtest returned None")

            # Get stats
            stats = portfolio.stats()

            if stats is None:
                raise ValueError("Portfolio stats returned None")

            # Store results (use first symbol as key for aggregated portfolio)
            portfolio_key = "Portfolio"
            self.portfolios[portfolio_key] = portfolio
            self.stats[portfolio_key] = stats

            # Step 4: Calculate metrics
            return_pct = stats.get('Total Return [%]', 0)
            sharpe = stats.get('Sharpe Ratio', 0)
            max_dd = stats.get('Max Drawdown [%]', 0)
            total_trades = stats.get('Total Trades', 0)

            self._worker_log(worker_id, "", "info")
            self._worker_log(worker_id, "✓ Portfolio Simulation Complete", "success")
            self._worker_log(worker_id, f"  Return: {return_pct:.2f}%", "success" if return_pct >= 0 else "error")
            self._worker_log(worker_id, f"  Sharpe: {sharpe:.2f}", "info")
            self._worker_log(worker_id, f"  Max DD: {max_dd:.2f}%", "info")
            self._worker_log(worker_id, f"  Trades: {total_trades}", "info")
            self._worker_log(worker_id, "", "info")
            self._worker_log(worker_id, "Step 4/4: Preparing reports...", "info")

            # Update portfolio progress (single entry)
            self.progress_queues[portfolio_key].put(ProgressUpdate(
                symbol=symbols_display,
                message="Preparing reports...",
                progress=0.95,
                timestamp=datetime.now()
            ))
            self.log_queues[portfolio_key].put(LogMessage(
                symbol=symbols_display,
                message=f"Portfolio: {return_pct:.2f}% return, {sharpe:.2f} Sharpe, {total_trades} trades",
                level="success" if return_pct >= 0 else "info",
                timestamp=datetime.now()
            ))

            log_info(f"GUIController: Multi-symbol portfolio backtest completed")

        except Exception as e:
            log_exception(e, "GUIController: Multi-symbol portfolio backtest failed")

            # Log error to worker panel
            if worker_id is not None:
                self._worker_log(worker_id, "", "error")
                self._worker_log(worker_id, f"✗ Portfolio Backtest Failed", "error")
                self._worker_log(worker_id, f"  Error: {str(e)}", "error")

            # Mark portfolio as failed (single entry)
            portfolio_key = "Portfolio"
            self.status[portfolio_key] = "failed"
            self.errors[portfolio_key] = str(e)
            self.log_queues[portfolio_key].put(LogMessage(
                symbol=f"Portfolio ({', '.join(symbols)})",
                message=f"Error: {str(e)}",
                level="error",
                timestamp=datetime.now()
            ))
            raise
        finally:
            # Release worker
            if worker_id is not None:
                self._release_worker_id(worker_id)

    # Callback handlers (push to queues)

    def _on_symbol_start(self, symbol: str):
        """
        Called when symbol starts (from worker thread).

        Thread Safety: Queue.put() is thread-safe.
        """
        # Phase 3: Claim worker ID for this thread
        worker_id = self._claim_worker_id()

        # Store thread -> worker mapping
        thread_id = threading.current_thread().ident
        with self._worker_lock:
            if not hasattr(self, '_thread_to_worker'):
                self._thread_to_worker = {}
            self._thread_to_worker[thread_id] = worker_id

        # Update worker assignment
        self._update_worker_assignment(worker_id, symbol)

        # Log detailed start info (match terminal output)
        self._worker_log(worker_id, f"=== {symbol} ===", "info")
        self._worker_log(worker_id, f"Period: {self._start_date} to {self._end_date}", "info")
        self._worker_log(worker_id, f"Capital: ${self._runner.engine.initial_capital:,.2f} | Fees: {self._runner.engine.fees*100:.2f}%", "info")

        self.status[symbol] = "running"
        self.progress_queues[symbol].put(ProgressUpdate(
            symbol=symbol,
            message="Starting...",
            progress=0.0,
            timestamp=datetime.now()
        ))

    def _on_symbol_progress(self, symbol: str, message: str, progress: float):
        """
        Called during symbol execution (from worker thread).

        Args:
            symbol: Symbol being processed
            message: Progress message (e.g., "Loading data...")
            progress: Progress value 0.0-1.0
        """
        # Phase 3: Log progress to worker queue
        thread_id = threading.current_thread().ident
        with self._worker_lock:
            worker_id = getattr(self, '_thread_to_worker', {}).get(thread_id)

        if worker_id is not None:
            # Log progress with more context
            if "Loading" in message:
                self._worker_log(worker_id, f"{message}", "info")
            elif "Computing" in message:
                self._worker_log(worker_id, f"{message}", "info")
            else:
                self._worker_log(worker_id, f"{message}", "info")

        self.progress_queues[symbol].put(ProgressUpdate(
            symbol=symbol,
            message=message,
            progress=progress,
            timestamp=datetime.now()
        ))

    def _on_symbol_complete(self, symbol: str, portfolio: Portfolio, stats: pd.Series):
        """
        Called when symbol completes (from worker thread).

        Args:
            symbol: Symbol that completed
            portfolio: Portfolio object with backtest results
            stats: Statistics Series
        """
        # Phase 3: Get worker ID and log completion
        thread_id = threading.current_thread().ident
        with self._worker_lock:
            worker_id = getattr(self, '_thread_to_worker', {}).get(thread_id)

        self.status[symbol] = "completed"
        self.portfolios[symbol] = portfolio
        self.stats[symbol] = stats

        self.progress_queues[symbol].put(ProgressUpdate(
            symbol=symbol,
            message="Complete",
            progress=1.0,
            timestamp=datetime.now()
        ))

        # Send completion log message and detailed stats (match terminal output)
        if stats is not None:
            return_pct = stats.get('Total Return [%]', 0)
            sharpe = stats.get('Sharpe Ratio', 0)
            max_dd = stats.get('Max Drawdown [%]', 0)
            win_rate = stats.get('Win Rate [%]', 0)
            total_trades = stats.get('Total Trades', 0)
            final_value = stats.get('End Value', 0)

            self.log_queues[symbol].put(LogMessage(
                symbol=symbol,
                message=f"Complete - Return: {return_pct:.2f}%, Sharpe: {sharpe:.2f}",
                level="success" if return_pct >= 0 else "info",
                timestamp=datetime.now()
            ))

            # Phase 3: Log detailed completion stats to worker queue (match terminal output)
            if worker_id is not None:
                # Extract data information from portfolio (match engine's "Loaded X bars" log)
                try:
                    if portfolio is not None and hasattr(portfolio, 'wrapper'):
                        wrapper = portfolio.wrapper
                        if hasattr(wrapper, 'index'):
                            index = wrapper.index
                            num_bars = len(index)
                            start_date = index[0].strftime('%Y-%m-%d') if len(index) > 0 else 'N/A'
                            end_date = index[-1].strftime('%Y-%m-%d') if len(index) > 0 else 'N/A'
                            self._worker_log(worker_id, f"Loaded {num_bars} bars from {start_date} to {end_date}", "info")
                except Exception:
                    pass  # Skip if data extraction fails

                # Results summary
                level = "success" if return_pct >= 0 else "info"
                self._worker_log(worker_id, f"--- {symbol} Results ---", level)

                # Total Return (colored)
                if return_pct >= 0:
                    self._worker_log(worker_id, f"Total Return:    {return_pct:+.2f}%", "success")
                else:
                    self._worker_log(worker_id, f"Total Return:    {return_pct:.2f}%", "error")

                # Sharpe Ratio (colored)
                if sharpe >= 1.0:
                    self._worker_log(worker_id, f"Sharpe Ratio:    {sharpe:.2f}", "success")
                elif sharpe >= 0:
                    self._worker_log(worker_id, f"Sharpe Ratio:    {sharpe:.2f}", "info")
                else:
                    self._worker_log(worker_id, f"Sharpe Ratio:    {sharpe:.2f}", "error")

                # Max Drawdown (colored)
                if max_dd > -10:
                    self._worker_log(worker_id, f"Max Drawdown:    {max_dd:.2f}%", "info")
                elif max_dd > -20:
                    self._worker_log(worker_id, f"Max Drawdown:    {max_dd:.2f}%", "warning")
                else:
                    self._worker_log(worker_id, f"Max Drawdown:    {max_dd:.2f}%", "error")

                # Win Rate (colored)
                if win_rate >= 50:
                    self._worker_log(worker_id, f"Win Rate:        {win_rate:.2f}%", "success")
                else:
                    self._worker_log(worker_id, f"Win Rate:        {win_rate:.2f}%", "info")

                # Trade count and final value
                self._worker_log(worker_id, f"Total Trades:    {int(total_trades)}", "info")
                self._worker_log(worker_id, f"Final Value:     ${final_value:,.2f}", "info")
                self._worker_log(worker_id, f"{'='*30}", "info")

        # Phase 3: Release worker ID and mark idle
        if worker_id is not None:
            self._update_worker_assignment(worker_id, None)
            self._release_worker_id(worker_id)

            # Remove from thread mapping
            with self._worker_lock:
                if hasattr(self, '_thread_to_worker') and thread_id in self._thread_to_worker:
                    del self._thread_to_worker[thread_id]

    def _on_symbol_error(self, symbol: str, error: Exception):
        """
        Called when symbol fails (from worker thread).

        Args:
            symbol: Symbol that failed
            error: Exception that occurred
        """
        # Phase 3: Get worker ID and log error
        thread_id = threading.current_thread().ident
        with self._worker_lock:
            worker_id = getattr(self, '_thread_to_worker', {}).get(thread_id)

        log_exception(error, f"GUIController: Symbol {symbol} failed")

        self.status[symbol] = "failed"
        self.errors[symbol] = str(error)

        self.log_queues[symbol].put(LogMessage(
            symbol=symbol,
            message=f"Error: {str(error)}",
            level="error",
            timestamp=datetime.now()
        ))

        self.progress_queues[symbol].put(ProgressUpdate(
            symbol=symbol,
            message="Failed",
            progress=0.0,
            timestamp=datetime.now()
        ))

        # Phase 3: Log error to worker queue
        if worker_id is not None:
            self._worker_log(worker_id, f"{symbol}: Error - {str(error)}", "error")

            # Release worker ID and mark idle
            self._update_worker_assignment(worker_id, None)
            self._release_worker_id(worker_id)

            # Remove from thread mapping
            with self._worker_lock:
                if hasattr(self, '_thread_to_worker') and thread_id in self._thread_to_worker:
                    del self._thread_to_worker[thread_id]

    # UI interface methods (called from main thread)

    def get_updates(self) -> Dict[str, Dict]:
        """
        Poll all queues for updates.

        Called from main UI thread periodically (e.g., every 100ms).

        Returns:
            Dict mapping symbol -> {
                'progress': [ProgressUpdate, ...],
                'logs': [LogMessage, ...],
                'status': str
            }

        Thread Safety:
        - Queue.get_nowait() is thread-safe
        - Drains up to 10 progress updates per symbol
        - Drains up to 20 log messages per symbol
        """
        updates = {}

        for symbol in self.progress_queues.keys():
            updates[symbol] = {
                'progress': [],
                'logs': [],
                'status': self.status[symbol]
            }

            # Drain progress queue (non-blocking, limit 10 per poll)
            for _ in range(10):
                try:
                    update = self.progress_queues[symbol].get_nowait()
                    updates[symbol]['progress'].append(update)
                except Empty:
                    break

            # Drain log queue (non-blocking, limit 20 per poll)
            for _ in range(20):
                try:
                    log = self.log_queues[symbol].get_nowait()
                    updates[symbol]['logs'].append(log)
                except Empty:
                    break

        return updates

    def get_worker_updates(self) -> Dict[str, any]:
        """
        Poll worker queues for updates (Phase 3).

        Called from main UI thread periodically (e.g., every 200ms).

        Returns:
            Dict with keys:
                'status_updates': List[WorkerStatusUpdate] - Worker status changes
                'logs': Dict[int, List[WorkerLogMessage]] - Logs per worker

        Thread Safety:
        - Queue.get_nowait() is thread-safe
        - Drains up to 5 status updates per poll
        - Drains up to 20 log messages per worker
        """
        updates = {
            'status_updates': [],
            'logs': {}
        }

        # Drain worker status queue (limit 5 per poll)
        for _ in range(5):
            try:
                status_update = self.worker_status_queue.get_nowait()
                updates['status_updates'].append(status_update)
            except Empty:
                break

        # Drain worker log queues (limit 20 per worker)
        for worker_id in self.worker_log_queues.keys():
            updates['logs'][worker_id] = []
            for _ in range(20):
                try:
                    log_msg = self.worker_log_queues[worker_id].get_nowait()
                    updates['logs'][worker_id].append(log_msg)
                except Empty:
                    break

        return updates

    def is_running(self) -> bool:
        """
        Check if backtests are still running.

        Returns:
            True if any backtests are pending or running
        """
        return self._running

    def get_results(self) -> Dict[str, pd.Series]:
        """
        Get final results (stats).

        Returns:
            Dictionary mapping symbol -> stats Series

        Note:
            Only complete after is_running() returns False.
        """
        return self.stats.copy()

    def get_portfolios(self) -> Dict[str, Portfolio]:
        """
        Get Portfolio objects for charts.

        Returns:
            Dictionary mapping symbol -> Portfolio object

        Note:
            Used for generating equity curves and trade analysis.
            Only complete after is_running() returns False.
        """
        return self.portfolios.copy()

    def get_status(self, symbol: str) -> str:
        """
        Get status for a specific symbol.

        Args:
            symbol: Symbol to get status for

        Returns:
            Status string: "pending", "running", "completed", or "failed"
        """
        return self.status.get(symbol, "pending")

    def get_progress_summary(self) -> Dict[str, int]:
        """
        Get progress summary.

        Returns:
            Dict with keys:
            - 'total': Total number of symbols
            - 'completed': Number of completed symbols
            - 'running': Number of currently running symbols
            - 'pending': Number of pending symbols
            - 'failed': Number of failed symbols
        """
        return {
            'total': len(self.status),
            'completed': sum(1 for s in self.status.values() if s == "completed"),
            'running': sum(1 for s in self.status.values() if s == "running"),
            'pending': sum(1 for s in self.status.values() if s == "pending"),
            'failed': sum(1 for s in self.status.values() if s == "failed")
        }

    def get_errors(self) -> Dict[str, str]:
        """
        Get error messages for failed symbols.

        Returns:
            Dictionary mapping symbol -> error message
        """
        return self.errors.copy()

    def get_tracked_items(self) -> List[str]:
        """
        Get list of tracked items (symbols or portfolio).

        For Single-Symbol mode: Returns list of symbols
        For Multi-Symbol Portfolio mode: Returns ["Portfolio"]

        Returns:
            List of tracked item names

        Note: This should be called AFTER run() to get the correct list
        """
        return list(self.progress_queues.keys())

    # Phase 3: Worker ID management helpers

    def _claim_worker_id(self) -> int:
        """
        Claim an available worker ID from the pool.

        Returns:
            worker_id (0 to max_workers-1)

        Thread Safety: Queue.get() is thread-safe (blocks if none available)
        """
        return self.available_worker_ids.get()

    def _release_worker_id(self, worker_id: int):
        """
        Release a worker ID back to the pool.

        Args:
            worker_id: Worker ID to release

        Thread Safety: Queue.put() is thread-safe
        """
        self.available_worker_ids.put(worker_id)

    def _update_worker_assignment(self, worker_id: int, symbol: Optional[str]):
        """
        Update worker assignment tracking.

        Args:
            worker_id: Worker ID
            symbol: Symbol being processed (None if idle)

        Thread Safety: Protected by _worker_lock
        """
        with self._worker_lock:
            self.worker_assignments[worker_id] = symbol

            # Push status update to queue
            status_update = WorkerStatusUpdate(
                worker_id=worker_id,
                symbol=symbol,
                status="started" if symbol else "idle",
                timestamp=datetime.now()
            )
            self.worker_status_queue.put(status_update)

    def _worker_log(self, worker_id: int, message: str, level: str = "info"):
        """
        Log a message from a worker thread.

        Args:
            worker_id: Worker ID
            message: Log message
            level: Log level ("info"|"success"|"warning"|"error")

        Thread Safety: Queue.put() is thread-safe
        """
        log_msg = WorkerLogMessage(
            worker_id=worker_id,
            message=message,
            level=level,
            timestamp=datetime.now()
        )
        self.worker_log_queues[worker_id].put(log_msg)

    def cancel(self):
        """
        Request cancellation of running backtests.

        This is a cooperative cancellation - already running symbols will
        complete, but no new symbols will start.

        Note:
            Safe to call even if backtests are not running.
        """
        if self._runner:
            self._runner.cancel()

    # Output generation helpers

    def _export_sweep_results(self, output_dir: Path, timestamp: str, symbols: List[str]):
        """
        Export sweep results (single-symbol mode).

        Args:
            output_dir: Output directory path
            timestamp: Timestamp string for filenames
            symbols: List of symbols
        """
        # Aggregate results to DataFrame
        df = ResultsAggregator.aggregate_results(self.stats)

        # Export CSV
        csv_path = output_dir / f"{timestamp}_{self._strategy_name}_sweep_results.csv"
        ResultsAggregator.export_to_csv(df, csv_path, include_summary=True)
        log_info(f"Exported CSV: {csv_path}")

        # Export HTML
        html_path = output_dir / f"{timestamp}_{self._strategy_name}_sweep_results.html"
        title = f"Backtest Sweep: {self._strategy_name} ({len(symbols)} symbols)"
        ResultsAggregator.export_to_html(df, html_path, title=title, portfolios=self.portfolios)
        log_info(f"Exported HTML: {html_path}")

        # Debug logging
        log_info(f"Portfolios available for tearsheet generation: {len(self.portfolios) if self.portfolios else 0}")
        log_info(f"Portfolio symbols: {list(self.portfolios.keys()) if self.portfolios else 'None'}")

        # Export detailed trade logs and tearsheets for each symbol
        if self.portfolios and len(self.portfolios) > 0:
            trades_dir = output_dir / "trades"
            trades_dir.mkdir(exist_ok=True)

            tearsheets_dir = output_dir / "tearsheets"
            tearsheets_dir.mkdir(exist_ok=True)

            for symbol, portfolio in self.portfolios.items():
                if portfolio is None:
                    continue

                symbol_prefix = f"{timestamp}_{symbol}"

                # Export trades CSV
                trades_csv = trades_dir / f"{symbol_prefix}_trades.csv"
                TradeLogger.export_trades_csv(portfolio, trades_csv, symbol=symbol)

                # Export equity curve
                equity_csv = trades_dir / f"{symbol_prefix}_equity_curve.csv"
                TradeLogger.export_equity_curve_csv(portfolio, equity_csv, symbol=symbol)

                # Export portfolio state
                state_csv = trades_dir / f"{symbol_prefix}_portfolio_state.csv"
                TradeLogger.export_portfolio_state_csv(portfolio, state_csv, symbol=symbol)

                # Generate QuantStats tearsheet for this symbol
                try:
                    import quantstats as qs
                    from config import get_tearsheet_frequency

                    tearsheet_path = tearsheets_dir / f"{symbol_prefix}_tearsheet.html"

                    # Get configured frequency for resampling (reduces file size)
                    freq = get_tearsheet_frequency()
                    returns = portfolio.returns(freq=freq)

                    # Log what we're doing
                    freq_label = {
                        None: 'full resolution',
                        'H': 'hourly',
                        'D': 'daily',
                        'W': 'weekly'
                    }.get(freq, freq)
                    log_info(f"Generating {symbol} tearsheet with {freq_label} data ({len(returns)} points)...")

                    qs.reports.html(
                        returns,
                        output=str(tearsheet_path),
                        title=f"{self._strategy_name} - {symbol}",
                        benchmark=None
                    )

                    # Log file size
                    size_mb = tearsheet_path.stat().st_size / 1024 / 1024
                    log_info(f"Generated tearsheet for {symbol}: {tearsheet_path} ({size_mb:.1f} MB)")
                except ImportError:
                    log_error(f"QuantStats not installed - skipping tearsheet for {symbol}")
                except Exception as e:
                    log_exception(e, f"Failed to generate tearsheet for {symbol}")

            log_info(f"Trade logs exported to: {trades_dir}")
            log_info(f"Tearsheets exported to: {tearsheets_dir}")
        else:
            log_error("WARNING: No portfolios available for tearsheet generation!")
            log_error("This means portfolio objects were not stored during the sweep.")
            log_error("Tearsheets and detailed trade logs will NOT be generated.")

    def _export_portfolio_results(self, output_dir: Path, timestamp: str, symbols: List[str]):
        """
        Export multi-symbol portfolio results.

        Args:
            output_dir: Output directory path
            timestamp: Timestamp string for filenames
            symbols: List of symbols in portfolio
        """
        # Get worker ID if available (for logging)
        worker_id = getattr(self, '_portfolio_worker_id', None)

        if worker_id is not None:
            self._worker_log(worker_id, "Generating reports...", "info")

        # Get the portfolio (stored with key "Portfolio")
        portfolio = self.portfolios.get("Portfolio")
        stats = self.stats.get("Portfolio")

        if portfolio is None or stats is None:
            log_error("No portfolio results to export")
            return

        # Export portfolio stats as CSV (stats is now pd.Series)
        stats_path = output_dir / f"{timestamp}_{self._strategy_name}_portfolio_stats.csv"
        stats.to_csv(stats_path, header=True)
        log_info(f"Exported portfolio stats CSV: {stats_path}")

        # Export equity curve using VectorBT-compatible .value() method
        import time
        start = time.time()
        equity_csv = output_dir / f"{timestamp}_Portfolio_equity_curve.csv"
        TradeLogger.export_equity_curve_csv(portfolio, equity_csv, symbol="Portfolio")
        log_info(f"Exported equity curve: {equity_csv} ({time.time() - start:.1f}s)")

        # Export trades using VectorBT-compatible .trades property
        start = time.time()
        trades_csv = output_dir / f"{timestamp}_Portfolio_trades.csv"
        TradeLogger.export_trades_csv(portfolio, trades_csv, symbol="Portfolio")
        log_info(f"Exported trades: {trades_csv} ({time.time() - start:.1f}s)")

        # Export portfolio state using VectorBT-compatible .wrapper property
        start = time.time()
        state_csv = output_dir / f"{timestamp}_Portfolio_portfolio_state.csv"
        TradeLogger.export_portfolio_state_csv(portfolio, state_csv, symbol="Portfolio")
        log_info(f"Exported portfolio state: {state_csv} ({time.time() - start:.1f}s)")

        # Export rebalancing events if available (MultiAssetPortfolio-specific)
        if hasattr(portfolio, 'rebalancing_events') and portfolio.rebalancing_events:
            rebal_csv = output_dir / f"{timestamp}_Portfolio_rebalancing_events.csv"
            rebal_df = pd.DataFrame(portfolio.rebalancing_events)
            rebal_df.to_csv(rebal_csv, index=False)
            log_info(f"Exported rebalancing events: {rebal_csv}")

        log_info("=" * 80)
        log_info("PORTFOLIO MODE EXPORT STARTED - CODE VERSION 2.0")
        log_info(f"Portfolio type: {type(portfolio).__name__}")
        log_info(f"Portfolio has position_count_history: {hasattr(portfolio, 'position_count_history')}")
        log_info("=" * 80)

        # Send progress update to portfolio queue
        portfolio_key = "Portfolio"
        symbols_display = f"Portfolio ({', '.join(symbols)})"
        if portfolio_key in self.log_queues:
            self.log_queues[portfolio_key].put(LogMessage(
                symbol=symbols_display,
                message="Generating QuantStats tearsheet...",
                level="info",
                timestamp=datetime.now()
            ))

        # Generate portfolio tearsheet using QuantStats and VectorBT-compatible .returns() method
        try:
            import quantstats as qs

            tearsheet_start = time.time()
            tearsheet_path = output_dir / f"{timestamp}_{self._strategy_name}_portfolio_tearsheet.html"
            symbols_str = ", ".join(symbols)

            # Get configured frequency for resampling (reduces file size)
            freq = get_tearsheet_frequency()
            returns = portfolio.returns(freq=freq)

            # Log what we're doing
            freq_label = {
                None: 'full resolution',
                'H': 'hourly',
                'D': 'daily',
                'W': 'weekly'
            }.get(freq, freq)
            log_info(f"Generating portfolio tearsheet with {freq_label} data ({len(returns)} points)...")

            # Generate tearsheet
            qs.reports.html(
                returns,
                output=str(tearsheet_path),
                title=f"{self._strategy_name} - Multi-Symbol Portfolio ({symbols_str})",
                benchmark=None
            )
            tearsheet_time = time.time() - tearsheet_start

            # Log file size
            size_mb = tearsheet_path.stat().st_size / 1024 / 1024
            log_info(f"Exported portfolio tearsheet: {tearsheet_path} ({size_mb:.1f} MB, {tearsheet_time:.1f}s)")
        except ImportError:
            log_error("QuantStats not installed - skipping tearsheet generation")
            log_error("Install with: pip install quantstats")
        except Exception as e:
            log_exception(e, "Failed to generate tearsheet")
            log_error("Continuing without tearsheet")

        # Generate comprehensive multi-symbol portfolio report with metrics and charts
        try:
            # Check if portfolio is MultiAssetPortfolio (has the new analytics attributes)
            if not hasattr(portfolio, 'position_count_history'):
                log_info("Portfolio is not MultiAssetPortfolio - skipping enhanced metrics")
                raise AttributeError("Portfolio missing multi-symbol attributes")

            log_info(f"Portfolio has {len(portfolio.position_count_history)} position count entries")
            log_info(f"Portfolio has {len(portfolio.symbol_weights_history)} weight history entries")
            log_info(f"Portfolio has {len(portfolio.closed_positions)} closed positions")

            # Send progress update to portfolio queue
            portfolio_key = "Portfolio"
            symbols_display = f"Portfolio ({', '.join(symbols)})"
            if portfolio_key in self.log_queues:
                self.log_queues[portfolio_key].put(LogMessage(
                    symbol=symbols_display,
                    message="Calculating portfolio metrics...",
                    level="info",
                    timestamp=datetime.now()
                ))

            # ============================================================
            # MULTITHREADED REPORT GENERATION
            # ============================================================
            if worker_id is not None:
                self._worker_log(worker_id, "  → Calculating metrics (parallel)...", "info")

            log_info("Starting parallel report generation...")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time

            all_metrics = None
            all_charts = None

            # Define parallel tasks with timing
            def calculate_metrics():
                start = time.time()
                log_info("Worker: Calculating multi-symbol portfolio metrics...")
                metrics = MultiSymbolMetrics.calculate_all_metrics(portfolio)
                elapsed = time.time() - start
                log_info(f"Worker: Calculated metrics categories: {list(metrics.keys())} in {elapsed:.1f}s")
                return metrics

            def generate_charts_task(metrics):
                start = time.time()
                log_info("Worker: Generating portfolio visualization charts...")
                charts = MultiSymbolChartGenerator.generate_all_charts(
                    portfolio,
                    metrics,
                    parallel=True,  # Enable parallel chart generation for performance
                    max_workers=9
                )
                elapsed = time.time() - start
                log_info(f"Worker: Generated {len(charts)} chart datasets in {elapsed:.1f}s")
                return charts

            # Execute metrics and charts in parallel
            overall_start = time.time()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ReportGen") as executor:
                # Submit metrics calculation (must run first)
                metrics_start = time.time()
                metrics_future = executor.submit(calculate_metrics)

                # Wait for metrics before generating charts
                all_metrics = metrics_future.result()
                metrics_time = time.time() - metrics_start

                if worker_id is not None:
                    self._worker_log(worker_id, f"  ✓ Metrics calculated in {metrics_time:.1f}s", "info")
                    self._worker_log(worker_id, "  → Generating 9 charts (parallel, 9 workers)...", "info")

                # Now generate charts (depends on metrics)
                charts_start = time.time()
                charts_future = executor.submit(generate_charts_task, all_metrics)
                all_charts = charts_future.result()
                charts_time = time.time() - charts_start

            overall_time = time.time() - overall_start
            if worker_id is not None:
                chart_count = len(all_charts)
                self._worker_log(worker_id, f"  ✓ Charts generated in {charts_time:.1f}s", "success")
                self._worker_log(worker_id, f"  ✓ Total report generation: {overall_time:.1f}s", "success")

            log_info(f"Parallel report generation complete in {overall_time:.1f}s (metrics: {metrics_time:.1f}s, charts: {charts_time:.1f}s)")

            # Send progress update to portfolio queue
            if portfolio_key in self.log_queues:
                self.log_queues[portfolio_key].put(LogMessage(
                    symbol=symbols_display,
                    message="Saving reports to disk...",
                    level="info",
                    timestamp=datetime.now()
                ))

            # ============================================================
            # PARALLEL FILE EXPORT & HTML GENERATION
            # ============================================================
            if worker_id is not None:
                self._worker_log(worker_id, "  → Exporting files (parallel)...", "info")

            log_info("Starting parallel file export...")

            # Prepare JSON-serializable metrics
            json_metrics = {
                'composition': all_metrics.get('composition', {}),
                'attribution': {
                    'per_symbol': {
                        k: dict(v) for k, v in all_metrics.get('attribution', {}).get('per_symbol', {}).items()
                    },
                    'best_symbol': all_metrics.get('attribution', {}).get('best_symbol'),
                    'worst_symbol': all_metrics.get('attribution', {}).get('worst_symbol'),
                    'total_pnl': all_metrics.get('attribution', {}).get('total_pnl', 0),
                },
                'diversification': all_metrics.get('diversification', {}),
                'rebalancing': all_metrics.get('rebalancing', {}),
                'trade_analysis': all_metrics.get('trade_analysis', {}),
            }

            # Create per-symbol stats for basic HTML report
            symbol_stats = {}
            attribution = all_metrics.get('attribution', {})
            per_symbol_stats = attribution.get('per_symbol', {})

            for symbol in symbols:
                stats = per_symbol_stats.get(symbol, {})
                stats_dict = {
                    'Total Return [%]': stats.get('Total Return [%]', 0),
                    'Sharpe Ratio': stats.get('Sharpe Ratio', 0),
                    'Win Rate [%]': stats.get('Win Rate [%]', 0),
                    'Total Trades': stats.get('Total Trades', 0),
                    'Avg Hold Duration [days]': stats.get('Avg Hold Duration [days]', 0),
                    'Total P&L': stats.get('Total P&L', 0),
                }
                symbol_stats[symbol] = pd.Series(stats_dict)

            # Define export tasks
            def export_basic_html():
                if not symbol_stats:
                    return None
                df = ResultsAggregator.aggregate_results(symbol_stats)
                html_report_path = output_dir / f"{timestamp}_{self._strategy_name}_portfolio_report.html"
                title = f"Multi-Symbol Portfolio: {self._strategy_name} ({len(symbols)} symbols)"
                log_info(f"Worker: Generating basic HTML report: {html_report_path}")
                ResultsAggregator.export_to_html(df, html_report_path, title=title, portfolios={})
                log_info(f"Worker: Exported basic HTML")
                return html_report_path

            def export_analytics_html():
                viewer_path = output_dir / f"{timestamp}_{self._strategy_name}_portfolio_analytics.html"
                log_info(f"Worker: Generating interactive analytics HTML: {viewer_path}")
                MultiSymbolHTMLViewer.generate_html(
                    metrics=json_metrics,
                    charts=all_charts,
                    output_path=viewer_path,
                    title=f"Portfolio Analytics: {self._strategy_name}"
                )
                log_info(f"Worker: Generated interactive HTML viewer")
                return viewer_path

            # Execute all exports in parallel
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="FileExport") as executor:
                futures = {
                    executor.submit(export_basic_html): "basic_html",
                    executor.submit(export_analytics_html): "analytics_html"
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        result = future.result()
                        log_info(f"Completed export: {task_name}")
                    except Exception as exc:
                        log_error(f"Export task {task_name} failed: {exc}")

            log_info("All file exports complete")

            if worker_id is not None:
                self._worker_log(worker_id, "  ✓ All reports exported", "success")
                self._worker_log(worker_id, "", "info")
                self._worker_log(worker_id, "✓ Portfolio Backtest Complete", "success")
                self._worker_log(worker_id, f"  Output: {output_dir}", "info")

        except Exception as e:
            log_exception(e, "Failed to generate portfolio report with metrics and charts")
            log_error("Continuing without enhanced HTML report")
            if worker_id is not None:
                self._worker_log(worker_id, "  ✗ Report generation failed", "error")
                self._worker_log(worker_id, f"    Error: {str(e)}", "error")

        # Generate symbol comparison CSV/HTML report showing per-symbol contribution
        try:
            comparison_path = output_dir / f"{timestamp}_{self._strategy_name}_symbol_comparison.csv"

            # Extract per-symbol metrics from closed_positions if available
            if hasattr(portfolio, 'closed_positions') and portfolio.closed_positions:
                # Group trades by symbol
                trades_by_symbol = {}
                for trade in portfolio.closed_positions:
                    symbol = trade.get('symbol', 'Unknown')
                    if symbol not in trades_by_symbol:
                        trades_by_symbol[symbol] = []
                    trades_by_symbol[symbol].append(trade)

                # Calculate per-symbol metrics
                symbol_metrics = []
                for symbol in symbols:
                    trades = trades_by_symbol.get(symbol, [])

                    total_pnl = sum(t.get('pnl', 0) for t in trades)
                    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]

                    metrics = {
                        'Symbol': symbol,
                        'Total Trades': len(trades),
                        'Winning Trades': len(winning_trades),
                        'Losing Trades': len(trades) - len(winning_trades),
                        'Win Rate [%]': (len(winning_trades) / len(trades) * 100) if trades else 0,
                        'Total P&L': total_pnl,
                        'Avg P&L per Trade': total_pnl / len(trades) if trades else 0,
                    }
                    symbol_metrics.append(metrics)

                # Create comparison DataFrame
                comparison_df = pd.DataFrame(symbol_metrics)

                # Add portfolio totals row
                totals = {
                    'Symbol': 'PORTFOLIO TOTAL',
                    'Total Trades': comparison_df['Total Trades'].sum(),
                    'Winning Trades': comparison_df['Winning Trades'].sum(),
                    'Losing Trades': comparison_df['Losing Trades'].sum(),
                    'Win Rate [%]': (comparison_df['Winning Trades'].sum() /
                                    comparison_df['Total Trades'].sum() * 100)
                                    if comparison_df['Total Trades'].sum() > 0 else 0,
                    'Total P&L': comparison_df['Total P&L'].sum(),
                    'Avg P&L per Trade': (comparison_df['Total P&L'].sum() /
                                         comparison_df['Total Trades'].sum())
                                         if comparison_df['Total Trades'].sum() > 0 else 0,
                }
                comparison_df = pd.concat([comparison_df, pd.DataFrame([totals])], ignore_index=True)

                # Export to CSV
                comparison_df.to_csv(comparison_path, index=False)
                log_info(f"Exported symbol comparison: {comparison_path}")

                # Also export as HTML for better viewing
                comparison_html = output_dir / f"{timestamp}_{self._strategy_name}_symbol_comparison.html"
                html_content = f"""
                <html>
                <head>
                    <title>Symbol Comparison - {self._strategy_name}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                        th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
                        td {{ border: 1px solid #ddd; padding: 8px; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        tr:last-child {{ font-weight: bold; background-color: #e7f3e7; }}
                        .positive {{ color: green; }}
                        .negative {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Multi-Symbol Portfolio - Symbol Comparison</h1>
                    <h2>{self._strategy_name}</h2>
                    <p><strong>Symbols:</strong> {symbols_str}</p>
                    {comparison_df.to_html(index=False, escape=False, classes='comparison-table')}
                </body>
                </html>
                """

                with open(comparison_html, 'w') as f:
                    f.write(html_content)
                log_info(f"Exported symbol comparison HTML: {comparison_html}")

            else:
                log_error("No trade data available for symbol comparison")

        except Exception as e:
            log_exception(e, "Failed to generate symbol comparison report")
            log_error("Continuing without comparison report")
