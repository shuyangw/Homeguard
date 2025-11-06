"""
Unit tests for GUI backtest controller.

Tests the enhanced SweepRunner with callbacks and the GUIBacktestController wrapper.
"""

import pytest
import time
from typing import List, Dict
from datetime import datetime

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization.sweep_runner import SweepRunner
from backtesting.engine.portfolio_simulator import Portfolio
from strategies.base_strategies.moving_average import MovingAverageCrossover
from gui.workers.gui_controller import GUIBacktestController
import pandas as pd


class TestSweepRunnerCallbacks:
    """Test SweepRunner with callback enhancements"""

    def test_sweep_runner_callbacks_fire(self):
        """Test that callbacks fire in correct order during backtest"""
        # Tracking callbacks
        callbacks_fired = {
            'start': [],
            'progress': [],
            'complete': [],
            'error': []
        }

        def on_start(symbol: str):
            callbacks_fired['start'].append(symbol)

        def on_progress(symbol: str, message: str, progress: float):
            callbacks_fired['progress'].append((symbol, message, progress))

        def on_complete(symbol: str, portfolio: Portfolio, stats: pd.Series):
            callbacks_fired['complete'].append((symbol, portfolio, stats))
            # Verify we get Portfolio object
            assert portfolio is not None
            assert isinstance(portfolio, Portfolio)
            assert stats is not None

        def on_error(symbol: str, error: Exception):
            callbacks_fired['error'].append((symbol, error))

        # Create engine and runner with callbacks
        engine = BacktestEngine(initial_capital=10000, fees=0.001)
        runner = SweepRunner(
            engine=engine,
            max_workers=2,
            show_progress=False,  # Disable console output
            on_symbol_start=on_start,
            on_symbol_progress=on_progress,
            on_symbol_complete=on_complete,
            on_symbol_error=on_error
        )

        # Run sweep
        strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        results = runner.run_sweep(
            strategy=strategy,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2024-03-01',
            parallel=True
        )

        # Verify callbacks fired
        assert len(callbacks_fired['start']) == 3
        assert all(sym in callbacks_fired['start'] for sym in symbols)

        assert len(callbacks_fired['progress']) >= 6  # At least 2 progress updates per symbol
        assert len(callbacks_fired['complete']) == 3

        # Verify portfolios accessible
        portfolios = runner.get_portfolios()
        assert len(portfolios) == 3
        for symbol in symbols:
            assert symbol in portfolios
            assert isinstance(portfolios[symbol], Portfolio)

    def test_sweep_runner_backward_compatible(self):
        """Test that SweepRunner works without callbacks (like CLI)"""
        # Create runner WITHOUT callbacks (like CLI mode)
        engine = BacktestEngine(initial_capital=10000, fees=0.001)
        runner = SweepRunner(
            engine=engine,
            max_workers=2,
            show_progress=False  # No callbacks, no console output
        )

        # Run sweep exactly like CLI does
        strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
        symbols = ['AAPL', 'MSFT']

        results = runner.run_sweep(
            strategy=strategy,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2024-03-01',
            parallel=True
        )

        # Verify results returned
        assert len(results) == 2
        for symbol in symbols:
            assert symbol in results
            assert results[symbol] is not None  # Stats returned

        # Portfolios still accessible
        portfolios = runner.get_portfolios()
        assert len(portfolios) == 2


class TestGUIBacktestController:
    """Test GUIBacktestController wrapper"""

    def test_gui_controller_queue_updates(self):
        """Test that GUI controller provides queue-based updates"""
        controller = GUIBacktestController(max_workers=2)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        # Start backtests
        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2024-03-01',
            initial_capital=10000.0,
            fees=0.001
        )

        # Poll for updates while running
        updates_received = []
        max_polls = 100  # Safety limit
        polls = 0

        while controller.is_running() and polls < max_polls:
            updates = controller.get_updates()
            updates_received.append(updates)
            time.sleep(0.1)  # 100ms poll interval
            polls += 1

        # Verify we got updates
        assert len(updates_received) > 0

        # Verify all symbols appeared in updates
        all_symbols_seen = set()
        for update_batch in updates_received:
            all_symbols_seen.update(update_batch.keys())

        assert all(sym in all_symbols_seen for sym in symbols)

        # Verify progress updates were received
        progress_updates_count = 0
        for update_batch in updates_received:
            for symbol_update in update_batch.values():
                progress_updates_count += len(symbol_update['progress'])

        assert progress_updates_count >= 3  # At least some progress updates

    def test_gui_controller_completion(self):
        """Test full backtest cycle with GUI controller"""
        controller = GUIBacktestController(max_workers=4)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
        symbols = ['AAPL', 'MSFT']

        # Start backtests
        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2024-03-01',
            initial_capital=10000.0,
            fees=0.001
        )

        # Wait for completion
        max_wait = 60  # 60 seconds timeout
        start_time = time.time()

        while controller.is_running():
            if time.time() - start_time > max_wait:
                pytest.fail("Backtest timed out")

            # Check progress summary
            summary = controller.get_progress_summary()
            assert summary['total'] == 2
            assert summary['completed'] + summary['running'] + summary['pending'] <= 2

            time.sleep(0.1)

        # Verify completion
        assert not controller.is_running()

        # Get final results
        results = controller.get_results()
        assert len(results) == 2
        for symbol in symbols:
            assert symbol in results
            assert results[symbol] is not None

        # Get portfolios
        portfolios = controller.get_portfolios()
        assert len(portfolios) == 2
        for symbol in symbols:
            assert symbol in portfolios
            assert isinstance(portfolios[symbol], Portfolio)

        # Check progress summary shows completion
        summary = controller.get_progress_summary()
        assert summary['completed'] == 2
        assert summary['running'] == 0
        assert summary['pending'] == 0

    def test_gui_controller_status_tracking(self):
        """Test that controller tracks symbol status correctly"""
        controller = GUIBacktestController(max_workers=2)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
        symbols = ['AAPL', 'MSFT']

        # Start backtests
        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2024-03-01'
        )

        # Track status changes
        status_seen = {sym: set() for sym in symbols}

        while controller.is_running():
            updates = controller.get_updates()
            for symbol, update in updates.items():
                status_seen[symbol].add(update['status'])
            time.sleep(0.1)

        # Verify status progression
        for symbol in symbols:
            # Should have seen: pending -> running -> completed
            assert 'pending' in status_seen[symbol] or 'running' in status_seen[symbol]
            assert 'completed' in status_seen[symbol]


@pytest.mark.slow
class TestGUIControllerPerformance:
    """Performance tests for GUI controller"""

    def test_many_symbols_parallel(self):
        """Test handling many symbols with multiple workers"""
        controller = GUIBacktestController(max_workers=8)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=50)

        # Test with 10 symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                   'NVDA', 'TSLA', 'JPM', 'V', 'WMT']

        start = time.time()

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2024-02-01',  # Shorter period for speed
            initial_capital=10000.0,
            fees=0.001
        )

        # Wait for completion
        while controller.is_running():
            summary = controller.get_progress_summary()
            # Print progress for debugging
            print(f"Progress: {summary['completed']}/{summary['total']}")
            time.sleep(0.5)

        elapsed = time.time() - start

        # Verify completion
        results = controller.get_results()
        assert len(results) == 10

        portfolios = controller.get_portfolios()
        assert len(portfolios) == 10

        # Performance check: Should complete in reasonable time
        # With 8 workers, 10 symbols should complete < 60 seconds
        assert elapsed < 60, f"Took too long: {elapsed:.1f}s"

        print(f"Completed 10 symbols in {elapsed:.1f}s with 8 workers")


class TestPhase3WorkerTracking:
    """Test Phase 3 worker tracking and logging features"""

    def test_worker_id_pool_initialization(self):
        """Test that worker ID pool is correctly initialized"""
        controller = GUIBacktestController(max_workers=4)

        # Start backtests to initialize worker pool
        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        symbols = ['AAPL']

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            fees=0.001
        )

        # Verify worker queues initialized
        assert len(controller.worker_log_queues) == 4
        assert len(controller.worker_assignments) == 4

        # Verify all workers initially idle
        for worker_id in range(4):
            assert controller.worker_assignments[worker_id] is None

        # Wait for completion
        while controller.is_running():
            time.sleep(0.1)

    def test_worker_updates_structure(self):
        """Test that get_worker_updates() returns correct structure"""
        controller = GUIBacktestController(max_workers=2)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        symbols = ['AAPL', 'MSFT']

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            fees=0.001
        )

        # Poll for worker updates
        time.sleep(0.5)  # Let workers start

        updates = controller.get_worker_updates()

        # Verify structure
        assert 'status_updates' in updates
        assert 'logs' in updates
        assert isinstance(updates['status_updates'], list)
        assert isinstance(updates['logs'], dict)

        # Verify worker logs have correct keys
        for worker_id in controller.worker_log_queues.keys():
            assert worker_id in updates['logs']
            assert isinstance(updates['logs'][worker_id], list)

        # Wait for completion
        while controller.is_running():
            time.sleep(0.1)

    def test_worker_status_updates_fire(self):
        """Test that worker status updates are generated"""
        controller = GUIBacktestController(max_workers=2)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        symbols = ['AAPL']

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            fees=0.001
        )

        # Collect all status updates
        all_status_updates = []

        while controller.is_running():
            updates = controller.get_worker_updates()
            all_status_updates.extend(updates['status_updates'])
            time.sleep(0.1)

        # Final poll after completion
        updates = controller.get_worker_updates()
        all_status_updates.extend(updates['status_updates'])

        # Verify we got status updates
        assert len(all_status_updates) > 0, "Should receive worker status updates"

        # Verify at least one "started" status
        started_updates = [u for u in all_status_updates if u.status == "started"]
        assert len(started_updates) > 0, "Should have at least one worker start"

        # Verify at least one "idle" status
        idle_updates = [u for u in all_status_updates if u.status == "idle"]
        assert len(idle_updates) > 0, "Should have at least one worker go idle"

    def test_worker_logs_generated(self):
        """Test that worker log messages are generated"""
        controller = GUIBacktestController(max_workers=2)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        symbols = ['AAPL', 'MSFT']

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            fees=0.001
        )

        # Collect all worker logs
        all_worker_logs = {0: [], 1: []}

        while controller.is_running():
            updates = controller.get_worker_updates()

            for worker_id, log_messages in updates['logs'].items():
                if worker_id in all_worker_logs:
                    all_worker_logs[worker_id].extend(log_messages)

            time.sleep(0.1)

        # Final poll after completion
        updates = controller.get_worker_updates()
        for worker_id, log_messages in updates['logs'].items():
            if worker_id in all_worker_logs:
                all_worker_logs[worker_id].extend(log_messages)

        # Verify at least one worker generated logs
        total_logs = sum(len(logs) for logs in all_worker_logs.values())
        assert total_logs > 0, "Should receive worker log messages"

        # Verify log messages have correct attributes
        for worker_id, logs in all_worker_logs.items():
            for log_msg in logs:
                assert hasattr(log_msg, 'worker_id')
                assert hasattr(log_msg, 'message')
                assert hasattr(log_msg, 'level')
                assert hasattr(log_msg, 'timestamp')
                assert log_msg.level in ['info', 'success', 'warning', 'error']

    def test_worker_id_claim_and_release(self):
        """Test worker ID claim and release mechanics"""
        controller = GUIBacktestController(max_workers=3)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        symbols = ['AAPL']

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            fees=0.001
        )

        # Track worker assignments over time
        assignment_snapshots = []

        while controller.is_running():
            snapshot = controller.worker_assignments.copy()
            assignment_snapshots.append(snapshot)
            time.sleep(0.1)

        # Verify worker IDs are in valid range
        for snapshot in assignment_snapshots:
            for worker_id, symbol in snapshot.items():
                assert 0 <= worker_id < 3, f"Worker ID {worker_id} out of range"
                # Symbol can be None (idle) or a string
                assert symbol is None or isinstance(symbol, str)

        # After completion, all workers should be idle
        final_assignments = controller.worker_assignments
        for worker_id, symbol in final_assignments.items():
            assert symbol is None, f"Worker {worker_id} should be idle after completion"

    def test_worker_tracking_with_multiple_symbols(self):
        """Test worker tracking with more symbols than workers"""
        controller = GUIBacktestController(max_workers=2)

        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # 4 symbols, 2 workers

        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            fees=0.001
        )

        # Track which symbols were processed by which workers
        worker_symbol_assignments = {0: set(), 1: set()}

        while controller.is_running():
            updates = controller.get_worker_updates()

            for status_update in updates['status_updates']:
                if status_update.symbol and status_update.status == "started":
                    worker_id = status_update.worker_id
                    if worker_id in worker_symbol_assignments:
                        worker_symbol_assignments[worker_id].add(status_update.symbol)

            time.sleep(0.1)

        # Verify all symbols were processed
        all_processed_symbols = set()
        for symbols_set in worker_symbol_assignments.values():
            all_processed_symbols.update(symbols_set)

        assert len(all_processed_symbols) == 4, "All 4 symbols should be processed"

        # Verify at least one worker processed multiple symbols
        # (since 4 symbols with 2 workers, at least one must handle 2+)
        multi_symbol_workers = [w for w, syms in worker_symbol_assignments.items() if len(syms) >= 2]
        assert len(multi_symbol_workers) > 0, "At least one worker should handle multiple symbols"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
