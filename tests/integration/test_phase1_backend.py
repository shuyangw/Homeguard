"""
Simple verification script for Phase 1 implementation.

Tests the enhanced SweepRunner and GUIBacktestController.
"""

import sys
import time
from pathlib import Path

# Add src to path (from tests/integration/ go up 2 levels to project root, then into src/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.engine.sweep_runner import SweepRunner
from strategies.base_strategies.moving_average import MovingAverageCrossover
from gui.workers.gui_controller import GUIBacktestController


def test_sweep_runner_backward_compatible():
    """Test that SweepRunner works without callbacks (backward compatible)"""
    print("\n" + "="*70)
    print("TEST 1: SweepRunner Backward Compatibility (CLI Mode)")
    print("="*70)

    engine = BacktestEngine(initial_capital=10000, fees=0.001)
    runner = SweepRunner(
        engine=engine,
        max_workers=2,
        show_progress=True  # Show console output like CLI
    )

    strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
    symbols = ['AAPL', 'MSFT']

    print(f"\nRunning backtest for {symbols}...")

    results = runner.run_sweep(
        strategy=strategy,
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-03-01',
        parallel=True
    )

    # Verify results
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for symbol in symbols:
        assert symbol in results, f"Symbol {symbol} not in results"
        assert results[symbol] is not None, f"No stats for {symbol}"

    print("\n[PASS] Test 1: SweepRunner works in CLI mode (backward compatible)")

    return True


def test_sweep_runner_with_callbacks():
    """Test that SweepRunner fires callbacks correctly"""
    print("\n" + "="*70)
    print("TEST 2: SweepRunner with Callbacks (GUI Mode)")
    print("="*70)

    callbacks_fired = {
        'start': [],
        'progress': [],
        'complete': []
    }

    def on_start(symbol):
        print(f"  [Callback] Started: {symbol}")
        callbacks_fired['start'].append(symbol)

    def on_progress(symbol, message, progress):
        print(f"  [Callback] Progress: {symbol} - {message} ({progress*100:.0f}%)")
        callbacks_fired['progress'].append((symbol, message, progress))

    def on_complete(symbol, portfolio, stats):
        print(f"  [Callback] Complete: {symbol}")
        callbacks_fired['complete'].append(symbol)
        assert portfolio is not None, f"No portfolio for {symbol}"
        assert stats is not None, f"No stats for {symbol}"

    engine = BacktestEngine(initial_capital=10000, fees=0.001)
    runner = SweepRunner(
        engine=engine,
        max_workers=2,
        show_progress=False,  # Disable console, use callbacks
        on_symbol_start=on_start,
        on_symbol_progress=on_progress,
        on_symbol_complete=on_complete
    )

    strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
    symbols = ['AAPL', 'MSFT']

    print(f"\nRunning backtest with callbacks for {symbols}...")

    results = runner.run_sweep(
        strategy=strategy,
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-03-01',
        parallel=True
    )

    # Verify callbacks fired
    assert len(callbacks_fired['start']) == 2, \
        f"Expected 2 start callbacks, got {len(callbacks_fired['start'])}"

    assert len(callbacks_fired['progress']) >= 4, \
        f"Expected at least 4 progress callbacks, got {len(callbacks_fired['progress'])}"

    assert len(callbacks_fired['complete']) == 2, \
        f"Expected 2 complete callbacks, got {len(callbacks_fired['complete'])}"

    # Verify portfolios accessible
    portfolios = runner.get_portfolios()
    assert len(portfolios) == 2, f"Expected 2 portfolios, got {len(portfolios)}"

    print(f"\n[PASS] Test 2: Callbacks fired correctly")
    print(f"  - Started: {len(callbacks_fired['start'])} symbols")
    print(f"  - Progress updates: {len(callbacks_fired['progress'])}")
    print(f"  - Completed: {len(callbacks_fired['complete'])} symbols")

    return True


def test_gui_controller():
    """Test GUIBacktestController wrapper"""
    print("\n" + "="*70)
    print("TEST 3: GUIBacktestController")
    print("="*70)

    controller = GUIBacktestController(max_workers=2)

    strategy = MovingAverageCrossover(fast_window=10, slow_window=50)
    symbols = ['AAPL', 'MSFT']

    print(f"\nStarting GUI controller for {symbols}...")

    controller.start_backtests(
        strategy=strategy,
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-03-01',
        initial_capital=10000.0,
        fees=0.001
    )

    # Poll for updates
    updates_count = 0
    max_wait = 60  # seconds
    start = time.time()

    while controller.is_running():
        if time.time() - start > max_wait:
            raise TimeoutError("Test timed out")

        updates = controller.get_updates()
        summary = controller.get_progress_summary()

        if updates:
            updates_count += 1
            print(f"  [Poll {updates_count}] Progress: {summary['completed']}/{summary['total']} completed, "
                  f"{summary['running']} running")

        time.sleep(0.2)  # 200ms poll

    # Verify completion
    assert not controller.is_running(), "Controller still running after completion"

    results = controller.get_results()
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    portfolios = controller.get_portfolios()
    assert len(portfolios) == 2, f"Expected 2 portfolios, got {len(portfolios)}"

    summary = controller.get_progress_summary()
    assert summary['completed'] == 2, f"Expected 2 completed, got {summary['completed']}"
    assert summary['running'] == 0, f"Expected 0 running, got {summary['running']}"

    print(f"\n[PASS] Test 3: GUIBacktestController works correctly")
    print(f"  - Received {updates_count} update batches")
    print(f"  - All symbols completed successfully")

    return True


def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("PHASE 1 VERIFICATION - Enhanced SweepRunner & GUIBacktestController")
    print("="*70)

    try:
        test_sweep_runner_backward_compatible()
        test_sweep_runner_with_callbacks()
        test_gui_controller()

        print("\n" + "="*70)
        print(">>> ALL TESTS PASSED <<<")
        print("="*70)
        print("\nPhase 1 implementation is complete and working!")
        print("\nNext steps:")
        print("  - Phase 2: Build Flet UI foundation")
        print("  - Phase 3: Real-time execution view")
        print("  - Phase 4: Results & export")
        print("  - Phase 5: Polish & packaging")
        print()

        return 0

    except Exception as e:
        print("\n" + "="*70)
        print(">>> TEST FAILED <<<")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
