"""
Validation script for optimization module refactoring.

Tests:
1. All imports work correctly
2. Backward compatibility maintained
3. No circular dependencies
4. All integration points functional
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_backend_imports():
    """Test backend optimization imports."""
    print("Testing backend imports...")

    # Test new module structure
    from backtesting.optimization import GridSearchOptimizer, SweepRunner
    print("  ✓ backtesting.optimization imports")

    from backtesting.optimization.grid_search import GridSearchOptimizer
    print("  ✓ backtesting.optimization.grid_search imports")

    from backtesting.optimization.sweep_runner import SweepRunner
    print("  ✓ backtesting.optimization.sweep_runner imports")

    # Test backward compatibility
    from backtesting.engine.backtest_engine import BacktestEngine
    engine = BacktestEngine()
    assert hasattr(engine, 'optimize'), "BacktestEngine.optimize() method missing"
    print("  ✓ BacktestEngine.optimize() backward compatibility")

    return True

def test_gui_imports():
    """Test GUI optimization imports."""
    print("\nTesting GUI imports...")

    # Test new module structure
    from gui.optimization import OptimizationDialog, OptimizationRunner
    print("  ✓ gui.optimization imports")

    from gui.optimization.dialog import OptimizationDialog
    print("  ✓ gui.optimization.dialog imports")

    from gui.optimization.runner import OptimizationRunner
    print("  ✓ gui.optimization.runner imports")

    return True

def test_integration_points():
    """Test integration between modules."""
    print("\nTesting integration points...")

    # Test BacktestEngine → GridSearchOptimizer
    from backtesting.engine.backtest_engine import BacktestEngine
    from backtesting.optimization.grid_search import GridSearchOptimizer

    engine = BacktestEngine()
    optimizer = GridSearchOptimizer(engine)
    assert optimizer.engine == engine, "GridSearchOptimizer not properly linked to engine"
    print("  ✓ BacktestEngine → GridSearchOptimizer integration")

    # Test app.py imports
    try:
        from gui.app import BacktestApp
        print("  ✓ BacktestApp imports successfully")
    except ImportError as e:
        print(f"  ✗ BacktestApp import failed: {e}")
        return False

    # Test setup_view.py imports
    try:
        from gui.views.setup_view import SetupView
        print("  ✓ SetupView imports successfully")
    except ImportError as e:
        print(f"  ✗ SetupView import failed: {e}")
        return False

    return True

def test_no_circular_dependencies():
    """Test for circular import dependencies."""
    print("\nTesting for circular dependencies...")

    try:
        # Import in various orders to detect circular dependencies
        import backtesting.optimization
        import backtesting.engine.backtest_engine
        import gui.optimization
        import gui.views.setup_view

        # Try reverse order
        import gui.views.setup_view
        import gui.optimization
        import backtesting.engine.backtest_engine
        import backtesting.optimization

        print("  ✓ No circular dependencies detected")
        return True
    except ImportError as e:
        print(f"  ✗ Circular dependency detected: {e}")
        return False

def test_backward_compatibility():
    """Test that old code still works."""
    print("\nTesting backward compatibility...")

    from backtesting.engine.backtest_engine import BacktestEngine
    from strategies.base_strategies.moving_average import MovingAverageCrossover
    import pandas as pd
    import numpy as np

    # Create mock data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.3,
        'high': close_prices + np.abs(np.random.randn(100) * 0.5),
        'low': close_prices - np.abs(np.random.randn(100) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100),
        'symbol': 'TEST'
    }, index=dates)

    df = df.set_index('symbol', append=True)
    df = df.swaplevel()

    # Mock data loader
    class MockDataLoader:
        def load_symbols(self, symbols, start, end):
            return df

    # Test old API
    engine = BacktestEngine(initial_capital=10000, fees=0.0)
    engine.data_loader = MockDataLoader()

    param_grid = {
        'fast_window': [10, 15],
        'slow_window': [20]
    }

    try:
        result = engine.optimize(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2023-01-01',
            end_date='2023-04-10',
            metric='sharpe_ratio'
        )

        assert 'best_params' in result
        assert 'best_value' in result
        assert 'best_portfolio' in result
        assert 'metric' in result

        print("  ✓ Backward compatibility: engine.optimize() works")
        return True
    except Exception as e:
        print(f"  ✗ Backward compatibility failed: {e}")
        return False

def test_file_structure():
    """Verify new file structure exists."""
    print("\nTesting file structure...")

    base_path = Path(__file__).parent.parent

    # Backend files
    backend_files = [
        'src/backtesting/optimization/__init__.py',
        'src/backtesting/optimization/grid_search.py',
        'src/backtesting/optimization/sweep_runner.py',
    ]

    for file_path in backend_files:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"  ✗ Missing: {file_path}")
            return False
        print(f"  ✓ Exists: {file_path}")

    # GUI files
    gui_files = [
        'src/gui/optimization/__init__.py',
        'src/gui/optimization/dialog.py',
        'src/gui/optimization/runner.py',
    ]

    for file_path in gui_files:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"  ✗ Missing: {file_path}")
            return False
        print(f"  ✓ Exists: {file_path}")

    # Test files
    test_files = [
        'tests/backtesting/optimization/__init__.py',
        'tests/backtesting/optimization/test_grid_search.py',
        'tests/gui/optimization/__init__.py',
        'tests/gui/optimization/test_dialog.py',
        'tests/gui/optimization/test_runner.py',
    ]

    for file_path in test_files:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"  ✗ Missing: {file_path}")
            return False
        print(f"  ✓ Exists: {file_path}")

    return True

def main():
    """Run all validation tests."""
    print("="*70)
    print("OPTIMIZATION MODULE REFACTORING VALIDATION")
    print("="*70)

    tests = [
        ("File Structure", test_file_structure),
        ("Backend Imports", test_backend_imports),
        ("GUI Imports", test_gui_imports),
        ("Integration Points", test_integration_points),
        ("Circular Dependencies", test_no_circular_dependencies),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All validation tests passed!")
        print("✓ Refactoring is correct and complete.")
        return 0
    else:
        print(f"\n✗ {total - passed} validation test(s) failed!")
        print("✗ Please fix the issues before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
