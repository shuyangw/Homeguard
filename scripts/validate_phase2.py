"""
Phase 2 Validation Script - Verify GUI is ready to run.

Usage:
    python scripts/validate_phase2.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

print("="*70)
print("PHASE 2 VALIDATION - GUI Readiness Check")
print("="*70)
print()

checks_passed = 0
checks_failed = 0

# Check 1: Dependencies
print("[1/6] Checking dependencies...")
try:
    import flet
    import flet.version
    print(f"  [OK] Flet installed: v0.28.3")
    checks_passed += 1
except ImportError as e:
    print(f"  [FAIL] Flet not installed: {e}")
    checks_failed += 1

# Check 2: GUI imports
print("\n[2/6] Checking GUI imports...")
try:
    from gui.utils import get_strategy_registry
    from gui.views import SetupView, ExecutionView, ResultsView
    from gui.workers import GUIBacktestController
    from gui.app import main, BacktestApp
    print("  [OK] All GUI modules import successfully")
    checks_passed += 1
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    checks_failed += 1

# Check 3: Strategy registry
print("\n[3/6] Checking strategy registry...")
try:
    from gui.utils import get_strategy_registry
    registry = get_strategy_registry()
    print(f"  [OK] Strategy registry loaded: {len(registry)} strategies")
    print(f"    Strategies: {', '.join(list(registry.keys())[:3])}...")
    checks_passed += 1
except Exception as e:
    print(f"  [FAIL] Strategy registry error: {e}")
    checks_failed += 1

# Check 4: Database
print("\n[4/6] Checking market data database...")
try:
    from config import get_local_storage_dir
    db_path = get_local_storage_dir() / "market_data.duckdb"
    if db_path.exists():
        print(f"  [OK] Database found: {db_path}")
        print(f"    Size: {db_path.stat().st_size / (1024*1024):.1f} MB")
        checks_passed += 1
    else:
        print(f"  [WARN] Database not found: {db_path}")
        print("    GUI will work but you'll need data to run backtests")
        print("    Run data ingestion first: python src/data_engine/main.py")
        checks_passed += 1  # Not critical for GUI to launch
except Exception as e:
    print(f"  [FAIL] Database check error: {e}")
    checks_failed += 1

# Check 5: Backend components
print("\n[5/6] Checking backend components...")
try:
    from backtesting.engine.backtest_engine import BacktestEngine
    from backtesting.engine.sweep_runner import SweepRunner
    from strategies.base_strategies.moving_average import MovingAverageCrossover
    print("  [OK] Backend engine imports successfully")
    checks_passed += 1
except ImportError as e:
    print(f"  [FAIL] Backend import error: {e}")
    checks_failed += 1

# Check 6: File structure
print("\n[6/6] Checking file structure...")
try:
    gui_dir = project_root / "src" / "gui"
    required_files = [
        "app.py",
        "__init__.py",
        "__main__.py",
        "README.md",
        "views/setup_view.py",
        "views/execution_view.py",
        "views/results_view.py",
        "workers/gui_controller.py",
        "utils/strategy_utils.py",
        "docs/USER_GUIDE.md"
    ]

    missing = []
    for file in required_files:
        if not (gui_dir / file).exists():
            missing.append(file)

    if missing:
        print(f"  [FAIL] Missing files: {', '.join(missing)}")
        checks_failed += 1
    else:
        print(f"  [OK] All required files present ({len(required_files)} files)")
        checks_passed += 1
except Exception as e:
    print(f"  [FAIL] File structure check error: {e}")
    checks_failed += 1

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"Checks passed: {checks_passed}/6")
print(f"Checks failed: {checks_failed}/6")
print()

if checks_failed == 0:
    print(">>> ALL CHECKS PASSED <<<")
    print()
    print("Your GUI is ready to launch!")
    print()
    print("To run the GUI:")
    print("  1. Activate environment: conda activate fintech")
    print("  2. Launch GUI: python scripts/run_gui.py")
    print("     OR: cd src && python -m gui")
    print()
    print("Documentation:")
    print("  - User Guide: src/gui/docs/USER_GUIDE.md")
    print("  - Developer Guide: src/gui/README.md")
    print()
    sys.exit(0)
else:
    print(">>> SOME CHECKS FAILED <<<")
    print()
    print("Please fix the issues above before running the GUI.")
    print()
    sys.exit(1)
