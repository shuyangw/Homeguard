"""Test script to verify GUI imports work correctly."""

import sys
from pathlib import Path

# Add src to path (from gui/tests/ we need to go up 2 levels to get to src/)
# test_imports.py is at: src/gui/tests/test_imports.py
# parent.parent = src/
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

print("Testing GUI imports...")
print(f"Python path: {src_dir}")
print()

try:
    from gui.utils.strategy_utils import get_strategy_registry
    print("[OK] Strategy utils imported")

    registry = get_strategy_registry()
    print(f"[OK] Strategy registry loaded: {len(registry)} strategies")

    from gui.views.setup_view import SetupView
    print("[OK] Setup view imported")

    from gui.views.execution_view import ExecutionView
    print("[OK] Execution view imported")

    from gui.views.results_view import ResultsView
    print("[OK] Results view imported")

    from gui.workers.gui_controller import GUIBacktestController
    print("[OK] GUI controller imported")

    print("\n" + "="*60)
    print("All imports successful!")
    print("="*60)
    print("\nRun GUI:")
    print("  python scripts/run_gui.py")
    print("  OR")
    print("  cd src && python -m gui")
    print("="*60)

except Exception as e:
    print(f"[FAIL] Import error: {e}")
    import traceback
    traceback.print_exc()
