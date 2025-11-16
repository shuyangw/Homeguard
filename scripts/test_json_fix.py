"""
Quick test to verify JSON serialization fix for strategy_class.
"""

import sys
from pathlib import Path

# conftest.py already adds src to path

from src.strategies.base_strategies.moving_average import MovingAverageCrossover
from src.gui.utils.config_manager import ConfigManager
import tempfile

# Create a test config with strategy_class
test_config = {
    'strategy_class': MovingAverageCrossover,  # This is a class object
    'strategy_params': {'fast_window': 10, 'slow_window': 50},
    'symbols': ['AAPL', 'MSFT'],
    'start_date': '2023-01-01',
    'end_date': '2024-01-01',
    'initial_capital': 100000,
    'fees': 0.001,
    'workers': 4,
    'parallel': True,
    'generate_full_output': True
}

# Create a temp config manager
temp_dir = Path(tempfile.mkdtemp())
config_manager = ConfigManager(config_dir=temp_dir)

print("Testing JSON serialization fix...")
print(f"Original strategy_class type: {type(test_config['strategy_class'])}")
print(f"Original strategy_class value: {test_config['strategy_class']}")

try:
    # Try to save as last_run (this should NOT crash)
    config_manager.save_last_run(test_config)
    print("[OK] save_last_run() succeeded - no JSON serialization error!")

    # Try to load it back
    loaded_config = config_manager.load_last_run()
    print(f"[OK] load_last_run() succeeded")
    print(f"Loaded strategy_class type: {type(loaded_config['strategy_class'])}")
    print(f"Loaded strategy_class value: {loaded_config['strategy_class']}")

    assert loaded_config['strategy_class'] == 'MovingAverageCrossover', "Strategy name should be preserved"
    print("[OK] Strategy class correctly serialized as string name")

    # Try preset save/load
    config_manager.save_preset('TestPreset', test_config)
    print("[OK] save_preset() succeeded")

    loaded_preset = config_manager.load_preset('TestPreset')
    print("[OK] load_preset() succeeded")
    assert loaded_preset['strategy_class'] == 'MovingAverageCrossover'
    print("[OK] Preset correctly serialized")

    print("\n=== ALL TESTS PASSED ===")
    print("JSON serialization fix is working correctly!")

except Exception as e:
    print(f"\n[FAIL] Test failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
