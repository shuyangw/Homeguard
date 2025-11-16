"""
Root conftest.py to configure pytest for all test discovery.

Adds src/ to Python path so all imports work correctly.
"""

import sys
from pathlib import Path

# Add src to path for all tests
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

# Exclude integration tests that require live API credentials
collect_ignore = [
    "backtest_scripts/test_vix_optimization.py",  # Integration test - requires Alpaca credentials
    "backtest_scripts/test_position_limits.py",   # Integration test - requires historical data
    "scripts/test_json_fix.py",                   # Manual verification script
]
