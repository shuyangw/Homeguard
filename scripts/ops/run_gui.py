"""
Launcher script for the backtesting GUI.

Usage:
    python scripts/run_gui.py
"""

import sys
from pathlib import Path

# Add src directory to Python path (from scripts/ go up to project root, then into src/)
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import and run the Flet app
from gui.app import main
import flet as ft

if __name__ == "__main__":
    print("Starting Backtest Runner GUI...")
    print("Press Ctrl+C to exit")
    ft.app(target=main)
