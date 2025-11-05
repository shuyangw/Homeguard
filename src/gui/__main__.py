"""
Entry point for running the GUI as a module: python -m gui
"""

import flet as ft
from gui.app import main

if __name__ == "__main__":
    print("Starting Backtest Runner GUI...")
    print("Press Ctrl+C to exit")
    ft.app(target=main)
