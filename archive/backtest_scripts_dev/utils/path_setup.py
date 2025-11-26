"""
Standardized path setup for all backtest scripts.

This module provides a single, consistent way to set up Python paths
for importing project modules. All backtest scripts should use this
instead of custom sys.path manipulation.

Usage:
    from utils.path_setup import setup_project_paths
    setup_project_paths()

This replaces various patterns like:
    - sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    - sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    - sys.path.insert(0, str(Path(__file__).parent.parent))
"""

import sys
from pathlib import Path


def setup_project_paths() -> Path:
    """
    Add project root to Python path if not already present.

    This function:
    1. Resolves the project root (two levels up from this file)
    2. Adds project root to sys.path if not already there
    3. Returns the project root Path for convenience

    Returns:
        Path: The project root directory

    Example:
        >>> from utils.path_setup import setup_project_paths
        >>> ROOT_DIR = setup_project_paths()
        >>> # Now can import from src/
        >>> from src.engine.backtester import Backtester
    """
    # Get project root (backtest_scripts/utils/__file__ -> backtest_scripts -> project_root)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Add to sys.path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root


def get_project_root() -> Path:
    """
    Get the project root directory without modifying sys.path.

    Returns:
        Path: The project root directory

    Example:
        >>> from utils.path_setup import get_project_root
        >>> ROOT_DIR = get_project_root()
        >>> data_dir = ROOT_DIR / "data" / "cache"
    """
    return Path(__file__).resolve().parent.parent.parent
