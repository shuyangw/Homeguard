"""
Run history tracker for GUI backtests.

Tracks past backtest executions with timestamps, configuration, and results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class RunHistory:
    """
    Tracks history of backtest runs.

    Stores:
    - Timestamp
    - Strategy name and parameters
    - Symbols tested
    - Date range
    - Summary results (if available)
    """

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize run history.

        Args:
            history_file: Path to history JSON file (defaults to gui_config/run_history.json)
        """
        if history_file is None:
            from config import PROJECT_ROOT
            config_dir = PROJECT_ROOT / "gui_config"
            config_dir.mkdir(parents=True, exist_ok=True)
            history_file = config_dir / "run_history.json"

        self.history_file = Path(history_file)

    def add_run(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: Dict[str, Any],
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a backtest run to history.

        Args:
            strategy_name: Name of the strategy
            symbols: List of symbols tested
            start_date: Start date of backtest
            end_date: End date of backtest
            config: Full configuration dict
            results: Optional results summary
        """
        history = self._load_history()

        run_entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'symbols': symbols,
            'num_symbols': len(symbols),
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': config.get('initial_capital', 100000),
            'fees': config.get('fees', 0.0),
            'workers': config.get('workers', 1),
            'results': results if results else {}
        }

        # Add to beginning of list (most recent first)
        history.insert(0, run_entry)

        # Keep only last 50 runs
        if len(history) > 50:
            history = history[:50]

        self._save_history(history)

    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent backtest runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run entries (most recent first)
        """
        history = self._load_history()
        return history[:limit]

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """
        Get all backtest runs.

        Returns:
            List of all run entries (most recent first)
        """
        return self._load_history()

    def clear_history(self) -> None:
        """Clear all run history."""
        self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load run history from disk."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        """Save run history to disk."""
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
