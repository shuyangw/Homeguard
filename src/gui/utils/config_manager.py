"""
Configuration manager for GUI presets and symbol lists.

Handles saving/loading:
- Full backtest configurations (presets)
- Symbol list presets
- Last run configuration (for quick re-run)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ConfigManager:
    """
    Manages GUI configuration presets and symbol lists.

    Storage location: <project_root>/gui_config/
    - presets.json: Saved backtest configurations
    - symbol_lists.json: Saved symbol groups
    - last_run.json: Last executed configuration
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.

        Args:
            config_dir: Optional custom config directory.
                       Defaults to project_root/gui_config/
        """
        if config_dir is None:
            # Use project root/gui_config
            from config import PROJECT_ROOT
            config_dir = PROJECT_ROOT / "gui_config"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.presets_file = self.config_dir / "presets.json"
        self.symbol_lists_file = self.config_dir / "symbol_lists.json"
        self.last_run_file = self.config_dir / "last_run.json"

    # ========== Configuration Presets ==========

    def _make_config_serializable(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make config JSON-serializable by converting class objects to names.

        Args:
            config: Original config with potential class objects

        Returns:
            Serializable config dict
        """
        serializable = config.copy()

        # Convert strategy_class to its name string
        if 'strategy_class' in serializable and serializable['strategy_class'] is not None:
            serializable['strategy_class'] = serializable['strategy_class'].__name__

        return serializable

    def save_preset(self, name: str, config: Dict[str, Any]) -> None:
        """
        Save a backtest configuration preset.

        Args:
            name: Preset name (user-defined)
            config: Configuration dictionary from setup view
        """
        presets = self._load_presets()

        # Make config JSON-serializable
        serializable_config = self._make_config_serializable(config)

        # Add metadata
        preset_data = {
            'config': serializable_config,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat()
        }

        # Update if exists, add if new
        if name in presets:
            preset_data['created'] = presets[name].get('created', datetime.now().isoformat())

        presets[name] = preset_data

        self._save_presets(presets)

    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration preset by name.

        Args:
            name: Preset name

        Returns:
            Configuration dictionary or None if not found
        """
        presets = self._load_presets()
        if name in presets:
            return presets[name]['config']
        return None

    def get_preset_names(self) -> List[str]:
        """
        Get list of all saved preset names.

        Returns:
            List of preset names (sorted alphabetically)
        """
        presets = self._load_presets()
        return sorted(presets.keys())

    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset.

        Args:
            name: Preset name

        Returns:
            True if deleted, False if not found
        """
        presets = self._load_presets()
        if name in presets:
            del presets[name]
            self._save_presets(presets)
            return True
        return False

    def _load_presets(self) -> Dict[str, Any]:
        """Load all presets from disk."""
        if not self.presets_file.exists():
            return {}

        try:
            with open(self.presets_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_presets(self, presets: Dict[str, Any]) -> None:
        """Save presets to disk."""
        with open(self.presets_file, 'w') as f:
            json.dump(presets, f, indent=2)

    # ========== Symbol Lists ==========

    def save_symbol_list(self, name: str, symbols: List[str]) -> None:
        """
        Save a symbol list preset.

        Args:
            name: List name (e.g., "Tech Stocks", "S&P 500")
            symbols: List of ticker symbols
        """
        symbol_lists = self._load_symbol_lists()

        symbol_lists[name] = {
            'symbols': symbols,
            'created': datetime.now().isoformat(),
            'count': len(symbols)
        }

        self._save_symbol_lists(symbol_lists)

    def load_symbol_list(self, name: str) -> Optional[List[str]]:
        """
        Load a symbol list by name.

        Args:
            name: List name

        Returns:
            List of symbols or None if not found
        """
        symbol_lists = self._load_symbol_lists()
        if name in symbol_lists:
            return symbol_lists[name]['symbols']
        return None

    def get_symbol_list_names(self) -> List[str]:
        """
        Get list of all saved symbol list names.

        Returns:
            List of symbol list names (sorted)
        """
        symbol_lists = self._load_symbol_lists()
        return sorted(symbol_lists.keys())

    def get_symbol_list_info(self) -> Dict[str, int]:
        """
        Get symbol list names with symbol counts.

        Returns:
            Dict mapping list name to symbol count
        """
        symbol_lists = self._load_symbol_lists()
        return {name: data['count'] for name, data in symbol_lists.items()}

    def delete_symbol_list(self, name: str) -> bool:
        """
        Delete a symbol list.

        Args:
            name: List name

        Returns:
            True if deleted, False if not found
        """
        symbol_lists = self._load_symbol_lists()
        if name in symbol_lists:
            del symbol_lists[name]
            self._save_symbol_lists(symbol_lists)
            return True
        return False

    def _load_symbol_lists(self) -> Dict[str, Any]:
        """Load all symbol lists from disk."""
        if not self.symbol_lists_file.exists():
            return {}

        try:
            with open(self.symbol_lists_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_symbol_lists(self, symbol_lists: Dict[str, Any]) -> None:
        """Save symbol lists to disk."""
        with open(self.symbol_lists_file, 'w') as f:
            json.dump(symbol_lists, f, indent=2)

    # ========== Last Run (Quick Re-run) ==========

    def save_last_run(self, config: Dict[str, Any]) -> None:
        """
        Save the last executed configuration for quick re-run.

        Args:
            config: Configuration dictionary
        """
        # Make config JSON-serializable
        serializable_config = self._make_config_serializable(config)

        last_run_data = {
            'config': serializable_config,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.last_run_file, 'w') as f:
            json.dump(last_run_data, f, indent=2)

    def load_last_run(self) -> Optional[Dict[str, Any]]:
        """
        Load the last executed configuration.

        Returns:
            Configuration dictionary or None if no last run
        """
        if not self.last_run_file.exists():
            return None

        try:
            with open(self.last_run_file, 'r') as f:
                data = json.load(f)
                return data.get('config')
        except Exception:
            return None

    def has_last_run(self) -> bool:
        """
        Check if a last run configuration exists.

        Returns:
            True if last run exists
        """
        return self.last_run_file.exists()
