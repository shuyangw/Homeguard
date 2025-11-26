"""
Configuration loader utility for backtest scripts.

This module provides utilities for loading and merging YAML configuration files
with support for inheritance (extends) and CLI overrides.

Usage:
    from utils.config_loader import load_config

    # Load config with CLI overrides
    config = load_config("config/pairs_trading.yaml", {
        "backtest.start_date": "2023-01-01",
        "symbols": ["SPY", "QQQ"]
    })
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary containing YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file contains invalid YAML
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary (base is not modified)
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def apply_overrides(config: Dict, overrides: Dict[str, Any]) -> Dict:
    """
    Apply dot-notation overrides to config dictionary.

    Args:
        config: Base configuration dictionary
        overrides: Overrides in dot notation (e.g., {"backtest.start_date": "2023-01-01"})

    Returns:
        Config with overrides applied

    Example:
        >>> config = {"backtest": {"start_date": "2020-01-01"}}
        >>> overrides = {"backtest.start_date": "2023-01-01"}
        >>> apply_overrides(config, overrides)
        {"backtest": {"start_date": "2023-01-01"}}
    """
    result = deepcopy(config)

    for key, value in overrides.items():
        # Split dot notation (e.g., "backtest.start_date" -> ["backtest", "start_date"])
        parts = key.split('.')

        # Navigate to the parent dictionary
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    return result


def resolve_extends(config: Dict, config_dir: Path) -> Dict:
    """
    Resolve 'extends' directive by loading parent config and merging.

    Args:
        config: Configuration dictionary that may contain 'extends' key
        config_dir: Directory containing config files

    Returns:
        Config with parent config merged in
    """
    if 'extends' not in config:
        return config

    parent_file = config_dir / config['extends']
    parent_config = load_yaml(parent_file)

    # Recursively resolve parent's extends
    parent_config = resolve_extends(parent_config, config_dir)

    # Remove 'extends' key from current config
    config_copy = deepcopy(config)
    del config_copy['extends']

    # Merge parent with current (current takes precedence)
    return merge_dicts(parent_config, config_copy)


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with support for inheritance and overrides.

    Features:
    - Loads YAML configuration file
    - Resolves 'extends' directive (config inheritance)
    - Applies CLI overrides using dot notation

    Args:
        config_path: Path to config file (relative to config_dir or absolute)
        overrides: Optional dictionary of overrides in dot notation
        config_dir: Base directory for config files (defaults to backtest_scripts/config)

    Returns:
        Fully resolved configuration dictionary

    Example:
        >>> # Load default config
        >>> config = load_config("default_backtest.yaml")

        >>> # Load with overrides
        >>> config = load_config("pairs_trading.yaml", {
        ...     "backtest.start_date": "2023-01-01",
        ...     "symbols": ["SPY", "QQQ"]
        ... })
    """
    # Default to backtest_scripts/config directory
    if config_dir is None:
        # Get backtest_scripts directory (parent of utils)
        utils_dir = Path(__file__).parent
        backtest_scripts_dir = utils_dir.parent
        config_dir = backtest_scripts_dir / "config"

    # Resolve config path
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = config_dir / config_file

    # Load base config
    config = load_yaml(config_file)

    # Resolve extends (inheritance)
    config = resolve_extends(config, config_dir)

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)

    return config


def get_nested(config: Dict, key: str, default: Any = None) -> Any:
    """
    Get nested value from config using dot notation.

    Args:
        config: Configuration dictionary
        key: Dot notation key (e.g., "backtest.start_date")
        default: Default value if key not found

    Returns:
        Value at key path or default

    Example:
        >>> config = {"backtest": {"start_date": "2020-01-01"}}
        >>> get_nested(config, "backtest.start_date")
        "2020-01-01"
        >>> get_nested(config, "backtest.missing", default="default")
        "default"
    """
    parts = key.split('.')
    current = config

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current
