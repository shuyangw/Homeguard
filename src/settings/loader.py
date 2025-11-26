"""
Configuration loader with YAML support and inheritance.

Provides utilities for loading and merging YAML configuration files
with support for inheritance (extends) and CLI overrides.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

from src.settings.schema import BacktestConfig
from src.settings.defaults import DEFAULT_CONFIG


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

    with open(file_path, 'r', encoding='utf-8') as f:
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
        parts = key.split('.')
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
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

    parent_ref = config['extends']

    # Handle special 'default' reference
    if parent_ref == 'default':
        parent_config = deepcopy(DEFAULT_CONFIG)
    else:
        parent_file = config_dir / parent_ref
        if not parent_file.suffix:
            parent_file = parent_file.with_suffix('.yaml')
        parent_config = load_yaml(parent_file)
        parent_config = resolve_extends(parent_config, config_dir)

    config_copy = deepcopy(config)
    del config_copy['extends']

    return merge_dicts(parent_config, config_copy)


def get_nested(config: Dict, key: str, default: Any = None) -> Any:
    """
    Get nested value from config using dot notation.

    Args:
        config: Configuration dictionary
        key: Dot notation key (e.g., "backtest.start_date")
        default: Default value if key not found

    Returns:
        Value at key path or default
    """
    parts = key.split('.')
    current = config

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


def load_config_dict(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML file as dictionary.

    Features:
    - Loads YAML configuration file
    - Resolves 'extends' directive (config inheritance)
    - Applies CLI overrides using dot notation
    - Merges with defaults

    Args:
        config_path: Path to config file
        overrides: Optional dictionary of overrides in dot notation
        config_dir: Base directory for config files

    Returns:
        Fully resolved configuration dictionary
    """
    config_file = Path(config_path)

    # Set config_dir for resolving 'extends' references
    if config_dir is None:
        if config_file.is_absolute():
            config_dir = config_file.parent
        else:
            # For relative paths, resolve relative to current working directory
            config_file = config_file.resolve()
            config_dir = config_file.parent

    # Start with defaults
    config = deepcopy(DEFAULT_CONFIG)

    # Load and merge file config
    file_config = load_yaml(config_file)
    file_config = resolve_extends(file_config, config_dir)
    config = merge_dicts(config, file_config)

    # Apply CLI overrides
    if overrides:
        config = apply_overrides(config, overrides)

    return config


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None
) -> BacktestConfig:
    """
    Load configuration from YAML file with Pydantic validation.

    Args:
        config_path: Path to config file
        overrides: Optional dictionary of overrides in dot notation
        config_dir: Base directory for config files

    Returns:
        Validated BacktestConfig object

    Raises:
        ValidationError: If config fails validation
        FileNotFoundError: If config file not found
    """
    config_dict = load_config_dict(config_path, overrides, config_dir)
    return BacktestConfig.model_validate(config_dict)


def validate_config(config_dict: Dict[str, Any]) -> BacktestConfig:
    """
    Validate a configuration dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Validated BacktestConfig object

    Raises:
        ValidationError: If config fails validation
    """
    return BacktestConfig.model_validate(config_dict)
