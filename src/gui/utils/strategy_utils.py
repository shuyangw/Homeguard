"""
Utilities for strategy introspection and parameter extraction.
"""

import inspect
from typing import Dict, Any, List, Type
from src.backtesting.base.strategy import BaseStrategy


def get_strategy_registry() -> Dict[str, Type[BaseStrategy]]:
    """
    Get all available strategy classes.

    Returns:
        Dictionary mapping strategy display name -> strategy class
    """
    from src.strategies.registry import (
        get_strategy_class,
        list_strategy_display_names
    )

    # Build registry from display names
    display_names = list_strategy_display_names()
    registry = {}

    for display_name, class_name in display_names.items():
        try:
            registry[display_name] = get_strategy_class(class_name)
        except (ImportError, AttributeError):
            # Skip strategies that can't be loaded
            pass

    return registry


def get_strategy_parameters(strategy_class: Type[BaseStrategy]) -> Dict[str, Any]:
    """
    Extract __init__ parameters and their ACTUAL default values from a strategy class.

    This function instantiates the strategy with default arguments to get the
    actual runtime defaults (e.g., [63, 126, 252] instead of None for lookback_periods).

    Args:
        strategy_class: Strategy class to introspect

    Returns:
        Dictionary mapping parameter name -> actual default value
        Skips 'self' parameter
    """
    try:
        # Instantiate strategy with all defaults to get actual runtime values
        strategy_instance = strategy_class()

        # Get parameters from the instance's params dict
        params = strategy_instance.params.copy()

        return params
    except Exception:
        # Fallback to signature inspection if instantiation fails
        sig = inspect.signature(strategy_class.__init__)
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            default = param.default
            if default is inspect.Parameter.empty:
                # No default value, skip or use None
                default = None

            params[param_name] = default

        return params


def get_strategy_param_types(strategy_class: Type[BaseStrategy]) -> Dict[str, type]:
    """
    Extract parameter type annotations from strategy class.

    Args:
        strategy_class: Strategy class to introspect

    Returns:
        Dictionary mapping parameter name -> type annotation
    """
    sig = inspect.signature(strategy_class.__init__)
    types = {}

    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue

        annotation = param.annotation
        if annotation is not inspect.Parameter.empty:
            types[param_name] = annotation
        else:
            # Infer type from default value
            default = param.default
            if default is not inspect.Parameter.empty and default is not None:
                types[param_name] = type(default)
            else:
                types[param_name] = str  # Default to string

    return types


def get_strategy_description(strategy_class: Type[BaseStrategy]) -> str:
    """
    Extract docstring description from strategy class.

    Args:
        strategy_class: Strategy class

    Returns:
        First line of docstring, or empty string if no docstring
    """
    doc = strategy_class.__doc__
    if not doc:
        return ""

    # Get first non-empty line
    lines = [line.strip() for line in doc.split('\n') if line.strip()]
    return lines[0] if lines else ""


def format_strategy_info(strategy_class: Type[BaseStrategy]) -> str:
    """
    Format strategy information for display.

    Args:
        strategy_class: Strategy class

    Returns:
        Formatted string with strategy name, description, and parameters
    """
    name = strategy_class.__name__
    desc = get_strategy_description(strategy_class)
    params = get_strategy_parameters(strategy_class)

    info = f"{name}\n{desc}\n"
    if params:
        info += "\nParameters:\n"
        for param_name, default in params.items():
            info += f"  - {param_name}: {default}\n"

    return info
