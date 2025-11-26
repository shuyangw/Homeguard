"""
Unified strategy registry for config-driven backtesting.

Provides dynamic strategy lookup by name with lazy imports to avoid
import chain issues between backtesting and live trading modules.

Usage:
    from src.strategies.registry import get_strategy_class, list_strategies

    # Get strategy by class name or display name
    strategy_cls = get_strategy_class("MovingAverageCrossover")
    strategy_cls = get_strategy_class("Moving Average Crossover")

    # List all available strategies
    strategies = list_strategies()
"""

from typing import Dict, Type, Any, Optional, List, Tuple
import inspect

from src.backtesting.base.strategy import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Registry mapping strategy name -> (module_path, class_name)
# Using lazy loading to avoid import chain issues
_STRATEGY_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Base strategies
    "MovingAverageCrossover": ("src.strategies.base_strategies.moving_average", "MovingAverageCrossover"),
    "TripleMovingAverage": ("src.strategies.base_strategies.moving_average", "TripleMovingAverage"),
    "MeanReversion": ("src.strategies.base_strategies.mean_reversion", "MeanReversion"),
    "RSIMeanReversion": ("src.strategies.base_strategies.mean_reversion", "RSIMeanReversion"),
    "MeanReversionLongShort": ("src.strategies.base_strategies.mean_reversion_long_short", "MeanReversionLongShort"),
    "MomentumStrategy": ("src.strategies.base_strategies.momentum", "MomentumStrategy"),
    "BreakoutStrategy": ("src.strategies.base_strategies.momentum", "BreakoutStrategy"),

    # Advanced strategies
    "VolatilityTargetedMomentum": ("src.strategies.advanced.volatility_targeted_momentum", "VolatilityTargetedMomentum"),
    "OvernightMeanReversion": ("src.strategies.advanced.overnight_mean_reversion", "OvernightMeanReversionStrategy"),
    "OvernightMeanReversionStrategy": ("src.strategies.advanced.overnight_mean_reversion", "OvernightMeanReversionStrategy"),
    "CrossSectionalMomentum": ("src.strategies.advanced.cross_sectional_momentum", "CrossSectionalMomentum"),
    "PairsTrading": ("src.strategies.advanced.pairs_trading", "PairsTrading"),
}

# Display name -> class name mapping for user-friendly config files
_DISPLAY_NAME_MAP: Dict[str, str] = {
    "Moving Average Crossover": "MovingAverageCrossover",
    "Triple Moving Average": "TripleMovingAverage",
    "Mean Reversion": "MeanReversion",
    "RSI Mean Reversion": "RSIMeanReversion",
    "Mean Reversion Long Short": "MeanReversionLongShort",
    "Momentum Strategy": "MomentumStrategy",
    "Momentum": "MomentumStrategy",
    "Breakout Strategy": "BreakoutStrategy",
    "Breakout": "BreakoutStrategy",
    "Volatility Targeted Momentum": "VolatilityTargetedMomentum",
    "Overnight Mean Reversion": "OvernightMeanReversion",
    "OMR": "OvernightMeanReversion",
    "Cross-Sectional Momentum": "CrossSectionalMomentum",
    "Cross Sectional Momentum": "CrossSectionalMomentum",
    "Pairs Trading": "PairsTrading",
    "Pairs": "PairsTrading",
}

# Cache for loaded strategy classes
_CLASS_CACHE: Dict[str, Type[BaseStrategy]] = {}


def _resolve_strategy_name(name: str) -> str:
    """
    Resolve a strategy name to its canonical class name.

    Args:
        name: Strategy name (class name or display name)

    Returns:
        Canonical class name
    """
    # Check if it's already a class name
    if name in _STRATEGY_REGISTRY:
        return name

    # Check display name map
    if name in _DISPLAY_NAME_MAP:
        return _DISPLAY_NAME_MAP[name]

    # Case-insensitive search in display names
    name_lower = name.lower()
    for display_name, class_name in _DISPLAY_NAME_MAP.items():
        if display_name.lower() == name_lower:
            return class_name

    # Case-insensitive search in class names
    for class_name in _STRATEGY_REGISTRY:
        if class_name.lower() == name_lower:
            return class_name

    raise ValueError(f"Unknown strategy: '{name}'. Available: {list_strategies()}")


def _load_strategy_class(class_name: str) -> Type[BaseStrategy]:
    """
    Dynamically load a strategy class by name.

    Args:
        class_name: Canonical class name from registry

    Returns:
        Strategy class
    """
    if class_name in _CLASS_CACHE:
        return _CLASS_CACHE[class_name]

    if class_name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{class_name}' not in registry")

    module_path, actual_class_name = _STRATEGY_REGISTRY[class_name]

    try:
        import importlib
        module = importlib.import_module(module_path)
        strategy_cls = getattr(module, actual_class_name)

        # Cache for future use
        _CLASS_CACHE[class_name] = strategy_cls
        return strategy_cls

    except ImportError as e:
        logger.error(f"Failed to import strategy module '{module_path}': {e}")
        raise ImportError(f"Could not load strategy '{class_name}' from '{module_path}': {e}")
    except AttributeError as e:
        logger.error(f"Strategy class '{actual_class_name}' not found in '{module_path}'")
        raise AttributeError(f"Strategy class '{actual_class_name}' not found in module '{module_path}'")


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    """
    Get a strategy class by name (class name or display name).

    Args:
        name: Strategy name (e.g., "MovingAverageCrossover" or "Moving Average Crossover")

    Returns:
        Strategy class

    Raises:
        ValueError: If strategy name is unknown
        ImportError: If strategy module cannot be loaded

    Example:
        >>> strategy_cls = get_strategy_class("MovingAverageCrossover")
        >>> strategy = strategy_cls(fast_period=10, slow_period=50)
    """
    class_name = _resolve_strategy_name(name)
    return _load_strategy_class(class_name)


def list_strategies() -> List[str]:
    """
    Get list of all available strategy class names.

    Returns:
        List of strategy class names
    """
    return sorted(_STRATEGY_REGISTRY.keys())


def list_strategy_display_names() -> Dict[str, str]:
    """
    Get mapping of display names to class names.

    Returns:
        Dictionary mapping display name -> class name
    """
    return _DISPLAY_NAME_MAP.copy()


def get_strategy_info(name: str) -> Dict[str, Any]:
    """
    Get information about a strategy including parameters.

    Args:
        name: Strategy name

    Returns:
        Dictionary with strategy info:
        - class_name: Canonical class name
        - module: Module path
        - description: First line of docstring
        - parameters: Dict of parameter name -> default value
    """
    class_name = _resolve_strategy_name(name)
    module_path, actual_class_name = _STRATEGY_REGISTRY[class_name]

    strategy_cls = get_strategy_class(class_name)

    # Get description from docstring
    description = ""
    if strategy_cls.__doc__:
        lines = [line.strip() for line in strategy_cls.__doc__.split('\n') if line.strip()]
        description = lines[0] if lines else ""

    # Get parameters
    parameters = _get_strategy_parameters(strategy_cls)

    return {
        "class_name": class_name,
        "module": module_path,
        "description": description,
        "parameters": parameters,
    }


def _get_strategy_parameters(strategy_cls: Type[BaseStrategy]) -> Dict[str, Any]:
    """
    Extract parameters and default values from strategy class.

    Args:
        strategy_cls: Strategy class

    Returns:
        Dictionary mapping parameter name -> default value
    """
    try:
        # Try to instantiate with defaults and get params
        strategy_instance = strategy_cls()
        return strategy_instance.params.copy()
    except Exception:
        # Fallback to signature inspection
        sig = inspect.signature(strategy_cls.__init__)
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            default = param.default
            if default is inspect.Parameter.empty:
                default = None

            params[param_name] = default

        return params


def register_strategy(
    name: str,
    strategy_cls: Type[BaseStrategy],
    display_name: Optional[str] = None
) -> None:
    """
    Register a custom strategy at runtime.

    Args:
        name: Class name for the strategy
        strategy_cls: Strategy class to register
        display_name: Optional user-friendly display name

    Example:
        >>> from src.strategies.registry import register_strategy
        >>> register_strategy("MyCustomStrategy", MyCustomStrategy, "My Custom Strategy")
    """
    # Add to class cache directly since we have the class
    _CLASS_CACHE[name] = strategy_cls

    # Add to registry with module path
    _STRATEGY_REGISTRY[name] = (strategy_cls.__module__, strategy_cls.__name__)

    # Add display name if provided
    if display_name:
        _DISPLAY_NAME_MAP[display_name] = name

    logger.info(f"Registered custom strategy: {name}")


def clear_cache() -> None:
    """Clear the strategy class cache (useful for testing)."""
    _CLASS_CACHE.clear()
