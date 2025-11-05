"""
Parameter validation utilities.
"""

from typing import Any, Dict, List, Optional


def validate_parameters(params: Dict[str, Any], rules: Dict[str, Dict[str, Any]]) -> None:
    """
    Validate strategy parameters against rules.

    Args:
        params: Dictionary of parameter values
        rules: Dictionary of validation rules per parameter

    Raises:
        ValueError: If validation fails

    Example:
        rules = {
            'window': {'type': int, 'min': 1, 'max': 200},
            'threshold': {'type': float, 'min': 0.0, 'max': 1.0}
        }
        validate_parameters({'window': 20, 'threshold': 0.5}, rules)
    """
    for param_name, rule in rules.items():
        if param_name not in params:
            if rule.get('required', False):
                raise ValueError(f"Required parameter '{param_name}' is missing")
            continue

        value = params[param_name]

        if 'type' in rule and not isinstance(value, rule['type']):
            raise ValueError(
                f"Parameter '{param_name}' must be of type {rule['type'].__name__}, "
                f"got {type(value).__name__}"
            )

        if 'min' in rule and value < rule['min']:
            raise ValueError(
                f"Parameter '{param_name}' must be >= {rule['min']}, got {value}"
            )

        if 'max' in rule and value > rule['max']:
            raise ValueError(
                f"Parameter '{param_name}' must be <= {rule['max']}, got {value}"
            )

        if 'choices' in rule and value not in rule['choices']:
            raise ValueError(
                f"Parameter '{param_name}' must be one of {rule['choices']}, got {value}"
            )


def validate_positive_int(value: Any, name: str) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_positive_float(value: Any, name: str) -> None:
    """
    Validate that a value is a positive float.

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(value: Any, name: str, min_val: float, max_val: float) -> None:
    """
    Validate that a value is within a range.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
