"""Futures validation checks across all 4 layers.

Importing this package registers all auto-registering checks.
"""
from src.data.validation.futures.checks import structural  # noqa: F401
from src.data.validation.futures.checks import statistical  # noqa: F401
from src.data.validation.futures.checks import cross_source  # noqa: F401
from src.data.validation.futures.checks import external  # noqa: F401
