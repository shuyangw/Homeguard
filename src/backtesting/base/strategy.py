"""
Base strategy class for defining trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement generate_signals() method.
    """

    def __init__(self, **params):
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
        self.validate_parameters()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on price data.

        Args:
            data: DataFrame with OHLCV columns and timestamp index

        Returns:
            Tuple of (entries, exits) as boolean Series
            - entries: True where strategy signals entry
            - exits: True where strategy signals exit
        """
        pass

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Override in subclass to add custom validation.
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.

        Returns:
            Dictionary of parameter names and values
        """
        return self.params.copy()

    def set_parameters(self, **params) -> None:
        """
        Update strategy parameters.

        Args:
            **params: Parameters to update
        """
        self.params.update(params)
        self.validate_parameters()

    def __repr__(self) -> str:
        """
        String representation of strategy.
        """
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"

    def __str__(self) -> str:
        """
        Human-readable string representation.
        """
        return self.__repr__()


class LongOnlyStrategy(BaseStrategy):
    """
    Base class for long-only strategies (no short positions).
    """

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long-only entry and exit signals.

        Returns:
            Tuple of (entries, exits) for long positions only
        """
        entries, exits = self.generate_long_signals(data)
        return entries, exits

    @abstractmethod
    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Tuple of (long_entries, long_exits) as boolean Series
        """
        pass


class LongShortStrategy(BaseStrategy):
    """
    Base class for long-short strategies.
    """

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry and exit signals.

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        return self.generate_long_short_signals(data)

    @abstractmethod
    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry and exit signals.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        pass
