"""
Base strategy class for defining trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, List
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


class MultiSymbolStrategy(BaseStrategy):
    """
    Base class for strategies that trade multiple symbols simultaneously.

    Multi-symbol strategies generate signals across multiple correlated assets.
    Examples: pairs trading, statistical arbitrage, cross-sectional momentum.

    Subclasses must implement:
    - get_required_symbols(): Specify symbol requirements
    - generate_multi_signals(): Generate signals across multiple symbols
    """

    @abstractmethod
    def get_required_symbols(self) -> Union[int, List[str]]:
        """
        Specify symbol requirements for this strategy.

        Returns:
            int: Number of symbols required (e.g., 2 for pairs trading)
            List[str]: Specific symbols required (e.g., ['SPY', 'TLT'])

        Examples:
            >>> class PairsStrategy(MultiSymbolStrategy):
            ...     def get_required_symbols(self):
            ...         return 2  # Any 2 symbols

            >>> class SPYTLTStrategy(MultiSymbolStrategy):
            ...     def get_required_symbols(self):
            ...         return ['SPY', 'TLT']  # Specific pair
        """
        pass

    @abstractmethod
    def generate_multi_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[
        Tuple[pd.Series, pd.Series],
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
    ]]:
        """
        Generate signals across multiple symbols.

        Args:
            data_dict: Dictionary mapping symbol names to OHLCV DataFrames
                Example: {'AAPL': df1, 'MSFT': df2}

        Returns:
            Dictionary mapping symbols to signal tuples:
            - Long-only: {symbol: (entries, exits)}
            - Long-short: {symbol: (long_entries, long_exits, short_entries, short_exits)}

        Example (Long-Short Pairs):
            >>> def generate_multi_signals(self, data_dict):
            ...     sym1, sym2 = list(data_dict.keys())
            ...     # Calculate spread, generate signals
            ...     return {
            ...         sym1: (long_entries, long_exits, short_entries, short_exits),
            ...         sym2: (long_entries, long_exits, short_entries, short_exits)
            ...     }
        """
        pass

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Prevent single-symbol usage of multi-symbol strategies.

        Raises:
            NotImplementedError: Always raised for multi-symbol strategies
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a multi-symbol strategy that requires "
            f"multiple symbols to generate signals. Use BacktestEngine.run() with "
            f"a list of symbols: engine.run(strategy, symbols=['SYM1', 'SYM2'], ...)"
        )
