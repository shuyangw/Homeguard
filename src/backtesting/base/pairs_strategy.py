"""
Pairs trading strategy base class.
"""

from abc import abstractmethod
from typing import Dict, Tuple
import pandas as pd

from backtesting.base.strategy import MultiSymbolStrategy


class PairsStrategy(MultiSymbolStrategy):
    """
    Base class for pairs trading strategies.

    Pairs trading strategies trade two correlated assets simultaneously,
    typically going long the undervalued asset and short the overvalued asset.

    The strategy profits from the convergence of the spread between the two assets.

    Key Features:
    - Always requires exactly 2 symbols
    - Generates long/short signals for both legs
    - Market-neutral (hedged) approach

    Subclasses must implement:
    - generate_pairs_signals(): Generate signals for the pair

    Examples:
        Statistical arbitrage, cointegration-based pairs trading, correlation trading
    """

    def get_required_symbols(self) -> int:
        """
        Pairs strategies always require exactly 2 symbols.

        Returns:
            2 (int): Fixed requirement for pairs trading
        """
        return 2

    @abstractmethod
    def generate_pairs_signals(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol1: str,
        symbol2: str
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """
        Generate trading signals for a pair of assets.

        This method receives price data for both symbols and should return
        synchronized long/short signals for both legs of the pair trade.

        Args:
            data1: OHLCV DataFrame for first symbol
            data2: OHLCV DataFrame for second symbol
            symbol1: Name of first symbol (e.g., 'AAPL')
            symbol2: Name of second symbol (e.g., 'MSFT')

        Returns:
            Dictionary mapping each symbol to a 4-tuple of signals:
            {
                symbol1: (long_entries, long_exits, short_entries, short_exits),
                symbol2: (long_entries, long_exits, short_entries, short_exits)
            }

            Each signal is a boolean pd.Series with same index as input data.

        Example:
            For a spread convergence strategy:
            - When spread is wide (z-score > 2.0):
              - Short the expensive asset (symbol1)
              - Long the cheap asset (symbol2)
            - When spread converges (z-score < 0.5):
              - Close both positions

            >>> def generate_pairs_signals(self, data1, data2, symbol1, symbol2):
            ...     spread = data2['close'] - hedge_ratio * data1['close']
            ...     z_score = (spread - spread.mean()) / spread.std()
            ...
            ...     # Long spread: short sym1, long sym2
            ...     long_spread_entry = z_score > 2.0
            ...     long_spread_exit = z_score < 0.5
            ...
            ...     return {
            ...         symbol1: (
            ...             pd.Series(False, index=data1.index),  # long_entries
            ...             pd.Series(False, index=data1.index),  # long_exits
            ...             long_spread_entry,  # short_entries
            ...             long_spread_exit    # short_exits
            ...         ),
            ...         symbol2: (
            ...             long_spread_entry,  # long_entries
            ...             long_spread_exit,   # long_exits
            ...             pd.Series(False, index=data2.index),  # short_entries
            ...             pd.Series(False, index=data2.index)   # short_exits
            ...         )
            ...     }

        Notes:
            - Signals for both symbols should be synchronized (same timestamps)
            - Typically, when one symbol is longed, the other is shorted (hedged pair)
            - Pairs should enter and exit simultaneously for proper hedging
            - Use hedge ratios to size positions appropriately (handled by portfolio simulator)
        """
        pass

    def generate_multi_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """
        Wrapper to call generate_pairs_signals() with pair data.

        This method validates that exactly 2 symbols are provided, then
        calls the strategy-specific generate_pairs_signals() method.

        Args:
            data_dict: Dictionary with exactly 2 symbols and their OHLCV data

        Returns:
            Dictionary of signals for both symbols

        Raises:
            ValueError: If data_dict does not contain exactly 2 symbols
        """
        if len(data_dict) != 2:
            raise ValueError(
                f"PairsStrategy requires exactly 2 symbols, got {len(data_dict)}: "
                f"{list(data_dict.keys())}"
            )

        symbols = list(data_dict.keys())
        return self.generate_pairs_signals(
            data_dict[symbols[0]],
            data_dict[symbols[1]],
            symbols[0],
            symbols[1]
        )
