"""
Abstract base class for pure strategy implementations.

No dependencies on:
- Backtesting engine (Portfolio, data access methods)
- Live trading infrastructure (BrokerInterface, ExecutionEngine)

Only depends on:
- pandas DataFrames (standard data structure)
- datetime (standard library)
- Signal (pure data structure)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd

from src.strategies.core.signal import Signal


class StrategySignals(ABC):
    """
    Pure signal generation strategy.

    Subclasses implement signal logic without any knowledge of
    how signals will be executed (backtest vs live trading).

    The adapter pattern is used to connect pure strategies to:
    - Backtesting engine (via BacktestAdapter)
    - Live trading infrastructure (via LiveTradingAdapter)
    """

    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate trading signals based on market data.

        This is the core method that implements the strategy logic.
        It should be pure - same inputs always produce same outputs.

        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
                        DataFrame columns: ['open', 'high', 'low', 'close', 'volume']
                        DataFrame index: DatetimeIndex
            timestamp: Current timestamp for signal generation

        Returns:
            List of Signal objects

        Example:
            ```python
            signals = strategy.generate_signals(
                market_data={
                    'AAPL': df_aapl,  # DataFrame with OHLCV
                    'MSFT': df_msft
                },
                timestamp=datetime.now()
            )
            ```
        """
        pass

    @abstractmethod
    def get_required_lookback(self) -> int:
        """
        Return number of periods needed for indicator calculation.

        This tells the data provider how much historical data is needed.

        Returns:
            Number of periods (bars) needed

        Examples:
            - MA crossover (50/200): return 200
            - RSI (14): return 14
            - Bollinger Bands (20): return 20
        """
        pass

    def validate_data(self, df: pd.DataFrame, symbol: str = None) -> Tuple[bool, str]:
        """
        Validate that DataFrame has required structure.

        Can be overridden by subclasses for custom validation.

        Args:
            df: DataFrame to validate
            symbol: Optional symbol name for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        symbol_str = f" for {symbol}" if symbol else ""

        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Missing columns{symbol_str}: {missing_columns}"

        # Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            return False, f"Index must be DatetimeIndex{symbol_str}, got {type(df.index)}"

        # Check data length
        required_length = self.get_required_lookback()
        if len(df) < required_length:
            return False, f"Insufficient data{symbol_str}: need {required_length} periods, got {len(df)}"

        # Check for NaN values in required columns
        nan_columns = [col for col in required_columns if df[col].isna().any()]
        if nan_columns:
            nan_counts = {col: df[col].isna().sum() for col in nan_columns}
            return False, f"NaN values found{symbol_str}: {nan_counts}"

        # Check for non-positive prices
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                return False, f"Non-positive {col} prices found{symbol_str}"

        # Check for invalid OHLC relationships
        invalid_high = (df['high'] < df['low']).any()
        if invalid_high:
            return False, f"Invalid OHLC: high < low{symbol_str}"

        invalid_close = ((df['close'] < df['low']) | (df['close'] > df['high'])).any()
        if invalid_close:
            return False, f"Invalid OHLC: close outside [low, high]{symbol_str}"

        invalid_open = ((df['open'] < df['low']) | (df['open'] > df['high'])).any()
        if invalid_open:
            return False, f"Invalid OHLC: open outside [low, high]{symbol_str}"

        return True, ""

    def validate_market_data(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate all DataFrames in market_data.

        Args:
            market_data: Dict of symbol -> DataFrame

        Returns:
            Tuple of (all_valid, error_messages)
        """
        errors = []

        for symbol, df in market_data.items():
            is_valid, error = self.validate_data(df, symbol)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors

    def get_name(self) -> str:
        """
        Get strategy name.

        Default implementation returns class name.
        Override for custom names.
        """
        return self.__class__.__name__

    def get_parameters(self) -> Dict:
        """
        Get strategy parameters for logging/debugging.

        Override to return strategy-specific parameters.

        Returns:
            Dict of parameter names -> values
        """
        return {}

    def __str__(self):
        """String representation."""
        return f"{self.get_name()}({self.get_parameters()})"

    def __repr__(self):
        """Detailed representation."""
        params = ", ".join(f"{k}={v}" for k, v in self.get_parameters().items())
        return f"{self.__class__.__name__}({params})"


class DataRequirements:
    """
    Specification of data requirements for a strategy.

    Tells the data provider what data to fetch and how much history is needed.
    """

    def __init__(self):
        self.daily_data: List[Tuple[str, int]] = []  # (symbol, lookback_days)
        self.intraday_data: List[Tuple[str, str, int]] = []  # (symbol, timeframe, lookback_periods)

    def add_daily_data(self, symbol: str, lookback_days: int):
        """
        Add requirement for daily data.

        Args:
            symbol: Trading symbol
            lookback_days: Number of days of history needed
        """
        self.daily_data.append((symbol, lookback_days))

    def add_intraday_data(self, symbol: str, timeframe: str, lookback_periods: int):
        """
        Add requirement for intraday data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe ('1Min', '5Min', '15Min', '1Hour', etc.)
            lookback_periods: Number of periods needed
        """
        self.intraday_data.append((symbol, timeframe, lookback_periods))

    def get_all_symbols(self) -> List[str]:
        """Get unique list of all symbols needed."""
        daily_symbols = [s for s, _ in self.daily_data]
        intraday_symbols = [s for s, _, _ in self.intraday_data]
        return list(set(daily_symbols + intraday_symbols))

    def __str__(self):
        """String representation."""
        daily_str = f"{len(self.daily_data)} daily"
        intraday_str = f"{len(self.intraday_data)} intraday"
        return f"DataRequirements({daily_str}, {intraday_str})"

    def __repr__(self):
        """Detailed representation."""
        return f"DataRequirements(daily={self.daily_data}, intraday={self.intraday_data})"
