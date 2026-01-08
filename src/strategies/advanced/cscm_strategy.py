"""
Cross-Sectional Crypto Momentum (CSCM) Strategy.

A multi-symbol strategy that ranks crypto assets by momentum and holds top performers.

Strategy Logic:
- Weekly rebalance (Sunday 0:00 UTC)
- 14-day momentum lookback
- BTC regime filter: BTC > 20-day SMA = bullish (hold positions), else go to cash
- Top N coins by cross-sectional momentum ranking
- Equal weight allocation: 100% / top_n per position

Key Features:
- Exploits crypto's strong momentum effect
- Regime awareness reduces drawdowns in bear markets
- Weekly rebalance balances turnover vs alpha capture

Usage:
    from src.strategies.advanced.cscm_strategy import CSCMStrategy

    strategy = CSCMStrategy(
        universe=['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD'],
        top_n=3,
        momentum_period=14,
        btc_sma_period=20,
    )

    # For backtesting - requires multi-symbol data
    signals = strategy.generate_multi_signals(data_dict)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import numpy as np
import pandas as pd

from src.backtesting.base.strategy import MultiSymbolStrategy
from src.strategies.advanced.cscm_indicators import CSCMIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSCMStrategy(MultiSymbolStrategy):
    """
    Cross-Sectional Crypto Momentum Strategy.

    Ranks crypto assets by 14-day returns, holds top N performers.
    Uses BTC regime filter to go to cash in bear markets.

    Parameters:
        universe: List of crypto symbols to trade (e.g., ['BTC/USD', 'ETH/USD'])
        top_n: Number of top coins to hold (default 5)
        momentum_period: Days for momentum calculation (default 14)
        btc_sma_period: BTC SMA period for regime (default 20)
        rebalance_day: Day of week to rebalance (default 'sunday')
        go_to_cash_in_bear: Exit all in bearish regime (default True)
        btc_symbol: Symbol for BTC regime filter (default 'BTC/USD')
    """

    def __init__(
        self,
        universe: List[str],
        top_n: int = 5,
        momentum_period: int = 14,
        btc_sma_period: int = 20,
        rebalance_day: str = 'sunday',
        go_to_cash_in_bear: bool = True,
        btc_symbol: str = 'BTC/USD',
        **params
    ):
        """
        Initialize CSCM strategy.

        Args:
            universe: List of crypto symbols to trade
            top_n: Number of top coins to hold
            momentum_period: Days for momentum calculation
            btc_sma_period: BTC SMA period for regime filter
            rebalance_day: Day of week to rebalance
            go_to_cash_in_bear: Exit all positions in bearish regime
            btc_symbol: Symbol for BTC regime filter
        """
        self.universe = universe
        self.top_n = top_n
        self.momentum_period = momentum_period
        self.btc_sma_period = btc_sma_period
        self.rebalance_day = rebalance_day
        self.go_to_cash_in_bear = go_to_cash_in_bear
        self.btc_symbol = btc_symbol

        # Call parent init with all params
        super().__init__(
            universe=universe,
            top_n=top_n,
            momentum_period=momentum_period,
            btc_sma_period=btc_sma_period,
            rebalance_day=rebalance_day,
            go_to_cash_in_bear=go_to_cash_in_bear,
            btc_symbol=btc_symbol,
            **params
        )

        logger.info(f"Initialized CSCM Strategy")
        logger.info(f"  Universe: {len(universe)} symbols")
        logger.info(f"  Top N: {top_n}")
        logger.info(f"  Momentum Period: {momentum_period} days")
        logger.info(f"  BTC SMA Period: {btc_sma_period}")
        logger.info(f"  Rebalance Day: {rebalance_day}")
        logger.info(f"  Go to Cash in Bear: {go_to_cash_in_bear}")

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if not self.universe:
            raise ValueError("Universe must contain at least one symbol")

        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")

        if self.top_n > len(self.universe):
            logger.warning(
                f"top_n ({self.top_n}) > universe size ({len(self.universe)}), "
                f"will hold all symbols when available"
            )

        if self.momentum_period < 1:
            raise ValueError(f"momentum_period must be >= 1, got {self.momentum_period}")

        if self.btc_sma_period < 1:
            raise ValueError(f"btc_sma_period must be >= 1, got {self.btc_sma_period}")

        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if self.rebalance_day.lower() not in valid_days:
            raise ValueError(f"rebalance_day must be one of {valid_days}")

    def get_required_symbols(self) -> List[str]:
        """
        Return the list of symbols required for this strategy.

        Returns:
            List of crypto symbols including BTC for regime filter
        """
        symbols = list(self.universe)
        # Ensure BTC is included for regime filter
        if self.btc_symbol not in symbols:
            symbols.append(self.btc_symbol)
        return symbols

    def generate_multi_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """
        Generate cross-sectional momentum signals for multiple symbols.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data

        Returns:
            Dictionary of {symbol: (entries, exits)} signals
        """
        # Validate we have BTC data for regime filter
        if self.btc_symbol not in data_dict:
            raise ValueError(f"BTC data ({self.btc_symbol}) required for regime filter")

        btc_data = data_dict[self.btc_symbol]

        # Filter to tradeable universe (excluding BTC if not in universe)
        trade_dict = {
            sym: df for sym, df in data_dict.items()
            if sym in self.universe and df is not None and not df.empty
        }

        if not trade_dict:
            logger.warning("No valid data for any symbols in universe")
            return {}

        # Generate signals using indicators
        selections, weights, regime = CSCMIndicators.generate_signals(
            data_dict=trade_dict,
            btc_data=btc_data,
            momentum_period=self.momentum_period,
            btc_sma_period=self.btc_sma_period,
            top_n=self.top_n,
            rebalance_day=self.rebalance_day,
            go_to_cash_in_bear=self.go_to_cash_in_bear
        )

        if selections.empty:
            return {}

        # Convert selections to entry/exit signals
        signals_dict = {}
        for symbol in selections.columns:
            selected = selections[symbol].astype(bool)

            # Entry: transition from not selected to selected
            entries = selected & ~selected.shift(1).fillna(False)

            # Exit: transition from selected to not selected
            exits = ~selected & selected.shift(1).fillna(False)

            signals_dict[symbol] = (entries, exits)

        return signals_dict

    def get_target_positions(
        self,
        data_dict: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, float]:
        """
        Get target position weights for a specific date.

        This method is useful for live trading to determine current allocations.

        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data
            date: Target date for positions

        Returns:
            Dictionary of {symbol: weight} where weight is 0.0 to 1.0
        """
        # Validate we have BTC data
        if self.btc_symbol not in data_dict:
            logger.error(f"BTC data ({self.btc_symbol}) required for regime filter")
            return {}

        btc_data = data_dict[self.btc_symbol]

        # Filter to tradeable universe
        trade_dict = {
            sym: df for sym, df in data_dict.items()
            if sym in self.universe and df is not None and not df.empty
        }

        if not trade_dict:
            return {}

        # Get current signal
        signal = CSCMIndicators.get_current_signal(
            data_dict=trade_dict,
            btc_data=btc_data,
            momentum_period=self.momentum_period,
            btc_sma_period=self.btc_sma_period,
            top_n=self.top_n
        )

        # If bearish regime and configured to go to cash, return empty
        if self.go_to_cash_in_bear and signal.regime == 'bearish':
            return {sym: 0.0 for sym in self.universe}

        # Calculate weights for top symbols
        positions = {}
        weight = 1.0 / self.top_n if signal.top_symbols else 0.0

        for sym in self.universe:
            if sym in signal.top_symbols:
                positions[sym] = weight
            else:
                positions[sym] = 0.0

        return positions

    def get_current_regime(
        self,
        btc_data: pd.DataFrame
    ) -> str:
        """
        Get current market regime based on BTC price vs SMA.

        Args:
            btc_data: BTC OHLCV DataFrame

        Returns:
            'bullish' or 'bearish'
        """
        btc_close = btc_data['close'] if 'close' in btc_data.columns else btc_data['Close']
        btc_sma = CSCMIndicators.calculate_sma(btc_close, self.btc_sma_period)

        if btc_close.iloc[-1] > btc_sma.iloc[-1]:
            return 'bullish'
        return 'bearish'

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'universe': self.universe,
            'top_n': self.top_n,
            'momentum_period': self.momentum_period,
            'btc_sma_period': self.btc_sma_period,
            'rebalance_day': self.rebalance_day,
            'go_to_cash_in_bear': self.go_to_cash_in_bear,
            'btc_symbol': self.btc_symbol,
        }
