"""
Alpaca Data Provider - Wraps existing AlpacaBroker.

This is a thin adapter that delegates to AlpacaBroker without modifying it.
"""

from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.brokers.alpaca_broker import AlpacaBroker


class AlpacaDataProvider(DataProviderInterface):
    """
    Data provider wrapping AlpacaBroker.

    No changes to AlpacaBroker are required - this is a pure wrapper.
    """

    def __init__(self, broker: "AlpacaBroker"):
        """
        Initialize with existing broker instance.

        Args:
            broker: Configured AlpacaBroker instance
        """
        self._broker = broker

    @property
    def name(self) -> str:
        return "Alpaca"

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False  # Accepted for API consistency, ignored (Alpaca always fresh)
    ) -> Optional[pd.DataFrame]:
        """Fetch bars via AlpacaBroker, returning None on failure."""
        try:
            df = self._broker.get_historical_bars(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe
            )

            if df is None or df.empty:
                logger.warning(f"[Alpaca] No data for {symbol}")
                return None

            # AlpacaBroker already returns correct format
            return df

        except Exception as e:
            logger.error(f"[Alpaca] Failed to fetch {symbol}: {e}")
            return None

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False  # Accepted for API consistency, ignored (Alpaca always fresh)
    ) -> Dict[str, pd.DataFrame]:
        """Batch fetch - delegates to single-symbol calls."""
        results = {}

        for symbol in symbols:
            df = self.get_historical_bars(symbol, start, end, timeframe)
            if df is not None:
                results[symbol] = df

        return results

    def is_available(self) -> bool:
        """Check Alpaca connection."""
        try:
            return self._broker.test_connection()
        except Exception:
            return False
