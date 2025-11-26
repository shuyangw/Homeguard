"""
Market Hours Interface - Market schedule operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple


class MarketHoursInterface(ABC):
    """
    Abstract interface for market hours operations.

    Provides access to market schedule information.
    """

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open, False otherwise

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass

    @abstractmethod
    def get_market_hours(self, date: datetime) -> Tuple[datetime, datetime]:
        """
        Get market hours for a specific date.

        Args:
            date: Date to check

        Returns:
            Tuple of (open_time, close_time)

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        pass
