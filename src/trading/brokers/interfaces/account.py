"""
Account Interface - Account information and connection management.
"""

from abc import ABC, abstractmethod
from typing import Dict


class AccountInterface(ABC):
    """
    Abstract interface for account operations.

    Provides access to account information and connection testing.
    All broker implementations must implement this interface.
    """

    @abstractmethod
    def get_account(self) -> Dict:
        """
        Get account information.

        Returns:
            Dict with standardized keys:
                - account_id (str): Account identifier
                - buying_power (float): Available buying power
                - cash (float): Cash balance
                - portfolio_value (float): Total portfolio value
                - equity (float): Total equity
                - currency (str): Account currency (USD, etc.)

        Raises:
            BrokerConnectionError: If broker connection fails
            BrokerAuthError: If authentication fails
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test broker connection.

        Returns:
            True if connection successful, False otherwise
        """
        pass
