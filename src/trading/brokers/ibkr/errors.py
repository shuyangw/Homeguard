"""
IBKR Error Handling - Maps IBKR error codes to Homeguard exceptions.

This module does NOT define its own exception hierarchy. All exceptions
raised to strategy code are from src.trading.brokers.interfaces.base.

The only internal exception is PacingViolationError, used within
pacing.py and never leaked outside the ibkr package.
"""

from src.trading.brokers.interfaces.base import (
    BrokerError,
    BrokerConnectionError,
    InvalidOrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    SymbolNotFoundError,
)


# Internal-only exception (never leaves this package)
class PacingViolationError(Exception):
    """IBKR historical data pacing limit exceeded. Internal to ibkr package."""
    pass


# Codes that are purely informational (log and ignore)
INFORMATIONAL_CODES = frozenset({
    2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110,
    2119, 2137, 2158,
    10167,  # Displaying delayed data
    10168,  # Not subscribed
    10197,  # No market data during competing session
})

# Codes related to pacing / throttling (handled internally)
PACING_CODES = frozenset({162, 366, 420})

# Codes related to connection state (handled by reconnection)
CONNECTION_CODES = frozenset({504, 1100, 1101, 1102})

# Codes related to order rejection
ORDER_ERROR_CODES = frozenset({
    103, 104, 105, 106, 107, 109, 110,
    135, 136, 161, 201, 202, 203, 399, 434,
})


def classify_error(error_code: int) -> str:
    """Classify an IBKR error code into a category."""
    if error_code in INFORMATIONAL_CODES:
        return "informational"
    if error_code in PACING_CODES:
        return "pacing"
    if error_code in CONNECTION_CODES:
        return "connection"
    if error_code in ORDER_ERROR_CODES:
        return "order"
    return "error"


def ibkr_code_to_exception(error_code: int, message: str) -> BrokerError:
    """Convert an IBKR error code to the appropriate Homeguard exception."""
    if error_code in (504, 1100, 1101, 1102):
        return BrokerConnectionError(f"[IBKR-{error_code}] {message}")
    if error_code in (200, 321):
        return SymbolNotFoundError(f"[IBKR-{error_code}] {message}")
    if error_code in (201, 203, 103, 104, 105, 106, 107, 109, 110, 399, 434):
        return InvalidOrderError(f"[IBKR-{error_code}] {message}")
    if error_code in (135, 136, 161):
        return OrderNotFoundError(f"[IBKR-{error_code}] {message}")
    if error_code == 202:
        return InvalidOrderError(f"[IBKR-{error_code}] Order cancelled: {message}")
    return BrokerError(f"[IBKR-{error_code}] {message}")
