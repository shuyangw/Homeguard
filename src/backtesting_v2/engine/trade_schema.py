"""
Standardized trade and portfolio state column definitions.

Defines the base columns that all trades must have, plus optional
columns for multi-asset portfolios and bar-by-bar state tracking.
"""

from enum import IntEnum
from typing import List


class TradeType(IntEnum):
    """Trade type codes for Numba compatibility."""
    ENTRY = 0
    EXIT = 1
    SHORT_ENTRY = 2
    COVER_SHORT = 3


class ExitReason(IntEnum):
    """Exit reason codes for Numba compatibility."""
    SIGNAL = 0
    STOP_LOSS = 1
    TIME_STOP = 2
    PROFIT_TARGET = 3
    SCHEDULED = 4  # For intraday scheduled exits


# Map codes to human-readable names
TRADE_TYPE_NAMES = {
    TradeType.ENTRY: "entry",
    TradeType.EXIT: "exit",
    TradeType.SHORT_ENTRY: "short_entry",
    TradeType.COVER_SHORT: "cover_short",
}

EXIT_REASON_NAMES = {
    ExitReason.SIGNAL: "strategy_signal",
    ExitReason.STOP_LOSS: "stop_loss",
    ExitReason.TIME_STOP: "time_stop",
    ExitReason.PROFIT_TARGET: "profit_target",
    ExitReason.SCHEDULED: "scheduled_exit",
}


# === Base Trade Columns ===
# Every trade record must have these columns
BASE_TRADE_COLUMNS: List[str] = [
    "timestamp",      # Trade execution timestamp
    "type",           # Trade type: entry, exit, short_entry, cover_short
    "price",          # Execution price
    "shares",         # Number of shares traded
    "pnl",            # P&L for exits (0 for entries)
    "pnl_pct",        # P&L % for exits (0 for entries)
    "exit_reason",    # Exit reason: strategy_signal, stop_loss, etc.
    "cost",           # Total cost including fees (for entries)
    "proceeds",       # Net proceeds after fees (for exits)
]


# === Multi-Asset Portfolio Columns ===
# Additional columns for multi-symbol portfolios
MULTI_ASSET_COLUMNS: List[str] = [
    "symbol",             # Asset symbol
    "entry_timestamp",    # Entry time (for round-trip trades)
    "exit_timestamp",     # Exit time (for round-trip trades)
    "entry_price",        # Entry price
    "exit_price",         # Exit price
    "holding_period",     # Bars held
]


# === Portfolio State Columns ===
# Bar-by-bar portfolio state tracking
STATE_COLUMNS: List[str] = [
    "timestamp",          # Bar timestamp
    "cash",               # Cash balance
    "position_shares",    # Current position in shares
    "position_value",     # Market value of position
    "position_entry_price",  # Entry price of current position
    "total_equity",       # Total portfolio value
    "exposure_pct",       # Position value / total equity
]


# === Multi-Asset State Columns ===
# Additional state columns for multi-symbol portfolios
MULTI_ASSET_STATE_COLUMNS: List[str] = [
    "timestamp",          # Bar timestamp
    "cash",               # Cash balance
    "position_count",     # Number of open positions
    "total_equity",       # Total portfolio value
    "positions",          # Dict of symbol -> shares
    "weights",            # Dict of symbol -> weight
    "exposure_pct",       # Total exposure percentage
]


def validate_trade_columns(columns: List[str]) -> List[str]:
    """
    Check which base columns are missing from a trade DataFrame.

    Args:
        columns: List of column names in DataFrame

    Returns:
        List of missing base columns (empty if all present)
    """
    return [col for col in BASE_TRADE_COLUMNS if col not in columns]


def validate_state_columns(columns: List[str]) -> List[str]:
    """
    Check which state columns are missing from a state DataFrame.

    Args:
        columns: List of column names in DataFrame

    Returns:
        List of missing state columns (empty if all present)
    """
    return [col for col in STATE_COLUMNS if col not in columns]
