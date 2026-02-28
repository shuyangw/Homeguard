"""
Shared exit-checking utility for intraday strategies.

Provides a common function to check stop-loss, profit target, time exit,
time stop (max hold bars), and max loss cap conditions. Used by ICT and ORB
strategies to eliminate duplicated exit logic.

BMSB strategy is excluded because it uses a fundamentally different exit
paradigm (band cross detection + percentage trailing stops, no fixed
stop/target levels).
"""

from datetime import time
from enum import Enum
from typing import Optional, Tuple


class ExitReason(Enum):
    """Reason for exiting a position."""
    STOP_LOSS = "stop_loss"
    TARGET = "target"
    TIME_EXIT = "time_exit"
    TIME_STOP = "time_stop"
    MAX_LOSS = "max_loss"


def check_exit(
    position_dir: int,
    high: float,
    low: float,
    stop: float,
    target: float,
    current_time: Optional[time] = None,
    exit_time: Optional[time] = None,
    entry_bar: int = 0,
    current_bar: int = 0,
    max_bars: int = 0,
    entry_price: float = 0.0,
    close: float = 0.0,
    max_loss_pct: float = 0.0,
) -> Tuple[Optional[ExitReason], str]:
    """Check if a position should be exited.

    Checks exit conditions in priority order:
    1. Time exit (end of day / session close)
    2. Time stop (max bars held)
    3. Max loss cap (percentage-based)
    4. Stop-loss (price breaches stop level)
    5. Profit target (price reaches target level)

    Args:
        position_dir: 1 for long, -1 for short.
        high: Current bar high price.
        low: Current bar low price.
        stop: Stop-loss price level.
        target: Profit target price level.
        current_time: Current bar time of day (optional, for time exit).
        exit_time: Forced exit time of day (optional, for time exit).
        entry_bar: Bar index when position was entered (for time stop).
        current_bar: Current bar index (for time stop).
        max_bars: Maximum bars to hold, 0 = disabled (for time stop).
        entry_price: Price at entry (for max loss calculation).
        close: Current bar close price (for max loss calculation).
        max_loss_pct: Maximum loss percentage before forced exit,
            0.0 = disabled (e.g. 0.15 = 15%).

    Returns:
        Tuple of (ExitReason, reason_string) if should exit,
        or (None, '') if position should be held.
    """
    # 1. Time exit (end of day)
    if current_time is not None and exit_time is not None:
        if current_time >= exit_time:
            return ExitReason.TIME_EXIT, ExitReason.TIME_EXIT.value

    # 2. Time stop (max hold bars)
    if max_bars > 0 and entry_bar > 0:
        bars_held = current_bar - entry_bar
        if bars_held >= max_bars:
            return ExitReason.TIME_STOP, ExitReason.TIME_STOP.value

    # 3. Max loss cap
    if max_loss_pct > 0 and entry_price > 0:
        if position_dir == 1:  # long
            current_pnl_pct = (close - entry_price) / entry_price
        else:  # short
            current_pnl_pct = (entry_price - close) / entry_price
        if current_pnl_pct <= -max_loss_pct:
            return ExitReason.MAX_LOSS, ExitReason.MAX_LOSS.value

    # 4 & 5. Stop-loss and target
    if position_dir == 1:  # long
        if low <= stop:
            return ExitReason.STOP_LOSS, ExitReason.STOP_LOSS.value
        if high >= target:
            return ExitReason.TARGET, ExitReason.TARGET.value
    else:  # short
        if high >= stop:
            return ExitReason.STOP_LOSS, ExitReason.STOP_LOSS.value
        if low <= target:
            return ExitReason.TARGET, ExitReason.TARGET.value

    return None, ''
