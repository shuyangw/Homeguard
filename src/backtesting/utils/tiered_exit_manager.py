"""
Tiered Exit Manager for intraday strategies.

Provides multi-stage exit logic:
1. Target 1 hit: Exit 50% of position, move stop to breakeven
2. Target 2 hit: Exit remaining position OR activate trailing stop
3. Trailing stop: Follow price with configurable offset
4. EOD exit: Force exit at specified time (default 3:55 PM)
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from typing import Optional, Tuple, Dict, List
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExitType(Enum):
    """Types of exits for tracking and reporting."""
    STOP_LOSS = "stop_loss"
    TARGET1 = "target1"
    TARGET2 = "target2"
    TRAILING = "trailing"
    EOD = "eod"
    MANUAL = "manual"


@dataclass
class TieredExitConfig:
    """Configuration for tiered exit strategy."""
    # Target multipliers (relative to opening range height)
    target1_multiplier: float = 1.0
    target2_multiplier: float = 2.0

    # Exit percentages at each target
    target1_exit_pct: float = 0.5  # Exit 50% at target 1
    target2_exit_pct: float = 1.0  # Exit remaining at target 2

    # Trailing stop configuration
    use_trailing: bool = True
    trailing_trigger_r: float = 1.0  # Trigger after 1R profit
    trailing_offset_pct: float = 0.02  # 2% trailing offset

    # End of day exit
    eod_exit_time: time = field(default_factory=lambda: time(15, 55))

    # Breakeven buffer after target 1 (as % of risk)
    breakeven_buffer_pct: float = 0.1  # 10% of risk above breakeven

    def __post_init__(self):
        """Validate configuration."""
        if self.target1_multiplier >= self.target2_multiplier:
            raise ValueError("target1_multiplier must be less than target2_multiplier")
        if not 0 < self.target1_exit_pct <= 1:
            raise ValueError("target1_exit_pct must be between 0 and 1")
        if self.trailing_offset_pct <= 0:
            raise ValueError("trailing_offset_pct must be positive")


@dataclass
class TieredPosition:
    """State tracking for a position with tiered exits."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_bar_idx: int
    entry_time: Optional[datetime] = None

    # Risk/reward levels
    stop_loss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    risk: float = 0.0  # Entry - Stop (or Stop - Entry for shorts)

    # Position tracking
    initial_shares: int = 0
    remaining_shares: int = 0
    shares_exited: int = 0

    # State flags
    target1_hit: bool = False
    target2_hit: bool = False
    trailing_active: bool = False

    # Trailing stop state
    trailing_stop: float = 0.0
    max_favorable_price: float = 0.0  # Highest (long) or lowest (short) since entry

    # Current stop (may be updated after target 1)
    current_stop: float = 0.0

    # PnL tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def __post_init__(self):
        """Initialize current stop and max price."""
        self.current_stop = self.stop_loss
        self.max_favorable_price = self.entry_price
        self.remaining_shares = self.initial_shares


@dataclass
class ExitResult:
    """Result of an exit check."""
    should_exit: bool
    exit_type: Optional[ExitType] = None
    exit_price: float = 0.0
    exit_shares: int = 0
    exit_pct: float = 0.0  # 0.5 for partial, 1.0 for full
    pnl: float = 0.0
    reason: str = ""


class TieredExitManager:
    """
    Manage tiered exits for intraday positions.

    Exit Flow:
    1. Check stop loss first (risk management priority)
    2. Check target 1 -> partial exit, move stop to breakeven
    3. Check target 2 -> full exit OR activate trailing
    4. Check trailing stop (if active)
    5. Check EOD exit (force exit at specified time)

    Usage:
        config = TieredExitConfig(target1_multiplier=1.0, target2_multiplier=2.0)
        manager = TieredExitManager(config)

        # Create position
        position = manager.create_position(
            symbol='TQQQ',
            direction='long',
            entry_price=50.0,
            shares=100,
            stop_loss=49.0,
            target1=51.0,
            target2=52.0
        )

        # Check exits each bar
        result = manager.check_exit(position, current_price=51.0, current_high=51.2,
                                     current_low=50.8, current_time=time(10, 30))
        if result.should_exit:
            # Execute exit
            position = manager.process_exit(position, result)
    """

    def __init__(self, config: Optional[TieredExitConfig] = None):
        """Initialize with configuration."""
        self.config = config or TieredExitConfig()

    def create_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        target1: float,
        target2: float,
        entry_bar_idx: int = 0,
        entry_time: Optional[datetime] = None
    ) -> TieredPosition:
        """
        Create a new position with tiered exit levels.

        Args:
            symbol: Stock symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            shares: Number of shares
            stop_loss: Initial stop loss price
            target1: First profit target
            target2: Second profit target
            entry_bar_idx: Index of entry bar
            entry_time: Datetime of entry

        Returns:
            TieredPosition object
        """
        if direction == 'long':
            risk = entry_price - stop_loss
        else:
            risk = stop_loss - entry_price

        return TieredPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_bar_idx=entry_bar_idx,
            entry_time=entry_time,
            stop_loss=stop_loss,
            target1=target1,
            target2=target2,
            risk=risk,
            initial_shares=shares,
            remaining_shares=shares
        )

    def check_exit(
        self,
        position: TieredPosition,
        current_price: float,
        current_high: float,
        current_low: float,
        current_time: time,
        bar_idx: int = 0
    ) -> ExitResult:
        """
        Check all exit conditions for a position.

        Priority order:
        1. Stop loss (includes trailing stop)
        2. Target 1 (partial exit)
        3. Target 2 (full exit or trail activation)
        4. EOD exit

        Args:
            position: Current position state
            current_price: Current close price
            current_high: Current bar high
            current_low: Current bar low
            current_time: Current time of day
            bar_idx: Current bar index

        Returns:
            ExitResult with exit details
        """
        if position.remaining_shares <= 0:
            return ExitResult(should_exit=False)

        # Update max favorable price for trailing stop calculation
        self._update_max_price(position, current_high, current_low)

        # 1. Check stop loss first (including trailing stop)
        stop_result = self._check_stop_loss(position, current_high, current_low)
        if stop_result.should_exit:
            return stop_result

        # 2. Check target 1 (if not yet hit)
        if not position.target1_hit:
            t1_result = self._check_target1(position, current_high, current_low)
            if t1_result.should_exit:
                return t1_result

        # 3. Check target 2 (if target 1 already hit)
        if position.target1_hit and not position.target2_hit:
            t2_result = self._check_target2(position, current_high, current_low)
            if t2_result.should_exit:
                return t2_result

        # 4. Check EOD exit
        eod_result = self._check_eod_exit(position, current_price, current_time)
        if eod_result.should_exit:
            return eod_result

        return ExitResult(should_exit=False)

    def process_exit(
        self,
        position: TieredPosition,
        result: ExitResult
    ) -> TieredPosition:
        """
        Process an exit and update position state.

        Args:
            position: Current position
            result: Exit result from check_exit()

        Returns:
            Updated position
        """
        if not result.should_exit:
            return position

        # Calculate PnL for this exit
        if position.direction == 'long':
            pnl = (result.exit_price - position.entry_price) * result.exit_shares
        else:
            pnl = (position.entry_price - result.exit_price) * result.exit_shares

        position.realized_pnl += pnl
        position.shares_exited += result.exit_shares
        position.remaining_shares -= result.exit_shares

        # Update state based on exit type
        if result.exit_type == ExitType.TARGET1:
            position.target1_hit = True
            # Move stop to breakeven + buffer
            self._move_stop_to_breakeven(position)

        elif result.exit_type == ExitType.TARGET2:
            position.target2_hit = True
            # Optionally activate trailing stop instead of full exit
            if self.config.use_trailing and position.remaining_shares > 0:
                self._activate_trailing_stop(position)

        return position

    def _update_max_price(
        self,
        position: TieredPosition,
        current_high: float,
        current_low: float
    ) -> None:
        """Update max favorable price for trailing stop."""
        if position.direction == 'long':
            position.max_favorable_price = max(
                position.max_favorable_price,
                current_high
            )
        else:
            position.max_favorable_price = min(
                position.max_favorable_price,
                current_low
            )

        # Update trailing stop if active
        if position.trailing_active:
            self._update_trailing_stop(position)

    def _check_stop_loss(
        self,
        position: TieredPosition,
        current_high: float,
        current_low: float
    ) -> ExitResult:
        """Check if stop loss is hit."""
        stop_price = position.current_stop

        if position.direction == 'long':
            if current_low <= stop_price:
                exit_type = (ExitType.TRAILING if position.trailing_active
                             else ExitType.STOP_LOSS)
                return ExitResult(
                    should_exit=True,
                    exit_type=exit_type,
                    exit_price=stop_price,
                    exit_shares=position.remaining_shares,
                    exit_pct=1.0,
                    reason=f"Stop hit at {stop_price:.2f}"
                )
        else:  # short
            if current_high >= stop_price:
                exit_type = (ExitType.TRAILING if position.trailing_active
                             else ExitType.STOP_LOSS)
                return ExitResult(
                    should_exit=True,
                    exit_type=exit_type,
                    exit_price=stop_price,
                    exit_shares=position.remaining_shares,
                    exit_pct=1.0,
                    reason=f"Stop hit at {stop_price:.2f}"
                )

        return ExitResult(should_exit=False)

    def _check_target1(
        self,
        position: TieredPosition,
        current_high: float,
        current_low: float
    ) -> ExitResult:
        """Check if target 1 is hit."""
        shares_to_exit = int(position.initial_shares * self.config.target1_exit_pct)
        shares_to_exit = min(shares_to_exit, position.remaining_shares)

        if position.direction == 'long':
            if current_high >= position.target1:
                return ExitResult(
                    should_exit=True,
                    exit_type=ExitType.TARGET1,
                    exit_price=position.target1,
                    exit_shares=shares_to_exit,
                    exit_pct=self.config.target1_exit_pct,
                    reason=f"Target 1 hit at {position.target1:.2f}"
                )
        else:  # short
            if current_low <= position.target1:
                return ExitResult(
                    should_exit=True,
                    exit_type=ExitType.TARGET1,
                    exit_price=position.target1,
                    exit_shares=shares_to_exit,
                    exit_pct=self.config.target1_exit_pct,
                    reason=f"Target 1 hit at {position.target1:.2f}"
                )

        return ExitResult(should_exit=False)

    def _check_target2(
        self,
        position: TieredPosition,
        current_high: float,
        current_low: float
    ) -> ExitResult:
        """Check if target 2 is hit."""
        if position.direction == 'long':
            if current_high >= position.target2:
                return ExitResult(
                    should_exit=True,
                    exit_type=ExitType.TARGET2,
                    exit_price=position.target2,
                    exit_shares=position.remaining_shares,
                    exit_pct=1.0,
                    reason=f"Target 2 hit at {position.target2:.2f}"
                )
        else:  # short
            if current_low <= position.target2:
                return ExitResult(
                    should_exit=True,
                    exit_type=ExitType.TARGET2,
                    exit_price=position.target2,
                    exit_shares=position.remaining_shares,
                    exit_pct=1.0,
                    reason=f"Target 2 hit at {position.target2:.2f}"
                )

        return ExitResult(should_exit=False)

    def _check_eod_exit(
        self,
        position: TieredPosition,
        current_price: float,
        current_time: time
    ) -> ExitResult:
        """Check if end-of-day exit time is reached."""
        if current_time >= self.config.eod_exit_time:
            return ExitResult(
                should_exit=True,
                exit_type=ExitType.EOD,
                exit_price=current_price,
                exit_shares=position.remaining_shares,
                exit_pct=1.0,
                reason=f"EOD exit at {current_time}"
            )

        return ExitResult(should_exit=False)

    def _move_stop_to_breakeven(self, position: TieredPosition) -> None:
        """Move stop to breakeven plus buffer after target 1 hit."""
        buffer = position.risk * self.config.breakeven_buffer_pct

        if position.direction == 'long':
            position.current_stop = position.entry_price + buffer
        else:
            position.current_stop = position.entry_price - buffer

        logger.debug(
            f"Moved stop to breakeven: {position.current_stop:.2f} "
            f"(entry: {position.entry_price:.2f}, buffer: {buffer:.4f})"
        )

    def _activate_trailing_stop(self, position: TieredPosition) -> None:
        """Activate trailing stop after target 2."""
        position.trailing_active = True
        self._update_trailing_stop(position)

        logger.debug(
            f"Activated trailing stop at {position.trailing_stop:.2f} "
            f"(max price: {position.max_favorable_price:.2f})"
        )

    def _update_trailing_stop(self, position: TieredPosition) -> None:
        """Update trailing stop based on max favorable price."""
        offset = position.max_favorable_price * self.config.trailing_offset_pct

        if position.direction == 'long':
            new_stop = position.max_favorable_price - offset
            # Only ratchet up, never down
            if new_stop > position.current_stop:
                position.current_stop = new_stop
                position.trailing_stop = new_stop
        else:  # short
            new_stop = position.max_favorable_price + offset
            # Only ratchet down, never up
            if new_stop < position.current_stop:
                position.current_stop = new_stop
                position.trailing_stop = new_stop


class DailyRiskManager:
    """
    Track daily risk limits for intraday strategies.

    Enforces:
    - Maximum daily loss percentage
    - Maximum number of trades per day
    - Maximum concurrent positions
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_daily_trades: int = 10,
        max_concurrent_positions: int = 5
    ):
        """
        Initialize risk manager.

        Args:
            max_daily_loss_pct: Maximum daily loss as percentage of capital
            max_daily_trades: Maximum trades allowed per day
            max_concurrent_positions: Maximum concurrent open positions
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_trades = max_daily_trades
        self.max_concurrent_positions = max_concurrent_positions

        # Daily tracking (reset each day)
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.open_positions: Dict[str, TieredPosition] = {}
        self.current_date = None

    def reset_daily(self, date=None) -> None:
        """Reset daily tracking for new trading day."""
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.open_positions.clear()
        self.current_date = date

    def can_trade(
        self,
        symbol: str,
        capital: float
    ) -> Tuple[bool, str]:
        """
        Check if a new trade is allowed.

        Args:
            symbol: Symbol to trade
            capital: Current account capital

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check daily loss limit
        max_loss = capital * self.max_daily_loss_pct
        if self.daily_pnl <= -max_loss:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"

        # Check trade count
        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Max daily trades reached ({self.daily_trade_count})"

        # Check concurrent positions
        if len(self.open_positions) >= self.max_concurrent_positions:
            return False, f"Max concurrent positions ({len(self.open_positions)})"

        # Check if already in this symbol
        if symbol in self.open_positions:
            return False, f"Already in position for {symbol}"

        return True, "OK"

    def record_entry(
        self,
        symbol: str,
        position: TieredPosition
    ) -> None:
        """Record a new trade entry."""
        self.open_positions[symbol] = position
        self.daily_trade_count += 1

    def record_exit(
        self,
        symbol: str,
        pnl: float
    ) -> None:
        """Record a trade exit."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        self.daily_pnl += pnl

    def get_stats(self) -> Dict[str, any]:
        """Get current daily stats."""
        return {
            'daily_pnl': self.daily_pnl,
            'trade_count': self.daily_trade_count,
            'open_positions': len(self.open_positions),
            'open_symbols': list(self.open_positions.keys())
        }
