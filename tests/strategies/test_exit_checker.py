"""
Unit tests for the shared ExitChecker utility.

Tests cover:
- Stop-loss exits for long and short positions
- Profit target exits for long and short positions
- Time exit (end of day)
- Time stop (max hold bars)
- Max loss cap
- Priority ordering of exit conditions
- No-exit cases (price in safe range)
"""

import pytest
from datetime import time

from src.strategies.advanced.exit_checker import ExitReason, check_exit


class TestStopLossExits:
    """Test stop-loss exit conditions."""

    def test_long_stop_loss_hit(self):
        """Long stop-loss triggers when low <= stop."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=101.0,
            low=97.0,   # Below stop at 98
            stop=98.0,
            target=104.0,
        )
        assert reason == ExitReason.STOP_LOSS
        assert reason_str == 'stop_loss'

    def test_long_stop_loss_exact(self):
        """Long stop-loss triggers when low == stop exactly."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=101.0,
            low=98.0,   # Exactly at stop
            stop=98.0,
            target=104.0,
        )
        assert reason == ExitReason.STOP_LOSS

    def test_short_stop_loss_hit(self):
        """Short stop-loss triggers when high >= stop."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=103.0,  # Above stop at 102
            low=100.0,
            stop=102.0,
            target=96.0,
        )
        assert reason == ExitReason.STOP_LOSS
        assert reason_str == 'stop_loss'

    def test_short_stop_loss_exact(self):
        """Short stop-loss triggers when high == stop exactly."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=102.0,  # Exactly at stop
            low=100.0,
            stop=102.0,
            target=96.0,
        )
        assert reason == ExitReason.STOP_LOSS


class TestTargetExits:
    """Test profit target exit conditions."""

    def test_long_target_hit(self):
        """Long target triggers when high >= target."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=105.0,  # Above target at 104
            low=103.0,
            stop=98.0,
            target=104.0,
        )
        assert reason == ExitReason.TARGET
        assert reason_str == 'target'

    def test_long_target_exact(self):
        """Long target triggers when high == target exactly."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=104.0,
            low=103.0,
            stop=98.0,
            target=104.0,
        )
        assert reason == ExitReason.TARGET

    def test_short_target_hit(self):
        """Short target triggers when low <= target."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=99.0,
            low=95.0,   # Below target at 96
            stop=102.0,
            target=96.0,
        )
        assert reason == ExitReason.TARGET
        assert reason_str == 'target'

    def test_short_target_exact(self):
        """Short target triggers when low == target exactly."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=99.0,
            low=96.0,   # Exactly at target
            stop=102.0,
            target=96.0,
        )
        assert reason == ExitReason.TARGET


class TestTimeExit:
    """Test time-based exit conditions."""

    def test_time_exit_at_exit_time(self):
        """Time exit triggers at or after exit time."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=101.0,
            low=99.0,
            stop=98.0,
            target=104.0,
            current_time=time(15, 45),
            exit_time=time(15, 45),
        )
        assert reason == ExitReason.TIME_EXIT
        assert reason_str == 'time_exit'

    def test_time_exit_after_exit_time(self):
        """Time exit triggers after exit time."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=101.0,
            low=99.0,
            stop=102.0,
            target=96.0,
            current_time=time(15, 50),
            exit_time=time(15, 45),
        )
        assert reason == ExitReason.TIME_EXIT

    def test_no_time_exit_before_exit_time(self):
        """No time exit before exit time."""
        reason, _ = check_exit(
            position_dir=1,
            high=101.0,
            low=99.0,
            stop=98.0,
            target=104.0,
            current_time=time(11, 0),
            exit_time=time(15, 45),
        )
        assert reason is None

    def test_no_time_exit_when_not_provided(self):
        """No time exit when time parameters are not provided."""
        reason, _ = check_exit(
            position_dir=1,
            high=101.0,
            low=99.0,
            stop=98.0,
            target=104.0,
        )
        assert reason is None


class TestTimeStop:
    """Test max hold bars (time stop) conditions."""

    def test_time_stop_triggers(self):
        """Time stop triggers when bars_held >= max_bars."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=101.0,
            low=99.0,
            stop=98.0,
            target=104.0,
            entry_bar=100,
            current_bar=200,
            max_bars=100,
        )
        assert reason == ExitReason.TIME_STOP
        assert reason_str == 'time_stop'

    def test_time_stop_does_not_trigger_when_disabled(self):
        """Time stop disabled when max_bars=0."""
        reason, _ = check_exit(
            position_dir=1,
            high=101.0,
            low=99.0,
            stop=98.0,
            target=104.0,
            entry_bar=100,
            current_bar=200,
            max_bars=0,
        )
        assert reason is None

    def test_time_stop_does_not_trigger_early(self):
        """Time stop does not trigger before max_bars reached."""
        reason, _ = check_exit(
            position_dir=1,
            high=101.0,
            low=99.0,
            stop=98.0,
            target=104.0,
            entry_bar=100,
            current_bar=150,
            max_bars=100,
        )
        assert reason is None


class TestMaxLoss:
    """Test maximum loss cap conditions."""

    def test_max_loss_long_triggers(self):
        """Max loss triggers for long when loss exceeds threshold."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=88.0,
            low=84.0,
            stop=80.0,
            target=120.0,
            entry_price=100.0,
            close=85.0,     # 15% loss
            max_loss_pct=0.15,
        )
        assert reason == ExitReason.MAX_LOSS
        assert reason_str == 'max_loss'

    def test_max_loss_short_triggers(self):
        """Max loss triggers for short when loss exceeds threshold."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=116.0,
            low=114.0,
            stop=120.0,
            target=80.0,
            entry_price=100.0,
            close=115.0,    # 15% loss
            max_loss_pct=0.15,
        )
        assert reason == ExitReason.MAX_LOSS

    def test_max_loss_disabled(self):
        """Max loss disabled when max_loss_pct=0."""
        reason, _ = check_exit(
            position_dir=1,
            high=88.0,
            low=84.0,
            stop=80.0,
            target=120.0,
            entry_price=100.0,
            close=85.0,
            max_loss_pct=0.0,
        )
        # Should not trigger max_loss; may trigger stop or target or nothing
        assert reason != ExitReason.MAX_LOSS


class TestNoExit:
    """Test cases where no exit should occur."""

    def test_no_exit_price_in_range_long(self):
        """No exit when price is between stop and target for long."""
        reason, reason_str = check_exit(
            position_dir=1,
            high=102.0,
            low=99.0,   # Above stop at 98
            stop=98.0,
            target=104.0,
            current_time=time(11, 0),
            exit_time=time(15, 45),
        )
        assert reason is None
        assert reason_str == ''

    def test_no_exit_price_in_range_short(self):
        """No exit when price is between stop and target for short."""
        reason, reason_str = check_exit(
            position_dir=-1,
            high=101.0,  # Below stop at 102
            low=97.0,    # Above target at 96
            stop=102.0,
            target=96.0,
            current_time=time(11, 0),
            exit_time=time(15, 45),
        )
        assert reason is None
        assert reason_str == ''


class TestPriorityOrdering:
    """Test that exit conditions are checked in correct priority order."""

    def test_time_exit_beats_stop_loss(self):
        """Time exit takes priority over stop-loss."""
        reason, _ = check_exit(
            position_dir=1,
            high=101.0,
            low=97.0,   # Would trigger stop at 98
            stop=98.0,
            target=104.0,
            current_time=time(15, 45),
            exit_time=time(15, 45),
        )
        assert reason == ExitReason.TIME_EXIT

    def test_time_exit_beats_target(self):
        """Time exit takes priority over target."""
        reason, _ = check_exit(
            position_dir=1,
            high=105.0,  # Would trigger target at 104
            low=99.0,
            stop=98.0,
            target=104.0,
            current_time=time(15, 45),
            exit_time=time(15, 45),
        )
        assert reason == ExitReason.TIME_EXIT

    def test_time_stop_beats_stop_loss(self):
        """Time stop takes priority over stop-loss."""
        reason, _ = check_exit(
            position_dir=1,
            high=101.0,
            low=97.0,   # Would trigger stop at 98
            stop=98.0,
            target=104.0,
            entry_bar=100,
            current_bar=200,
            max_bars=100,
        )
        assert reason == ExitReason.TIME_STOP

    def test_stop_loss_beats_target_when_both_hit(self):
        """Stop-loss checked before target when both conditions met."""
        # Long: low hits stop AND high hits target on same bar
        reason, _ = check_exit(
            position_dir=1,
            high=105.0,  # Hits target at 104
            low=97.0,    # Hits stop at 98
            stop=98.0,
            target=104.0,
        )
        assert reason == ExitReason.STOP_LOSS


class TestExitReasonEnum:
    """Test ExitReason enum values."""

    def test_enum_values(self):
        """Verify all expected enum values exist."""
        assert ExitReason.STOP_LOSS.value == 'stop_loss'
        assert ExitReason.TARGET.value == 'target'
        assert ExitReason.TIME_EXIT.value == 'time_exit'
        assert ExitReason.TIME_STOP.value == 'time_stop'
        assert ExitReason.MAX_LOSS.value == 'max_loss'
