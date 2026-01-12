"""Tests for OpEx calendar and phase detection."""

import pytest
from datetime import date, timedelta

from src.strategies.opex.calendar import OpExCalendar, OpExPhase, OpExCycle


class TestOpExCalendarThirdFriday:
    """Tests for third Friday calculation."""

    def test_third_friday_january_2024(self):
        """Test third Friday of January 2024."""
        result = OpExCalendar.get_third_friday(2024, 1)
        assert result == date(2024, 1, 19)

    def test_third_friday_february_2024(self):
        """Test third Friday of February 2024."""
        result = OpExCalendar.get_third_friday(2024, 2)
        assert result == date(2024, 2, 16)

    def test_third_friday_march_2024(self):
        """Test third Friday of March 2024."""
        result = OpExCalendar.get_third_friday(2024, 3)
        assert result == date(2024, 3, 15)

    def test_third_friday_all_2024(self):
        """Test all third Fridays of 2024."""
        expected = [
            date(2024, 1, 19),
            date(2024, 2, 16),
            date(2024, 3, 15),
            date(2024, 4, 19),
            date(2024, 5, 17),
            date(2024, 6, 21),
            date(2024, 7, 19),
            date(2024, 8, 16),
            date(2024, 9, 20),
            date(2024, 10, 18),
            date(2024, 11, 15),
            date(2024, 12, 20),
        ]

        for month, exp in enumerate(expected, start=1):
            result = OpExCalendar.get_third_friday(2024, month)
            assert result == exp, f"Month {month}: expected {exp}, got {result}"

    def test_third_friday_always_friday(self):
        """Verify third Friday is always a Friday."""
        for year in range(2020, 2030):
            for month in range(1, 13):
                result = OpExCalendar.get_third_friday(year, month)
                assert result.weekday() == 4, f"{result} is not Friday"


class TestOpExCalendarExpirations:
    """Tests for expiration listing."""

    def test_monthly_expirations_full_year(self):
        """Test getting all expirations for a year."""
        calendar = OpExCalendar()
        expirations = calendar.get_monthly_expirations(2024)

        assert len(expirations) == 12
        assert expirations[0] == date(2024, 1, 19)
        assert expirations[-1] == date(2024, 12, 20)

    def test_monthly_expirations_partial_year(self):
        """Test getting expirations for part of a year."""
        calendar = OpExCalendar()
        expirations = calendar.get_monthly_expirations(2024, start_month=3, end_month=6)

        assert len(expirations) == 4
        assert expirations[0] == date(2024, 3, 15)
        assert expirations[-1] == date(2024, 6, 21)

    def test_expirations_in_range(self):
        """Test getting expirations within a date range."""
        calendar = OpExCalendar()
        expirations = calendar.get_expirations_in_range(
            date(2024, 2, 1),
            date(2024, 4, 30),
        )

        assert len(expirations) == 3
        assert date(2024, 2, 16) in expirations
        assert date(2024, 3, 15) in expirations
        assert date(2024, 4, 19) in expirations

    def test_expirations_in_range_crosses_year(self):
        """Test range that crosses year boundary."""
        calendar = OpExCalendar()
        expirations = calendar.get_expirations_in_range(
            date(2023, 11, 1),
            date(2024, 2, 29),
        )

        assert date(2023, 11, 17) in expirations
        assert date(2023, 12, 15) in expirations
        assert date(2024, 1, 19) in expirations
        assert date(2024, 2, 16) in expirations


class TestOpExCalendarNextPrevious:
    """Tests for next/previous expiration."""

    def test_next_expiration_before_opex(self):
        """Test next expiration when before OpEx in month."""
        calendar = OpExCalendar()

        # January 15, 2024 -> next is January 19, 2024
        result = calendar.get_next_expiration(date(2024, 1, 15))
        assert result == date(2024, 1, 19)

    def test_next_expiration_on_opex(self):
        """Test next expiration when on OpEx day."""
        calendar = OpExCalendar()

        # January 19, 2024 (OpEx) -> should return same day
        result = calendar.get_next_expiration(date(2024, 1, 19))
        assert result == date(2024, 1, 19)

    def test_next_expiration_after_opex(self):
        """Test next expiration when after OpEx in month."""
        calendar = OpExCalendar()

        # January 20, 2024 -> next is February 16, 2024
        result = calendar.get_next_expiration(date(2024, 1, 20))
        assert result == date(2024, 2, 16)

    def test_previous_expiration(self):
        """Test previous expiration."""
        calendar = OpExCalendar()

        # January 25, 2024 -> previous is January 19, 2024
        result = calendar.get_previous_expiration(date(2024, 1, 25))
        assert result == date(2024, 1, 19)

    def test_previous_expiration_crosses_year(self):
        """Test previous expiration at year start."""
        calendar = OpExCalendar()

        # January 5, 2024 -> previous is December 15, 2023
        result = calendar.get_previous_expiration(date(2024, 1, 5))
        assert result == date(2023, 12, 15)


class TestOpExCalendarPhases:
    """Tests for phase detection."""

    def test_phase_normal(self):
        """Test NORMAL phase (far from OpEx)."""
        calendar = OpExCalendar()

        # January 5, 2024 is 10 trading days before Jan 19
        phase = calendar.get_phase(date(2024, 1, 5))
        assert phase == OpExPhase.NORMAL

    def test_phase_pre_opex(self):
        """Test PRE_OPEX phase (3-5 trading days before)."""
        calendar = OpExCalendar()

        # January 12, 2024 is 5 trading days before Jan 19
        phase = calendar.get_phase(date(2024, 1, 12))
        assert phase == OpExPhase.PRE_OPEX

    def test_phase_pinning(self):
        """Test PINNING phase (1-2 trading days before)."""
        calendar = OpExCalendar()

        # January 17, 2024 is 2 trading days before Jan 19
        phase = calendar.get_phase(date(2024, 1, 17))
        assert phase == OpExPhase.PINNING

        # January 18, 2024 is 1 trading day before Jan 19
        phase = calendar.get_phase(date(2024, 1, 18))
        assert phase == OpExPhase.PINNING

    def test_phase_opex_day(self):
        """Test OPEX_DAY phase."""
        calendar = OpExCalendar()

        phase = calendar.get_phase(date(2024, 1, 19))
        assert phase == OpExPhase.OPEX_DAY

    def test_phase_post_opex(self):
        """Test POST_OPEX phase (1-2 trading days after)."""
        calendar = OpExCalendar()

        # January 22, 2024 is 1 trading day after Jan 19
        phase = calendar.get_phase(date(2024, 1, 22))
        assert phase == OpExPhase.POST_OPEX

        # January 23, 2024 is 2 trading days after Jan 19
        phase = calendar.get_phase(date(2024, 1, 23))
        assert phase == OpExPhase.POST_OPEX

    def test_phase_info(self):
        """Test detailed phase information."""
        calendar = OpExCalendar()

        # During pinning
        phase, opex, days = calendar.get_phase_info(date(2024, 1, 17))
        assert phase == OpExPhase.PINNING
        assert opex == date(2024, 1, 19)
        assert days == 2  # 2 trading days to OpEx

        # During post-opex
        phase, opex, days = calendar.get_phase_info(date(2024, 1, 22))
        assert phase == OpExPhase.POST_OPEX
        assert opex == date(2024, 1, 19)
        assert days == 1  # 1 trading day since OpEx


class TestOpExCalendarCycle:
    """Tests for OpEx cycle information."""

    def test_get_cycle(self):
        """Test getting cycle boundaries."""
        calendar = OpExCalendar()
        cycle = calendar.get_cycle(date(2024, 1, 19))

        assert cycle.expiry == date(2024, 1, 19)
        # PRE_OPEX starts 5 trading days before
        assert cycle.pre_opex_start == date(2024, 1, 12)
        # PINNING starts 2 trading days before
        assert cycle.pinning_start == date(2024, 1, 17)
        # POST_OPEX ends 2 trading days after
        assert cycle.post_opex_end == date(2024, 1, 23)

    def test_cycle_properties(self):
        """Test cycle property accessors."""
        calendar = OpExCalendar()
        cycle = calendar.get_cycle(date(2024, 1, 19))

        assert cycle.cycle_start == date(2024, 1, 12)  # PRE_OPEX start
        assert cycle.cycle_end == date(2024, 1, 23)    # POST_OPEX end


class TestOpExCalendarUtilities:
    """Tests for utility methods."""

    def test_is_opex_week(self):
        """Test OpEx week detection."""
        calendar = OpExCalendar()

        # Monday of OpEx week
        assert calendar.is_opex_week(date(2024, 1, 15)) is True
        # Friday (OpEx day)
        assert calendar.is_opex_week(date(2024, 1, 19)) is True
        # Previous week
        assert calendar.is_opex_week(date(2024, 1, 12)) is False
        # Following week
        assert calendar.is_opex_week(date(2024, 1, 22)) is False

    def test_is_monthly_expiration(self):
        """Test monthly expiration check."""
        # Monthly expirations
        assert OpExCalendar.is_monthly_expiration(date(2024, 1, 19)) is True
        assert OpExCalendar.is_monthly_expiration(date(2024, 2, 16)) is True

        # Not monthly (weekly)
        assert OpExCalendar.is_monthly_expiration(date(2024, 1, 12)) is False
        assert OpExCalendar.is_monthly_expiration(date(2024, 1, 26)) is False

        # Not even Friday
        assert OpExCalendar.is_monthly_expiration(date(2024, 1, 18)) is False
