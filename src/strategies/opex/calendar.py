"""
OpEx Calendar and Phase Detection.

Tracks monthly options expiration dates (third Friday of each month)
and determines the current OpEx phase for trading decisions.

OpEx Phases:
- NORMAL: More than 5 trading days to expiration
- PRE_OPEX: 3-5 trading days before (early gamma effects)
- PINNING: 1-2 trading days before (strongest pinning)
- OPEX_DAY: Expiration day (high uncertainty)
- POST_OPEX: 1-2 trading days after (gamma release)
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay


class OpExPhase(str, Enum):
    """
    Current phase relative to monthly options expiration.

    Trading behavior differs by phase:
    - NORMAL: No special OpEx considerations
    - PRE_OPEX: Early gamma effects, moderate pinning
    - PINNING: Strongest pin effect, fade moves away from pin
    - OPEX_DAY: High uncertainty, avoid new positions
    - POST_OPEX: Gamma released, momentum opportunity
    """
    NORMAL = "NORMAL"
    PRE_OPEX = "PRE_OPEX"
    PINNING = "PINNING"
    OPEX_DAY = "OPEX_DAY"
    POST_OPEX = "POST_OPEX"


@dataclass
class OpExCycle:
    """
    Information about a single OpEx cycle.

    Attributes:
        expiry: The monthly expiration date (third Friday)
        pre_opex_start: First day of PRE_OPEX phase
        pinning_start: First day of PINNING phase
        post_opex_end: Last day of POST_OPEX phase
    """
    expiry: date
    pre_opex_start: date
    pinning_start: date
    post_opex_end: date

    @property
    def cycle_start(self) -> date:
        """First day when OpEx effects begin."""
        return self.pre_opex_start

    @property
    def cycle_end(self) -> date:
        """Last day when OpEx effects persist."""
        return self.post_opex_end


class OpExCalendar:
    """
    Calendar utility for monthly options expiration.

    Identifies third Friday of each month (standard monthly OpEx)
    and determines current OpEx phase for any given date.

    Usage:
        calendar = OpExCalendar()

        # Get next monthly expiration
        next_opex = calendar.get_next_expiration(date.today())

        # Get current phase
        phase = calendar.get_phase(date.today())

        # Get all expirations for a year
        expirations = calendar.get_monthly_expirations(2024)
    """

    # Phase boundaries (in trading days relative to OpEx)
    PRE_OPEX_START = 5   # T-5 trading days
    PINNING_START = 2    # T-2 trading days
    POST_OPEX_END = 2    # T+2 trading days

    def __init__(self):
        """Initialize OpEx calendar."""
        # Cache for computed expirations
        self._expiration_cache: dict = {}

    @staticmethod
    def get_third_friday(year: int, month: int) -> date:
        """
        Get the third Friday of a given month.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Date of third Friday
        """
        # First day of month
        first_day = date(year, month, 1)

        # Find first Friday
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)

        # Third Friday is 14 days after first Friday
        third_friday = first_friday + timedelta(days=14)

        return third_friday

    def get_monthly_expirations(
        self,
        year: int,
        start_month: int = 1,
        end_month: int = 12,
    ) -> List[date]:
        """
        Get all monthly expirations for a year.

        Args:
            year: Year to get expirations for
            start_month: Starting month (1-12)
            end_month: Ending month (1-12)

        Returns:
            List of expiration dates
        """
        cache_key = (year, start_month, end_month)
        if cache_key in self._expiration_cache:
            return self._expiration_cache[cache_key]

        expirations = []
        for month in range(start_month, end_month + 1):
            exp = self.get_third_friday(year, month)
            expirations.append(exp)

        self._expiration_cache[cache_key] = expirations
        return expirations

    def get_expirations_in_range(
        self,
        start_date: date,
        end_date: date,
    ) -> List[date]:
        """
        Get all monthly expirations within a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            List of expiration dates in range
        """
        expirations = []

        # Iterate through months in range
        current = date(start_date.year, start_date.month, 1)
        end = date(end_date.year, end_date.month, 1)

        while current <= end:
            exp = self.get_third_friday(current.year, current.month)
            if start_date <= exp <= end_date:
                expirations.append(exp)

            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        return expirations

    def get_next_expiration(self, from_date: date) -> date:
        """
        Get the next monthly expiration on or after from_date.

        Args:
            from_date: Reference date

        Returns:
            Next expiration date
        """
        # Check current month
        exp = self.get_third_friday(from_date.year, from_date.month)
        if exp >= from_date:
            return exp

        # Move to next month
        if from_date.month == 12:
            return self.get_third_friday(from_date.year + 1, 1)
        else:
            return self.get_third_friday(from_date.year, from_date.month + 1)

    def get_previous_expiration(self, from_date: date) -> date:
        """
        Get the most recent monthly expiration before from_date.

        Args:
            from_date: Reference date

        Returns:
            Previous expiration date
        """
        # Check current month
        exp = self.get_third_friday(from_date.year, from_date.month)
        if exp < from_date:
            return exp

        # Move to previous month
        if from_date.month == 1:
            return self.get_third_friday(from_date.year - 1, 12)
        else:
            return self.get_third_friday(from_date.year, from_date.month - 1)

    def get_phase(
        self,
        current_date: date,
        next_opex: Optional[date] = None,
    ) -> OpExPhase:
        """
        Determine the OpEx phase for a given date.

        Args:
            current_date: Date to check
            next_opex: Next expiration (computed if not provided)

        Returns:
            Current OpEx phase
        """
        if next_opex is None:
            next_opex = self.get_next_expiration(current_date)

        # Check if we're past this OpEx (in POST_OPEX of previous)
        prev_opex = self.get_previous_expiration(current_date)

        # Calculate trading days to next OpEx
        days_to_opex = self._trading_days_between(current_date, next_opex)

        # Calculate trading days since previous OpEx
        days_since_opex = self._trading_days_between(prev_opex, current_date)

        # Determine phase
        if current_date == next_opex:
            return OpExPhase.OPEX_DAY

        elif days_since_opex <= self.POST_OPEX_END and current_date > prev_opex:
            return OpExPhase.POST_OPEX

        elif days_to_opex <= self.PINNING_START:
            return OpExPhase.PINNING

        elif days_to_opex <= self.PRE_OPEX_START:
            return OpExPhase.PRE_OPEX

        else:
            return OpExPhase.NORMAL

    def get_phase_info(
        self,
        current_date: date,
    ) -> Tuple[OpExPhase, date, int]:
        """
        Get detailed phase information.

        Args:
            current_date: Date to check

        Returns:
            Tuple of (phase, relevant_opex, days_to_or_from_opex)
        """
        next_opex = self.get_next_expiration(current_date)
        prev_opex = self.get_previous_expiration(current_date)
        phase = self.get_phase(current_date, next_opex)

        if phase == OpExPhase.POST_OPEX:
            days = self._trading_days_between(prev_opex, current_date)
            return phase, prev_opex, days
        else:
            days = self._trading_days_between(current_date, next_opex)
            return phase, next_opex, days

    def get_cycle(self, opex_date: date) -> OpExCycle:
        """
        Get the full OpEx cycle for an expiration.

        Args:
            opex_date: The monthly expiration date

        Returns:
            OpExCycle with all phase boundaries
        """
        # Calculate phase boundaries in trading days
        pre_opex_start = self._subtract_trading_days(opex_date, self.PRE_OPEX_START)
        pinning_start = self._subtract_trading_days(opex_date, self.PINNING_START)
        post_opex_end = self._add_trading_days(opex_date, self.POST_OPEX_END)

        return OpExCycle(
            expiry=opex_date,
            pre_opex_start=pre_opex_start,
            pinning_start=pinning_start,
            post_opex_end=post_opex_end,
        )

    def is_opex_week(self, check_date: date) -> bool:
        """
        Check if date is within an OpEx week (Mon-Fri of expiration week).

        Args:
            check_date: Date to check

        Returns:
            True if in OpEx week
        """
        next_opex = self.get_next_expiration(check_date)

        # Get Monday of OpEx week
        days_to_monday = next_opex.weekday()  # Friday = 4, so 4 days back to Monday
        opex_monday = next_opex - timedelta(days=days_to_monday)

        # Check if date is in that week
        return opex_monday <= check_date <= next_opex

    def _trading_days_between(self, start: date, end: date) -> int:
        """
        Count trading days from start to end (exclusive of start, inclusive of end).

        This counts "how many trading days until end" from start's perspective.
        Example: Jan 17 (Wed) to Jan 19 (Fri) = 2 trading days (Thu, Fri)

        Args:
            start: Start date (not counted)
            end: End date (counted)

        Returns:
            Number of trading days (0 if same day or end before start)
        """
        if start >= end:
            return 0

        # Use pandas business day counting - exclusive start, inclusive end
        return len(pd.bdate_range(start, end, inclusive='right'))

    def _subtract_trading_days(self, from_date: date, n_days: int) -> date:
        """Subtract n trading days from a date."""
        result = pd.Timestamp(from_date) - BDay(n_days)
        return result.date()

    def _add_trading_days(self, from_date: date, n_days: int) -> date:
        """Add n trading days to a date."""
        result = pd.Timestamp(from_date) + BDay(n_days)
        return result.date()

    @staticmethod
    def is_monthly_expiration(check_date: date) -> bool:
        """
        Check if a date is a monthly expiration (third Friday).

        Args:
            check_date: Date to check

        Returns:
            True if third Friday of its month
        """
        if check_date.weekday() != 4:  # Not Friday
            return False

        third_friday = OpExCalendar.get_third_friday(
            check_date.year, check_date.month
        )
        return check_date == third_friday
