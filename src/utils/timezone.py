"""
Centralized timezone utility for consistent timestamp handling.

All live trading logging and timestamp generation should use this module
to ensure consistent EST/ET timezone usage across the application.

Usage:
    from src.utils.timezone import tz

    # Get current time in configured timezone (default: US/Eastern)
    now = tz.now()

    # Format timestamps
    timestamp = tz.timestamp()           # "2025-01-15 15:50:23"
    iso_timestamp = tz.iso_timestamp()   # "2025-01-15T15:50:23.123456-05:00"
    date_str = tz.date_str()             # "20250115"
    time_str = tz.time_str()             # "15:50:23"
    datetime_str = tz.datetime_str()     # "20250115_155023"

    # Get today's date
    today = tz.today()

    # Convert UTC datetime to configured timezone
    est_time = tz.from_utc(utc_datetime)

    # Configure timezone (if needed)
    tz.set_timezone('US/Pacific')
"""

from datetime import datetime, date, time as dt_time
from typing import Optional
import pytz


class TimezoneManager:
    """
    Centralized timezone manager for consistent timestamp handling.

    Default timezone is US/Eastern (EST/EDT) for trading applications.
    All timestamps are timezone-aware and consistent across environments.
    """

    # Default timezone for trading (EST/EDT)
    DEFAULT_TIMEZONE = 'US/Eastern'

    def __init__(self, timezone: str = None):
        """
        Initialize timezone manager.

        Args:
            timezone: Timezone string (e.g., 'US/Eastern', 'UTC', 'US/Pacific')
                     Defaults to US/Eastern for trading applications.
        """
        self._timezone_str = timezone or self.DEFAULT_TIMEZONE
        self._timezone = pytz.timezone(self._timezone_str)

    @property
    def timezone(self) -> pytz.BaseTzInfo:
        """Get the current timezone object."""
        return self._timezone

    @property
    def timezone_name(self) -> str:
        """Get the current timezone name."""
        return self._timezone_str

    def set_timezone(self, timezone: str) -> None:
        """
        Set the timezone for all timestamp operations.

        Args:
            timezone: Timezone string (e.g., 'US/Eastern', 'UTC', 'US/Pacific')

        Raises:
            pytz.UnknownTimeZoneError: If timezone string is invalid
        """
        self._timezone = pytz.timezone(timezone)
        self._timezone_str = timezone

    def now(self) -> datetime:
        """
        Get current datetime in configured timezone.

        Returns:
            Timezone-aware datetime in configured timezone
        """
        return datetime.now(pytz.UTC).astimezone(self._timezone)

    def today(self) -> date:
        """
        Get today's date in configured timezone.

        Returns:
            Date object for today in configured timezone
        """
        return self.now().date()

    def time(self) -> dt_time:
        """
        Get current time in configured timezone.

        Returns:
            Time object for current time in configured timezone
        """
        return self.now().time()

    def from_utc(self, utc_dt: datetime) -> datetime:
        """
        Convert UTC datetime to configured timezone.

        Args:
            utc_dt: UTC datetime (can be naive or aware)

        Returns:
            Timezone-aware datetime in configured timezone
        """
        if utc_dt.tzinfo is None:
            # Assume naive datetime is UTC
            utc_dt = pytz.UTC.localize(utc_dt)
        return utc_dt.astimezone(self._timezone)

    def to_utc(self, local_dt: datetime) -> datetime:
        """
        Convert local datetime to UTC.

        Args:
            local_dt: Datetime in configured timezone (can be naive or aware)

        Returns:
            Timezone-aware datetime in UTC
        """
        if local_dt.tzinfo is None:
            # Assume naive datetime is in configured timezone
            local_dt = self._timezone.localize(local_dt)
        return local_dt.astimezone(pytz.UTC)

    def localize(self, naive_dt: datetime) -> datetime:
        """
        Add timezone info to a naive datetime.

        Args:
            naive_dt: Naive datetime to localize

        Returns:
            Timezone-aware datetime in configured timezone
        """
        return self._timezone.localize(naive_dt)

    # =========================================================================
    # Formatting Methods - Standard formats for logging
    # =========================================================================

    def timestamp(self, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
        """
        Get formatted timestamp string.

        Args:
            fmt: strftime format string (default: '%Y-%m-%d %H:%M:%S')

        Returns:
            Formatted timestamp string (e.g., "2025-01-15 15:50:23")
        """
        return self.now().strftime(fmt)

    def iso_timestamp(self) -> str:
        """
        Get ISO 8601 formatted timestamp with timezone.

        Returns:
            ISO timestamp string (e.g., "2025-01-15T15:50:23.123456-05:00")
        """
        return self.now().isoformat()

    def date_str(self, fmt: str = '%Y%m%d') -> str:
        """
        Get formatted date string for filenames.

        Args:
            fmt: strftime format string (default: '%Y%m%d')

        Returns:
            Formatted date string (e.g., "20250115")
        """
        return self.now().strftime(fmt)

    def time_str(self, fmt: str = '%H:%M:%S') -> str:
        """
        Get formatted time string.

        Args:
            fmt: strftime format string (default: '%H:%M:%S')

        Returns:
            Formatted time string (e.g., "15:50:23")
        """
        return self.now().strftime(fmt)

    def datetime_str(self, fmt: str = '%Y%m%d_%H%M%S') -> str:
        """
        Get formatted datetime string for filenames.

        Args:
            fmt: strftime format string (default: '%Y%m%d_%H%M%S')

        Returns:
            Formatted datetime string (e.g., "20250115_155023")
        """
        return self.now().strftime(fmt)

    def format(self, dt: datetime, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
        """
        Format a datetime object to string.

        Args:
            dt: Datetime to format
            fmt: strftime format string

        Returns:
            Formatted datetime string
        """
        return dt.strftime(fmt)

    # =========================================================================
    # Comparison Helpers
    # =========================================================================

    def is_same_day(self, dt1: datetime, dt2: datetime = None) -> bool:
        """
        Check if two datetimes are on the same day in configured timezone.

        Args:
            dt1: First datetime
            dt2: Second datetime (defaults to now)

        Returns:
            True if same day, False otherwise
        """
        if dt2 is None:
            dt2 = self.now()

        # Convert both to configured timezone
        if dt1.tzinfo is None:
            dt1 = self.localize(dt1)
        else:
            dt1 = dt1.astimezone(self._timezone)

        if dt2.tzinfo is None:
            dt2 = self.localize(dt2)
        else:
            dt2 = dt2.astimezone(self._timezone)

        return dt1.date() == dt2.date()

    def seconds_since(self, dt: datetime) -> float:
        """
        Get seconds elapsed since a datetime.

        Args:
            dt: Datetime to measure from

        Returns:
            Seconds elapsed (float)
        """
        now = self.now()
        if dt.tzinfo is None:
            dt = self.localize(dt)
        return (now - dt).total_seconds()

    def hours_since(self, dt: datetime) -> float:
        """
        Get hours elapsed since a datetime.

        Args:
            dt: Datetime to measure from

        Returns:
            Hours elapsed (float)
        """
        return self.seconds_since(dt) / 3600


# Global instance with default timezone (US/Eastern for trading)
tz = TimezoneManager()


# Convenience functions that delegate to global instance
def now() -> datetime:
    """Get current datetime in configured timezone."""
    return tz.now()


def today() -> date:
    """Get today's date in configured timezone."""
    return tz.today()


def timestamp(fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Get formatted timestamp string."""
    return tz.timestamp(fmt)


def iso_timestamp() -> str:
    """Get ISO 8601 formatted timestamp with timezone."""
    return tz.iso_timestamp()


def date_str(fmt: str = '%Y%m%d') -> str:
    """Get formatted date string for filenames."""
    return tz.date_str(fmt)


def time_str(fmt: str = '%H:%M:%S') -> str:
    """Get formatted time string."""
    return tz.time_str(fmt)


def datetime_str(fmt: str = '%Y%m%d_%H%M%S') -> str:
    """Get formatted datetime string for filenames."""
    return tz.datetime_str(fmt)


def from_utc(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to configured timezone."""
    return tz.from_utc(utc_dt)


def set_timezone(timezone: str) -> None:
    """Set the global timezone for all timestamp operations."""
    tz.set_timezone(timezone)


def get_timezone() -> str:
    """Get the current timezone name."""
    return tz.timezone_name


def ensure_et_index(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Ensure DataFrame index is in Eastern Time.

    This is a utility for validating/converting DataFrame timestamps.
    Primarily used to verify broker data contracts.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with ET-converted index

    Raises:
        ValueError: If index is not a DatetimeIndex
    """
    import pandas as pd

    if df.empty:
        return df

    ET = pytz.timezone('America/New_York')

    if isinstance(df.index, pd.MultiIndex):
        ts_level = df.index.get_level_values(1)
        if hasattr(ts_level, 'tz') and ts_level.tz is not None:
            new_ts = ts_level.tz_convert(ET)
            df.index = pd.MultiIndex.from_arrays(
                [df.index.get_level_values(0), new_ts],
                names=df.index.names
            )
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert(ET)
        else:
            df.index = df.index.tz_localize('UTC').tz_convert(ET)

    return df


def assert_et_timezone(df: 'pd.DataFrame', context: str = "") -> None:
    """
    Assert that DataFrame index is in Eastern Time.

    Use this to validate broker data contracts. Raises if timezone is wrong.

    Args:
        df: DataFrame to validate
        context: Description for error messages (e.g., "get_bars output")

    Raises:
        AssertionError: If index is not in Eastern Time
    """
    import pandas as pd

    if df.empty:
        return

    # Get the timestamp index
    if isinstance(df.index, pd.MultiIndex):
        ts_index = df.index.get_level_values(1)
    else:
        ts_index = df.index

    if not isinstance(ts_index, pd.DatetimeIndex):
        return  # Can't validate non-datetime indexes

    if ts_index.tz is None:
        raise AssertionError(
            f"DataFrame index has no timezone{' (' + context + ')' if context else ''}. "
            "Expected US/Eastern."
        )

    tz_name = str(ts_index.tz)
    if 'Eastern' not in tz_name and 'New_York' not in tz_name:
        raise AssertionError(
            f"DataFrame index timezone is {tz_name}{' (' + context + ')' if context else ''}. "
            "Expected US/Eastern."
        )
