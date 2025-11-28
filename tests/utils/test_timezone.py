"""
Unit tests for the centralized timezone utility module.

Tests verify that:
1. Timezone manager defaults to US/Eastern (EST/EDT)
2. All datetime operations return EST timestamps
3. Timezone can be reconfigured
4. UTC conversions work correctly
5. Formatting methods produce correct output
"""

import pytest
from datetime import datetime, date, time as dt_time, timedelta
from unittest.mock import patch
import pytz

from src.utils.timezone import (
    TimezoneManager,
    tz,
    now,
    today,
    timestamp,
    iso_timestamp,
    date_str,
    time_str,
    datetime_str,
    from_utc,
    set_timezone,
    get_timezone
)


class TestTimezoneManager:
    """Tests for the TimezoneManager class."""

    def test_default_timezone_is_eastern(self):
        """Default timezone should be US/Eastern."""
        manager = TimezoneManager()
        assert manager.timezone_name == 'US/Eastern'

    def test_custom_timezone_initialization(self):
        """Can initialize with custom timezone."""
        manager = TimezoneManager('US/Pacific')
        assert manager.timezone_name == 'US/Pacific'

    def test_set_timezone(self):
        """Can change timezone after initialization."""
        manager = TimezoneManager()
        manager.set_timezone('UTC')
        assert manager.timezone_name == 'UTC'

    def test_invalid_timezone_raises_error(self):
        """Invalid timezone should raise UnknownTimeZoneError."""
        manager = TimezoneManager()
        with pytest.raises(pytz.UnknownTimeZoneError):
            manager.set_timezone('Invalid/Timezone')

    def test_now_returns_aware_datetime(self):
        """now() should return timezone-aware datetime."""
        manager = TimezoneManager()
        result = manager.now()
        assert result.tzinfo is not None

    def test_now_returns_eastern_time(self):
        """now() should return time in Eastern timezone."""
        manager = TimezoneManager('US/Eastern')
        result = manager.now()
        # Timezone name should be EST or EDT depending on DST
        tz_name = result.tzname()
        assert tz_name in ('EST', 'EDT')

    def test_today_returns_date(self):
        """today() should return a date object."""
        manager = TimezoneManager()
        result = manager.today()
        assert isinstance(result, date)

    def test_time_returns_time(self):
        """time() should return a time object."""
        manager = TimezoneManager()
        result = manager.time()
        assert isinstance(result, dt_time)


class TestTimezoneConversions:
    """Tests for timezone conversion methods."""

    def test_from_utc_with_naive_datetime(self):
        """from_utc() should treat naive datetime as UTC."""
        manager = TimezoneManager('US/Eastern')
        utc_time = datetime(2025, 1, 15, 20, 0, 0)  # 8 PM UTC
        result = manager.from_utc(utc_time)

        # 8 PM UTC should be 3 PM EST (5 hour difference in winter)
        assert result.hour == 15
        assert result.tzinfo is not None

    def test_from_utc_with_aware_datetime(self):
        """from_utc() should convert aware UTC datetime."""
        manager = TimezoneManager('US/Eastern')
        utc_time = datetime(2025, 1, 15, 20, 0, 0, tzinfo=pytz.UTC)
        result = manager.from_utc(utc_time)

        assert result.hour == 15  # 3 PM EST
        assert result.tzinfo is not None

    def test_to_utc_with_naive_datetime(self):
        """to_utc() should treat naive datetime as local timezone."""
        manager = TimezoneManager('US/Eastern')
        local_time = datetime(2025, 1, 15, 15, 0, 0)  # 3 PM EST
        result = manager.to_utc(local_time)

        # 3 PM EST should be 8 PM UTC (5 hour difference in winter)
        assert result.hour == 20
        assert result.tzinfo == pytz.UTC

    def test_localize_adds_timezone(self):
        """localize() should add timezone info to naive datetime."""
        manager = TimezoneManager('US/Eastern')
        naive_dt = datetime(2025, 1, 15, 15, 0, 0)
        result = manager.localize(naive_dt)

        assert result.tzinfo is not None
        assert result.hour == 15  # Hour should remain the same


class TestTimezoneFormatting:
    """Tests for timestamp formatting methods."""

    def test_timestamp_default_format(self):
        """timestamp() should return YYYY-MM-DD HH:MM:SS format."""
        manager = TimezoneManager()
        result = manager.timestamp()
        # Should match pattern like "2025-01-15 15:30:00"
        assert len(result) == 19
        assert result[4] == '-'
        assert result[7] == '-'
        assert result[10] == ' '
        assert result[13] == ':'
        assert result[16] == ':'

    def test_timestamp_custom_format(self):
        """timestamp() should accept custom format."""
        manager = TimezoneManager()
        result = manager.timestamp('%Y/%m/%d')
        assert '/' in result
        assert len(result) == 10

    def test_iso_timestamp_includes_timezone(self):
        """iso_timestamp() should include timezone offset."""
        manager = TimezoneManager('US/Eastern')
        result = manager.iso_timestamp()
        # Should end with timezone offset like -05:00 or -04:00
        assert '-0' in result or '+0' in result

    def test_date_str_default_format(self):
        """date_str() should return YYYYMMDD format."""
        manager = TimezoneManager()
        result = manager.date_str()
        assert len(result) == 8
        assert result.isdigit()

    def test_time_str_default_format(self):
        """time_str() should return HH:MM:SS format."""
        manager = TimezoneManager()
        result = manager.time_str()
        assert len(result) == 8
        assert result[2] == ':'
        assert result[5] == ':'

    def test_datetime_str_default_format(self):
        """datetime_str() should return YYYYMMDD_HHMMSS format."""
        manager = TimezoneManager()
        result = manager.datetime_str()
        assert len(result) == 15
        assert result[8] == '_'


class TestTimezoneComparison:
    """Tests for comparison helper methods."""

    def test_is_same_day_true(self):
        """is_same_day() should return True for same day."""
        manager = TimezoneManager('US/Eastern')
        dt1 = manager.now()
        dt2 = manager.now()
        assert manager.is_same_day(dt1, dt2)

    def test_is_same_day_false(self):
        """is_same_day() should return False for different days."""
        manager = TimezoneManager('US/Eastern')
        dt1 = datetime(2025, 1, 15, 12, 0, 0)
        dt2 = datetime(2025, 1, 16, 12, 0, 0)
        dt1 = manager.localize(dt1)
        dt2 = manager.localize(dt2)
        assert not manager.is_same_day(dt1, dt2)

    def test_seconds_since(self):
        """seconds_since() should return elapsed seconds."""
        manager = TimezoneManager()
        past_time = manager.now() - timedelta(seconds=60)
        result = manager.seconds_since(past_time)
        # Should be approximately 60 seconds (with small tolerance)
        assert 59 <= result <= 61

    def test_hours_since(self):
        """hours_since() should return elapsed hours."""
        manager = TimezoneManager()
        past_time = manager.now() - timedelta(hours=2)
        result = manager.hours_since(past_time)
        # Should be approximately 2 hours (with small tolerance)
        assert 1.99 <= result <= 2.01


class TestGlobalInstance:
    """Tests for the global tz instance and convenience functions."""

    def test_global_tz_is_eastern(self):
        """Global tz instance should default to US/Eastern."""
        assert tz.timezone_name == 'US/Eastern'

    def test_now_convenience_function(self):
        """now() convenience function should work."""
        result = now()
        assert result.tzinfo is not None

    def test_today_convenience_function(self):
        """today() convenience function should work."""
        result = today()
        assert isinstance(result, date)

    def test_timestamp_convenience_function(self):
        """timestamp() convenience function should work."""
        result = timestamp()
        assert isinstance(result, str)
        assert len(result) == 19

    def test_iso_timestamp_convenience_function(self):
        """iso_timestamp() convenience function should work."""
        result = iso_timestamp()
        assert 'T' in result

    def test_date_str_convenience_function(self):
        """date_str() convenience function should work."""
        result = date_str()
        assert len(result) == 8

    def test_time_str_convenience_function(self):
        """time_str() convenience function should work."""
        result = time_str()
        assert ':' in result

    def test_datetime_str_convenience_function(self):
        """datetime_str() convenience function should work."""
        result = datetime_str()
        assert '_' in result

    def test_from_utc_convenience_function(self):
        """from_utc() convenience function should work."""
        utc_time = datetime(2025, 1, 15, 20, 0, 0, tzinfo=pytz.UTC)
        result = from_utc(utc_time)
        assert result.tzinfo is not None

    def test_get_timezone_returns_name(self):
        """get_timezone() should return timezone name."""
        result = get_timezone()
        assert result == 'US/Eastern'


class TestESTConsistency:
    """Tests to verify EST is used consistently."""

    def test_utc_to_est_conversion_winter(self):
        """Test UTC to EST conversion during winter (standard time)."""
        manager = TimezoneManager('US/Eastern')
        # January is winter - EST (UTC-5)
        utc_time = datetime(2025, 1, 15, 20, 50, 0, tzinfo=pytz.UTC)  # 8:50 PM UTC
        est_time = manager.from_utc(utc_time)

        assert est_time.hour == 15  # 3:50 PM EST
        assert est_time.minute == 50

    def test_utc_to_est_conversion_summer(self):
        """Test UTC to EST conversion during summer (daylight time)."""
        manager = TimezoneManager('US/Eastern')
        # July is summer - EDT (UTC-4)
        utc_time = datetime(2025, 7, 15, 19, 50, 0, tzinfo=pytz.UTC)  # 7:50 PM UTC
        edt_time = manager.from_utc(utc_time)

        assert edt_time.hour == 15  # 3:50 PM EDT
        assert edt_time.minute == 50

    def test_trading_time_conversion(self):
        """Test that 3:50 PM EST trading time is correctly handled."""
        manager = TimezoneManager('US/Eastern')
        # Create 3:50 PM EST
        eastern = pytz.timezone('US/Eastern')
        trading_time = eastern.localize(datetime(2025, 1, 15, 15, 50, 0))

        # Convert to UTC and back
        utc_time = manager.to_utc(trading_time)
        back_to_est = manager.from_utc(utc_time)

        assert back_to_est.hour == 15
        assert back_to_est.minute == 50


class TestLogTimestampFormat:
    """Tests to verify log timestamp formatting matches expectations."""

    def test_log_filename_format(self):
        """Log filename should use YYYYMMDD_HHMMSS format."""
        result = datetime_str()
        # Should be like "20251128_103045"
        parts = result.split('_')
        assert len(parts) == 2
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_log_timestamp_format(self):
        """Log timestamp should use YYYY-MM-DD HH:MM:SS format."""
        result = timestamp()
        # Should be like "2025-11-28 10:30:45"
        assert result[4] == '-'
        assert result[7] == '-'
        assert result[10] == ' '
        assert result[13] == ':'
        assert result[16] == ':'

    def test_iso_timestamp_for_csv(self):
        """ISO timestamp should be suitable for CSV logging."""
        result = iso_timestamp()
        # Should be parseable by datetime.fromisoformat()
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
