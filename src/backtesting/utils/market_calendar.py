"""
Market calendar utilities for filtering trading days.

Uses pandas_market_calendars for accurate NYSE trading days including:
- Weekends (Saturday, Sunday)
- Federal holidays (New Year's Day, MLK Day, Presidents Day, Good Friday, Memorial Day,
  Independence Day, Labor Day, Thanksgiving, Christmas)
- Early closes
- Special market closures

This ensures backtests only trade on actual market open days.
"""

import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from typing import Union, List


class MarketCalendar:
    """
    NYSE market calendar for filtering valid trading days.

    Uses pandas_market_calendars which includes:
    - All NYSE market holidays
    - Weekend filtering
    - Historical market closures
    - No web requests needed (static calendar data)
    """

    def __init__(self, exchange: str = 'NYSE'):
        """
        Initialize market calendar.

        Args:
            exchange: Market exchange (default: 'NYSE')
                     Other options: 'NASDAQ', 'LSE', 'TSX', etc.
        """
        self.calendar = mcal.get_calendar(exchange)
        self.exchange = exchange

    def is_trading_day(self, date: Union[str, datetime, pd.Timestamp]) -> bool:
        """
        Check if a specific date is a trading day.

        Args:
            date: Date to check (string 'YYYY-MM-DD', datetime, or Timestamp)

        Returns:
            True if market is open, False otherwise (weekend or holiday)

        Example:
            >>> cal = MarketCalendar()
            >>> cal.is_trading_day('2023-07-04')  # Independence Day
            False
            >>> cal.is_trading_day('2023-07-05')  # Wednesday
            True
        """
        date_ts = pd.Timestamp(date)
        schedule = self.calendar.schedule(start_date=date_ts, end_date=date_ts)
        return len(schedule) > 0

    def get_trading_days(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """
        Get all trading days in a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DatetimeIndex of valid trading days

        Example:
            >>> cal = MarketCalendar()
            >>> days = cal.get_trading_days('2023-01-01', '2023-01-31')
            >>> len(days)  # ~21 trading days in January
            21
        """
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
        return schedule.index

    def filter_trading_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a DataFrame to only include rows on valid trading days.

        Assumes DataFrame has either:
        - A DatetimeIndex
        - A MultiIndex with 'timestamp' or datetime level

        Args:
            df: DataFrame with datetime index or MultiIndex

        Returns:
            Filtered DataFrame containing only trading day data

        Example:
            >>> cal = MarketCalendar()
            >>> df = load_data('AAPL', '2023-01-01', '2023-12-31')
            >>> df_filtered = cal.filter_trading_days(df)
            >>> # df_filtered now excludes weekends and holidays
        """
        # Handle MultiIndex case
        if isinstance(df.index, pd.MultiIndex):
            # Find datetime level (usually 'timestamp' or second level)
            datetime_level = None
            for i, level in enumerate(df.index.levels):
                if isinstance(level, pd.DatetimeIndex):
                    datetime_level = i
                    break

            if datetime_level is None:
                raise ValueError("No datetime level found in MultiIndex")

            # Get datetime values from that level
            datetime_values = df.index.get_level_values(datetime_level)
        else:
            # Single index case
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be DatetimeIndex or MultiIndex with datetime level")
            datetime_values = df.index

        # Get trading days for the full range
        min_date = datetime_values.min()
        max_date = datetime_values.max()
        trading_days = self.get_trading_days(min_date, max_date)

        # Convert to timezone-naive date-only for comparison
        # Strip timezone if present (Alpaca data is UTC), then normalize to midnight
        if datetime_values.tz is not None:
            datetime_dates = datetime_values.tz_localize(None).normalize()
        else:
            datetime_dates = datetime_values.normalize()

        trading_days_dates = trading_days.normalize()

        # Create mask for rows on trading days
        mask = datetime_dates.isin(trading_days_dates)

        return df[mask]

    def get_holiday_list(self, year: int) -> List[pd.Timestamp]:
        """
        Get list of market holidays for a specific year.

        Args:
            year: Calendar year

        Returns:
            List of holiday dates as Timestamps

        Example:
            >>> cal = MarketCalendar()
            >>> holidays = cal.get_holiday_list(2023)
            >>> print(holidays)
            [Timestamp('2023-01-02'), Timestamp('2023-01-16'), ...]
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Get all calendar days
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')

        # Get trading days
        trading_days = self.get_trading_days(start_date, end_date)

        # Holidays are weekdays that aren't trading days
        weekdays = all_days[all_days.dayofweek < 5]  # Monday=0, Friday=4
        holidays = weekdays[~weekdays.isin(trading_days)]

        return holidays.tolist()


# Global instance for convenience
NYSE = MarketCalendar('NYSE')


def is_trading_day(date: Union[str, datetime, pd.Timestamp]) -> bool:
    """
    Quick check if a date is a NYSE trading day.

    Args:
        date: Date to check

    Returns:
        True if NYSE is open, False otherwise

    Example:
        >>> from backtesting.utils.market_calendar import is_trading_day
        >>> is_trading_day('2023-12-25')  # Christmas
        False
    """
    return NYSE.is_trading_day(date)


def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only NYSE trading days.

    Args:
        df: DataFrame with datetime index

    Returns:
        Filtered DataFrame

    Example:
        >>> from backtesting.utils.market_calendar import filter_trading_days
        >>> df_clean = filter_trading_days(df)
    """
    return NYSE.filter_trading_days(df)
