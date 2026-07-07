"""Seasonal / calendar futures strategies (SP-A).

#16 turn-of-month: long index roots from the last trading day of a month through
the third trading day of the next month (payment-cycle liquidity flow)."""
from __future__ import annotations

import pandas as pd

from src.strategies.advanced.futures_signal_base import CalendarMaskStrategy

_TOM_LEAD = 1    # last trading day of the month (1 trading day before month end)
_TOM_LAG = 3     # first three trading days of the new month


class FuturesTurnOfMonthStrategy(CalendarMaskStrategy):
    """Long index roots across the turn-of-month window (last day .. first +3)."""

    def _active_and_sign(self, index: pd.DatetimeIndex) -> pd.Series:
        dt_index = pd.DatetimeIndex(index)
        s = pd.Series(0.0, index=index)
        by_month = pd.Series(index, index=dt_index).groupby([dt_index.year, dt_index.month])
        active_days: set = set()
        for _, days in by_month:
            days = list(days)
            active_days.update(days[-_TOM_LEAD:])  # last trading day(s) of this month
            active_days.update(days[:_TOM_LAG])     # first trading day(s) of this month
        s.loc[s.index.isin(active_days)] = 1.0
        return s
