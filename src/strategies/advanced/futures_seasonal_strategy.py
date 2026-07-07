"""Seasonal / calendar futures strategies (SP-A).

#16 turn-of-month: long index roots from the last trading day of a month through
the third trading day of the next month (payment-cycle liquidity flow).

#15 same-calendar-month seasonality: rank each commodity root by its expanding
mean return in the current calendar month, using strictly-prior years only
(causal), across the commodity block."""
from __future__ import annotations

import pandas as pd

from src.strategies.advanced.futures_signal_base import CalendarMaskStrategy, CrossSectionalRankStrategy

_TOM_LEAD = 1    # last trading day of the month (1 trading day before month end)
_TOM_LAG = 3     # first three trading days of the new month
_SEASONAL_GROUP = "commodity"


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


class FuturesSameMonthSeasonalityStrategy(CrossSectionalRankStrategy):
    """Rank each root by its historical mean return in the current calendar month,
    using strictly-prior years only (causal), across the commodity block.

    Signal at a date in month m = mean over prior years of (that root's return in
    month m). Returns are month-over-month; the current year's month-m return is
    excluded so the estimate never sees the outcome it predicts."""

    def __init__(self, universe, cap: float = 20.0, **params):
        super().__init__(universe, group_fn=lambda r: _SEASONAL_GROUP, cap=cap)

    def _raw_signal_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        cols = [r for r in self.universe if r in close_panel.columns]
        px = close_panel[cols].astype(float)
        px = px.copy()
        px.index = pd.DatetimeIndex(px.index)
        monthly = px.resample("ME").last()
        m_ret = monthly.pct_change(fill_method=None)
        sig_m = pd.DataFrame(index=monthly.index, columns=cols, dtype=float)
        for col in cols:
            r = m_ret[col]
            for cal_m in range(1, 13):
                same = r[r.index.month == cal_m]
                # expanding mean of STRICTLY-PRIOR years' same-month returns (shift(1)
                # drops the current year) -> known at the start of month cal_m, causal
                prior_mean = same.shift(1).expanding(min_periods=1).mean()
                sig_m.loc[same.index, col] = prior_mean.values
        # map the monthly signal to daily rows by (year, month), NOT by timestamp:
        # calendar month-end (ME) falls after business month-end (BME), so a
        # timestamp ffill would wrongly assign the prior month's signal.
        sig_m.index = pd.MultiIndex.from_arrays([sig_m.index.year, sig_m.index.month])
        daily_key = pd.MultiIndex.from_arrays([px.index.year, px.index.month])
        sig_daily = sig_m.reindex(daily_key)
        # Restore the ORIGINAL close_panel.index (not the DatetimeIndex-converted
        # copy): the futures engine does `d in forecast_panel.index` membership
        # checks using close_panel's own index objects, and a value-equal but
        # dtype-different index (e.g. datetime64 vs. object) fails that lookup,
        # silently zeroing every position.
        sig_daily.index = close_panel.index
        return sig_daily.reindex(columns=self.universe)
