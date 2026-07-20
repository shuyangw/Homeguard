"""FxTurnOfMonth: seasonal turn-of-month USD basket (#33).

Unconditional, every calendar month: long AUDUSD / short USDJPY at equal
Carver-scale forecast magnitude (+-10.0), held over a fixed 5-trading-day
window -- T-1 through T+3, where T is the first trading day of the month as
observed in close_panel.index (position-based, not calendar-day-based). The
forecast is 0 outside that window, giving a naturally periodic
flat -> full -> flat cycle with no separate exit bookkeeping.

ATR-proxy stop (approximation, documented): forecast_panel only receives
close prices (no OHLC), so a true intraday/high-low ATR cannot be computed
through this interface. `_basket_atr_proxy` substitutes a close-only proxy
(rolling mean of |close.pct_change()| per leg, summed across the two legs
since they form a spread -- fractional, not raw price units, so the two
legs are on a comparable scale despite AUDUSD and USDJPY trading at very
different price levels). The basket ATR proxy value is captured once, at window
entry (T-1's close), and held fixed for the rest of that window -- using a
value that updates intra-window would let the stop threshold react to the
same adverse move it's supposed to be gating against, which is both
non-causal in spirit and destabilizing. If the basket's cumulative adverse
close-to-close move since entry exceeds `atr_stop_mult` x that fixed
ATR proxy, the forecast is forced to 0 for the remainder of the window
(early exit; no re-entry until next month's window).

This is a zero-tunable, pre-registered specification: forecast_scale,
atr_window, and atr_stop_mult are exposed as constructor kwargs for
testability only, not for sweeping/optimization.
"""
from __future__ import annotations

import pandas as pd

_AUDUSD = "AUDUSD"
_USDJPY = "USDJPY"


class FxTurnOfMonth:
    def __init__(self, universe, forecast_scale: float = 10.0, atr_window: int = 14,
                 atr_stop_mult: float = 1.0, **params):
        self.universe = list(universe)
        self.forecast_scale = float(forecast_scale)
        self.atr_window = int(atr_window)
        self.atr_stop_mult = float(atr_stop_mult)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        cols = list(close_panel.columns)
        out = pd.DataFrame(0.0, index=close_panel.index, columns=cols)
        if _AUDUSD not in cols or _USDJPY not in cols:
            return out.fillna(0.0)

        audusd = close_panel[_AUDUSD].astype(float)
        usdjpy = close_panel[_USDJPY].astype(float)
        basket_atr = self._basket_atr_proxy(audusd, usdjpy)

        n = len(close_panel.index)
        dt_index = pd.DatetimeIndex(close_panel.index)
        by_month = pd.Series(range(n), index=dt_index).groupby([dt_index.year, dt_index.month])

        audusd_col, usdjpy_col = out.columns.get_loc(_AUDUSD), out.columns.get_loc(_USDJPY)
        for _, positions in by_month:
            t_pos = int(positions.iloc[0])  # T = first trading day of this month (by index position)
            # Window entry is T-1; if T is the very first row in the whole panel
            # there is no prior day to enter on, so the window starts at T itself
            # (documented edge case -- affects at most the first month in the data).
            start_pos = max(0, t_pos - 1)
            end_pos = min(n - 1, t_pos + 3)
            self._fill_window(out, audusd, usdjpy, basket_atr, start_pos, end_pos,
                               audusd_col, usdjpy_col)
        return out.fillna(0.0)

    def _basket_atr_proxy(self, audusd: pd.Series, usdjpy: pd.Series) -> pd.Series:
        # Close-only proxy: forecast_panel never sees OHLC, so a true ATR (using
        # intraday high/low) is not computable here. Trailing mean absolute
        # close-to-close move per leg, summed across the two legs (the position
        # is a spread, so leg risk is additive), stands in for basket range.
        audusd_range = audusd.pct_change().abs().rolling(self.atr_window).mean()
        usdjpy_range = usdjpy.pct_change().abs().rolling(self.atr_window).mean()
        return audusd_range + usdjpy_range

    def _fill_window(self, out: pd.DataFrame, audusd: pd.Series, usdjpy: pd.Series,
                      basket_atr: pd.Series, start_pos: int, end_pos: int,
                      audusd_col: int, usdjpy_col: int) -> None:
        atr_entry = basket_atr.iloc[start_pos]
        audusd_entry = audusd.iloc[start_pos]
        usdjpy_entry = usdjpy.iloc[start_pos]
        stopped = False
        for pos in range(start_pos, end_pos + 1):
            if not stopped:
                # Long AUDUSD / short USDJPY: profit proxy = leg gain minus leg
                # loss since entry; adverse move is the shortfall below breakeven.
                pnl_proxy = (audusd.iloc[pos] / audusd_entry - 1.0) - (usdjpy.iloc[pos] / usdjpy_entry - 1.0)
                adverse = max(0.0, -pnl_proxy)
                if pd.notna(atr_entry) and adverse > self.atr_stop_mult * atr_entry:
                    stopped = True
            if not stopped:
                out.iloc[pos, audusd_col] = self.forecast_scale
                out.iloc[pos, usdjpy_col] = -self.forecast_scale
