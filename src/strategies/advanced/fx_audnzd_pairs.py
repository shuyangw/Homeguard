"""AudNzdPairs (#35): AUDUSD/NZDUSD cointegration-residual pairs spread.

Weekly, causal. At each rebalance date a trailing ``lookback``-day OLS of
``ln(AUDUSD)`` on ``ln(NZDUSD)`` (data up to that date only) yields the hedge
ratio ``beta`` and a residual series; the residual z-score at that date drives a
mean-reversion state machine. Enter when ``|z| > entry_z`` (signed against the
divergence, ``-sign(z)``: a positive z means AUD is rich, so short the spread);
hold until ``|z| < target_z`` (reverted), ``|z| > stop_z`` (blown out), or
``max_days`` held; then flat. New entries inside 7 calendar days of an RBA or
RBNZ decision are skipped. Active rebalance dates emit a single
``Spread("AUDUSD", "NZDUSD", beta, strength)`` with the trailing spread vol from
the base class. All computation uses only rows up to the current date.
"""
from __future__ import annotations

import numpy as np

from src.backtesting.engine.spread_sizing import Spread
from src.data.macro_calendar import load_cb_decisions
from src.strategies.advanced.fx_spread_base import SpreadStrategy

_LEG_A = "AUDUSD"
_LEG_B = "NZDUSD"
_STRENGTH_CAP = 20.0


class AudNzdPairs(SpreadStrategy):
    def __init__(self, lookback=120, entry_z=2.0, target_z=0.5, stop_z=3.25, max_days=20):
        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.target_z = float(target_z)
        self.stop_z = float(stop_z)
        self.max_days = int(max_days)

    def _blackout_dates(self):
        cb = load_cb_decisions()
        return list(cb.get("RBA", [])) + list(cb.get("RBNZ", []))

    @staticmethod
    def _is_rebalance(d, prev_d) -> bool:
        if prev_d is None:
            return True
        return d.isocalendar()[1] != prev_d.isocalendar()[1]

    def _regression_z(self, ln_a, ln_b, i):
        lo = i - self.lookback + 1
        if lo < 0:
            return None
        y, x = ln_a[lo:i + 1], ln_b[lo:i + 1]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
            return None
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        sd = resid.std()
        if not np.isfinite(sd) or sd <= 0:
            return None
        z = (resid[-1] - resid.mean()) / sd
        return (float(slope), float(z)) if np.isfinite(z) else None

    def _in_blackout(self, d, blackout_dates) -> bool:
        return any(abs((d - dd).days) <= 7 for dd in blackout_dates)

    def _strength(self, z: float) -> float:
        scaled = -z * (10.0 / self.entry_z)
        return float(np.clip(scaled, -_STRENGTH_CAP, _STRENGTH_CAP))

    def spread_book(self, close_panel):
        close_panel = close_panel.sort_index()
        dates = list(close_panel.index)
        ln_a = np.log(close_panel[_LEG_A].astype(float).values)
        ln_b = np.log(close_panel[_LEG_B].astype(float).values)
        blackout_dates = self._blackout_dates()

        book, sigma = {}, {}
        position, entry_idx = 0, None
        prev_d = None

        for i, d in enumerate(dates):
            is_reb = self._is_rebalance(d, prev_d)
            prev_d = d
            if not is_reb:
                continue

            reg = self._regression_z(ln_a, ln_b, i)
            if reg is None:
                continue
            beta, z = reg
            abs_z = abs(z)

            if position != 0:
                held = i - entry_idx
                if abs_z < self.target_z or abs_z > self.stop_z or held >= self.max_days:
                    position, entry_idx = 0, None

            if position == 0 and abs_z > self.entry_z and not self._in_blackout(d, blackout_dates):
                position, entry_idx = -1 if z > 0 else 1, i

            if position == 0:
                continue

            sig = self._spread_sigma(close_panel, _LEG_A, _LEG_B, beta, i)
            if sig is None:
                continue
            book[d] = [Spread(_LEG_A, _LEG_B, beta, self._strength(z))]
            sigma[d] = {(_LEG_A, _LEG_B): sig}

        return book, sigma
