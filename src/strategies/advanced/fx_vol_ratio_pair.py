"""VolRatioPair (#30): simplified symmetric vol-ratio reversion spread.

Weekly, causal. For each coupled set ``(leg_a, leg_b)`` the trailing realized
vol of each leg (std of daily returns over ``rv_window`` days) is measured at the
current date; ``r = ln(RV_a / RV_b)`` is z-scored against its own trailing
``z_window`` (~2yr) distribution. When ``|z| > entry_z`` the vol ratio is bet to
revert -- LONG the low-vol leg, SHORT the high-vol leg. With the
``Spread(leg_a, leg_b, hedge_ratio, strength)`` convention (positive strength =
long A / short B), a hot leg A (RV_a > RV_b, z > 0) yields NEGATIVE strength
(short A / long B), mirroring #35's ``-z * (10 / entry_z)`` scaling clipped to
+-20. The position is held until ``|z| < exit_z``. The hedge ratio comes from a
trailing OLS of ``ln(A)`` on ``ln(B)`` (1.0 if degenerate). Active rebalance
dates emit one ``Spread`` per set with the trailing spread vol from the base
class. All computation uses only rows up to the current date.
"""
from __future__ import annotations

import numpy as np

from src.backtesting.engine.spread_sizing import Spread
from src.strategies.advanced.fx_spread_base import SpreadStrategy

_STRENGTH_CAP = 20.0


class VolRatioPair(SpreadStrategy):
    def __init__(self, coupled_sets=(("EURNOK", "EURSEK"), ("AUDUSD", "NZDUSD"),
                                     ("XAUUSD", "XAGUSD")),
                 rv_window=10, z_window=504, entry_z=2.0, exit_z=1.0):
        self.coupled_sets = tuple((a, b) for a, b in coupled_sets)
        self.rv_window = int(rv_window)
        self.z_window = int(z_window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)

    @staticmethod
    def _is_rebalance(d, prev_d) -> bool:
        if prev_d is None:
            return True
        return d.isocalendar()[1] != prev_d.isocalendar()[1]

    def _strength(self, z: float) -> float:
        scaled = -z * (10.0 / self.entry_z)
        return float(np.clip(scaled, -_STRENGTH_CAP, _STRENGTH_CAP))

    def _hedge_ratio(self, ln_a, ln_b, i) -> float:
        lo = max(0, i - self.z_window + 1)
        y, x = ln_a[lo:i + 1], ln_b[lo:i + 1]
        if len(x) < 2 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            return 1.0
        beta = np.polyfit(x, y, 1)[0]
        return float(beta) if np.isfinite(beta) else 1.0

    def _ratio_z(self, r, i):
        if not np.isfinite(r[i]):
            return None
        lo = max(0, i - self.z_window + 1)
        w = r[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) < 2:
            return None
        sd = w.std()
        if not np.isfinite(sd) or sd <= 0:
            return None
        z = (r[i] - w.mean()) / sd
        return float(z) if np.isfinite(z) else None

    def _set_series(self, close_panel):
        data = {}
        for a, b in self.coupled_sets:
            if a not in close_panel.columns or b not in close_panel.columns:
                continue
            ra = close_panel[a].astype(float).pct_change(fill_method=None)
            rb = close_panel[b].astype(float).pct_change(fill_method=None)
            rv_a = ra.rolling(self.rv_window).std().values
            rv_b = rb.rolling(self.rv_window).std().values
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(rv_a / rv_b)
            data[(a, b)] = {
                "r": r,
                "ln_a": np.log(close_panel[a].astype(float).values),
                "ln_b": np.log(close_panel[b].astype(float).values),
                "pos": 0,
            }
        return data

    def spread_book(self, close_panel):
        close_panel = close_panel.sort_index()
        dates = list(close_panel.index)
        data = self._set_series(close_panel)

        book, sigma = {}, {}
        prev_d = None

        for i, d in enumerate(dates):
            is_reb = self._is_rebalance(d, prev_d)
            prev_d = d
            if not is_reb:
                continue

            day_spreads, day_sigma = [], {}
            for (a, b), st in data.items():
                z = self._ratio_z(st["r"], i)
                if z is None:
                    continue
                abs_z = abs(z)

                if st["pos"] != 0 and abs_z < self.exit_z:
                    st["pos"] = 0
                if st["pos"] == 0 and abs_z > self.entry_z:
                    st["pos"] = -1 if z > 0 else 1
                if st["pos"] == 0:
                    continue

                beta = self._hedge_ratio(st["ln_a"], st["ln_b"], i)
                sig = self._spread_sigma(close_panel, a, b, beta, i)
                if sig is None:
                    continue
                day_spreads.append(Spread(a, b, beta, self._strength(z)))
                day_sigma[(a, b)] = sig

            if day_spreads:
                book[d] = day_spreads
                sigma[d] = day_sigma

        return book, sigma
