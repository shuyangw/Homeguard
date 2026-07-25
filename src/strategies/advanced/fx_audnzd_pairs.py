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
from src.backtesting.signals.kalman_beta import causal_dynamic_beta, select_warmup_indices
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

            exited = False
            if position != 0:
                held = i - entry_idx
                if abs_z < self.target_z or abs_z > self.stop_z or held >= self.max_days:
                    position, entry_idx, exited = 0, None, True

            if position == 0 and not exited and abs_z > self.entry_z and not self._in_blackout(d, blackout_dates):
                position, entry_idx = -1 if z > 0 else 1, i

            if position == 0:
                continue

            sig = self._spread_sigma(close_panel, _LEG_A, _LEG_B, beta, i)
            if sig is None:
                continue
            book[d] = [Spread(_LEG_A, _LEG_B, beta, self._strength(z))]
            sigma[d] = {(_LEG_A, _LEG_B): sig}

        return book, sigma


class AudNzdPairsKalman(AudNzdPairs):
    """ARM B of the Kalman hedge-ratio scoping diagnostic (pre-registration:
    docs/strategies/research/20260722_fx_kalman_hedge_ratio_preregistration.md).

    Identical to AudNzdPairs in every respect -- universe, weekly rebalance, entry/
    target/stop z, max_days, RBA/RBNZ blackout, strength scaling, spread sigma --
    EXCEPT that the hedge ratio comes from a causal Kalman filter (time-varying
    beta) instead of a trailing-window OLS. The z-score is then built the SAME way
    as Arm A (residual over the same trailing window, standardized), so any
    difference in results is attributable to the estimator alone.

    Q and R are FIXED per the pre-registration and must not be tuned: Q from
    delta=1e-4, R from the OLS residual variance over the first `lookback`
    FINITE, aligned (AUDUSD, NZDUSD) rows of the supplied panel (the start of
    the walk-forward TRAINING window). Selecting by finiteness rather than a
    literal `[:lookback]` slice makes the R seed robust to a panel that opens
    before real price history begins, or to a handful of data-quality-nulled
    bars landing in the seed window -- without this, a single non-finite row
    poisons the seed to NaN and silently zeroes out the entire window's filter
    (see `docs/strategies/research/20260722_fx_kalman_hedge_ratio_results.md`
    Section 3a for the windows 1/13 defect this fixes).
    """

    delta = 1e-4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._beta_cache: dict[int, np.ndarray] = {}

    def _beta_path(self, ln_a, ln_b):
        key = (len(ln_a), float(ln_a[0]), float(ln_a[-1]))
        cached = self._beta_cache.get(key)
        if cached is not None:
            return cached
        seed_idx = select_warmup_indices(ln_a, ln_b, self.lookback)
        r_var = np.nan
        if seed_idx is not None:
            y0, x0 = ln_a[seed_idx], ln_b[seed_idx]
            s0, i0 = np.polyfit(x0, y0, 1)
            r_var = float(np.var(y0 - (s0 * x0 + i0)))
        paths = causal_dynamic_beta(ln_a, ln_b, self.delta, r_var, self.lookback)
        self._beta_cache[key] = paths
        return paths

    def _regression_z(self, ln_a, ln_b, i):
        lo = i - self.lookback + 1
        if lo < 0:
            return None
        y, x = ln_a[lo:i + 1], ln_b[lo:i + 1]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
            return None
        alpha_path, beta_path = self._beta_path(ln_a, ln_b)
        slope, intercept = beta_path[i], alpha_path[i]
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return None
        resid = y - (slope * x + intercept)
        sd = resid.std()
        if not np.isfinite(sd) or sd <= 0:
            return None
        z = (resid[-1] - resid.mean()) / sd
        return (float(slope), float(z)) if np.isfinite(z) else None
