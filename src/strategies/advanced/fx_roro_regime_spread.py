"""FxRoroRegimeSpread: macro risk-on/risk-off regime spread (#42).

A composite RORO (risk-on/risk-off) score built from AUDJPY, CHFJPY, and
XAUUSD 5-day-return z-scores drives an event-driven entry into a
beta-weighted AUDJPY/CHFJPY spread: long AUDJPY / short CHFJPY on a
risk-on ignition, mirrored short AUDJPY / long CHFJPY on a risk-off
ignition. XAUUSD is a score-only input -- it is never traded (its
forecast column is always exactly 0.0).

Composite score (daily, causal)::

    ret5 = close.pct_change(5, fill_method=None)
    z(x) = (ret5 - ret5.rolling(zscore_window).mean()) / ret5.rolling(zscore_window).std(ddof=0)
    score = z(AUDJPY) - z(CHFJPY) + z(-1 * XAUUSD ret5)

``z(-1 * XAUUSD ret5)`` is used as ``-1 * z(XAUUSD ret5)`` directly: negating
a series negates both the numerator (``x - mean(x)``, since ``mean(-x) =
-mean(x)``) while leaving the rolling standard deviation unchanged
(``std(-x) = std(x)``), so ``z(-x) = -z(x)`` exactly. This is simpler and
numerically identical to z-scoring a separately negated series.

Entry (event-driven; fires only on the crossing day, comparing
``score[t-1]`` against ``score[t]``):

- **Risk-on ignition**: ``score[t-1] <= entry_threshold and score[t] >
  entry_threshold`` -> long AUDJPY / short CHFJPY, beta-weighted.
- **Risk-off mirror**: ``score[t-1] >= -entry_threshold and score[t] <
  -entry_threshold`` -> short AUDJPY / long CHFJPY.

An opposite-direction entry signal while already in a position is a normal
flip (exit then enter, same day) -- this falls out of the exit-then-entry
evaluation order below without special-casing, because crossing past
``-entry_threshold`` (or ``+entry_threshold``) always also satisfies the
score-recross-0 exit condition for the currently-held position. A
same-direction re-entry signal while already in that same position (e.g.
score dips into the [0, entry_threshold] dead zone without exiting, then
re-crosses above entry_threshold) is treated as a no-op: the existing trade
(and its fixed-at-entry beta/ATR) continues unchanged. This is a documented
design choice, not explicitly pinned down in the pre-registration.

Sizing ("beta-weighted"): the AUDJPY leg forecast is
``+-forecast_scale``. The CHFJPY leg forecast is
``-+forecast_scale * beta``, where ``beta`` is the trailing
``beta_window``-day OLS beta of CHFJPY daily returns regressed on AUDJPY
daily returns (``Cov(CHFJPY_ret, AUDJPY_ret) / Var(AUDJPY_ret)``),
estimated ONCE at entry (causal, using data through the entry day) and
held fixed for the life of the trade. ``beta`` is clipped to
``beta_clip`` as a fixed a-priori sanity guard against pathological
regression estimates (not a tunable).

Exit (checked daily, causal; first of the following three to fire wins --
this priority order is a documented design choice for the (rare) case
multiple conditions are true on the same day):

1. **signal_reversal**: the score recrosses 0 (``score <= 0`` for a
   risk-on position, ``score >= 0`` for a risk-off position).
2. **time_fixed**: exactly ``max_hold_days`` trading days after entry.
3. **vol_scaled_stop**: the AUDJPY/CHFJPY log-spread
   (``log(AUDJPY_close) - log(CHFJPY_close)``) has moved against the
   position by more than ``atr_stop_mult * atr_proxy_at_entry`` from its
   entry-day level, where ``atr_proxy`` is the rolling ``atr_window``-day
   mean of ``abs(spread.diff())``, evaluated at ENTRY and held fixed for
   the trade (same causal-at-entry convention as beta).

On the exit day both legs return to forecast 0 and stay flat until the
next entry signal fires (not necessarily the opposite-direction one).

Per backtesting methodology Section 11.1, this strategy combines
signal_reversal + time_fixed + vol_scaled_stop exits, all checked once per
day on close-only daily data (Section 11.2: time_fixed and
signal_reversal are valid on daily bars; the vol_scaled_stop here is a
close-only ATR proxy checked once per day at the close -- consistent with
this framework's close-only daily FX panel, but NOT validated at intrabar
resolution, so an intraday gap through and back from the stop level within
the same day is not distinguishable from one that stays through the
close. This is a documented limitation, not a bug.) ``forecast_panel``
returns forecasts only, not an exit-reason column; the state machine below
tracks which of the three exit types would fire (via the `if/elif`
ordering) for auditability, even though that reason is not currently
persisted to output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FxRoroRegimeSpread:
    def __init__(self, universe, zscore_window: int = 252, beta_window: int = 90,
                 entry_threshold: float = 1.0, max_hold_days: int = 20,
                 atr_window: int = 14, atr_stop_mult: float = 2.5,
                 beta_clip=(0.3, 3.0), forecast_scale: float = 10.0, **params):
        self.universe = list(universe)
        self.zscore_window = int(zscore_window)
        self.beta_window = int(beta_window)
        self.entry_threshold = float(entry_threshold)
        self.max_hold_days = int(max_hold_days)
        self.atr_window = int(atr_window)
        self.atr_stop_mult = float(atr_stop_mult)
        self.beta_clip = (float(beta_clip[0]), float(beta_clip[1]))
        self.forecast_scale = float(forecast_scale)

    def _ret5_zscore(self, close: pd.Series) -> pd.Series:
        ret5 = close.pct_change(5, fill_method=None)
        mean = ret5.rolling(self.zscore_window).mean()
        std = ret5.rolling(self.zscore_window).std(ddof=0)
        return (ret5 - mean) / std

    def _composite_score(self, close_panel: pd.DataFrame) -> pd.Series:
        z_audjpy = self._ret5_zscore(close_panel["AUDJPY"].astype(float))
        z_chfjpy = self._ret5_zscore(close_panel["CHFJPY"].astype(float))
        z_xauusd = self._ret5_zscore(close_panel["XAUUSD"].astype(float))
        return z_audjpy - z_chfjpy - z_xauusd

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in close_panel.columns if c in self.universe]
        index = close_panel.index

        if "AUDJPY" not in cols or "CHFJPY" not in cols or "XAUUSD" not in cols:
            return pd.DataFrame(0.0, index=index, columns=cols).fillna(0.0)

        audjpy = close_panel["AUDJPY"].astype(float)
        chfjpy = close_panel["CHFJPY"].astype(float)
        score = self._composite_score(close_panel)

        audjpy_ret = audjpy.pct_change(fill_method=None)
        chfjpy_ret = chfjpy.pct_change(fill_method=None)
        cov = chfjpy_ret.rolling(self.beta_window).cov(audjpy_ret)
        var = audjpy_ret.rolling(self.beta_window).var()
        beta_series = (cov / var).clip(lower=self.beta_clip[0], upper=self.beta_clip[1])

        log_spread = np.log(audjpy) - np.log(chfjpy)
        atr_proxy = log_spread.diff().abs().rolling(self.atr_window).mean()

        audjpy_fc, chfjpy_fc = self._run_state_machine(
            score.values, log_spread.values, beta_series.values, atr_proxy.values)

        out = {c: pd.Series(0.0, index=index) for c in cols}
        out["AUDJPY"] = pd.Series(audjpy_fc, index=index)
        out["CHFJPY"] = pd.Series(chfjpy_fc, index=index)
        return pd.DataFrame(out)[cols].fillna(0.0)

    def _run_state_machine(self, score_vals, spread_vals, beta_vals, atr_vals):
        n = len(score_vals)
        audjpy_fc = np.zeros(n)
        chfjpy_fc = np.zeros(n)

        position = None  # None, "risk_on", "risk_off"
        entry_i = None
        beta_at_entry = None
        entry_spread = None
        atr_at_entry = None

        for i in range(1, n):
            prev_score = score_vals[i - 1]
            cur_score = score_vals[i]

            if position is not None:
                exit_now = self._check_exit(
                    position, cur_score, i, entry_i, spread_vals[i],
                    entry_spread, atr_at_entry)
                if exit_now:
                    position = entry_i = beta_at_entry = entry_spread = atr_at_entry = None

            if position is None and not np.isnan(prev_score) and not np.isnan(cur_score):
                new_position = self._check_entry(prev_score, cur_score)
                if new_position is not None and not np.isnan(beta_vals[i]):
                    position = new_position
                    entry_i = i
                    beta_at_entry = beta_vals[i]
                    entry_spread = spread_vals[i]
                    atr_at_entry = atr_vals[i]

            if position == "risk_on":
                audjpy_fc[i] = self.forecast_scale
                chfjpy_fc[i] = -self.forecast_scale * beta_at_entry
            elif position == "risk_off":
                audjpy_fc[i] = -self.forecast_scale
                chfjpy_fc[i] = self.forecast_scale * beta_at_entry

        return audjpy_fc, chfjpy_fc

    def _check_entry(self, prev_score: float, cur_score: float):
        if prev_score <= self.entry_threshold and cur_score > self.entry_threshold:
            return "risk_on"
        if prev_score >= -self.entry_threshold and cur_score < -self.entry_threshold:
            return "risk_off"
        return None

    def _check_exit(self, position: str, cur_score: float, i: int, entry_i: int,
                     cur_spread: float, entry_spread: float, atr_at_entry: float) -> bool:
        if not np.isnan(cur_score):
            if position == "risk_on" and cur_score <= 0.0:
                return True
            if position == "risk_off" and cur_score >= 0.0:
                return True

        if (i - entry_i) >= self.max_hold_days:
            return True

        if atr_at_entry is not None and not np.isnan(atr_at_entry) and not np.isnan(cur_spread):
            stop_dist = self.atr_stop_mult * atr_at_entry
            if position == "risk_on" and cur_spread <= entry_spread - stop_dist:
                return True
            if position == "risk_off" and cur_spread >= entry_spread + stop_dist:
                return True

        return False
