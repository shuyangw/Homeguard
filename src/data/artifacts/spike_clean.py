"""Spike-and-revert data-cleaning for daily FX closes.

Thin-minute-data bad-close prints inject a large single-day move that fully
reverses the next day (e.g. USDCAD 2024-12-20 close 1.098 vs ~1.44 neighbors).
Real regime breaks (SNB 2015-01-15 EUR/CHF) do NOT revert. This module flags
the reverting spikes -- a large move whose next-day return cancels most of it --
and nulls the offending close so no fake tail reaches strategy returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ABS_FLOOR = 0.10       # min |daily log-return| to consider (physical implausibility)
_REVERT_FRAC = 0.4      # flag if |r_t + r_{t+1}| < this * |r_t| (>60% reverses)


def scrub_spike_reverts(close: pd.Series, abs_floor: float = _ABS_FLOOR,
                        revert_frac: float = _REVERT_FRAC,
                        causal: bool = False) -> tuple[pd.Series, list]:
    """Flag and null spike-and-revert bad closes.

    `causal=False` (default) uses the REVERT test, which reads `r[t+1]` to decide
    whether to null bar `t`. That is forward-looking: a live feed cannot know at
    time t that t's print was bad. It is retained as the default DELIBERATELY,
    because it is what makes the cleaner precise -- it nulls only moves that
    REVERSE, and therefore PRESERVES real regime breaks (SNB 2015, the TRY and
    ZAR crisis moves). A same-day-only rule cannot distinguish "bad tick" from
    "real 15% devaluation" and would delete genuine tail risk, which is a far
    worse distortion (it flatters every backtest's drawdowns) than the ~0.035%
    of bars this look-ahead touches.

    `causal=True` uses only information available at t: a move is flagged when
    it exceeds `abs_floor` AND is a large multiple of the trailing volatility.
    Honest for live-replay/paper-trading parity, but it WILL null real crisis
    bars -- do not use it to produce risk statistics.

    Returns (cleaned_series, flagged_index_labels).
    """
    r = np.log(close.astype(float)).diff()
    flagged: list = []
    if causal:
        trailing_sd = r.rolling(60, min_periods=20).std().shift(1)
        for i in range(1, len(close)):
            rt, sd = r.iloc[i], trailing_sd.iloc[i]
            if pd.isna(rt) or pd.isna(sd) or sd <= 0:
                continue
            if abs(rt) > abs_floor and abs(rt) > 8.0 * sd:
                flagged.append(close.index[i])
    else:
        for i in range(1, len(close) - 1):
            rt = r.iloc[i]
            rnext = r.iloc[i + 1]
            if pd.isna(rt) or pd.isna(rnext):
                continue
            if abs(rt) > abs_floor and abs(rt + rnext) < revert_frac * abs(rt):
                flagged.append(close.index[i])
    cleaned = close.copy()
    if flagged:
        cleaned.loc[flagged] = np.nan
    return cleaned, flagged
