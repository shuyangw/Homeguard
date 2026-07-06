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
                        revert_frac: float = _REVERT_FRAC) -> tuple[pd.Series, list]:
    r = np.log(close.astype(float)).diff()
    flagged: list = []
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
