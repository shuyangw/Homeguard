"""M2 -- the mark convention (handoff spec v2 Section 4 M2, amended by A3).

Registered rules, verbatim in effect:

  * All EOD-level marks are the derived 15:45 snapshot record.
  * PROHIBITED: trade-print-derived option OHLC bars as marks for out-of-the-
    money or low-volume strikes, and raw session "close" prints.
  * Marking uses `mid` from VALID quotes (crossed / zero-bid rows excluded per
    V3, FLAGGED not dropped silently).
  * On low-delta strikes the smoothed surface is the CROSS-CHECK -- a mid that
    diverges materially from the surface is FLAGGED, not silently used.

Amendment A3 (binding) adds: for OPT-047 and OPT-030 the surface may NOT be a
mark source at all. Those candidates are returned to this module's general rule
(raw quote mid). Where a valid quote is absent the position is NOT marked and
the session is REPORTED as unmarked -- never filled from the surface.

--------------------------------------------------------------------------
Three traps this module exists to close
--------------------------------------------------------------------------
1. `close` / `vwap` / `open` / `high` / `low` are TRADE PRINTS. An option can
   print once a day, anywhere on or off the spread, or not at all. They are
   never a mark. `assert_not_trade_print` makes that a loud failure.
2. The 09:30 bar carries `bid == ask == 0` universally. That is a structural
   artifact of the session open, not a genuine zero bid, and it gets its own
   reject reason so census counts stay interpretable.
3. Greek columns are NaN-valued float64, NOT sql NULL: arrow reports
   `null_count == 0` for a 100%-NaN column. Every usability test here is a
   VALUE test (`np.isfinite`). This trap has defeated four analyses.

NO imputation, forward-fill, interpolation or smoothing anywhere. Rejected rows
keep their place in the frame with `mark = NaN` and a reason.
"""
from __future__ import annotations

from datetime import time as _time
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Registered constants
# --------------------------------------------------------------------------

#: Trade-derived columns. Never a mark, for any strike, at any liquidity.
PROHIBITED_MARK_COLUMNS: Set[str] = {
    "open", "high", "low", "close", "vwap", "volume", "trade_count", "day_volume",
}

#: P1's low-delta threshold, reused here as the cross-check trigger.
LOW_DELTA_XCHECK_THRESHOLD = 0.10

#: Registered from D-047/030: the surface is judged against the market in VOL
#: POINTS, and 1.5 was the pre-registered "median divergence" bound.
SURFACE_DIVERGENCE_VOL_PTS = 1.5

#: A3 Section 3.1 -- these two candidates mark off raw quote mid, never the surface.
A3_RAW_MARK_CANDIDATES: Set[str] = {"OPT-047", "OPT-030"}

#: The universally-zero quote bar at the session open.
OPEN_BAR_ET: _time = _time(9, 30)

REJECT_ZERO_BID = "zero_bid"
REJECT_CROSSED = "crossed"
REJECT_NONFINITE = "nonfinite_quote"
REJECT_ZERO_QUOTE_OPEN_BAR = "zero_quote_open_bar"

# Registered usability bounds for the NaN-valued greek columns (promoted from
# the Wave-0 diagnostic layer; there is exactly one implementation).
IV_MIN = 1e-6
IV_MAX = 5.0
ABS_DELTA_MIN = 1e-6
ABS_DELTA_MAX = 1.0  # EXCLUSIVE: |delta| == 1 is a saturated/degenerate mark

__all__ = [
    "A3_RAW_MARK_CANDIDATES",
    "ABS_DELTA_MAX",
    "ABS_DELTA_MIN",
    "IV_MAX",
    "IV_MIN",
    "LOW_DELTA_XCHECK_THRESHOLD",
    "OPEN_BAR_ET",
    "PROHIBITED_MARK_COLUMNS",
    "REJECT_CROSSED",
    "REJECT_NONFINITE",
    "REJECT_ZERO_BID",
    "REJECT_ZERO_QUOTE_OPEN_BAR",
    "SURFACE_DIVERGENCE_VOL_PTS",
    "MarkSourceError",
    "ProhibitedMarkError",
    "assert_mark_source_allowed",
    "assert_not_trade_print",
    "greek_usable_mask",
    "mark_chain",
    "unmarked_census",
]


class ProhibitedMarkError(ValueError):
    """Raised when a trade-print column is offered as a mark."""


class MarkSourceError(ValueError):
    """Raised when a candidate's registered mark source is violated."""


def assert_not_trade_print(column: str) -> None:
    """Fail loud if `column` is trade-derived. Call this at every mark site."""
    if column in PROHIBITED_MARK_COLUMNS:
        raise ProhibitedMarkError(
            f"[-] {column!r} is a TRADE PRINT and is prohibited as a mark (M2). "
            f"The only sanctioned mark is `mid` from a valid quote."
        )


def assert_mark_source_allowed(candidate_id: str, source: str) -> None:
    """Enforce A3 Section 3.1 for OPT-047 / OPT-030.

    `source` is the intended mark source, e.g. 'quote_mid' or 'iv_smooth'.
    """
    assert_not_trade_print(source)
    if candidate_id in A3_RAW_MARK_CANDIDATES and source != "quote_mid":
        raise MarkSourceError(
            f"[-] amendment A3 binds {candidate_id}: marks come from raw quote "
            f"mid only, never {source!r}. Where a valid quote is absent the "
            f"position is left UNMARKED and the session is reported."
        )


def greek_usable_mask(df: pd.DataFrame, require_delta: bool = True) -> pd.Series:
    """Row mask of contracts whose greeks are actually USABLE.

    Tests VALUES (`np.isfinite`), never `null_count` / `is_null` / column
    presence: a 100%-NaN float64 column reports `null_count == 0` in arrow and
    would pass any null-based check. Also rejects degenerate marks (iv <= 0,
    iv > 5.0, |delta| == 0 or 1) and rows whose quote was not valid.
    """
    iv = pd.to_numeric(df.get("implied_vol"), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(iv) & (iv > IV_MIN) & (iv <= IV_MAX)
    if require_delta:
        d = pd.to_numeric(df.get("delta"), errors="coerce").to_numpy(dtype=float)
        ad = np.abs(d)
        mask &= np.isfinite(d) & (ad > ABS_DELTA_MIN) & (ad < ABS_DELTA_MAX)
    if "quote_valid" in df.columns:
        qv = df["quote_valid"].fillna(False).to_numpy(dtype=bool)
        mask &= qv
    return pd.Series(mask, index=df.index, name="greek_usable")


def _reject_reason(bid: float, ask: float, bar_time: Optional[_time]) -> Optional[str]:
    if not (np.isfinite(bid) and np.isfinite(ask)):
        return REJECT_NONFINITE
    if bid == 0.0 and ask == 0.0:
        if bar_time is not None and bar_time == OPEN_BAR_ET:
            return REJECT_ZERO_QUOTE_OPEN_BAR
        return REJECT_ZERO_BID
    if ask < bid:
        return REJECT_CROSSED
    if bid <= 0.0:
        return REJECT_ZERO_BID
    return None


def mark_chain(
    chain: pd.DataFrame,
    bid_col: str = "bid",
    ask_col: str = "ask",
    delta_col: str = "delta",
    iv_col: str = "implied_vol",
    iv_smooth_col: str = "iv_smooth",
    bar_time_col: str = "bar_time_et",
) -> pd.DataFrame:
    """Attach M2 marks to an option chain frame.

    Adds, without dropping or reordering a single row:
        mark                      (bid + ask) / 2 on valid quotes, else NaN
        mark_valid                bool
        mark_reject_reason        one of the REJECT_* codes, else None
        surface_divergence_vol_pts  |iv_shipped - iv_smooth| * 100, low delta only
        surface_divergence_flag   True where that exceeds the registered bound

    The surface is a CROSS-CHECK. A flagged row keeps its quote mid; the
    surface never replaces it.
    """
    out = chain.copy()
    bid = pd.to_numeric(out.get(bid_col), errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(out.get(ask_col), errors="coerce").to_numpy(dtype=float)
    if bar_time_col in out.columns:
        times = list(out[bar_time_col])
    else:
        times = [None] * len(out)

    reasons = [_reject_reason(b, a, t) for b, a, t in zip(bid, ask, times)]
    valid = np.array([r is None for r in reasons], dtype=bool)

    mark = np.full(len(out), np.nan)
    mark[valid] = (bid[valid] + ask[valid]) / 2.0

    out["mark"] = mark
    out["mark_valid"] = valid
    out["mark_reject_reason"] = reasons

    div = np.full(len(out), np.nan)
    flag = np.zeros(len(out), dtype=bool)
    if iv_smooth_col in out.columns and delta_col in out.columns:
        d = pd.to_numeric(out[delta_col], errors="coerce").to_numpy(dtype=float)
        iv = pd.to_numeric(out.get(iv_col), errors="coerce").to_numpy(dtype=float)
        sm = pd.to_numeric(out[iv_smooth_col], errors="coerce").to_numpy(dtype=float)
        low = np.isfinite(d) & (np.abs(d) < LOW_DELTA_XCHECK_THRESHOLD)
        both = low & np.isfinite(iv) & np.isfinite(sm)
        div[both] = np.abs(iv[both] - sm[both]) * 100.0
        flag = np.isfinite(div) & (div > SURFACE_DIVERGENCE_VOL_PTS)

    out["surface_divergence_vol_pts"] = div
    out["surface_divergence_flag"] = flag

    n_bad = int((~valid).sum())
    if n_bad:
        logger.debug(
            f"[!] {n_bad:,} of {len(out):,} rows unmarked (flagged, not dropped)"
        )
    return out


def unmarked_census(marked: pd.DataFrame) -> pd.DataFrame:
    """Count of unmarked rows by reason -- the report A3 requires for 047/030."""
    bad = marked[~marked["mark_valid"].to_numpy(dtype=bool)]
    if bad.empty:
        return pd.DataFrame(columns=["mark_reject_reason", "n"])
    return (
        bad.groupby("mark_reject_reason", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
