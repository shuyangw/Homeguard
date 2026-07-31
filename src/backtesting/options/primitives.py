"""P1-P10 -- the ten shared primitives (handoff spec v2 Section 3, amended A3).

Build once. Candidates reference these BY NAME; duplicated logic is a
correctness risk and a silent-divergence risk.

Nothing here computes P&L, holds a position, or produces a verdict. P3, P6 and
P7 describe DECISIONS and FILLS (the registered primitive surface); attribution
(M4), regime slicing (M5) and validation (M6) sit on top and are Phase 2b.

--------------------------------------------------------------------------
Deviations from the registered text, all reported in the Phase-2a doc
--------------------------------------------------------------------------
P2  the spec fixes the DTE window but never says which expiry inside it wins.
    Registered here: nearest to `target_dte` (default = window midpoint), ties
    to the SHORTER dte. Deterministic and logged, never a silent straddle.
P4  the "single shared vega budget constant" has NO registered value anywhere
    in the chain. `shared_vega_budget()` therefore RAISES rather than inventing
    one -- a silently substituted parameter is an extra trial.
P6  the "slippage schedule tiered by realized-vol percentile" likewise has no
    registered bps values. `cost_bps_schedule` is REQUIRED, never defaulted.
P9  enforces the 15:45 truncation, so its daily OHLC is NOT the full-session
    OHLC used by the Wave-0 `build_rv_daily`. The two differ by design.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date as _date
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.backtesting.options.marks import (
    PROHIBITED_MARK_COLUMNS,
    assert_not_trade_print,
    greek_usable_mask,
)
from src.backtesting.options.snapshot import (
    EASTERN_TZ,
    SNAPSHOT_TIME_ET,
    truncate_to_snapshot,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252

#: Columns that carry the same-session end-of-day value and are a hard leak at
#: the 15:45 snapshot (V6). Consume the `_lag1` forms instead.
UNLAGGED_EOD_COLUMNS = frozenset({"oi_eod", "gamma_eod", "open_interest_eod"})

#: P1's registered low-delta threshold. STRICTLY less than -- 0.10 exactly does
#: not bind (A3 Section 3.3, recorded to remove ambiguity).
P1_LOW_DELTA_THRESHOLD = 0.10

DEFAULT_PROFIT_TAKE = 0.50
DEFAULT_DTE_EXIT = 21

#: P10's spec is FROZEN. No order selection, no refit-schedule tuning.
HAR_SPEC_FROZEN: Tuple[int, int, int] = (1, 5, 22)

MIN_WINDOW_COVERAGE = 0.95

REGIME_STATES = frozenset(
    {"STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "UNPREDICTABLE", "BEAR"}
)


class LeakageError(RuntimeError):
    """Raised when a computation would consult data it cannot have known."""


class UnregisteredParameterError(RuntimeError):
    """Raised when a registered-but-unvalued parameter is requested.

    Substituting a nearby value would be an unlogged trial. Fail loud instead.
    """


# ==========================================================================
# Backward-looking window primitives (shared by P8; promoted from Wave 0)
# ==========================================================================


def trailing_percentile(
    s: pd.Series, window: int, min_coverage: float = MIN_WINDOW_COVERAGE
) -> pd.Series:
    """Fraction of the `window` STRICTLY-PRIOR observations below today's value.

    The current observation is EXCLUDED from its own comparison window and the
    window spans exactly `window` prior positions, so nothing at t is a function
    of any observation at t+1 or later. Missing sessions are never imputed --
    the comparison uses the finite observations actually present, provided they
    cover at least `min_coverage` of the window.
    """
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    n = len(v)
    need = max(1, int(np.ceil(min_coverage * window)))
    out = np.full(n, np.nan)
    for i in range(window, n):
        cur = v[i]
        if not np.isfinite(cur):
            continue
        w = v[i - window:i]
        w = w[np.isfinite(w)]
        if len(w) < need:
            continue
        out[i] = float((w < cur).sum()) / float(len(w))
    return pd.Series(out, index=s.index, name=f"pctile_{window}")


def trailing_rank(
    s: pd.Series, window: int, min_coverage: float = MIN_WINDOW_COVERAGE
) -> pd.Series:
    """(x - min) / (max - min) over the `window` STRICTLY-PRIOR observations."""
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    n = len(v)
    need = max(1, int(np.ceil(min_coverage * window)))
    out = np.full(n, np.nan)
    for i in range(window, n):
        cur = v[i]
        if not np.isfinite(cur):
            continue
        w = v[i - window:i]
        w = w[np.isfinite(w)]
        if len(w) < need:
            continue
        lo, hi = float(w.min()), float(w.max())
        if hi <= lo:
            continue
        out[i] = (cur - lo) / (hi - lo)
    return pd.Series(out, index=s.index, name=f"rank_{window}")


def assert_backward_looking(
    fn: Callable[[pd.Series], pd.Series],
    series: pd.Series,
    cuts: Optional[Sequence[int]] = None,
) -> None:
    """Falsifier for "strictly backward looking" (spec v2 Section 1.3 rule 4).

    Perturbs the FUTURE of the input and requires every output at or before the
    cut to be bit-identical. A full-sample percentile fails this immediately,
    which is the point: without this control the rank tests pass vacuously.
    """
    n = len(series)
    if cuts is None:
        cuts = [int(n * 0.75), int(n * 0.9)]
    base = fn(series)
    for cut in cuts:
        if cut <= 0 or cut >= n:
            continue
        tampered = series.copy()
        tampered.iloc[cut + 1:] = tampered.iloc[cut + 1:] * 50.0 + 1234.5
        after = fn(tampered)
        a = base.iloc[: cut + 1].to_numpy(dtype=float)
        b = after.iloc[: cut + 1].to_numpy(dtype=float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        finite = ~np.isnan(a) & ~np.isnan(b)
        if not same_nan or not np.allclose(a[finite], b[finite], equal_nan=False):
            raise LeakageError(
                f"[-] output at or before index {cut} changed when FUTURE input "
                f"was perturbed -- the computation is not backward looking."
            )


# ==========================================================================
# Monthly expiry helpers (promoted from Wave 0; one implementation only)
# ==========================================================================


def third_friday(year: int, month: int) -> _date:
    """Standard monthly expiry date before any holiday shift."""
    cal = calendar.Calendar()
    fridays = [
        d for d in cal.itermonthdates(year, month)
        if d.month == month and d.weekday() == 4
    ]
    return fridays[2]


def monthly_expiry(
    year: int, month: int, trading_days: Optional[Set[_date]] = None
) -> _date:
    """Third Friday, shifted BACK to the prior trading day when it is a holiday."""
    d = third_friday(year, month)
    if trading_days is None:
        return d
    for _ in range(7):
        if d in trading_days:
            return d
        d = d - pd.Timedelta(days=1).to_pytimedelta()
    return third_friday(year, month)


def monthly_expiry_set(
    start: _date, end: _date, trading_days: Optional[Set[_date]] = None
) -> Set[_date]:
    """All standard monthly expiries in [start, end].

    Both listing conventions are accepted: from Feb-2015 the standard monthly is
    dated the third FRIDAY; before that it was dated the SATURDAY following it.
    """
    out: Set[_date] = set()
    y, m = start.year, start.month
    one = pd.Timedelta(days=1).to_pytimedelta()
    while (y, m) <= (end.year, end.month):
        for d in (monthly_expiry(y, m, trading_days=trading_days),
                  third_friday(y, m) + one):
            if start <= d <= end:
                out.add(d)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# ==========================================================================
# P1 -- select_strike_by_delta
# ==========================================================================


class DeltaSource(str, Enum):
    """Which delta the SELECTION reads.

    AUTO is P1 as registered: the smoothed surface below |0.10|, shipped greeks
    at or above it. A3_SMOOTH_NONEXTRAP is amendment A3's rule and binds ONLY
    OPT-047 / OPT-030 -- A3 Section 4 explicitly declines a slate-wide change.
    """

    AUTO = "auto"
    SHIPPED = "shipped"
    SMOOTH = "smooth"
    A3_SMOOTH_NONEXTRAP = "a3_smooth_nonextrap"


@dataclass
class StrikeSelection:
    """P1's return value. Strategies log BOTH the strike and the realized delta,
    plus which source selected it -- the split is a reportable result (A3)."""

    strike: float
    expiry: _date
    right: str
    dte: int
    realized_delta: float
    target_delta: float
    delta_source: DeltaSource
    extrapolated_fallback: bool
    mark: float
    bid: float
    ask: float
    n_candidates: int
    n_excluded_unusable: int
    tie_break: str


def _resolve_delta_policy(target_delta: float, policy: DeltaSource) -> DeltaSource:
    if policy is not DeltaSource.AUTO:
        return policy
    if abs(target_delta) < P1_LOW_DELTA_THRESHOLD:
        return DeltaSource.SMOOTH
    return DeltaSource.SHIPPED


def _selection_deltas(
    df: pd.DataFrame, policy: DeltaSource
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (delta values, per-row 'used smooth' flags) for the policy."""
    shipped = pd.to_numeric(df.get("delta"), errors="coerce").to_numpy(dtype=float)
    if policy is DeltaSource.SHIPPED:
        return shipped, np.zeros(len(df), dtype=bool)

    smooth = pd.to_numeric(df.get("delta_smooth"), errors="coerce").to_numpy(dtype=float)
    if policy is DeltaSource.SMOOTH:
        return smooth, np.isfinite(smooth)

    # A3: the surface where it is NOT extrapolated, shipped (flagged) otherwise.
    if "extrapolated" in df.columns:
        extrap = (
            df["extrapolated"].astype("object").where(
                df["extrapolated"].notna(), True
            ).to_numpy(dtype=bool)
        )
    else:
        extrap = np.ones(len(df), dtype=bool)
    use_smooth = np.isfinite(smooth) & ~extrap
    vals = np.where(use_smooth, smooth, shipped)
    return vals, use_smooth


def select_strike_by_delta(
    chain: pd.DataFrame,
    right: str,
    target_delta: float,
    dte_window: Tuple[int, int],
    delta_source: DeltaSource = DeltaSource.AUTO,
    tie_break_col: Optional[str] = "oi_eod_lag1",
) -> Optional[StrikeSelection]:
    """P1 -- nearest available |delta| to target within the DTE window.

    Ties break to the more liquid strike, using LAGGED open interest only; the
    same-session `oi_eod` is a hard leak at the snapshot minute (V6) and is
    refused outright. Returns the strike PLUS the realized delta actually
    selected, and the source that selected it.
    """
    if tie_break_col in UNLAGGED_EOD_COLUMNS:
        raise LeakageError(
            f"[-] {tie_break_col!r} carries the SAME-SESSION end-of-day value and "
            f"is not known at the 15:45 snapshot (V6). Use 'oi_eod_lag1'."
        )
    if tie_break_col is not None:
        assert_not_trade_print(tie_break_col)

    lo, hi = int(dte_window[0]), int(dte_window[1])
    df = chain[(chain["right"] == right) & (chain["dte"] >= lo) & (chain["dte"] <= hi)]
    if df.empty:
        return None

    policy = _resolve_delta_policy(target_delta, delta_source)
    deltas, used_smooth = _selection_deltas(df, policy)

    quote_ok = (
        df["quote_valid"].fillna(False).to_numpy(dtype=bool)
        if "quote_valid" in df.columns
        else np.ones(len(df), dtype=bool)
    )
    usable = np.isfinite(deltas) & quote_ok
    n_excluded = int((~usable).sum())
    if not usable.any():
        return None

    cand = df[usable].reset_index(drop=True)
    cand_delta = deltas[usable]
    cand_smooth = used_smooth[usable]

    dist = np.abs(np.abs(cand_delta) - abs(target_delta))
    best = float(dist.min())
    at_best = np.flatnonzero(dist <= best + 1e-12)

    tie_break = "none"
    pick = int(at_best[0])
    if len(at_best) > 1:
        pick, tie_break = _break_tie(cand, at_best, tie_break_col)

    row = cand.iloc[pick]
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    mark = (bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) else float("nan")

    source = DeltaSource.SMOOTH if bool(cand_smooth[pick]) else DeltaSource.SHIPPED
    fallback = bool(
        policy is DeltaSource.A3_SMOOTH_NONEXTRAP and not cand_smooth[pick]
    )

    return StrikeSelection(
        strike=float(row["strike"]),
        expiry=row["expiry"],
        right=right,
        dte=int(row["dte"]),
        realized_delta=float(cand_delta[pick]),
        target_delta=float(target_delta),
        delta_source=source,
        extrapolated_fallback=fallback,
        mark=mark,
        bid=bid,
        ask=ask,
        n_candidates=int(len(cand)),
        n_excluded_unusable=n_excluded,
        tie_break=tie_break,
    )


def select_nearest_abs_delta(
    df: pd.DataFrame, target: float, tolerance: float
) -> Optional[pd.Series]:
    """Row whose |delta| is nearest `target`, or None beyond `tolerance`.

    The tolerance-bounded form used by the DESCRIPTIVE daily tables (skew,
    delta-IV). `select_strike_by_delta` is P1 proper -- it adds the DTE window,
    the smoothed-surface rule, the liquidity tie-break and the source log, and
    is what strategies call. This helper is kept because the descriptive tables
    genuinely need "nearest within tolerance, else nothing"; there is exactly
    one implementation of it and it lives here.
    """
    if df is None or len(df) == 0:
        return None
    d = pd.to_numeric(df["delta"], errors="coerce").to_numpy(dtype=float)
    dist = np.abs(np.abs(d) - target)
    ok = np.isfinite(dist)
    if not ok.any():
        return None
    dist = np.where(ok, dist, np.inf)
    i = int(np.argmin(dist))
    if dist[i] > tolerance:
        return None
    return df.iloc[i]


def _break_tie(
    cand: pd.DataFrame, at_best: np.ndarray, tie_break_col: Optional[str]
) -> Tuple[int, str]:
    """More liquid first: lagged OI desc, then tighter quoted spread, then the
    lower strike so the result is deterministic and never order-dependent."""
    sub = cand.iloc[at_best]
    if tie_break_col and tie_break_col in sub.columns:
        oi = pd.to_numeric(sub[tie_break_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(oi).any() and len(np.unique(oi[np.isfinite(oi)])) > 1:
            return int(at_best[int(np.nanargmax(oi))]), tie_break_col
    if "spread_abs" in sub.columns:
        sp = pd.to_numeric(sub["spread_abs"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(sp).any() and len(np.unique(sp[np.isfinite(sp)])) > 1:
            return int(at_best[int(np.nanargmin(sp))]), "spread_abs"
    strikes = sub["strike"].to_numpy(dtype=float)
    return int(at_best[int(np.argmin(strikes))]), "strike"


# ==========================================================================
# P2 -- select_expiry
# ==========================================================================


@dataclass
class ExpirySelection:
    expiry: _date
    dte: int
    n_candidates: int
    prefer: str
    target_dte: float


def select_expiry(
    chain: pd.DataFrame,
    dte_min: int,
    dte_max: int,
    prefer: str = "any",
    target_dte: Optional[float] = None,
) -> Optional[ExpirySelection]:
    """P2 -- pick ONE expiry inside the DTE window and log it.

    `prefer` in {monthly, any}. Monthly-only where the spec says OpEx.
    A distribution of expiries is never straddled silently: exactly one expiry
    comes back, with the candidate count that produced it.
    """
    if prefer not in ("monthly", "any"):
        raise ValueError(f"[-] prefer must be 'monthly' or 'any', got {prefer!r}")

    win = chain[(chain["dte"] >= int(dte_min)) & (chain["dte"] <= int(dte_max))]
    if win.empty:
        return None

    pairs = (
        win.groupby("expiry", sort=True)["dte"].first().reset_index()
    )
    if prefer == "monthly":
        monthlies = monthly_expiry_set(pairs["expiry"].min(), pairs["expiry"].max())
        pairs = pairs[pairs["expiry"].isin(monthlies)]
        if pairs.empty:
            return None

    tgt = float(target_dte) if target_dte is not None else (dte_min + dte_max) / 2.0
    d = pairs["dte"].to_numpy(dtype=float)
    order = np.lexsort((d, np.abs(d - tgt)))  # nearest to target, ties to shorter
    i = int(order[0])
    chosen = pairs.iloc[i]
    logger.debug(
        f"[+] P2 chose expiry {chosen['expiry']} (dte {int(chosen['dte'])}) from "
        f"{len(pairs)} candidate(s), prefer={prefer}, target_dte={tgt}"
    )
    return ExpirySelection(
        expiry=chosen["expiry"],
        dte=int(chosen["dte"]),
        n_candidates=int(len(pairs)),
        prefer=prefer,
        target_dte=tgt,
    )


# ==========================================================================
# P3 -- standard_exit
# ==========================================================================


@dataclass
class ExitDecision:
    should_exit: bool
    reason: Optional[str]
    fraction_captured: float
    unmarked: bool = False


def standard_exit(
    entry_credit: float,
    current_value: float,
    dte: int,
    profit_take: float = DEFAULT_PROFIT_TAKE,
    dte_exit: int = DEFAULT_DTE_EXIT,
) -> ExitDecision:
    """P3 -- close at `profit_take` of the credit captured OR at `dte_exit`,
    whichever comes first. Evaluated ONCE DAILY at the snapshot.

    Stateless by construction: it takes the credit received and the current
    mark, never a position object. An UNMARKED position (no valid quote) does
    not exit -- it is reported, per M2/A3.
    """
    if not np.isfinite(entry_credit) or entry_credit <= 0:
        raise ValueError(
            f"[-] standard_exit is a credit rule; entry_credit must be > 0, "
            f"got {entry_credit}. Debit structures name their own exit."
        )
    if not np.isfinite(current_value):
        return ExitDecision(False, None, float("nan"), unmarked=True)

    captured = (entry_credit - current_value) / entry_credit
    if captured >= profit_take:
        return ExitDecision(True, "profit_take", float(captured))
    if dte <= dte_exit:
        return ExitDecision(True, "dte_exit", float(captured))
    return ExitDecision(False, None, float(captured))


# ==========================================================================
# P4 -- sizers
# ==========================================================================


@dataclass
class SizingResult:
    units: float
    basis: str
    net_vega_per_unit: float = float("nan")
    budget: float = float("nan")
    nav: float = float("nan")

    @property
    def vega_per_nav(self) -> float:
        if not np.isfinite(self.nav) or self.nav <= 0:
            return float("nan")
        return self.budget / self.nav


def shared_vega_budget() -> float:
    """The single shared vega budget across F1/F3/F6.

    The chain registers that this constant EXISTS, is shared, and is never
    tuned -- but never states its value. Returning an invented number would
    silently create an unlogged degree of freedom, so this raises until an
    amendment supplies one.
    """
    raise UnregisteredParameterError(
        "[-] P4's shared vega budget has no registered value in the doc chain "
        "(spec v2 Section 3 P4 names the constant but never sets it). Pass "
        "`budget_vega` explicitly from a logged amendment; do not substitute."
    )


def size_by_vega_budget(
    structure: Sequence[dict], budget_vega: float, nav: float
) -> SizingResult:
    """P4 -- units such that |net vega| ~= budget.

    `structure` is one UNIT of the structure: dicts with `side` ('buy'/'sell'),
    `qty` (contracts per unit) and `vega` (per share, as quoted).
    """
    if not structure:
        raise ValueError("[-] structure must have at least one leg")
    if not np.isfinite(budget_vega) or budget_vega <= 0:
        raise ValueError(f"[-] budget_vega must be > 0, got {budget_vega}")

    net = 0.0
    for leg in structure:
        sign = -1.0 if leg["side"] == "sell" else 1.0
        net += sign * float(leg["qty"]) * float(leg["vega"]) * 100.0
    if abs(net) < 1e-9:
        raise ValueError(
            "[-] structure is vega neutral per unit; it cannot be sized by a "
            "vega budget. Name `size_by_debit` or `size_by_notional` instead."
        )
    return SizingResult(
        units=float(budget_vega / abs(net)),
        basis="vega_budget",
        net_vega_per_unit=float(net),
        budget=float(budget_vega),
        nav=float(nav),
    )


def size_by_debit(debit_per_unit: float, budget_usd: float) -> SizingResult:
    if not np.isfinite(debit_per_unit) or debit_per_unit <= 0:
        raise ValueError(f"[-] debit_per_unit must be > 0, got {debit_per_unit}")
    return SizingResult(units=float(budget_usd / debit_per_unit), basis="debit",
                        budget=float(budget_usd))


def size_by_notional(notional_per_unit: float, budget_usd: float) -> SizingResult:
    if not np.isfinite(notional_per_unit) or notional_per_unit <= 0:
        raise ValueError(f"[-] notional_per_unit must be > 0, got {notional_per_unit}")
    return SizingResult(units=float(budget_usd / notional_per_unit), basis="notional",
                        budget=float(budget_usd))


# ==========================================================================
# P5 -- regime_gate
# ==========================================================================

_REGIME_CACHE: Dict[str, pd.DataFrame] = {}


def load_regime_state_daily() -> pd.DataFrame:
    """The causal-replay classifier state log.

    CAVEAT (reported, not worked around): the row for session t is built from
    session t's CLOSE, so consuming it at the 15:45 snapshot on session t is a
    partial same-session look-forward. `regime_state_at(..., lag_sessions=1)`
    removes it. The house convention (RAMP in production) reads same-day, so
    lag 0 is the default and the choice is made explicit at every call site.
    """
    cached = _REGIME_CACHE.get("regime")
    if cached is None:
        from src.settings import get_local_storage_dir

        path = (
            get_local_storage_dir() / "alt_data" / "regime" / "regime_state_daily.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"[-] no regime state log at {path}")
        df = pd.read_parquet(path)
        df["session_date"] = pd.to_datetime(df["date"]).dt.date
        cached = df.sort_values("session_date").reset_index(drop=True)
        _REGIME_CACHE["regime"] = cached
    return cached


def regime_state_at(
    session: _date, lag_sessions: int = 0, table: Optional[pd.DataFrame] = None
) -> Optional[str]:
    """Classifier state for `session`, optionally lagged by whole sessions."""
    df = table if table is not None else load_regime_state_daily()
    idx = df.index[df["session_date"] <= session]
    if len(idx) == 0:
        return None
    pos = int(idx[-1]) - int(lag_sessions)
    if pos < 0:
        return None
    return str(df.iloc[pos]["regime"])


def regime_gate(state: Optional[str], allowed_states: Sequence[str]) -> bool:
    """P5 -- is `state` in `allowed_states`?

    A missing state BLOCKS rather than passes: an unknown regime is not a
    licence to trade. An unrecognised state name fails loud.
    """
    allowed = set(allowed_states)
    unknown = allowed - REGIME_STATES
    if unknown:
        raise ValueError(f"[-] unknown allowed_states {sorted(unknown)}")
    if state is None:
        return False
    if state not in REGIME_STATES:
        raise ValueError(
            f"[-] unknown regime state {state!r}; expected one of {sorted(REGIME_STATES)}"
        )
    return state in allowed


# ==========================================================================
# P6 -- hedge_ledger
# ==========================================================================


class HedgeMode(str, Enum):
    DAILY_1545 = "daily_1545"
    BAND = "band"


HEDGE_LEDGER_COLUMNS = [
    "session_date", "ts_et", "mode", "trigger", "spot", "target_shares",
    "prior_shares", "traded_shares", "slippage_bps", "cost_usd",
]


def _bps_for(rv_percentile: Optional[float], schedule: Dict[str, float]) -> float:
    from src.backtesting.options.cost_model import vol_state_bucket

    key = vol_state_bucket(rv_percentile)
    if key in schedule:
        return float(schedule[key])
    if "vol_unknown" in schedule:
        return float(schedule["vol_unknown"])
    raise UnregisteredParameterError(
        f"[-] cost_bps_schedule has no entry for {key!r} and no 'vol_unknown' fallback"
    )


def hedge_ledger(
    position: pd.DataFrame,
    bars_1m: pd.DataFrame,
    mode: HedgeMode = HedgeMode.DAILY_1545,
    band: Optional[float] = None,
    cost_bps_schedule: Optional[Dict[str, float]] = None,
    ts_col: str = "timestamp",
    price_col: str = "close",
) -> pd.DataFrame:
    """P6 -- delta hedging in the underlying, emitting a ledger of EVERY fill.

    `position` carries one row per session with `session_date` and
    `net_delta_shares` (the option position's delta expressed in shares), plus
    optionally `rv_percentile` (tiers the slippage) and, for band mode,
    `sigma_daily` and `gamma_shares_per_point`.

    `mode`:
      daily_1545  one hedge per session at the registered snapshot minute
      band        re-hedge whenever the underlying has moved +/- `band` sigma
                  since the last hedge

    The slippage schedule is TIERED BY REALIZED-VOL PERCENTILE and is REQUIRED:
    hedge frequency spikes exactly when underlying spreads widen, and a flat bps
    assumption flatters every gamma strategy. The chain registers the tiering
    but no bps values, so there is deliberately no default.

    Attribution reads THIS ledger, never a reconstruction. (Attribution itself
    is M4 -- Phase 2b.)
    """
    if cost_bps_schedule is None:
        raise UnregisteredParameterError(
            "[-] P6 requires an explicit `cost_bps_schedule` keyed by vol state "
            "('vol_low'/'vol_mid'/'vol_high'/'vol_unknown'). The doc chain "
            "registers the TIERING but no bps values; a flat default would "
            "flatter every gamma strategy."
        )
    if mode is HedgeMode.BAND and (band is None or band <= 0):
        raise ValueError("[-] band mode requires band > 0")

    bars = truncate_to_snapshot(bars_1m, ts_col=ts_col)
    if bars.empty:
        return pd.DataFrame(columns=HEDGE_LEDGER_COLUMNS)

    by_session = {d: g.reset_index(drop=True) for d, g in bars.groupby("session_date")}
    rows: List[dict] = []
    prior = 0.0

    for _, pos in position.sort_values("session_date").iterrows():
        sess = pos["session_date"]
        g = by_session.get(sess)
        if g is None or g.empty:
            logger.warning(f"[!] no bars for {sess}; no hedge emitted (not imputed)")
            continue
        bps = _bps_for(pos.get("rv_percentile"), cost_bps_schedule)
        net_delta = float(pos["net_delta_shares"])

        if mode is HedgeMode.DAILY_1545:
            last = g.iloc[-1]
            prior = _emit(rows, sess, last, "daily_1545", "snapshot",
                          -net_delta, prior, bps)
            continue

        sigma = float(pos.get("sigma_daily", np.nan))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"[-] band mode needs a positive sigma_daily for {sess}")
        gamma = float(pos.get("gamma_shares_per_point", 0.0))
        ref_spot = float(g.iloc[0][price_col])
        last_spot = ref_spot
        for i in range(len(g)):
            bar = g.iloc[i]
            spot = float(bar[price_col])
            moved = abs(np.log(spot / last_spot)) >= band * sigma
            if i == 0 or moved:
                target = -(net_delta + gamma * (spot - ref_spot))
                trigger = "initial" if i == 0 else "band"
                new_prior = _emit(rows, sess, bar, "band", trigger, target, prior, bps)
                if new_prior != prior or i == 0:
                    last_spot = spot
                prior = new_prior

    return pd.DataFrame(rows, columns=HEDGE_LEDGER_COLUMNS)


def _emit(rows, sess, bar, mode, trigger, target, prior, bps) -> float:
    traded = float(target) - float(prior)
    if abs(traded) < 1e-12 and trigger != "initial":
        return prior
    spot = float(bar["close"]) if "close" in bar else float(bar.iloc[0])
    rows.append(
        {
            "session_date": sess,
            "ts_et": bar["ts_et"],
            "mode": mode,
            "trigger": trigger,
            "spot": spot,
            "target_shares": float(target),
            "prior_shares": float(prior),
            "traded_shares": traded,
            "slippage_bps": float(bps),
            "cost_usd": abs(traded) * spot * float(bps) / 1e4,
        }
    )
    return float(target)


# ==========================================================================
# P7 -- roll
# ==========================================================================


@dataclass
class RollTrigger:
    dte_at_or_below: Optional[int] = None
    abs_delta_at_or_above: Optional[float] = None


@dataclass
class RollEvent:
    reason: str
    closed: dict
    opened: Optional[StrikeSelection]
    cost_events: List[dict] = field(default_factory=list)


def should_roll(
    trigger: RollTrigger, dte: Optional[int], abs_delta: Optional[float]
) -> Tuple[bool, Optional[str]]:
    if trigger.dte_at_or_below is not None and dte is not None:
        if int(dte) <= int(trigger.dte_at_or_below):
            return True, "dte"
    if trigger.abs_delta_at_or_above is not None and abs_delta is not None:
        if abs(float(abs_delta)) >= float(trigger.abs_delta_at_or_above):
            return True, "abs_delta"
    return False, None


def roll(
    current: dict, trigger: RollTrigger, new_selection: Optional[StrikeSelection]
) -> Optional[RollEvent]:
    """P7 -- mechanical roll. EACH roll logs TWO separate cost events (the close
    and the open); they are never netted into one transit."""
    fired, reason = should_roll(trigger, current.get("dte"), current.get("abs_delta"))
    if not fired:
        return None
    if new_selection is None:
        raise ValueError(
            "[-] roll trigger fired but no replacement selection was supplied; "
            "a roll that cannot open a new leg is a CLOSE, not a roll."
        )
    return RollEvent(
        reason=reason,
        closed=dict(current),
        opened=new_selection,
        cost_events=[
            {"action": "close", "leg": dict(current), "reason": reason},
            {"action": "open", "leg": new_selection, "reason": reason},
        ],
    )


# ==========================================================================
# P8 -- iv_rank / iv_percentile
# ==========================================================================


def _iv_series(
    ticker: str, dte_bucket: int, table: Optional[pd.DataFrame]
) -> pd.DataFrame:
    if table is None:
        from src.data.options.derived_store import load_derived_table

        table = load_derived_table("iv_rank_daily")
    df = table[(table["root"] == ticker) & (table["dte_bucket"] == dte_bucket)]
    return df.sort_values("session_date").reset_index(drop=True)


def _lookup(df: pd.DataFrame, column: pd.Series, session: _date) -> float:
    hit = np.flatnonzero(df["session_date"].to_numpy() == session)
    if len(hit) == 0:
        return float("nan")
    return float(column.iloc[int(hit[0])])


def iv_rank(
    ticker: str,
    date: _date,
    window_years: int = 1,
    dte_bucket: int = 30,
    table: Optional[pd.DataFrame] = None,
) -> float:
    """P8 -- trailing IV rank, STRICTLY backward looking. No full-sample ranks."""
    df = _iv_series(ticker, dte_bucket, table)
    if df.empty:
        return float("nan")
    col = f"iv_rank_{window_years}y"
    series = (
        df[col] if col in df.columns
        else trailing_rank(df["atm_iv"], TRADING_DAYS_PER_YEAR * window_years)
    )
    return _lookup(df, series, date)


def iv_percentile(
    ticker: str,
    date: _date,
    window_years: int = 1,
    dte_bucket: int = 30,
    table: Optional[pd.DataFrame] = None,
) -> float:
    """P8 -- trailing IV percentile, STRICTLY backward looking."""
    df = _iv_series(ticker, dte_bucket, table)
    if df.empty:
        return float("nan")
    col = f"iv_pctile_{window_years}y"
    series = (
        df[col] if col in df.columns
        else trailing_percentile(df["atm_iv"], TRADING_DAYS_PER_YEAR * window_years)
    )
    return _lookup(df, series, date)


# ==========================================================================
# P9 -- Yang-Zhang realized volatility
# ==========================================================================


def snapshot_daily_ohlc(bars_1m: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Daily OHLC built ONLY from bars at or before the 15:45 snapshot.

    Deliberately NOT the full-session OHLC: snapshot symmetry means every signal
    sees the same clock the marks and hedges do.
    """
    bars = truncate_to_snapshot(bars_1m, ts_col=ts_col)
    if bars.empty:
        return pd.DataFrame(columns=["session_date", "open", "high", "low", "close"])
    g = bars.sort_values("ts_et").groupby("session_date", sort=True)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).reset_index()
    return out


def yang_zhang_rv_from_bars(
    bars_1m: pd.DataFrame,
    window_days: int,
    ts_col: str = "timestamp",
    annualization_factor: float = 252.0,
) -> pd.Series:
    """P9 -- Yang-Zhang realized vol, with the 15:45 truncation enforced INSIDE.

    Reuses `src.features.volatility.yang_zhang_rv`; no forked estimator.
    """
    if window_days < 2:
        raise ValueError(f"[-] window_days must be >= 2, got {window_days}")
    from src.features.volatility import yang_zhang_rv

    daily = snapshot_daily_ohlc(bars_1m, ts_col=ts_col)
    if daily.empty:
        return pd.Series(dtype=float, name=f"yz_{window_days}d")
    rv = yang_zhang_rv(
        daily[["open", "high", "low", "close"]], window_days, annualization_factor
    )
    rv.index = pd.Index(daily["session_date"], name="session_date")
    return rv.rename(f"yz_{window_days}d")


def snapshot_daily_realized_variance(
    bars_1m: pd.DataFrame, ts_col: str = "timestamp", price_col: str = "close"
) -> pd.Series:
    """Sum of squared 1-minute log CLOSE-to-CLOSE returns inside the truncated
    session. Close-to-close within the session structurally excludes the
    overnight gap -- the gap is not intraday realized variance."""
    bars = truncate_to_snapshot(bars_1m, ts_col=ts_col)
    if bars.empty:
        return pd.Series(dtype=float, name="rv_1m_daily_var")
    out = {}
    for sess, g in bars.sort_values("ts_et").groupby("session_date", sort=True):
        px = g[price_col].to_numpy(dtype=float)
        if len(px) < 2:
            continue
        r = np.diff(np.log(px))
        out[sess] = float(np.nansum(r ** 2))
    return pd.Series(out, name="rv_1m_daily_var")


# ==========================================================================
# P10 -- HAR-RV forecast
# ==========================================================================


@dataclass
class HarForecast:
    """P10's return value.

    `vix_correlation` is the REGISTERED spurious-path check (OPT-019): a
    forecast that is merely a repackaged VIX is not an independent signal, so
    the correlation is reported alongside every forecast, never separately.
    """

    forecast: pd.Series
    vix_correlation: float
    n_forecast: int
    spec: Tuple[int, int, int]
    min_train: int


def har_rv_forecast(
    bars_1m: Optional[pd.DataFrame] = None,
    rv_daily: Optional[pd.Series] = None,
    spec: Tuple[int, int, int] = HAR_SPEC_FROZEN,
    horizon: int = 1,
    min_train: int = 252,
    vix: Optional[pd.Series] = None,
    ts_col: str = "timestamp",
) -> HarForecast:
    """P10 -- HAR-RV forecast on P9 inputs.

    Spec FROZEN at (1,5,22): no order selection, no refit-schedule tuning.
    Fit on an EXPANDING window with no forward data. The 15:45 truncation is
    enforced internally via `snapshot_daily_realized_variance`.
    """
    if tuple(spec) != HAR_SPEC_FROZEN:
        raise ValueError(
            f"[-] the HAR spec is FROZEN at {HAR_SPEC_FROZEN}; {tuple(spec)} would "
            f"be order selection, which the registration forbids."
        )
    if int(horizon) != 1:
        raise ValueError(
            f"[-] the registered HAR target is next-day RV (horizon 1); "
            f"got {horizon}. A different horizon is a different specification."
        )
    if rv_daily is None:
        if bars_1m is None:
            raise ValueError("[-] pass either bars_1m or rv_daily")
        rv_daily = snapshot_daily_realized_variance(bars_1m, ts_col=ts_col)

    from src.backtesting.vol.har_rv import har_forecast

    rv = pd.Series(rv_daily).astype(float)
    fc = har_forecast(rv, min_train=min_train)

    corr = float("nan")
    if vix is not None:
        v = pd.Series(vix).astype(float).reindex(fc.index)
        vol_pct = np.sqrt(fc.to_numpy(dtype=float) * 252.0) * 100.0
        pair = pd.DataFrame({"f": vol_pct, "v": v.to_numpy(dtype=float)}).dropna()
        if len(pair) > 2 and pair["f"].std() > 0 and pair["v"].std() > 0:
            corr = float(pair["f"].corr(pair["v"]))

    return HarForecast(
        forecast=fc,
        vix_correlation=corr,
        n_forecast=int(np.isfinite(fc.to_numpy(dtype=float)).sum()),
        spec=HAR_SPEC_FROZEN,
        min_train=int(min_train),
    )
