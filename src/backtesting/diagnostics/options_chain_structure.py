"""Wave-0 Group B2 option-chain STRUCTURE measurements.

Scope boundary (load-bearing): everything here measures PROPERTIES OF THE
OPTION-CHAIN DATA -- premium ratios, constraint-binding frequencies, entry
availability counts, distance distributions. There is NO position, NO fill, NO
equity curve, NO P&L and NO performance metric anywhere in this module. Keeping
that boundary explicit is what keeps these measurements out of the project's
multiple-testing trial count.

Registered rulings honoured here:
  * marks are `mid` on `quote_valid` rows ONLY; the trade-derived
    open/high/low/close/vwap columns are NEVER used as a mark;
  * greeks ship as NaN-valued float64 (NOT SQL NULL) -- every usability test
    below uses is_nan() AND is_null(), never null_count();
  * `oi_eod` / `gamma_eod` are leak-bearing at the snapshot minute; only the
    T-1-lagged values (`oi_eod_lag1`, restitched across month seams via
    `canonical.add_eod_lag`) are consumed;
  * greek coverage is bounded by the per-partition census -- non-OK partitions
    are excluded EXPLICITLY and the exclusions are counted and reported.

No imputation, no forward-fill, no smoothing, no interpolation. Rows that fail
a criterion are counted, never silently dropped.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.data.options.canonical import add_eod_lag
from src.features.volatility import yang_zhang_rv
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHAIN_STORE_NAME = "options_chain_eod"

GREEK_CENSUS_PATH = Path(
    "docs/strategies/research/options-slate/"
    "20260728_options_greek_coverage_census.csv"
)

#: Pre-commitment: a delta selection whose |realized| deviates from the target
#: by more than this is DROPPED (counted, never imputed).
MAX_DELTA_DEVIATION = 0.05

#: P1's low-delta rule: below this the spec requires a smoothed surface (M7),
#: which does not exist in this repo. Selections below it are OUT OF SPEC.
LOW_DELTA_SPEC_FLOOR = 0.10

TRADING_DAYS_PER_YEAR = 252

_CHAIN_COLUMNS = [
    "root", "session_date", "expiry", "strike", "right",
    "bid", "ask", "mid", "spread_rel", "quote_valid",
    "dte", "volume", "implied_vol", "delta", "underlying_px",
    "oi_eod", "oi_eod_lag1",
]


# --------------------------------------------------------------------------
# Partition discovery / greek census
# --------------------------------------------------------------------------

def _chain_root_dir(root: str) -> Path:
    return get_local_storage_dir() / "options" / CHAIN_STORE_NAME / f"root={root}"


def available_chain_months(root: str) -> List[Tuple[int, int]]:
    """(year, month) partitions of options_chain_eod present on disk for `root`."""
    base = _chain_root_dir(root)
    found: List[Tuple[int, int]] = []
    if not base.exists():
        return found
    for ydir in sorted(base.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            if (mdir / "data.parquet").exists():
                found.append((int(ydir.name.split("=")[1]),
                              int(mdir.name.split("=")[1])))
    return sorted(found)


def greek_ok_months(root: str) -> Set[Tuple[int, int]]:
    """(year, month) partitions whose census greek_status is OK."""
    census = pd.read_csv(GREEK_CENSUS_PATH)
    sub = census[(census["root"] == root) & (census["greek_status"] == "OK")]
    return {(int(r.year), int(r.month)) for r in sub.itertuples()}


def usable_chain_months(root: str) -> Tuple[List[Tuple[int, int]],
                                            List[Tuple[int, int]]]:
    """Split the on-disk partitions into (greek-OK, excluded-by-census)."""
    ok = greek_ok_months(root)
    on_disk = available_chain_months(root)
    usable = [m for m in on_disk if m in ok]
    excluded = [m for m in on_disk if m not in ok]
    return usable, excluded


def load_chain_months(root: str,
                      months: Sequence[Tuple[int, int]],
                      columns: Optional[Sequence[str]] = None,
                      restitch_oi_lag: bool = False) -> pl.DataFrame:
    """Read the given options_chain_eod partitions, selecting columns BY NAME.

    `restitch_oi_lag=True` recomputes `oi_eod_lag1` over the concatenated frame
    so the first session of each month is no longer NULL purely because the lag
    was computed within a single month file.
    """
    cols = list(columns) if columns is not None else list(_CHAIN_COLUMNS)
    frames = []
    for (y, m) in months:
        fp = _chain_root_dir(root) / f"year={y:04d}" / f"month={m:02d}" / "data.parquet"
        if not fp.exists():
            continue
        f = pl.read_parquet(fp, columns=cols)
        # Reality divergence (measured 2026-07-28): some options_chain_eod
        # partitions ship the float columns as Float32 and others as Float64,
        # which makes a naive vertical concat raise. Normalize to Float64 --
        # a widening cast only; no value is altered, dropped or imputed.
        narrow = [c for c, t in f.schema.items() if t == pl.Float32]
        if narrow:
            f = f.with_columns([pl.col(c).cast(pl.Float64) for c in narrow])
        frames.append(f)
    if not frames:
        return pl.DataFrame(schema={c: pl.Null for c in cols})
    out = pl.concat(frames, how="vertical")
    if restitch_oi_lag:
        needed = {"root", "expiry", "strike", "right", "session_date", "oi_eod"}
        missing = needed - set(out.columns)
        if missing:
            raise ValueError(f"[-] restitch_oi_lag needs columns {sorted(missing)}")
        has_gamma = "gamma_eod" in out.columns
        if not has_gamma:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("gamma_eod"))
        out = add_eod_lag(out.drop([c for c in ("oi_eod_lag1", "gamma_eod_lag1")
                                    if c in out.columns]))
        if not has_gamma:
            out = out.drop(["gamma_eod", "gamma_eod_lag1"])
    return out


# --------------------------------------------------------------------------
# Primitive P1 -- delta strike selection (single implementation, used everywhere)
# --------------------------------------------------------------------------

def _finite(col: str) -> pl.Expr:
    return (pl.col(col).is_not_null()
            & pl.col(col).is_not_nan()
            & pl.col(col).is_finite())


def eligible_quotes(chain: pl.DataFrame) -> pl.DataFrame:
    """Rows usable as marks: valid two-sided quote AND a finite `mid`.

    NaN-vs-NULL trap: `mid` is a real NULL when quote_valid is False, but the
    greeks are NaN-valued floats -- both are tested explicitly.
    """
    return chain.filter(pl.col("quote_valid") & _finite("mid"))


def select_expiry_by_dte(chain: pl.DataFrame,
                         dte_lo: int,
                         dte_hi: int) -> pl.DataFrame:
    """Keep ONE expiry per session: the DTE nearest the window midpoint.

    Spec P2 -- never silently straddle a distribution of expiries. Ties break
    to the SHORTER dte.
    """
    if chain.height == 0:
        return chain
    mid = (dte_lo + dte_hi) / 2.0
    c = chain.filter((pl.col("dte") >= dte_lo) & (pl.col("dte") <= dte_hi))
    if c.height == 0:
        return c
    picks = (c.select(["session_date", "expiry", "dte"]).unique()
             .with_columns((pl.col("dte") - mid).abs().alias("_dev"))
             .sort(["session_date", "_dev", "dte"])
             .group_by("session_date", maintain_order=True)
             .first()
             .select(["session_date", "expiry"]))
    return c.join(picks, on=["session_date", "expiry"], how="inner")


def select_strike_by_delta(chain: pl.DataFrame,
                           right: str,
                           target_delta: float,
                           dte_window: Tuple[int, int],
                           max_deviation: float = MAX_DELTA_DEVIATION
                           ) -> pl.DataFrame:
    """Nearest available |delta| to `target_delta` within `dte_window`.

    One row per session. Tie-break (registered): higher `volume`, then higher
    `oi_eod_lag1`, then lower `spread_rel`.

    Returns the realized delta actually selected alongside the target, plus
    `delta_dev` and `within_tolerance` so selection quality is visible and
    out-of-tolerance selections are COUNTED rather than silently dropped.
    """
    tgt = abs(float(target_delta))
    c = eligible_quotes(chain).filter((pl.col("right") == right) & _finite("delta"))
    c = select_expiry_by_dte(c, *dte_window)
    if c.height == 0:
        return c.with_columns(
            pl.lit(tgt).alias("target_delta"),
            pl.lit(None, dtype=pl.Float64).alias("realized_delta"),
            pl.lit(None, dtype=pl.Float64).alias("delta_dev"),
            pl.lit(None, dtype=pl.Boolean).alias("within_tolerance"),
        )
    c = c.with_columns(
        (pl.col("delta").abs() - tgt).abs().alias("delta_dev"),
    )
    sort_cols = ["session_date", "delta_dev", "volume"]
    descending = [False, False, True]
    if "oi_eod_lag1" in c.columns:
        sort_cols.append("oi_eod_lag1")
        descending.append(True)
    sort_cols.append("spread_rel")
    descending.append(False)
    c = (c.sort(sort_cols, descending=descending, nulls_last=True)
         .group_by("session_date", maintain_order=True)
         .first())
    return c.with_columns(
        pl.lit(tgt).alias("target_delta"),
        pl.col("delta").alias("realized_delta"),
        (pl.col("delta_dev") <= max_deviation).alias("within_tolerance"),
    ).sort("session_date")


def select_strike_nearest_price(chain: pl.DataFrame,
                                right: str,
                                target_strikes: pl.DataFrame,
                                dte_window: Tuple[int, int],
                                label: str) -> pl.DataFrame:
    """Strike nearest a per-session TARGET STRIKE LEVEL (sigma-defined legs).

    `target_strikes` must carry columns [session_date, target_strike].
    Same expiry rule as `select_strike_by_delta`.
    """
    c = eligible_quotes(chain).filter(pl.col("right") == right)
    c = select_expiry_by_dte(c, *dte_window)
    if c.height == 0:
        return c
    c = c.join(target_strikes, on="session_date", how="inner")
    if c.height == 0:
        return c
    c = c.with_columns(
        (pl.col("strike") - pl.col("target_strike")).abs().alias("_strike_dev"))
    c = (c.sort(["session_date", "_strike_dev", "volume", "spread_rel"],
                descending=[False, False, True, False], nulls_last=True)
         .group_by("session_date", maintain_order=True)
         .first())
    rename = {
        "strike": f"{label}_strike", "mid": f"{label}_mid",
        "delta": f"{label}_delta", "spread_rel": f"{label}_spread_rel",
        "_strike_dev": f"{label}_strike_dev", "expiry": f"{label}_expiry",
        "dte": f"{label}_dte",
    }
    keep = ["session_date"] + [k for k in rename if k in c.columns]
    return c.select(keep).rename(rename).sort("session_date")


# --------------------------------------------------------------------------
# Calendar helpers
# --------------------------------------------------------------------------

def third_friday(year: int, month: int) -> date:
    """Third Friday of the month (the monthly-expiry anchor, unshifted)."""
    fridays = [d for d in range(1, calendar.monthrange(year, month)[1] + 1)
               if date(year, month, d).weekday() == 4]
    return date(year, month, fridays[2])


def monthly_expiry_anchors(years: Iterable[int]) -> List[date]:
    return [third_friday(y, m) for y in years for m in range(1, 13)]


def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def is_in_opex_week(d: date, opex_dates: Set[date]) -> bool:
    """True when `d` falls in the Mon-Fri week containing a monthly expiry."""
    return week_monday(d) in {week_monday(x) for x in opex_dates}


# --------------------------------------------------------------------------
# Underlying helpers
# --------------------------------------------------------------------------

def daily_ohlc_from_minutes(symbol: str, years: Sequence[int]) -> pd.DataFrame:
    """Regular-hours daily OHLC built from the sip_split 1-minute store.

    Reuses `session_bars.load_minute_bars` (no forked loader). Bars are limited
    to 09:30-16:00 ET. No imputation of missing sessions.
    """
    from src.backtesting.diagnostics.session_bars import load_minute_bars

    bars = load_minute_bars(symbol, years)
    ny = bars["ny"]
    rth = bars[(ny.dt.time >= pd.Timestamp("09:30").time())
               & (ny.dt.time <= pd.Timestamp("16:00").time())].copy()
    rth["session_date"] = rth["ny"].dt.normalize().dt.tz_localize(None)
    g = rth.groupby("session_date")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
    })
    out.index = pd.DatetimeIndex(out.index)
    return out.sort_index()


def sigma_for_dte(annualized_vol: float, dte: float) -> float:
    """Scale an annualized vol to a DTE horizon: sigma_dte = vol * sqrt(dte/252)."""
    return float(annualized_vol) * float(np.sqrt(dte / TRADING_DAYS_PER_YEAR))


def pullback_flags_1session(closes: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """close(t) <= (1 - threshold) * close(t-1). First element is False."""
    c = np.asarray(closes, dtype=float)
    out = np.zeros(c.shape, dtype=bool)
    out[1:] = c[1:] <= (1.0 - threshold) * c[:-1]
    return out


def pullback_flags_trailing_max(closes: np.ndarray,
                                window: int = 20,
                                threshold: float = 0.02) -> np.ndarray:
    """close(t) <= (1 - threshold) * max(close over the trailing `window`)."""
    s = pd.Series(np.asarray(closes, dtype=float))
    trailing = s.rolling(window=window, min_periods=window).max()
    return (s <= (1.0 - threshold) * trailing).fillna(False).to_numpy()


def realized_vol_forward(daily: pd.DataFrame,
                         start: date,
                         horizon_days: int,
                         window_min: int = 5) -> Optional[float]:
    """Annualized Yang-Zhang RV over the forward window (start, start+horizon].

    DIAGNOSTIC ONLY -- forward-looking by construction, produces no signal and
    no P&L. Returns None when the window runs past the data edge.
    """
    idx = daily.index
    start_ts = pd.Timestamp(start).normalize()
    pos = int(idx.searchsorted(start_ts, side="left"))
    if pos >= len(idx) or idx[pos] != start_ts:
        return None
    end_cal = start_ts + pd.Timedelta(days=horizon_days)
    end_pos = int(idx.searchsorted(end_cal, side="right"))
    seg = daily.iloc[pos:end_pos]
    n = len(seg)
    if n < window_min + 1:
        return None
    if end_pos >= len(idx) and idx[-1] < end_cal:
        return None
    vals = yang_zhang_rv(seg, window=n - 1, annualization_factor=TRADING_DAYS_PER_YEAR)
    v = vals.iloc[-1]
    return None if not np.isfinite(v) else float(v)


# --------------------------------------------------------------------------
# Entry cadence (registered): non-overlapping entries
# --------------------------------------------------------------------------

def non_overlapping_entries(sessions: Sequence[date],
                            qualifies: Sequence[bool],
                            hold_days: int) -> List[date]:
    """Walk sessions in date order; on a qualifying session record an entry and
    skip forward `hold_days` CALENDAR days before resuming."""
    entries: List[date] = []
    blocked_until: Optional[date] = None
    for d, q in zip(sessions, qualifies):
        if blocked_until is not None and d < blocked_until:
            continue
        if bool(q):
            entries.append(d)
            blocked_until = d + timedelta(days=hold_days)
    return entries


def entries_per_year(entries: Sequence[date],
                     session_dates: Sequence[date]) -> pd.DataFrame:
    """Entries per calendar year alongside the session count in that year."""
    ent = pd.Series(pd.to_datetime(list(entries))) if len(entries) else pd.Series(
        dtype="datetime64[ns]")
    ses = pd.Series(pd.to_datetime(list(session_dates)))
    ec = ent.dt.year.value_counts() if len(ent) else pd.Series(dtype=int)
    sc = ses.dt.year.value_counts()
    years = sorted(sc.index)
    return pd.DataFrame({
        "year": years,
        "entries": [int(ec.get(y, 0)) for y in years],
        "sessions": [int(sc.get(y, 0)) for y in years],
    })


# --------------------------------------------------------------------------
# Bootstrap helpers (difference of group means)
# --------------------------------------------------------------------------

def _diff_of_means(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.nan
    return float(values[mask].mean() - values.mean())


def bootstrap_diff_ci(values: np.ndarray,
                      mask: np.ndarray,
                      n_boot: int = 10000,
                      seed: int = 42,
                      block: Optional[int] = None) -> Tuple[float, float, float]:
    """(point, lo95, hi95) for mean(conditional) - mean(all).

    `block=None` -> iid resampling (anticonservative under overlapping forward
    windows). `block=k` -> circular block bootstrap with block length k.
    """
    v = np.asarray(values, dtype=float)
    m = np.asarray(mask, dtype=bool)
    n = len(v)
    point = _diff_of_means(v, m)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    if block is None or block <= 1:
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[i] = _diff_of_means(v[idx], m[idx])
    else:
        n_blocks = int(np.ceil(n / block))
        offs = np.arange(block)
        for i in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            idx = ((starts[:, None] + offs[None, :]).ravel() % n)[:n]
            draws[i] = _diff_of_means(v[idx], m[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, np.nan, np.nan
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def describe(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
           "std": float(v.std(ddof=1)) if v.size > 1 else np.nan}
    for q in range(1, 10):
        out[f"d{q}"] = float(np.percentile(v, q * 10))
    return out
