"""Wave 0 Group B1 -- option-chain IV-state measurement layer.

DATA-PROPERTY MEASUREMENTS ONLY. No positions, no fills, no P&L, no equity
curves. Everything here reduces the canonical EOD option chain and the 1-minute
underlying bars to descriptive daily state tables, then measures transition /
persistence / episode-count properties of those states.

Registered conventions honoured here:
  M2  marks are `mid` from VALID quotes only; trade-derived open/high/low/close
      /vwap are PROHIBITED as marks.
  NaN-vs-NULL trap: implied_vol / delta / theta / vega / gamma_eod are
      NaN-valued float64, NOT sql NULL. Arrow reports null_count == 0 for a
      100%-NaN column. Every usability test here is a VALUE test.
  Greeks boundary: ThetaData serves no IV/greeks for ETF/index roots before
      2017-01, except QQQ (usable 2012+). The per-partition census is read and
      non-OK partitions are excluded EXPLICITLY and counted.
  oi_eod / gamma_eod are a lookahead leak at the snapshot minute -- unused here.
  No imputation / forward-fill / smoothing / interpolation across missing
      sessions. Every exclusion is quarantined, COUNTED and REPORTED.
  No full-sample percentile anywhere: every percentile / rank is strictly
      backward-looking and excludes the current observation from its window.
"""
from __future__ import annotations

import calendar
import glob
import os
from datetime import date as _date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)

EOD_ROOT = Path("H:/Stock_Data/options/options_chain_eod")
CENSUS_CSV = Path("docs/strategies/research/options-slate/"
                  "20260728_options_greek_coverage_census.csv")

DTE_BUCKETS = (7, 14, 30, 45, 60, 90)

# Registered usability bounds for the NaN-valued greek columns.
IV_MIN = 1e-6           # strictly positive
IV_MAX = 5.0            # 500% annualized -- above this the quote is garbage
ABS_DELTA_MIN = 1e-6    # strictly positive
ABS_DELTA_MAX = 1.0     # EXCLUSIVE: |delta| == 1.0 is a saturated/degenerate mark

EOD_COLUMNS = ["session_date", "expiry", "strike", "right", "mid",
               "implied_vol", "delta", "underlying_px", "dte", "quote_valid"]


# ---------------------------------------------------------------------------
# Backward-looking window primitives (anti-lookahead)
# ---------------------------------------------------------------------------

MIN_WINDOW_COVERAGE = 0.95


def trailing_percentile(s: pd.Series, window: int,
                        min_coverage: float = MIN_WINDOW_COVERAGE) -> pd.Series:
    """Fraction of the `window` STRICTLY-PRIOR observations below today's value.

    The current observation is EXCLUDED from its own comparison window, and the
    window always SPANS exactly `window` prior positions (NaN before that), so
    nothing at t is a function of any observation at t+1 or later.

    REGISTERED COVERAGE RULE: the window may contain non-measurable sessions
    (e.g. a session with no bracketing expiry). The comparison is made against
    the FINITE observations actually present, provided they cover at least
    `min_coverage` of the window; otherwise the output is NaN. Missing sessions
    are NEVER imputed, forward-filled or interpolated -- they are simply absent
    from the comparison set, and callers count them.
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


def trailing_rank(s: pd.Series, window: int,
                  min_coverage: float = MIN_WINDOW_COVERAGE) -> pd.Series:
    """(x - min) / (max - min) over the `window` STRICTLY-PRIOR observations.

    Same registered coverage rule as `trailing_percentile`.
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
        lo, hi = float(w.min()), float(w.max())
        if hi <= lo:
            continue
        out[i] = (cur - lo) / (hi - lo)
    return pd.Series(out, index=s.index, name=f"rank_{window}")


# ---------------------------------------------------------------------------
# NaN-greek filtering -- defeats the null_count trap
# ---------------------------------------------------------------------------

def greek_usable_mask(df: pd.DataFrame, require_delta: bool = True) -> pd.Series:
    """Row mask of contracts whose greeks are actually USABLE.

    Tests VALUES (np.isfinite), never null_count / is_null / column presence:
    a 100%-NaN float64 column reports null_count == 0 in arrow and would pass
    any null-based check. Also rejects degenerate marks (iv <= 0, iv > 5.0,
    |delta| == 0 or 1) and rows whose quote was not valid.
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


def select_nearest_abs_delta(df: pd.DataFrame,
                             target: float,
                             tolerance: float) -> Optional[pd.Series]:
    """Row whose |delta| is nearest `target`, or None if beyond `tolerance`."""
    if df is None or len(df) == 0:
        return None
    d = pd.to_numeric(df["delta"], errors="coerce").to_numpy(dtype=float)
    ad = np.abs(d)
    dist = np.abs(ad - target)
    ok = np.isfinite(dist)
    if not ok.any():
        return None
    dist = np.where(ok, dist, np.inf)
    i = int(np.argmin(dist))
    if dist[i] > tolerance:
        return None
    return df.iloc[i]


# ---------------------------------------------------------------------------
# Monthly (third-Friday) expiry identification
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> _date:
    """Standard monthly expiry date before any holiday shift."""
    cal = calendar.Calendar()
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return fridays[2]


def monthly_expiry(year: int, month: int,
                   trading_days: Optional[Set[_date]] = None) -> _date:
    """Third Friday, shifted BACK to the prior trading day when it is a holiday.

    (Good Friday is the operative case: the monthly contract then expires on
    the Thursday.) With no `trading_days` set supplied the raw third Friday is
    returned unshifted.
    """
    d = third_friday(year, month)
    if trading_days is None:
        return d
    for _ in range(7):
        if d in trading_days:
            return d
        d = d - pd.Timedelta(days=1).to_pytimedelta()
    return third_friday(year, month)


def monthly_expiry_set(start: _date, end: _date,
                       trading_days: Optional[Set[_date]] = None) -> Set[_date]:
    """All standard monthly expiries in [start, end]."""
    out: Set[_date] = set()
    y, m = start.year, start.month
    one = pd.Timedelta(days=1).to_pytimedelta()
    while (y, m) <= (end.year, end.month):
        # Both listing conventions are accepted: from Feb-2015 the standard
        # monthly is dated the third FRIDAY, before that it was dated the
        # SATURDAY following the third Friday (OCC expiration-date change).
        # A third Friday that is an exchange holiday shifts BACK to Thursday.
        for d in (monthly_expiry(y, m, trading_days=trading_days),
                  third_friday(y, m) + one):
            if start <= d <= end:
                out.add(d)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# ---------------------------------------------------------------------------
# Episode counting
# ---------------------------------------------------------------------------

def find_episodes(dates: Sequence,
                  flag: Sequence,
                  merge_gap_sessions: int,
                  valid: Optional[Sequence] = None) -> List[dict]:
    """Non-overlapping episodes of a boolean state over an ordered session axis.

    REGISTERED EPISODE DEFINITION (fixed before any counting):
      * an episode STARTS on the first session in the state;
      * it ENDS on the last session before the first session out of the state;
      * two consecutive episodes separated by FEWER THAN `merge_gap_sessions`
        sessions are MERGED into one (the candidate structure is still held
        across the gap);
      * a session whose state is NOT MEASURABLE (`valid` False -- missing IV,
        missing bracketing expiry, ...) is treated as NOT-in-state, i.e. it
        terminates a run. Such sessions are counted and reported separately;
        they are never imputed.
    `merge_gap_sessions=0` performs no merging at all.
    """
    f = np.asarray(flag, dtype=bool)
    if valid is not None:
        f = f & np.asarray(valid, dtype=bool)
    n = len(f)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j + 1 < n and f[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    merged: List[Tuple[int, int]] = []
    for r in runs:
        if merged and (r[0] - merged[-1][1] - 1) < merge_gap_sessions:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return [{"start": dates[a], "end": dates[b], "start_idx": a, "end_idx": b,
             "duration": b - a + 1} for a, b in merged]


# ---------------------------------------------------------------------------
# ATM IV term interpolation
# ---------------------------------------------------------------------------

def interp_atm_iv(per_expiry: pd.DataFrame,
                  target_dte: int) -> Tuple[float, str]:
    """ATM IV at `target_dte`, LINEARLY interpolated in DTE between the two
    listed expiries that BRACKET the target.

    REGISTERED RULE: exact dte match is used as-is; otherwise the nearest
    listed expiry strictly below and the nearest strictly above the target are
    interpolated linearly in calendar DTE. NO extrapolation: when no bracketing
    pair exists the value is NaN (never imputed) and the session is counted.
    """
    if per_expiry is None or len(per_expiry) == 0:
        return float("nan"), "no_bracket"
    d = per_expiry.dropna(subset=["atm_iv"]).sort_values("dte")
    if len(d) == 0:
        return float("nan"), "no_bracket"
    dte = d["dte"].to_numpy(dtype=float)
    iv = d["atm_iv"].to_numpy(dtype=float)
    exact = np.where(dte == target_dte)[0]
    if len(exact):
        return float(iv[exact[0]]), f"exact:{target_dte}"
    lo_i = np.where(dte < target_dte)[0]
    hi_i = np.where(dte > target_dte)[0]
    if len(lo_i) == 0 or len(hi_i) == 0:
        return float("nan"), "no_bracket"
    a, b = int(lo_i[-1]), int(hi_i[0])
    w = (target_dte - dte[a]) / (dte[b] - dte[a])
    return float(iv[a] + (iv[b] - iv[a]) * w), f"interp:{int(dte[a])}-{int(dte[b])}"


# ---------------------------------------------------------------------------
# Partition discovery + greek-census gating
# ---------------------------------------------------------------------------

def available_partitions(root: str) -> List[Tuple[int, int, Path]]:
    pat = str(EOD_ROOT / f"root={root}" / "year=*" / "month=*" / "data.parquet")
    out = []
    for fp in sorted(glob.glob(pat)):
        parts = Path(fp).parts
        y = int([p for p in parts if p.startswith("year=")][0].split("=")[1])
        m = int([p for p in parts if p.startswith("month=")][0].split("=")[1])
        out.append((y, m, Path(fp)))
    return sorted(out)


def greek_ok_partitions(root: str, census_csv: Path = CENSUS_CSV) -> Set[Tuple[int, int]]:
    c = pd.read_csv(census_csv)
    c = c[(c["root"] == root) & (c["greek_status"] == "OK")]
    return {(int(y), int(m)) for y, m in zip(c["year"], c["month"])}


def partition_plan(root: str, year_lo: int, year_hi: int,
                   census_csv: Path = CENSUS_CSV) -> Tuple[List[Tuple[int, int, Path]], dict]:
    """Partitions to read, plus an explicit census of what was excluded and why."""
    avail = available_partitions(root)
    ok = greek_ok_partitions(root, census_csv)
    keep, drop_year, drop_greek = [], [], []
    for y, m, fp in avail:
        if not (year_lo <= y <= year_hi):
            drop_year.append((y, m))
        elif (y, m) not in ok:
            drop_greek.append((y, m))
        else:
            keep.append((y, m, fp))
    census = {
        "root": root,
        "n_on_disk": len(avail),
        "n_used": len(keep),
        "n_excluded_out_of_window": len(drop_year),
        "n_excluded_greeks_not_ok": len(drop_greek),
        "excluded_greeks_not_ok": drop_greek,
        "first_used": f"{keep[0][0]}-{keep[0][1]:02d}" if keep else None,
        "last_used": f"{keep[-1][0]}-{keep[-1][1]:02d}" if keep else None,
    }
    return keep, census


# ---------------------------------------------------------------------------
# Per-month chain reduction
# ---------------------------------------------------------------------------

def _read_month(fp: Path) -> pd.DataFrame:
    df = pl.read_parquet(fp, columns=EOD_COLUMNS).to_pandas()
    for c in ("session_date", "expiry"):
        df[c] = pd.to_datetime(df[c]).dt.date
    return df


def _atm_per_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """Per (session_date, expiry): IV at the strike nearest the underlying,
    averaging the call and the put at that strike.

    REGISTERED RULE: the ATM strike is the listed strike minimising
    |strike - underlying_px| among strikes that have at least one USABLE side.
    Both sides are averaged when both are usable; a single usable side is used
    alone and flagged (`n_sides`).
    """
    d = df.copy()
    d["dist"] = (d["strike"] - d["underlying_px"]).abs()
    key = ["session_date", "expiry"]
    atm_dist = d.groupby(key, sort=False)["dist"].transform("min")
    a = d[d["dist"] <= atm_dist + 1e-9]
    g = a.groupby(key, sort=False).agg(
        atm_iv=("implied_vol", "mean"),
        n_sides=("implied_vol", "size"),
        atm_strike=("strike", "first"),
        underlying_px=("underlying_px", "first"),
        dte=("dte", "first"),
    ).reset_index()
    return g


def build_atm_iv_daily(root: str, year_lo: int, year_hi: int) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`atm_iv_daily` + the per-expiry ATM surface + a stage-by-stage row census.

    Returns (atm_iv_daily, atm_per_expiry, stats).
    """
    parts, census = partition_plan(root, year_lo, year_hi)
    stats = dict(census)
    stats.update(rows_read=0, rows_quote_invalid=0, rows_greek_unusable=0,
                 rows_kept=0)
    frames = []
    for y, m, fp in parts:
        raw = _read_month(fp)
        stats["rows_read"] += len(raw)
        qv = raw["quote_valid"].fillna(False).to_numpy(dtype=bool)
        stats["rows_quote_invalid"] += int((~qv).sum())
        usable = greek_usable_mask(raw, require_delta=False).to_numpy()
        stats["rows_greek_unusable"] += int((qv & ~usable).sum())
        d = raw[usable]
        d = d[np.isfinite(pd.to_numeric(d["underlying_px"], errors="coerce"))]
        d = d[d["dte"] >= 0]
        stats["rows_kept"] += len(d)
        if len(d):
            frames.append(_atm_per_expiry(d))
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), stats
    per_exp = pd.concat(frames, ignore_index=True)
    per_exp["root"] = root

    rows = []
    n_no_bracket = 0
    for sess, g in per_exp.groupby("session_date", sort=True):
        for bucket in DTE_BUCKETS:
            val, src = interp_atm_iv(g[["dte", "atm_iv"]], bucket)
            if not np.isfinite(val):
                n_no_bracket += 1
            rows.append({"root": root, "session_date": sess, "dte_bucket": bucket,
                         "atm_iv": val, "source_dte": src, "n_contracts": int(len(g))})
    out = pd.DataFrame(rows).sort_values(["dte_bucket", "session_date"]).reset_index(drop=True)
    stats["n_sessions"] = int(per_exp["session_date"].nunique())
    stats["n_bucket_cells_no_bracket"] = int(n_no_bracket)
    stats["n_bucket_cells"] = int(len(out))
    return out, per_exp, stats


def _representative_expiry(g: pd.DataFrame, target_dte: int) -> Optional[int]:
    """REGISTERED: the bucket's representative listed expiry is the one whose
    dte is nearest the bucket target, required within max(7, 0.25*target) days.
    """
    tol = max(7.0, 0.25 * target_dte)
    d = (g["dte"] - target_dte).abs()
    if len(d) == 0:
        return None
    i = int(d.values.argmin())
    if d.values[i] > tol:
        return None
    return i


def build_skew_daily(root: str, year_lo: int, year_hi: int,
                     buckets: Iterable[int] = DTE_BUCKETS) -> Tuple[pd.DataFrame, dict]:
    """25-delta skew per (session, dte bucket).

    REGISTERED RULES:
      * the bucket is represented by the single listed expiry nearest the
        target dte (tolerance max(7, 0.25*target) days) -- skew is NOT
        interpolated across expiries;
      * within that expiry the put and the call whose |delta| is nearest 0.25
        are selected independently; a session/bucket is DROPPED (and counted)
        when the nearest available |delta| on either side is further than 0.05
        from 0.25;
      * curvature uses the ATM IV of the SAME representative expiry, so all
        three legs come from one term point.
    """
    parts, census = partition_plan(root, year_lo, year_hi)
    stats = dict(census)
    stats.update(n_dropped_delta_tol=0, n_dropped_no_expiry=0, n_rows=0)
    buckets = list(buckets)
    rows = []
    for y, m, fp in parts:
        raw = _read_month(fp)
        usable = greek_usable_mask(raw, require_delta=True).to_numpy()
        d = raw[usable]
        d = d[d["dte"] >= 0]
        if not len(d):
            continue
        d = d.copy()
        d["absdist"] = (d["strike"] - d["underlying_px"]).abs()
        for sess, gs in d.groupby("session_date", sort=True):
            exp_dte = gs.groupby("expiry", sort=True)["dte"].first().reset_index()
            for bucket in buckets:
                i = _representative_expiry(exp_dte, bucket)
                if i is None:
                    stats["n_dropped_no_expiry"] += 1
                    continue
                exp = exp_dte["expiry"].iloc[i]
                rep_dte = int(exp_dte["dte"].iloc[i])
                ge = gs[gs["expiry"] == exp]
                puts = ge[ge["right"] == "P"]
                calls = ge[ge["right"] == "C"]
                rp = select_nearest_abs_delta(puts, 0.25, 0.05)
                rc = select_nearest_abs_delta(calls, 0.25, 0.05)
                if rp is None or rc is None:
                    stats["n_dropped_delta_tol"] += 1
                    continue
                atm_i = int(ge["absdist"].values.argmin())
                atm_strike = float(ge["strike"].values[atm_i])
                atm_rows = ge[np.isclose(ge["strike"], atm_strike)]
                atm_iv_rep = float(atm_rows["implied_vol"].mean())
                ivp, ivc = float(rp["implied_vol"]), float(rc["implied_vol"])
                rows.append({
                    "root": root, "session_date": sess, "dte_bucket": bucket,
                    "rep_expiry": exp, "rep_dte": rep_dte,
                    "iv_25d_put": ivp, "iv_25d_call": ivc,
                    "skew_25d": ivp - ivc,
                    "atm_iv_rep": atm_iv_rep,
                    "curvature": (ivp + ivc) / 2.0 - atm_iv_rep,
                    "realized_delta_put": float(abs(rp["delta"])),
                    "realized_delta_call": float(abs(rc["delta"])),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["dte_bucket", "session_date"]).reset_index(drop=True)
    stats["n_rows"] = int(len(out))
    return out, stats


def build_delta_iv_daily(root: str, year_lo: int, year_hi: int,
                         right: str,
                         targets: Sequence[float],
                         buckets: Sequence[int],
                         tolerance: float = 0.07) -> Tuple[pd.DataFrame, dict]:
    """IV of the `right` contract whose |delta| is nearest each target, per bucket.

    Same representative-expiry rule as `build_skew_daily`. Used by the
    scope-reduced D-011 supplementary measurement (0.50 / 0.30 delta puts).
    """
    parts, census = partition_plan(root, year_lo, year_hi)
    stats = dict(census)
    stats.update(n_dropped_delta_tol=0, n_dropped_no_expiry=0)
    rows = []
    for y, m, fp in parts:
        raw = _read_month(fp)
        d = raw[greek_usable_mask(raw, require_delta=True).to_numpy()]
        d = d[(d["dte"] >= 0) & (d["right"] == right)]
        if not len(d):
            continue
        for sess, gs in d.groupby("session_date", sort=True):
            exp_dte = gs.groupby("expiry", sort=True)["dte"].first().reset_index()
            for bucket in buckets:
                i = _representative_expiry(exp_dte, bucket)
                if i is None:
                    stats["n_dropped_no_expiry"] += 1
                    continue
                ge = gs[gs["expiry"] == exp_dte["expiry"].iloc[i]]
                rec = {"root": root, "session_date": sess, "dte_bucket": bucket,
                       "rep_dte": int(exp_dte["dte"].iloc[i]), "right": right}
                ok = True
                for tgt in targets:
                    r = select_nearest_abs_delta(ge, tgt, tolerance)
                    if r is None:
                        ok = False
                        break
                    key = f"iv_{int(round(tgt * 100)):02d}d"
                    rec[key] = float(r["implied_vol"])
                    rec[f"realized_delta_{int(round(tgt * 100)):02d}"] = float(abs(r["delta"]))
                if not ok:
                    stats["n_dropped_delta_tol"] += 1
                    continue
                rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["dte_bucket", "session_date"]).reset_index(drop=True)
    stats["n_rows"] = int(len(out))
    return out, stats


def build_term_slope_daily(root: str, year_lo: int, year_hi: int,
                           trading_days: Optional[Set[_date]] = None
                           ) -> Tuple[pd.DataFrame, dict]:
    """M2/M1 ATM-IV term slope on STANDARD MONTHLY expiries.

    REGISTERED RULES:
      * "monthly" = the third Friday of the month, shifted BACK to the prior
        trading day when that Friday is an exchange holiday (Good Friday);
        the shifted date must be a date actually listed in the chain;
      * M1 = the nearest monthly expiry with dte >= 7 (so a monthly in its
        final expiry week is skipped); M2 = the next monthly after M1;
      * ATM IV per monthly = IV at the strike nearest the underlying, averaging
        the usable call and put at that strike;
      * sessions where M1 or M2 cannot be identified are COUNTED, never imputed.
    """
    parts, census = partition_plan(root, year_lo, year_hi)
    stats = dict(census)
    stats.update(n_sessions=0, n_missing_m1=0, n_missing_m2=0, n_rows=0)
    rows = []
    for y, m, fp in parts:
        raw = _read_month(fp)
        usable = greek_usable_mask(raw, require_delta=False).to_numpy()
        d = raw[usable]
        d = d[d["dte"] >= 0]
        if not len(d):
            continue
        atm = _atm_per_expiry(d)
        exp_lo = atm["expiry"].min()
        exp_hi = atm["expiry"].max()
        monthlies = monthly_expiry_set(exp_lo, exp_hi, trading_days=trading_days)
        for sess, g in atm.groupby("session_date", sort=True):
            stats["n_sessions"] += 1
            gm = g[g["expiry"].isin(monthlies) & (g["dte"] >= 7)].sort_values("dte")
            if len(gm) < 1:
                stats["n_missing_m1"] += 1
                continue
            if len(gm) < 2:
                stats["n_missing_m2"] += 1
                continue
            m1, m2 = gm.iloc[0], gm.iloc[1]
            iv1, iv2 = float(m1["atm_iv"]), float(m2["atm_iv"])
            if not (np.isfinite(iv1) and np.isfinite(iv2) and iv1 > 0):
                stats["n_missing_m1"] += 1
                continue
            rows.append({"root": root, "session_date": sess,
                         "m1_expiry": m1["expiry"], "m2_expiry": m2["expiry"],
                         "m1_dte": int(m1["dte"]), "m2_dte": int(m2["dte"]),
                         "m1_atm_iv": iv1, "m2_atm_iv": iv2, "slope": iv2 / iv1})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("session_date").reset_index(drop=True)
    stats["n_rows"] = int(len(out))
    return out, stats


def build_iv_rank_daily(atm_iv_daily: pd.DataFrame) -> pd.DataFrame:
    """Trailing 252/504-session IV rank + percentile, strictly backward looking."""
    frames = []
    for (root, bucket), g in atm_iv_daily.groupby(["root", "dte_bucket"], sort=True):
        g = g.sort_values("session_date").reset_index(drop=True)
        iv = g["atm_iv"]
        out = pd.DataFrame({
            "root": root, "session_date": g["session_date"], "dte_bucket": bucket,
            "atm_iv": iv,
            "iv_rank_1y": trailing_rank(iv, 252).values,
            "iv_pctile_1y": trailing_percentile(iv, 252).values,
            "iv_pctile_2y": trailing_percentile(iv, 504).values,
        })
        out["n_obs_window"] = np.arange(len(out))
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Underlying realized-vol table
# ---------------------------------------------------------------------------

def build_rv_daily(symbol: str, years: Sequence[int]) -> Tuple[pd.DataFrame, dict]:
    """Daily realized-vol table from 1-minute underlying bars.

    REGISTERED RULES:
      * daily OHLC is aggregated from REGULAR-SESSION 1m bars only (09:30
        through the last bar before the real session close from the project
        market calendar -- so early closes are respected and pre/post-market
        bars are excluded);
      * `rv_1m_daily_var` = sum of squared 1-minute log CLOSE-to-CLOSE returns
        INSIDE the session. Close-to-close within the session structurally
        excludes the overnight gap (identical to the Group A definition);
      * `yz_10d` / `yz_20d` = src.features.volatility.yang_zhang_rv on that
        daily OHLC (ANNUALIZED vol, first `window` rows NaN);
      * `har_forecast_var` = src.backtesting.vol.har_rv.har_forecast on
        `rv_1m_daily_var` -- the FROZEN (1,5,22) spec, expanding causal OLS,
        252-session warmup, next-day E[RV] in DAILY-VARIANCE units.
    """
    from src.backtesting.diagnostics.session_bars import (
        load_minute_bars, session_close_schedule)
    from src.backtesting.vol.har_rv import har_forecast
    from src.features.volatility import yang_zhang_rv

    bars = load_minute_bars(symbol, years)
    lo = bars["ny"].min().date().isoformat()
    hi = bars["ny"].max().date().isoformat()
    sched = session_close_schedule(lo, hi)

    b = bars.copy()
    b["session_date"] = b["ny"].dt.date
    b["hhmm"] = b["ny"].dt.strftime("%H:%M")
    b = b[(b["hhmm"] >= "09:30") & (b["hhmm"] <= "15:59")]

    rows = []
    n_no_schedule = 0
    for sess, g in b.groupby("session_date", sort=True):
        rc = sched.get(sess)
        if rc is None:
            n_no_schedule += 1
            continue
        last_label = (rc - pd.Timedelta(minutes=1)).strftime("%H:%M")
        g = g[g["hhmm"] <= last_label]
        if len(g) < 2:
            continue
        px = g["close"].to_numpy(dtype=float)
        r = np.diff(np.log(px))
        rows.append({
            "session_date": sess,
            "open": float(g["open"].iloc[0]), "high": float(g["high"].max()),
            "low": float(g["low"].min()), "close": float(g["close"].iloc[-1]),
            "rv_1m_daily_var": float((r ** 2).sum()), "n_bars": int(len(g)),
            "is_early_close": bool(rc.strftime("%H:%M") < "16:00"),
        })
    d = pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)
    d["yz_10d"] = yang_zhang_rv(d[["open", "high", "low", "close"]], 10).values
    d["yz_20d"] = yang_zhang_rv(d[["open", "high", "low", "close"]], 20).values

    rv = pd.Series(d["rv_1m_daily_var"].values,
                   index=pd.to_datetime(d["session_date"]))
    fc = har_forecast(rv)
    d["har_forecast_var"] = fc.reindex(pd.to_datetime(d["session_date"])).values
    d["root"] = symbol

    # 20-session normalized price range, used by D-048's compression definition.
    hi20 = d["high"].rolling(20, min_periods=20).max()
    lo20 = d["low"].rolling(20, min_periods=20).min()
    d["range_20d_norm"] = ((hi20 - lo20) / d["close"]).values

    stats = {"symbol": symbol, "n_sessions": int(len(d)),
             "n_sessions_no_calendar_entry": int(n_no_schedule),
             "first_session": str(d["session_date"].iloc[0]) if len(d) else None,
             "last_session": str(d["session_date"].iloc[-1]) if len(d) else None,
             "n_har_valid": int(np.isfinite(d["har_forecast_var"]).sum())}
    return d, stats


# ---------------------------------------------------------------------------
# Small shared helpers used by the diagnostics
# ---------------------------------------------------------------------------

def forward_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    """Worst peak-to-trough move over the NEXT `horizon` sessions.

    Measurement of the price series only -- no position, no P&L.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n):
        j = min(i + horizon, n - 1)
        if j <= i:
            continue
        w = c[i:j + 1]
        run_max = np.maximum.accumulate(w)
        out[i] = float((w / run_max - 1.0).min())
    return pd.Series(out, index=close.index, name=f"fwd_mdd_{horizon}")


def describe(x: pd.Series) -> dict:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()),
            "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "p10": float(v.quantile(0.10)), "p25": float(v.quantile(0.25)),
            "p75": float(v.quantile(0.75)), "p90": float(v.quantile(0.90))}
