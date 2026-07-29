"""Wave-0 Group B2 driver -- option-chain STRUCTURE measurements.

DATA-PROPERTY MEASUREMENTS ONLY. No positions, no fills, no equity curves, no
P&L, no performance metrics. See src/backtesting/diagnostics/options_chain_structure.py.

Diagnostics:
  D-003  financing ratio distribution (OPT-003 gate)
  D-005  post-pullback vs unconditional put VRP at 0.30 delta (OPT-005 gate)
  D-029  jade-lizard starvation census (OPT-029 gate)
  D-037  broken-wing-fly no-cost starvation census (OPT-037 gate)
  D-043  pin distance, OpEx week vs control (OPT-043 gate)
  D-047  BLOCKED as specified; supplementary raw-quote 0.05-delta integrity census

Usage:
  python scripts/backtest_scripts/run_wave0_b2_diagnostics.py --diagnostic d003 --root SPY
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from src.backtesting.diagnostics import options_chain_structure as ocs
from src.features.volatility import yang_zhang_rv
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT_DIR = Path("output") / "wave0" / "groupB2"
WIDE_DTE = (0, 10000)


def _write(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / f"{name}.csv"
    df.to_csv(fp, index=False)
    logger.info(f"[+] wrote {len(df):,} rows -> {fp}")
    return fp


def _months(root: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    usable, excluded = ocs.usable_chain_months(root)
    logger.info(f"[+] {root}: {len(usable)} greek-OK partitions on disk, "
                f"{len(excluded)} excluded by census: {excluded}")
    return usable, excluded


def _daily(symbol: str, years: Sequence[int]) -> pd.DataFrame:
    have = sorted({y for y in years if (
        get_local_storage_dir() / "equities" / "sip_split" / "1min"
        / f"symbol={symbol}" / f"year={y}").exists()})
    if not have:
        raise FileNotFoundError(f"[-] no 1m equity bars for {symbol} in {years}")
    daily = ocs.daily_ohlc_from_minutes(symbol, have)
    logger.info(f"[+] {symbol} daily OHLC: {len(daily)} sessions "
                f"{daily.index[0]} .. {daily.index[-1]} (bar years {have[0]}-{have[-1]})")
    return daily


def _regime() -> pd.DataFrame:
    fp = get_local_storage_dir() / "alt_data" / "regime" / "regime_state_daily.parquet"
    df = pd.read_parquet(fp, columns=["date", "regime"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _yz20(daily: pd.DataFrame) -> pd.Series:
    return yang_zhang_rv(daily, window=20, annualization_factor=252)


# --------------------------------------------------------------------------
# D-003 -- financing ratio
# --------------------------------------------------------------------------

def run_d003(root: str) -> None:
    """(1M 0.20D call premium x 3) / (3M 0.15D put cost), monthly.

    PRE-COMMITMENT (registered before measuring): the monthly observation date
    is the MONTHLY OPEX FRIDAY (third Friday); if that session is absent from
    the chain, the latest session strictly before it in that month is used and
    the substitution is COUNTED. Rationale: OPT-003 rolls a monthly call, so
    the natural reset date is the monthly expiry.
    """
    usable, excluded = _months(root)
    rows = []
    obs_sub = 0
    for (y, m) in usable:
        ch = ocs.load_chain_months(root, [(y, m)])
        if ch.height == 0:
            continue
        sessions = sorted(set(ch["session_date"].to_list()))
        anchor = ocs.third_friday(y, m)
        if anchor in sessions:
            obs = anchor
        else:
            prior = [s for s in sessions if s < anchor]
            if not prior:
                continue
            obs = prior[-1]
            obs_sub += 1
        day = ch.filter(pl.col("session_date") == obs)
        call = ocs.select_strike_by_delta(day, "C", 0.20, (25, 40))
        put = ocs.select_strike_by_delta(day, "P", 0.15, (80, 100))
        rec: Dict = {"root": root, "year": y, "month": m, "obs_date": obs,
                     "obs_is_anchor": obs == anchor}
        if call.height == 1:
            rec.update(call_strike=call["strike"][0], call_mid=call["mid"][0],
                       call_delta=call["realized_delta"][0],
                       call_dev=call["delta_dev"][0], call_dte=call["dte"][0],
                       call_expiry=call["expiry"][0],
                       call_ok=bool(call["within_tolerance"][0]),
                       underlying_px=call["underlying_px"][0])
        if put.height == 1:
            rec.update(put_strike=put["strike"][0], put_mid=put["mid"][0],
                       put_delta=put["realized_delta"][0],
                       put_dev=put["delta_dev"][0], put_dte=put["dte"][0],
                       put_expiry=put["expiry"][0],
                       put_ok=bool(put["within_tolerance"][0]))
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["complete"] = df.get("call_mid").notna() & df.get("put_mid").notna() if len(df) else []
    df["in_tolerance"] = (df["call_ok"].astype("boolean").fillna(False)
                          & df["put_ok"].astype("boolean").fillna(False))
    df["ratio"] = np.where(df["complete"] & df["in_tolerance"],
                           3.0 * df["call_mid"] / df["put_mid"], np.nan)
    reg = _regime().set_index("date")["regime"]
    df["regime"] = [reg.get(d) for d in df["obs_date"]]
    _write(df, f"d003_financing_ratio_{root}")

    good = df[df["ratio"].notna()]
    summary = {
        "root": root, "n_months_attempted": len(df),
        "n_months_measured": len(good),
        "n_dropped_missing_leg": int((~df["complete"]).sum()),
        "n_dropped_delta_tolerance": int((df["complete"] & ~df["in_tolerance"]).sum()),
        "n_obs_date_substituted": obs_sub,
        "frac_ge_0p7": float((good["ratio"] >= 0.7).mean()) if len(good) else np.nan,
        "gate_pass": bool(len(good) and (good["ratio"] >= 0.7).mean() >= 0.60),
        "excluded_partitions": str(excluded),
    }
    summary.update({f"ratio_{k}": v for k, v in ocs.describe(good["ratio"]).items()})
    _write(pd.DataFrame([summary]), f"d003_summary_{root}")

    by_year = good.groupby("year")["ratio"].agg(
        n="size", median="median", mean="mean",
        frac_ge_0p7=lambda s: float((s >= 0.7).mean()))
    _write(by_year.reset_index(), f"d003_by_year_{root}")
    by_reg = good.groupby("regime")["ratio"].agg(
        n="size", median="median", mean="mean",
        frac_ge_0p7=lambda s: float((s >= 0.7).mean()))
    _write(by_reg.reset_index(), f"d003_by_regime_{root}")
    dev = pd.DataFrame({"leg": ["call_0.20", "put_0.15"]})
    dev["mean_abs_dev"] = [df["call_dev"].mean(), df["put_dev"].mean()]
    dev["median_abs_dev"] = [df["call_dev"].median(), df["put_dev"].median()]
    dev["p95_abs_dev"] = [df["call_dev"].quantile(0.95), df["put_dev"].quantile(0.95)]
    _write(dev, f"d003_delta_quality_{root}")
    logger.info(f"[+] D-003 {root}: frac>=0.7 = {summary['frac_ge_0p7']}, "
                f"gate_pass={summary['gate_pass']}")


# --------------------------------------------------------------------------
# D-005 -- put VRP, post-pullback vs unconditional
# --------------------------------------------------------------------------

def run_d005(root: str) -> None:
    """VRP = implied_vol(t) - Yang-Zhang RV over the option's forward life.

    Forward-looking BY DESIGN: this is a diagnostic comparison, not a tradable
    signal, and it produces no P&L.

    PRE-COMMITMENT: the pullback trigger is PRIMARY = close(t) <= 0.98*close(t-1)
    (a 1-session drop, the natural reading of "after a >= 2% underlying
    pullback"); SECONDARY (clearly labelled) = close(t) <= 0.98 * trailing
    20-session max close. Both are reported; the gate is read off the PRIMARY.
    """
    usable, excluded = _months(root)
    years = sorted({y for y, _ in usable})
    daily = _daily(root, range(min(years), max(years) + 2))

    recs = []
    all_sessions = set()
    for (y, m) in usable:
        ch = ocs.load_chain_months(root, [(y, m)])
        if ch.height == 0:
            continue
        all_sessions.update(ch["session_date"].unique().to_list())
        sel = ocs.select_strike_by_delta(ch, "P", 0.30, (21, 35))
        if sel.height == 0:
            continue
        recs.append(sel.select(["session_date", "expiry", "strike", "dte", "mid",
                                "implied_vol", "realized_delta", "delta_dev",
                                "within_tolerance", "underlying_px",
                                "spread_rel"]).to_pandas())
    sel = pd.concat(recs, ignore_index=True).sort_values("session_date")
    n_all = len(sel)
    sel["iv_finite"] = np.isfinite(sel["implied_vol"])
    n_iv_nan = int((~sel["iv_finite"]).sum())
    n_tol = int((~sel["within_tolerance"]).sum())
    sel = sel[sel["iv_finite"] & sel["within_tolerance"]].copy()

    fwd = [ocs.realized_vol_forward(daily, d, int(h))
           for d, h in zip(sel["session_date"], sel["dte"])]
    sel["rv_forward"] = fwd
    n_fwd_drop = int(sel["rv_forward"].isna().sum())
    sel = sel[sel["rv_forward"].notna()].copy()
    sel["vrp"] = sel["implied_vol"] - sel["rv_forward"]

    closes = daily["close"]
    pb1 = pd.Series(ocs.pullback_flags_1session(closes.to_numpy(), 0.02),
                    index=closes.index)
    pb2 = pd.Series(ocs.pullback_flags_trailing_max(closes.to_numpy(), 20, 0.02),
                    index=closes.index)
    sel["pullback_1session"] = [bool(pb1.get(d, False)) for d in sel["session_date"]]
    sel["pullback_trailmax20"] = [bool(pb2.get(d, False)) for d in sel["session_date"]]
    _write(sel, f"d005_put_vrp_{root}")

    out = []
    for label, col in (("primary_1session", "pullback_1session"),
                       ("secondary_trailmax20", "pullback_trailmax20")):
        v = sel["vrp"].to_numpy(dtype=float)
        mask = sel[col].to_numpy(dtype=bool)
        cond, unc = v[mask], v
        dte_med = float(np.median(sel["dte"]))
        t, p = stats.ttest_ind(cond, v[~mask], equal_var=False) if mask.sum() > 1 \
            else (np.nan, np.nan)
        pt, lo, hi = ocs.bootstrap_diff_ci(v, mask, 10000, 42, block=None)
        ptb, lob, hib = ocs.bootstrap_diff_ci(v, mask, 10000, 42, block=int(dte_med))
        out.append({
            "root": root, "trigger": label,
            "n_conditional": int(mask.sum()), "n_unconditional": int(len(v)),
            "mean_cond": float(cond.mean()) if mask.sum() else np.nan,
            "mean_unc": float(unc.mean()),
            "median_cond": float(np.median(cond)) if mask.sum() else np.nan,
            "median_unc": float(np.median(unc)),
            "diff_mean": pt, "diff_median": (float(np.median(cond)) - float(np.median(unc)))
            if mask.sum() else np.nan,
            "welch_t_vs_nontrigger": float(t), "welch_p": float(p),
            "iid_boot_lo95": lo, "iid_boot_hi95": hi,
            "block_len": int(dte_med), "block_boot_lo95": lob, "block_boot_hi95": hib,
            "gate_pass_mean_cond_gt_unc": bool(mask.sum() and cond.mean() > unc.mean()),
            "gate_pass_median_cond_gt_unc": bool(
                mask.sum() and np.median(cond) > np.median(unc)),
            "diff_mean_ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                               and (lo > 0 or hi < 0)),
            "diff_mean_blockci_excludes_zero": bool(
                np.isfinite(lob) and np.isfinite(hib) and (lob > 0 or hib < 0)),
            "n_chain_sessions": len(all_sessions),
            "n_sessions_no_selection": len(all_sessions) - n_all,
            "n_selections_raw": n_all, "n_dropped_iv_nan": n_iv_nan,
            "n_dropped_delta_tolerance": n_tol, "n_dropped_forward_edge": n_fwd_drop,
            "excluded_partitions": str(excluded),
        })
    _write(pd.DataFrame(out), f"d005_summary_{root}")

    sel["year"] = pd.to_datetime(sel["session_date"]).dt.year
    by_year = sel.groupby("year").apply(lambda g: pd.Series({
        "n": len(g), "mean_vrp_all": g["vrp"].mean(),
        "n_cond": int(g["pullback_1session"].sum()),
        "mean_vrp_cond": g.loc[g["pullback_1session"], "vrp"].mean(),
    }), include_groups=False)
    _write(by_year.reset_index(), f"d005_by_year_{root}")
    logger.info(f"[+] D-005 {root}: done")


# --------------------------------------------------------------------------
# D-029 -- jade lizard starvation census
# --------------------------------------------------------------------------

def run_d029(root: str) -> None:
    """credit(0.25P) + credit(0.25C) - debit(0.15C) > width(0.15C - 0.25C).

    PRE-COMMITMENT: all three legs share ONE expiry per session (the 40-50 DTE
    expiry nearest 45); entry cadence for the gate is NON-OVERLAPPING with a
    45-CALENDAR-DAY skip after each entry. Both the non-overlapping entry rate
    and the raw qualifying fraction are reported.
    """
    usable, excluded = _months(root)
    recs = []
    all_sessions = set()
    for (y, m) in usable:
        ch = ocs.load_chain_months(root, [(y, m)])
        if ch.height == 0:
            continue
        all_sessions.update(ch["session_date"].unique().to_list())
        one = ocs.select_expiry_by_dte(ocs.eligible_quotes(ch), 40, 50)
        if one.height == 0:
            continue
        p25 = ocs.select_strike_by_delta(one, "P", 0.25, WIDE_DTE)
        c25 = ocs.select_strike_by_delta(one, "C", 0.25, WIDE_DTE)
        c15 = ocs.select_strike_by_delta(one, "C", 0.15, WIDE_DTE)
        if min(p25.height, c25.height, c15.height) == 0:
            continue
        j = (p25.select(["session_date", "expiry", "dte", "underlying_px",
                         pl.col("strike").alias("p25_strike"),
                         pl.col("mid").alias("p25_mid"),
                         pl.col("realized_delta").alias("p25_delta"),
                         pl.col("delta_dev").alias("p25_dev"),
                         pl.col("within_tolerance").alias("p25_ok")])
             .join(c25.select(["session_date",
                               pl.col("strike").alias("c25_strike"),
                               pl.col("mid").alias("c25_mid"),
                               pl.col("realized_delta").alias("c25_delta"),
                               pl.col("delta_dev").alias("c25_dev"),
                               pl.col("within_tolerance").alias("c25_ok")]),
                   on="session_date", how="inner")
             .join(c15.select(["session_date",
                               pl.col("strike").alias("c15_strike"),
                               pl.col("mid").alias("c15_mid"),
                               pl.col("realized_delta").alias("c15_delta"),
                               pl.col("delta_dev").alias("c15_dev"),
                               pl.col("within_tolerance").alias("c15_ok")]),
                   on="session_date", how="inner"))
        recs.append(j.to_pandas())

    df = pd.concat(recs, ignore_index=True).sort_values("session_date")
    n_raw = len(df)
    df["all_in_tolerance"] = df["p25_ok"] & df["c25_ok"] & df["c15_ok"]
    df["degenerate_width"] = df["c15_strike"] <= df["c25_strike"]
    n_tol = int((~df["all_in_tolerance"]).sum())
    n_deg = int(df["degenerate_width"].sum())
    ok = df[df["all_in_tolerance"] & ~df["degenerate_width"]].copy()
    ok["credit"] = ok["p25_mid"] + ok["c25_mid"] - ok["c15_mid"]
    ok["width"] = ok["c15_strike"] - ok["c25_strike"]
    ok["qualifies"] = ok["credit"] > ok["width"]
    ok["credit_minus_width"] = ok["credit"] - ok["width"]
    _write(ok, f"d029_jade_lizard_{root}")

    entries = ocs.non_overlapping_entries(list(ok["session_date"]),
                                          list(ok["qualifies"]), 45)
    ey = ocs.entries_per_year(entries, list(ok["session_date"]))
    ok["year"] = pd.to_datetime(ok["session_date"]).dt.year
    ey = ey.merge(ok.groupby("year")["qualifies"].agg(
        n_sessions="size", n_qualifying="sum").reset_index(), on="year", how="left")
    ey["raw_qualify_frac"] = ey["n_qualifying"] / ey["n_sessions"]
    _write(ey, f"d029_by_year_{root}")

    full_years = ey[ey["sessions"] >= 200]
    summary = {
        "root": root, "n_chain_sessions": len(all_sessions),
        "n_sessions_no_complete_selection": len(all_sessions) - n_raw,
        "n_sessions_raw": n_raw,
        "n_dropped_delta_tolerance": n_tol, "n_dropped_degenerate_width": n_deg,
        "n_sessions_measured": len(ok),
        "raw_qualify_frac": float(ok["qualifies"].mean()),
        "n_nonoverlapping_entries": len(entries),
        "entries_per_year_mean_all": float(ey["entries"].mean()),
        "entries_per_year_mean_fullyears": float(full_years["entries"].mean())
        if len(full_years) else np.nan,
        "gate_pass_ge6_per_yr": bool(len(full_years) and full_years["entries"].mean() >= 6),
        "excluded_partitions": str(excluded),
    }
    summary.update({f"cmw_{k}": v for k, v in
                    ocs.describe(ok["credit_minus_width"]).items()})
    _write(pd.DataFrame([summary]), f"d029_summary_{root}")
    logger.info(f"[+] D-029 {root}: entries/yr(full)={summary['entries_per_year_mean_fullyears']}")


# --------------------------------------------------------------------------
# D-037 -- broken-wing-fly no-cost census
# --------------------------------------------------------------------------

def run_d037(root: str) -> None:
    """Put BWB net entry cost <= 0 census.

    PRE-COMMITMENT (geometry; the registered text gives only two sigma levels
    for three strikes, so this is an explicit operationalization):
      sigma_dte = yang_zhang_rv(underlying, 20d, ann=252) * sqrt(dte/252)
      near long wing K1  = strike nearest  spot - 1.00 * sigma_dte * spot
      far  long wing K3  = strike nearest  spot - 2.50 * sigma_dte * spot
      short body (x2) K2 = strike nearest  spot - (1.75 + 0.25*sign(drift20))
                                                   * sigma_dte * spot
      drift20 = log(close_t / close_{t-20}); a positive drift pushes the body
      AWAY from spot, i.e. the wide wing sits opposite the drift sign.
      net_cost = mid(K1) - 2*mid(K2) + mid(K3); qualifying <=> net_cost <= 0.
    Degenerate structures (any two legs snapping to the same strike) are
    QUARANTINED and COUNTED, never repaired. Entry cadence = non-overlapping
    with a 38-calendar-day skip (the 30-45 DTE window midpoint).
    """
    usable, excluded = _months(root)
    years = sorted({y for y, _ in usable})
    daily = _daily(root, range(min(years), max(years) + 1))
    yz = _yz20(daily)
    logret = np.log(daily["close"] / daily["close"].shift(20))

    recs = []
    all_sessions = set()
    for (y, m) in usable:
        ch = ocs.load_chain_months(root, [(y, m)])
        if ch.height == 0:
            continue
        all_sessions.update(ch["session_date"].unique().to_list())
        one = ocs.select_expiry_by_dte(ocs.eligible_quotes(ch), 30, 45)
        one = one.filter(pl.col("right") == "P")
        if one.height == 0:
            continue
        per = (one.group_by("session_date")
               .agg(pl.col("dte").first(), pl.col("expiry").first(),
                    pl.col("underlying_px").drop_nulls().first().alias("spot"))
               .sort("session_date")).to_pandas()
        per["session_date"] = pd.to_datetime(per["session_date"])
        per["yz20"] = [yz.get(d, np.nan) for d in per["session_date"]]
        per["drift20"] = [logret.get(d, np.nan) for d in per["session_date"]]
        per = per[np.isfinite(per["yz20"]) & np.isfinite(per["drift20"])
                  & per["spot"].notna()].copy()
        if per.empty:
            continue
        per["sigma_dte"] = [ocs.sigma_for_dte(v, d)
                            for v, d in zip(per["yz20"], per["dte"])]
        sgn = np.sign(per["drift20"])
        per["k1"] = per["spot"] * (1.0 - 1.00 * per["sigma_dte"])
        per["k2"] = per["spot"] * (1.0 - (1.75 + 0.25 * sgn) * per["sigma_dte"])
        per["k3"] = per["spot"] * (1.0 - 2.50 * per["sigma_dte"])
        legs = {}
        for label in ("k1", "k2", "k3"):
            tgt = pl.DataFrame({
                "session_date": [d.date() for d in per["session_date"]],
                "target_strike": list(per[label].astype(float))})
            legs[label] = ocs.select_strike_nearest_price(
                one, "P", tgt, WIDE_DTE, label).to_pandas()
        j = per[["session_date", "expiry", "dte", "spot", "yz20", "drift20",
                 "sigma_dte", "k1", "k2", "k3"]].copy()
        for label in ("k1", "k2", "k3"):
            leg = legs[label]
            leg["session_date"] = pd.to_datetime(leg["session_date"])
            j = j.merge(leg, on="session_date", how="left")
        recs.append(j)

    df = pd.concat(recs, ignore_index=True).sort_values("session_date")
    n_raw = len(df)
    have = df[["k1_mid", "k2_mid", "k3_mid"]].notna().all(axis=1)
    n_missing = int((~have).sum())
    df["degenerate"] = (df["k1_strike"] == df["k2_strike"]) | \
                       (df["k2_strike"] == df["k3_strike"]) | \
                       (df["k1_strike"] == df["k3_strike"])
    n_deg = int((have & df["degenerate"]).sum())
    ok = df[have & ~df["degenerate"]].copy()
    ok["net_cost"] = ok["k1_mid"] - 2.0 * ok["k2_mid"] + ok["k3_mid"]
    ok["qualifies"] = ok["net_cost"] <= 0.0
    _write(ok, f"d037_bwb_{root}")

    entries = ocs.non_overlapping_entries(list(ok["session_date"]),
                                          list(ok["qualifies"]), 38)
    ey = ocs.entries_per_year(entries, list(ok["session_date"]))
    ok["year"] = pd.to_datetime(ok["session_date"]).dt.year
    ey = ey.merge(ok.groupby("year")["qualifies"].agg(
        n_sessions="size", n_qualifying="sum").reset_index(), on="year", how="left")
    ey["raw_qualify_frac"] = ey["n_qualifying"] / ey["n_sessions"]
    _write(ey, f"d037_by_year_{root}")

    full_years = ey[ey["sessions"] >= 200]
    # far-wing quote quality: 2.5-sigma leg is deep OTM and OUT OF SPEC (M7 absent)
    far_valid = float(df["k3_mid"].notna().mean())
    summary = {
        "root": root, "n_chain_sessions": len(all_sessions),
        "n_sessions_no_complete_selection": len(all_sessions) - n_raw,
        "n_sessions_raw": n_raw,
        "n_dropped_missing_leg_quote": n_missing,
        "n_dropped_degenerate_strikes": n_deg,
        "n_sessions_measured": len(ok),
        "raw_qualify_frac": float(ok["qualifies"].mean()),
        "n_nonoverlapping_entries": len(entries),
        "entries_per_year_mean_all": float(ey["entries"].mean()),
        "entries_per_year_mean_fullyears": float(full_years["entries"].mean())
        if len(full_years) else np.nan,
        "gate_pass_ge6_per_yr": bool(len(full_years) and full_years["entries"].mean() >= 6),
        "farwing_valid_quote_frac": far_valid,
        "farwing_median_spread_rel": float(ok["k3_spread_rel"].median()),
        "farwing_median_abs_delta": float(ok["k3_delta"].abs().median()),
        "farwing_frac_below_0p10_delta": float((ok["k3_delta"].abs() < 0.10).mean()),
        "excluded_partitions": str(excluded),
    }
    summary.update({f"netcost_{k}": v for k, v in ocs.describe(ok["net_cost"]).items()})
    _write(pd.DataFrame([summary]), f"d037_summary_{root}")
    logger.info(f"[+] D-037 {root}: entries/yr(full)={summary['entries_per_year_mean_fullyears']}")


# --------------------------------------------------------------------------
# D-043 -- pin distance, OpEx week vs control
# --------------------------------------------------------------------------

def _monthly_expiry_map(expiries: Sequence[date]) -> Dict[Tuple[int, int], date]:
    """Actual monthly expiry per (year, month): the chain expiry closest to the
    third-Friday anchor within +/-3 days (handles holiday shifts)."""
    out: Dict[Tuple[int, int], date] = {}
    exp = sorted(set(expiries))
    for (y, m) in sorted({(e.year, e.month) for e in exp}):
        anchor = ocs.third_friday(y, m)
        cands = [e for e in exp if abs((e - anchor).days) <= 3]
        if cands:
            out[(y, m)] = min(cands, key=lambda e: (abs((e - anchor).days), e))
    return out


def run_d043(root: str) -> None:
    """pin_distance = |spot - max-OI strike|, OpEx week vs control weeks.

    T-1-LAGGED OI ONLY (`oi_eod_lag1`, restitched across month seams by
    re-running add_eod_lag over month+prior-month). Same-session oi_eod is a
    hard lookahead leak and is never read.

    PRE-COMMITMENTS:
      * OI is SUMMED across calls and puts at each strike;
      * OpEx week = the Mon-Fri week containing the monthly expiry; the
        referenced expiry is that expiring monthly. Control weeks reference the
        NEAREST monthly expiry with expiry >= session_date;
      * PRIMARY metric is the vol-normalized distance
        |spot - K| / (spot * sigma_dte); the spot-fraction and raw-dollar
        versions are secondary;
      * "materially smaller" is operationalized as: OpEx-week MEDIAN normalized
        distance is at least 20% below the control median AND the Mann-Whitney
        two-sided p < 0.05.
    """
    usable, excluded = _months(root)
    years = sorted({y for y, _ in usable})
    daily = _daily(root, range(min(years), max(years) + 1))
    yz = _yz20(daily)

    lag_stats = {"rows": 0, "lag_null": 0, "recovered_by_stitch": 0}
    recs = []
    for i, (y, m) in enumerate(usable):
        prev = usable[i - 1] if i > 0 else None
        load = [prev, (y, m)] if prev is not None else [(y, m)]
        cols = ["root", "session_date", "expiry", "strike", "right",
                "quote_valid", "dte", "underlying_px", "oi_eod", "oi_eod_lag1"]
        raw = ocs.load_chain_months(root, load, columns=cols)
        if raw.height == 0:
            continue
        within = raw.filter(pl.col("session_date").dt.year() == y).filter(
            pl.col("session_date").dt.month() == m)
        n_within_null = int(within["oi_eod_lag1"].is_null().sum())
        st = ocs.load_chain_months(root, load, columns=cols, restitch_oi_lag=True)
        st = st.filter((pl.col("session_date").dt.year() == y)
                       & (pl.col("session_date").dt.month() == m))
        lag_stats["rows"] += st.height
        n_after = int(st["oi_eod_lag1"].is_null().sum())
        lag_stats["lag_null"] += n_after
        lag_stats["recovered_by_stitch"] += n_within_null - n_after

        st = st.filter(pl.col("oi_eod_lag1").is_not_null()
                       & (pl.col("oi_eod_lag1") > 0))
        if st.height == 0:
            continue
        mexp = _monthly_expiry_map(st["expiry"].unique().to_list())
        monthlies = sorted(mexp.values())
        sessions = sorted(set(st["session_date"].to_list()))
        opex_set = set(monthlies)
        target = {}
        for d in sessions:
            if ocs.is_in_opex_week(d, opex_set):
                same = [e for e in monthlies if ocs.week_monday(e) == ocs.week_monday(d)]
                target[d] = same[0]
            else:
                fwd = [e for e in monthlies if e >= d]
                if fwd:
                    target[d] = fwd[0]
        tgt = pl.DataFrame({"session_date": list(target.keys()),
                            "target_expiry": list(target.values())})
        st = st.join(tgt, on="session_date", how="inner").filter(
            pl.col("expiry") == pl.col("target_expiry"))
        if st.height == 0:
            continue
        agg = (st.group_by(["session_date", "expiry", "strike"])
               .agg(pl.col("oi_eod_lag1").sum().alias("oi_sum"),
                    pl.col("dte").first(),
                    pl.col("underlying_px").drop_nulls().first().alias("spot")))
        pick = (agg.sort(["session_date", "oi_sum", "strike"],
                         descending=[False, True, False])
                .group_by("session_date", maintain_order=True).first()
                .sort("session_date")).to_pandas()
        recs.append(pick)

    df = pd.concat(recs, ignore_index=True).sort_values("session_date")
    df = df[df["spot"].notna()].copy()
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["yz20"] = [yz.get(pd.Timestamp(d), np.nan) for d in df["session_date"]]
    n_before_yz = len(df)
    df = df[np.isfinite(df["yz20"])].copy()
    lag_stats["dropped_no_underlying_yz20"] = n_before_yz - len(df)
    df["sigma_dte"] = [ocs.sigma_for_dte(v, max(d, 1))
                       for v, d in zip(df["yz20"], df["dte"])]
    df["dist_usd"] = (df["spot"] - df["strike"]).abs()
    df["dist_frac"] = df["dist_usd"] / df["spot"]
    df["dist_sigma"] = df["dist_frac"] / df["sigma_dte"]
    monthlies_all = sorted(set(df["expiry"]))
    mexp_all = _monthly_expiry_map(monthlies_all)
    opex_all = set(mexp_all.values())
    df["opex_week"] = [ocs.is_in_opex_week(d, opex_all) for d in df["session_date"]]
    df["weekday"] = pd.to_datetime(df["session_date"]).dt.day_name()
    df["year"] = pd.to_datetime(df["session_date"]).dt.year
    _write(df, f"d043_pin_distance_{root}")

    def _cmp(sub: pd.DataFrame, tag: str) -> Dict:
        a = sub.loc[sub["opex_week"], "dist_sigma"].to_numpy(dtype=float)
        b = sub.loc[~sub["opex_week"], "dist_sigma"].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return {"group": tag, "n_opex": len(a), "n_control": len(b)}
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        med_a, med_b = float(np.median(a)), float(np.median(b))
        return {
            "group": tag, "n_opex": len(a), "n_control": len(b),
            "median_sigma_opex": med_a, "median_sigma_control": med_b,
            "ratio_median": med_a / med_b if med_b else np.nan,
            "mean_sigma_opex": float(a.mean()), "mean_sigma_control": float(b.mean()),
            "median_frac_opex": float(sub.loc[sub["opex_week"], "dist_frac"].median()),
            "median_frac_control": float(sub.loc[~sub["opex_week"], "dist_frac"].median()),
            "median_usd_opex": float(sub.loc[sub["opex_week"], "dist_usd"].median()),
            "median_usd_control": float(sub.loc[~sub["opex_week"], "dist_usd"].median()),
            "median_dte_opex": float(sub.loc[sub["opex_week"], "dte"].median()),
            "median_dte_control": float(sub.loc[~sub["opex_week"], "dte"].median()),
            "mannwhitney_p": float(p),
            "gate_pass": bool(med_b and med_a <= 0.8 * med_b and p < 0.05),
            "gate_pass_on_spot_fraction": bool(
                float(sub.loc[~sub["opex_week"], "dist_frac"].median())
                and float(sub.loc[sub["opex_week"], "dist_frac"].median())
                <= 0.8 * float(sub.loc[~sub["opex_week"], "dist_frac"].median())
                and float(stats.mannwhitneyu(
                    sub.loc[sub["opex_week"], "dist_frac"].to_numpy(dtype=float),
                    sub.loc[~sub["opex_week"], "dist_frac"].to_numpy(dtype=float),
                    alternative="two-sided")[1]) < 0.05),
        }

    rows = [_cmp(df, "all_sessions")]
    for wd in ("Wednesday", "Friday"):
        rows.append(_cmp(df[df["weekday"] == wd], wd))
    dte_lo, dte_hi = 0, 4
    rows.append(_cmp(df[(df["dte"] >= dte_lo) & (df["dte"] <= dte_hi)],
                     f"dte_matched_{dte_lo}_{dte_hi}"))
    rows.append(_cmp(df[df["year"] <= 2020], "pre2021"))
    rows.append(_cmp(df[df["year"] >= 2021], "post2020"))
    comp = pd.DataFrame(rows)
    comp["lag_rows"] = lag_stats["rows"]
    comp["lag_null_rows_after_stitch"] = lag_stats["lag_null"]
    comp["lag_rows_recovered_by_stitch"] = lag_stats["recovered_by_stitch"]
    comp["sessions_dropped_no_underlying_yz20"] = lag_stats["dropped_no_underlying_yz20"]
    comp["excluded_partitions"] = str(excluded)
    _write(comp, f"d043_summary_{root}")

    by_year = df.groupby(["year", "opex_week"])["dist_sigma"].agg(
        n="size", median="median", mean="mean").reset_index()
    _write(by_year, f"d043_by_year_{root}")
    dte_dist = df.groupby("opex_week")["dte"].describe().reset_index()
    _write(dte_dist, f"d043_dte_distribution_{root}")
    logger.info(f"[+] D-043 {root}: done")


# --------------------------------------------------------------------------
# D-047/030 supplementary -- raw-quote 0.05-delta integrity census
# --------------------------------------------------------------------------

def run_d047_supplementary(root: str) -> None:
    """NOT the D-047/030 gate. The registered diagnostic compares a SMOOTHED
    surface (module M7) against raw quotes; M7 does not exist and ORATS was
    deferred, so the gate CANNOT RUN. This is a raw-quote availability census
    at ~0.05 delta that will be an INPUT to the real diagnostic once M7 lands.

    PRE-COMMITMENT: DTE window 21-45, one expiry per session (nearest 33 DTE).
    Strike chosen on |delta| nearest 0.05 over ALL rows carrying a finite
    delta -- including rows whose quote is invalid -- so quote availability is
    measurable rather than assumed away.
    """
    usable, excluded = _months(root)
    recs, arb = [], []
    for (y, m) in usable:
        ch = ocs.load_chain_months(root, [(y, m)])
        if ch.height == 0:
            continue
        c = ch.filter(pl.col("delta").is_not_null() & pl.col("delta").is_not_nan()
                      & pl.col("delta").is_finite())
        one = ocs.select_expiry_by_dte(c, 21, 45)
        if one.height == 0:
            continue
        for right in ("C", "P"):
            r = one.filter(pl.col("right") == right)
            if r.height == 0:
                continue
            r = r.with_columns((pl.col("delta").abs() - 0.05).abs().alias("dev"))
            pick = (r.sort(["session_date", "dev", "strike"])
                    .group_by("session_date", maintain_order=True).first()
                    .sort("session_date")).to_pandas()
            pick["right"] = right
            recs.append(pick)
        # static no-arbitrage on the valid-quote strike ladder of that expiry
        v = one.filter(pl.col("quote_valid"))
        for right in ("C", "P"):
            sub = v.filter(pl.col("right") == right).sort(["session_date", "strike"])
            if sub.height == 0:
                continue
            p = sub.to_pandas()
            for d, g in p.groupby("session_date"):
                if len(g) < 3:
                    continue
                dm = np.diff(g["mid"].to_numpy(dtype=float))
                bad = int((dm > 1e-9).sum()) if right == "C" else int((dm < -1e-9).sum())
                arb.append({"session_date": d, "right": right,
                            "n_pairs": len(dm), "n_violations": bad,
                            "expiry": g["expiry"].iloc[0]})

    df = pd.concat(recs, ignore_index=True)
    df["year"] = pd.to_datetime(df["session_date"]).dt.year
    _write(df, f"d047supp_lowdelta_census_{root}")
    arbdf = pd.DataFrame(arb)
    _write(arbdf.groupby("right").agg(
        sessions=("session_date", "nunique"), pairs=("n_pairs", "sum"),
        violations=("n_violations", "sum")).reset_index(),
        f"d047supp_noarb_{root}")

    rows = []
    for right, g in df.groupby("right"):
        valid = g["quote_valid"].astype(bool)
        rows.append({
            "root": root, "right": right, "n_sessions": len(g),
            "median_abs_delta_selected": float(g["delta"].abs().median()),
            "frac_valid_two_sided_quote": float(valid.mean()),
            "median_spread_rel": float(g.loc[valid, "spread_rel"].median()),
            "iqr_lo_spread_rel": float(g.loc[valid, "spread_rel"].quantile(0.25)),
            "iqr_hi_spread_rel": float(g.loc[valid, "spread_rel"].quantile(0.75)),
            "zero_bid_frac": float((g["bid"].fillna(0.0) <= 0.0).mean()),
            "excluded_partitions": str(excluded),
        })
    _write(pd.DataFrame(rows), f"d047supp_summary_{root}")
    by_year = df.groupby(["year", "right"]).agg(
        n=("session_date", "size"),
        frac_valid=("quote_valid", "mean")).reset_index()
    _write(by_year, f"d047supp_by_year_{root}")
    logger.info(f"[+] D-047 supplementary {root}: done")


DIAGS = {"d003": run_d003, "d005": run_d005, "d029": run_d029,
         "d037": run_d037, "d043": run_d043, "d047supp": run_d047_supplementary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic", required=True, choices=sorted(DIAGS))
    ap.add_argument("--root", required=True)
    args = ap.parse_args(argv)
    with RunStatus("wave0_b2", meta={"diagnostic": args.diagnostic,
                                     "root": args.root}) as status:
        status.heartbeat(note=f"{args.diagnostic} {args.root} start")
        DIAGS[args.diagnostic](args.root)
        status.heartbeat(note=f"{args.diagnostic} {args.root} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
