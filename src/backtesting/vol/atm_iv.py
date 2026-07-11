"""Daily ATM implied volatility for ES/NQ from databento futures-option prints.

Per day: pick the expiry nearest target_dte (constant-maturity term point), take
the most-active near-ATM strikes by volume, volume-weight their session close,
invert Black-76 against the day's front-future close (underlying) and the FRED
DFF rate. Validate the ES series against VIX before any downstream use.
"""
from __future__ import annotations

from datetime import date
import calendar

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

from src.settings import get_local_storage_dir
from src.data.rates.fred_reader import get_fred_series
from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.backtesting.vol.option_symbol import parse_option_symbol
from src.utils.logger import get_logger

logger = get_logger()

_N_ATM_STRIKES = 3


def _b76_price(F: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if right == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_iv(price: float, F: float, K: float, T: float, r: float, right: str) -> float:
    """Implied vol via bounded Brent root-find. NaN if no solution / price below intrinsic."""
    if price <= 0 or T <= 0 or F <= 0 or K <= 0:
        return float("nan")
    intrinsic = np.exp(-r * T) * (max(F - K, 0.0) if right == "C" else max(K - F, 0.0))
    if price < intrinsic - 1e-6:
        return float("nan")
    try:
        return float(brentq(lambda s: _b76_price(F, K, T, r, s, right) - price, 1e-3, 5.0))
    except ValueError:
        return float("nan")


def _options_month(y: int, m: int) -> pl.DataFrame | None:
    fp = (get_local_storage_dir() / "futures" / "databento" / "options_1min"
          / f"year={y}" / f"month={m}" / "data.parquet")
    if not fp.exists():
        return None
    return pl.read_parquet(fp)


def _daily_bars(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse 1-min prints to one row per (day, symbol): session close + summed volume."""
    df = df.with_columns(pl.col("timestamp").dt.date().alias("d"))
    return (
        df.sort("timestamp")
        .group_by(["d", "symbol"])
        .agg(pl.col("close").last().alias("close"), pl.col("volume").sum().alias("volume"))
    )


def _atm_iv_for_day(d: date, root: str, F: float, r: float, day_rows: list[dict],
                     target_dte: int) -> float:
    cands = []
    for row in day_rows:
        o = parse_option_symbol(row["symbol"], ref_year=d.year)
        if o is None or o.root != root:
            continue
        exp = date(o.expiry_year, o.expiry_month, calendar.monthrange(o.expiry_year, o.expiry_month)[1])
        dte = (exp - d).days
        if dte <= 5:
            continue
        cands.append((abs(dte - target_dte), abs(o.strike - F), dte, o, row["close"], row["volume"]))
    if not cands:
        return float("nan")

    best_dte_gap = min(c[0] for c in cands)
    near = [c for c in cands if c[0] == best_dte_gap]
    best_dte = near[0][2]
    same_expiry = [c for c in cands if c[2] == best_dte]
    same_expiry.sort(key=lambda c: (c[1], -c[5]))  # nearest ATM strike, then most volume
    picks = same_expiry[:_N_ATM_STRIKES]

    T = best_dte / 365.0
    ivs, wts = [], []
    for _, _, _, o, close, vol in picks:
        iv = black76_iv(close, F, o.strike, T, r, o.right)
        if not np.isnan(iv):
            ivs.append(iv)
            wts.append(max(vol, 1.0))
    if not ivs:
        return float("nan")
    return float(np.average(ivs, weights=wts))


def atm_iv_series(root: str, start: date, end: date, target_dte: int = 30) -> pd.Series:
    """Date-indexed daily ATM implied vol for `root` over [start, end].

    The Black-76 underlying F is sourced from the RAW (unadjusted) front-future
    daily close, not the ratio-adjusted continuous close. Option strikes/prices
    are quoted against the real futures level that day; the ratio-adjustment
    back-splices historical price levels for pct_change continuity and is off
    by up to ~7% (2020 ES) versus the real level, which corrupts near-ATM
    strike selection, the intrinsic bound, and the resulting IV.
    """
    raw_daily = ContinuousContractDataLoader().aggregate_to_daily(
        root, method="raw", start=start, end=end,
    )
    raw_pd = raw_daily.to_pandas()
    raw_pd["d"] = pd.to_datetime(raw_pd["timestamp"]).dt.date
    underlying = raw_pd.set_index("d")["close"]

    out: dict[pd.Timestamp, float] = {}
    months = pd.period_range(start, end, freq="M")
    for period in months:
        y, m = period.year, period.month
        raw = _options_month(y, m)
        if raw is None:
            continue
        daily = _daily_bars(raw)
        for (d,), day_df in daily.group_by(["d"]):
            if d < start or d > end:
                continue
            if d not in underlying.index or np.isnan(underlying.loc[d]):
                continue
            F = float(underlying.loc[d])
            try:
                r = get_fred_series("DFF", d) / 100.0
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"[atm_iv_series] no DFF rate for {d}: {e}")
                continue
            iv = _atm_iv_for_day(d, root, F, r, day_df.to_dicts(), target_dte)
            if not np.isnan(iv):
                out[pd.Timestamp(d)] = iv

    return pd.Series(out, name=f"{root}_iv_atm", dtype=float).sort_index()


def validate_iv_against_vix(es_iv: pd.Series) -> dict:
    """Sanity-check an ES ATM-IV series against the VX1 futures-implied vol proxy.

    Returns dict with `corr`, `median_ratio`, `ok` (bool: corr > 0.6 and
    0.5 < median_ratio < 2.0). This is a load-bearing data-quality gate: a
    failing result means the IV series is not fit for downstream use and
    MUST be reported as a stop, not worked around by loosening thresholds.
    """
    curve = pl.read_parquet(get_local_storage_dir() / "alt_data" / "vix" / "vx_curve.parquet").to_pandas()
    curve["date"] = pd.to_datetime(curve["date"]).dt.date
    vix = curve.set_index("date")["vx1_settle"] / 100.0  # VX1 as a vol proxy (fractional)
    vix.index = pd.to_datetime(vix.index)

    es_iv = es_iv.copy()
    es_iv.index = pd.to_datetime(es_iv.index)

    j = pd.concat([es_iv.rename("iv"), vix.rename("vix")], axis=1, join="inner").dropna()
    if len(j) < 30:
        return {"corr": float("nan"), "median_ratio": float("nan"), "ok": False}
    corr = float(j["iv"].corr(j["vix"]))
    ratio = float((j["iv"] / j["vix"]).median())
    return {"corr": corr, "median_ratio": ratio, "ok": bool(corr > 0.6 and 0.5 < ratio < 2.0)}
