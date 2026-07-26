"""Currency short-rate panel + FX carry rate differentials from FRED.

Carry accrual on spot FX is the overnight interest-rate differential
(r_base - r_quote). This module maps each currency to a FRED short-rate series
(policy or short-tenor bill rate), builds a daily decimal-rate panel aligned to
the backtest's FX dates, and computes per-pair rate differentials. Metals
(XAU/XAG) have no interest rate -> base rate 0.0, so gold carry is pure USD
funding.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils import logger

# Currency -> FRED short-rate series id (percent units in FRED).
# USD/EUR use their daily policy rates (DFF, ECB deposit). All other currencies
# use the OECD 3-month interbank rate (IR3TIB01*M156N), a single consistent
# family that is CURRENT for every currency (verified ends 2026-04/05).
# This replaced the earlier IRSTCI01* (call-money) family, which OECD
# DISCONTINUED for several currencies -- SEK ended 2020-10 (5.7yr stale, stuck
# at 0.10% while Riksbank hiked to ~4%), CHF ended 2024-03, NZD ended 2024-12 --
# silently producing wrong carry for those legs. IR3TIB01 carries a small
# (~10-30bp) term premium over the USD/EUR overnight rates; acceptable for carry
# differentials and vastly better than multi-year-stale data. A bad ID raises
# FredValidationError (guards the 2026 CHF-series HTML-error bug). SGD has no
# free FRED short-rate series -- omitted, falls back to 0.0 with a WARNING.
CURRENCY_FRED_SERIES: dict[str, str] = {
    "USD": "DFF",              # Effective Federal Funds Rate (daily policy)
    "EUR": "ECBDFR",           # ECB Deposit Facility Rate (daily policy)
    "CHF": "IR3TIB01CHM156N",  # 3-month interbank, monthly, ffilled to daily
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "NOK": "IR3TIB01NOM156N",
    "SEK": "IR3TIB01SEM156N",
    "MXN": "IR3TIB01MXM156N",
    "ZAR": "IR3TIB01ZAM156N",
    "PLN": "IR3TIB01PLM156N",
    "HUF": "IR3TIB01HUM156N",
    "CNH": "IR3TIB01CNM156N",   # onshore China 3M interbank as offshore-CNH proxy
    "TRY": "INTDSRTRM193N",     # CBRT discount rate (OECD interbank stale since 2008)
    "INR": "IRSTCI01INM156N",   # India call money (OECD interbank absent)
}
_METALS = {"XAU", "XAG"}

# Publication lag applied before a FRED observation may be used (added 2026-07-25).
# FRED stamps a MONTHLY series at the FIRST of the month, but the value dated
# 2026-05-01 is May's AVERAGE -- unknowable until May ends, and OECD/CBRT publish
# it weeks later still. Forward-filling from the stamp date therefore let a carry
# backtest see the current month's rate from day 1 of that month: a 1-2 month
# lookahead on the CARRY SIGNAL ITSELF. Monthly observations are now delayed by
# 60 days (month completes ~30d, publication ~30d). Daily policy rates (DFF,
# ECBDFR) publish next-day and take a 1-day lag.
_MONTHLY_PUBLICATION_LAG_DAYS = 60
_DAILY_PUBLICATION_LAG_DAYS = 1


def _publication_lag_days(idx: pd.DatetimeIndex) -> int:
    """Lag for a FRED series, inferred from its observation spacing."""
    if len(idx) < 2:            # one observation: no spacing to infer, be conservative
        return _MONTHLY_PUBLICATION_LAG_DAYS
    spacing = pd.Series(idx).diff().dt.days.median()
    return (_MONTHLY_PUBLICATION_LAG_DAYS if spacing > 20
            else _DAILY_PUBLICATION_LAG_DAYS)


def load_fx_rate_panel(currencies: list[str], index: pd.Index) -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "alt_data" / "fred"
    out: dict[str, pd.Series] = {}
    idx_dt = pd.to_datetime(pd.Index(index))
    for ccy in currencies:
        if ccy in _METALS:
            out[ccy] = pd.Series(0.0, index=index)
            continue
        series_id = CURRENCY_FRED_SERIES.get(ccy)
        if series_id is None:
            logger.warning(f"[load_fx_rate_panel] no FRED series for {ccy}; rate=0")
            out[ccy] = pd.Series(0.0, index=index)
            continue
        fp = base / series_id / "daily.parquet"
        if not fp.exists():
            logger.warning(f"[load_fx_rate_panel] FRED file missing for {ccy} ({series_id}); rate=0")
            out[ccy] = pd.Series(0.0, index=index)
            continue
        raw = pd.read_parquet(fp)
        s = pd.Series(raw["value"].values, index=pd.to_datetime(raw["date"].values)) / 100.0
        s = s.sort_index()
        s.index = s.index + pd.to_timedelta(_publication_lag_days(s.index), unit="D")
        s = s.reindex(idx_dt.union(s.index)).ffill().reindex(idx_dt)
        s.index = index
        out[ccy] = s
    return pd.DataFrame(out)


def build_rate_diff_panel(pairs: list[str], rate_panel: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for pair in pairs:
        base_ccy, quote_ccy = pair[:3], pair[3:]
        out[pair] = rate_panel[base_ccy] - rate_panel[quote_ccy]
    return pd.DataFrame(out)


def currencies_for_pairs(pairs: list[str]) -> list[str]:
    ccys: set[str] = set()
    for pair in pairs:
        ccys.add(pair[:3])
        ccys.add(pair[3:])
    return sorted(ccys)
