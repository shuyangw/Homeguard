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

# Currency -> FRED daily short-rate series id (percent units in FRED).
# G10 + select EM short-rate coverage (Task 4). All IRSTCI01* IDs below were
# live-verified via FREDRatesPlugin.fetch_series before being added here (each
# returned >0 rows over 2011-2026); a bad ID raises FredValidationError instead
# of silently writing an HTML error page (the 2026 CHF-series bug this guards
# against). SGD has no free short-rate series on FRED (SIBOR/SORA are not
# published there) and is intentionally omitted -- falls back to 0.0 with a
# WARNING like any other currency missing from this map.
CURRENCY_FRED_SERIES: dict[str, str] = {
    "USD": "DFF",              # Effective Federal Funds Rate
    "EUR": "ECBDFR",           # ECB Deposit Facility Rate
    "CHF": "IRSTCI01CHM156N",  # Switzerland call-money (overnight) rate. Monthly,
                               # ffilled to daily; the series was discontinued
                               # 2024-03 so the last ~2yrs carry the last value.
    "JPY": "IRSTCI01JPM156N",  # Japan call-money (overnight) rate. Monthly,
                               # ffilled to daily; current through 2026.
    "GBP": "IRSTCI01GBM156N",  # UK call-money (overnight) rate. Monthly, ffilled.
    "CAD": "IRSTCI01CAM156N",  # Canada call-money (overnight) rate. Monthly, ffilled.
    "AUD": "IRSTCI01AUM156N",  # Australia call-money (overnight) rate. Monthly, ffilled.
    "NZD": "IRSTCI01NZM156N",  # New Zealand call-money (overnight) rate. Monthly, ffilled.
    "NOK": "IRSTCI01NOM156N",  # Norway call-money (overnight) rate. Monthly, ffilled.
    "SEK": "IRSTCI01SEM156N",  # Sweden call-money (overnight) rate. Monthly, ffilled.
    "MXN": "IRSTCI01MXM156N",  # Mexico call-money (overnight) rate. Monthly, ffilled.
    "ZAR": "IRSTCI01ZAM156N",  # South Africa call-money (overnight) rate. Monthly, ffilled.
}
_METALS = {"XAU", "XAG"}


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
        s = s.sort_index().reindex(idx_dt.union(s.index)).ffill().reindex(idx_dt)
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
