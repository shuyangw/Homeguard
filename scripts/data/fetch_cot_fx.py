"""Fetch CME FX Commitments-of-Traders (legacy futures-only) from the CFTC public
Socrata API into alt_data/cot/cot_fx.parquet.

Keyless. Net speculative positioning per pair, signed "bullish-the-pair" as
net%OI = sign * (noncomm_long - noncomm_short) / open_interest, where `sign` maps
the USD-per-foreign CME future to our spot pair (JPY/CAD/CHF/MXN futures -> short
the USDxxx pair). Consumed by src/data/cot.py.

Usage: PYTHONPATH=$(pwd) python scripts/data/fetch_cot_fx.py
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils import logger

# CME futures market name (aliases across history) -> (spot pair, sign)
MARKETS: dict[str, tuple[str, int]] = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": ("EURUSD", +1),
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": ("USDJPY", -1),
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": ("GBPUSD", +1),
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": ("GBPUSD", +1),
    "POUND STERLING - CHICAGO MERCANTILE EXCHANGE": ("GBPUSD", +1),
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": ("USDCAD", -1),
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": ("USDCHF", -1),
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": ("AUDUSD", +1),
    "AUSTRALIAN DOLLARS - CHICAGO MERCANTILE EXCHANGE": ("AUDUSD", +1),
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": ("NZDUSD", +1),
    "NEW ZEALAND DOLLARS - CHICAGO MERCANTILE EXCHANGE": ("NZDUSD", +1),
    "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE": ("NZDUSD", +1),
    "MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE": ("USDMXN", -1),
}
_BASE = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_FIELDS = ["market_and_exchange_names", "report_date_as_yyyy_mm_dd",
           "open_interest_all", "noncomm_positions_long_all", "noncomm_positions_short_all"]


def fetch_cot_fx() -> str:
    frames = []
    for name, (pair, sign) in MARKETS.items():
        q = {"$select": ",".join(_FIELDS),
             "$where": f"market_and_exchange_names='{name}'",
             "$order": "report_date_as_yyyy_mm_dd", "$limit": "50000"}
        url = _BASE + "?" + urllib.parse.urlencode(q)
        rows = json.load(urllib.request.urlopen(url, timeout=60))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["pair"], df["sign"] = pair, sign
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    for c in _FIELDS[2:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["date"] = pd.to_datetime(raw["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    net = raw["noncomm_positions_long_all"] - raw["noncomm_positions_short_all"]
    raw["noncomm_net_pct_oi"] = raw["sign"] * net / raw["open_interest_all"]
    out = (raw.groupby(["pair", "date"], as_index=False)
              .agg(noncomm_net_pct_oi=("noncomm_net_pct_oi", "mean"),
                   open_interest=("open_interest_all", "sum"),
                   sign=("sign", "first")))
    dst = get_local_storage_dir() / "alt_data" / "cot"
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "cot_fx.parquet"
    out.to_parquet(path, index=False)
    logger.success(f"[cot] wrote {path}: {len(out)} rows across {out['pair'].nunique()} pairs")
    return str(path)


if __name__ == "__main__":
    fetch_cot_fx()
