"""Dukascopy free historical FX/metals backfill (keyless).

Fills the gaps the Massive/Polygon source genuinely lacks -- the selective
2020-11/12 vendor outage (NZD, Nordic crosses, metals, JPY-crosses) and XAUUSD
pre-2019 (Polygon has no metals in its flat archive that far back). Dukascopy's
public data feed (no account, no API key) has all of it as tick/1-min history.

Cross-validated against the existing Polygon minute data on control days:
sub-pip agreement (NZDUSD 0.40 pips, EURUSD 0.10 pips median). Bid+ask are
mid-ed to match the existing convention. Output is the canonical 8-col
per-symbol-per-month parquet under fx_1min/, so build_fx_daily_cache consumes it
transparently.

Three crosses (NOKSEK/NOKJPY/SEKJPY) are not direct Dukascopy instruments; they
are derived by triangulation from USD legs (validated: NOKSEK vs real median
3.3 bps). The derived OHLC is an approximation (per-field ratio of aligned
legs), acceptable for the daily cache these minor crosses feed.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

import dukascopy_python as dk
import dukascopy_python.instruments as ins
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.settings import FX_MASSIVE_1MIN, get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_OHLC = ["open", "high", "low", "close"]

# hg_symbol -> Dukascopy instrument, OR ("derive", numerator, denominator).
# Cross = numerator / denominator (e.g. NOKSEK = USDSEK / USDNOK).
DUKA_MAP: dict[str, object] = {
    "NZDUSD": ins.INSTRUMENT_FX_MAJORS_NZD_USD,
    "USDCHF": ins.INSTRUMENT_FX_MAJORS_USD_CHF,
    "EURCHF": ins.INSTRUMENT_FX_CROSSES_EUR_CHF,
    "EURNOK": ins.INSTRUMENT_FX_CROSSES_EUR_NOK,
    "EURSEK": ins.INSTRUMENT_FX_CROSSES_EUR_SEK,
    "USDNOK": ins.INSTRUMENT_FX_CROSSES_USD_NOK,
    "USDSEK": ins.INSTRUMENT_FX_CROSSES_USD_SEK,
    "AUDNZD": ins.INSTRUMENT_FX_CROSSES_AUD_NZD,
    "NZDJPY": ins.INSTRUMENT_FX_CROSSES_NZD_JPY,
    "XAUUSD": ins.INSTRUMENT_FX_METALS_XAU_USD,
    "XAGUSD": ins.INSTRUMENT_FX_METALS_XAG_USD,
    "USDMXN": ins.INSTRUMENT_FX_CROSSES_USD_MXN,
    "USDZAR": ins.INSTRUMENT_FX_CROSSES_USD_ZAR,
    "USDCNH": ins.INSTRUMENT_FX_CROSSES_USD_CNH,
    "USDTRY": ins.INSTRUMENT_FX_CROSSES_USD_TRY,
    "USDPLN": ins.INSTRUMENT_FX_CROSSES_USD_PLN,
    "USDHUF": ins.INSTRUMENT_FX_CROSSES_USD_HUF,
    "NOKSEK": ("derive", ins.INSTRUMENT_FX_CROSSES_USD_SEK, ins.INSTRUMENT_FX_CROSSES_USD_NOK),
    "NOKJPY": ("derive", ins.INSTRUMENT_FX_MAJORS_USD_JPY, ins.INSTRUMENT_FX_CROSSES_USD_NOK),
    "SEKJPY": ("derive", ins.INSTRUMENT_FX_MAJORS_USD_JPY, ins.INSTRUMENT_FX_CROSSES_USD_SEK),
}


def mid_ohlc(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Average bid and ask OHLC on their shared timestamps; carry bid volume."""
    idx = bid.index.intersection(ask.index)
    mid = (bid.loc[idx, _OHLC] + ask.loc[idx, _OHLC]) / 2.0
    mid["volume"] = bid.loc[idx, "volume"] if "volume" in bid else 0.0
    return mid


def derive_ohlc(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """Derive a cross's OHLC as the per-field ratio of aligned legs (num/den)."""
    idx = num.index.intersection(den.index)
    out = pd.DataFrame(index=idx)
    for c in _OHLC:
        out[c] = num.loc[idx, c] / den.loc[idx, c]
    out["volume"] = num.loc[idx, "volume"] if "volume" in num else 0.0
    return out


def to_canonical(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Map a timestamp-indexed OHLC(+volume) frame to the canonical 8 columns."""
    if ohlc.empty:
        return pd.DataFrame(columns=CANONICAL_OHLCV_SCHEMA)
    vol = pd.to_numeric(ohlc.get("volume", pd.Series(0.0, index=ohlc.index)),
                        errors="coerce").fillna(0.0)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(ohlc.index, utc=True),
        "open": ohlc["open"].astype(float).to_numpy(),
        "high": ohlc["high"].astype(float).to_numpy(),
        "low": ohlc["low"].astype(float).to_numpy(),
        "close": ohlc["close"].astype(float).to_numpy(),
        "volume": vol.round().astype(int).to_numpy(),
        "trade_count": 0,
        "vwap": ohlc["close"].astype(float).to_numpy(),
    })
    return out[CANONICAL_OHLCV_SCHEMA]


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1)
    end = datetime(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    return start, end


def _fetch_mid(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    bid = dk.fetch(instrument, dk.INTERVAL_MIN_1, dk.OFFER_SIDE_BID, start, end)
    ask = dk.fetch(instrument, dk.INTERVAL_MIN_1, dk.OFFER_SIDE_ASK, start, end)
    return mid_ohlc(bid, ask)


def fetch_pair_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Fetch one (symbol, year, month) as canonical 8-col rows (mid prices)."""
    spec = DUKA_MAP[symbol]
    start, end = _month_bounds(year, month)
    if isinstance(spec, tuple) and spec[0] == "derive":
        _, num_inst, den_inst = spec
        num = _fetch_mid(num_inst, start, end)
        den = _fetch_mid(den_inst, start, end)
        ohlc = derive_ohlc(num, den)
    else:
        ohlc = _fetch_mid(spec, start, end)
    return to_canonical(ohlc)


def write_month(rows: pd.DataFrame, symbol: str, year: int, month: int) -> int:
    """Atomically write canonical rows to fx_1min/symbol=.../year=/month=/data.parquet."""
    out_dir = (get_local_storage_dir() / FX_MASSIVE_1MIN
               / f"symbol={symbol}" / f"year={year}" / f"month={month}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.from_pandas(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="ns", time_zone="UTC")),
    ).select(CANONICAL_OHLCV_SCHEMA).sort("timestamp").unique(subset=["timestamp"], keep="last")
    tmp = out_dir / "data.parquet.tmp"
    df.write_parquet(tmp)
    os.replace(tmp, out_dir / "data.parquet")
    return df.height


def backfill_gaps(gaps: list[tuple[str, int, int]]) -> dict:
    """Backfill each (symbol, year, month) gap. Returns a per-gap row-count summary."""
    summary: dict[str, int] = {}
    for symbol, year, month in gaps:
        if symbol not in DUKA_MAP:
            logger.warning(f"[dukascopy] no mapping for {symbol}, skipping")
            continue
        try:
            rows = fetch_pair_month(symbol, year, month)
        except Exception as e:
            logger.error(f"[dukascopy] {symbol} {year}-{month} fetch failed: {e}")
            continue
        if rows.empty:
            logger.warning(f"[dukascopy] {symbol} {year}-{month} returned empty")
            continue
        n = write_month(rows, symbol, year, month)
        summary[f"{symbol} {year}-{month:02d}"] = n
        logger.info(f"[dukascopy] wrote {symbol} {year}-{month:02d}: {n} minute bars")
    return summary
