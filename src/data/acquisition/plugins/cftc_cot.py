"""CFTC Commitments of Traders (TFF: Traders in Financial Futures) downloader.

Pulls the weekly TFF report from cftc.gov, filters to FX-relevant futures contracts,
writes to alt_data/cot/{instrument}/weekly.parquet.

Schema (canonical, simplified from the 80+ TFF columns):
  report_date (pl.Date, weekly Tuesday-of-positions)
  dealer_long, dealer_short, asset_mgr_long, asset_mgr_short,
  leveraged_long, leveraged_short, other_rep_long, other_rep_short,
  non_rep_long, non_rep_short
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import polars as pl
import requests

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# CFTC TFF historical archive (annual files)
TFF_HISTORICAL_BASE = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_"

# CFTC contract market codes for FX futures
COT_INSTRUMENTS = {
    "6E": "099741",  # Euro FX
    "6J": "097741",  # Japanese Yen
    "6B": "096742",  # British Pound
    "6S": "092741",  # Swiss Franc
    "6C": "090741",  # Canadian Dollar
    "6A": "232741",  # Australian Dollar
    "6N": "112741",  # New Zealand Dollar
    "6M": "095741",  # Mexican Peso
    "6L": "102741",  # Brazilian Real
    "6Z": "122741",  # South African Rand
    "6R": "089741",  # Russian Ruble (pre-2022)
}

CANONICAL_COLUMNS = [
    "report_date", "dealer_long", "dealer_short",
    "asset_mgr_long", "asset_mgr_short",
    "leveraged_long", "leveraged_short",
    "other_rep_long", "other_rep_short",
    "non_rep_long", "non_rep_short",
]


def parse_tff_csv(raw: bytes, contract_market_code: str) -> pl.DataFrame:
    """Parse a TFF report CSV (one annual or weekly file)."""
    df = pl.read_csv(io.BytesIO(raw), ignore_errors=True, schema_overrides={"CFTC_Contract_Market_Code": pl.Utf8})
    df = df.filter(pl.col("CFTC_Contract_Market_Code").cast(pl.Utf8).str.zfill(6) == contract_market_code)
    # Map raw column names to canonical
    column_map = {
        "Report_Date_as_MM_DD_YYYY": "report_date_raw",
        "Dealer_Positions_Long_All": "dealer_long",
        "Dealer_Positions_Short_All": "dealer_short",
    }
    df = df.rename({k: v for k, v in column_map.items() if k in df.columns})
    if "report_date_raw" in df.columns:
        df = df.with_columns(
            pl.col("report_date_raw").str.strptime(pl.Date, format="%m/%d/%Y").dt.strftime("%Y-%m-%d").alias("report_date"),
        )
    cols = [c for c in ["report_date", "dealer_long", "dealer_short"] if c in df.columns]
    return df.select(cols)


class CFTCCOTPlugin:
    """Pull CFTC TFF report and write per-instrument parquet."""

    def __init__(self, storage_root: Path | None = None) -> None:
        self._root = storage_root if storage_root is not None else (get_local_storage_dir() / "alt_data")

    def fetch_year(self, year: int) -> bytes:
        """Download one year's TFF historical archive ZIP (returns CSV bytes)."""
        url = f"{TFF_HISTORICAL_BASE}{year}.zip"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = [n for n in zf.namelist() if n.endswith((".txt", ".csv"))]
        return zf.read(names[0])

    def write_instrument(self, instrument: str, df: pl.DataFrame) -> Path:
        """Write canonical parquet for one instrument."""
        out_dir = self._root / "cot" / instrument
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "weekly.parquet"
        tmp = out.with_suffix(out.suffix + ".tmp")
        df.write_parquet(tmp)
        os.replace(tmp, out)
        return out

    def fetch_all_instruments(self, start_year: int = 2010, end_year: int = 2026,
                              *, skip_existing: bool = True) -> dict:
        summary = {}
        for instrument, code in COT_INSTRUMENTS.items():
            out = self._root / "cot" / instrument / "weekly.parquet"
            if skip_existing and out.exists():
                summary[instrument] = "skipped"
                continue
            all_rows = []
            for year in range(start_year, end_year + 1):
                try:
                    raw = self.fetch_year(year)
                    parsed = parse_tff_csv(raw, code)
                    all_rows.append(parsed)
                except Exception as e:
                    logger.warning(f"  {instrument} {year}: {e}")
            if all_rows:
                combined = pl.concat([d for d in all_rows if d.height > 0])
                self.write_instrument(instrument, combined)
                summary[instrument] = f"wrote {combined.height} rows"
            else:
                summary[instrument] = "no data"
        return summary
