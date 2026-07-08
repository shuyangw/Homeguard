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
TFF_HISTORICAL_BASE = "https://www.cftc.gov/files/dea/history/fut_fin_txt_"

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
        "Report_Date_as_YYYY-MM-DD": "report_date_raw",
        "Dealer_Positions_Long_All": "dealer_long",
        "Dealer_Positions_Short_All": "dealer_short",
    }
    df = df.rename({k: v for k, v in column_map.items() if k in df.columns})
    if "report_date_raw" in df.columns:
        df = df.with_columns(
            pl.col("report_date_raw").str.strptime(pl.Date, format="%Y-%m-%d").dt.strftime("%Y-%m-%d").alias("report_date"),
        )
    cols = [c for c in ["report_date", "dealer_long", "dealer_short"] if c in df.columns]
    return df.select(cols)


# CFTC Legacy Futures-Only historical archive (annual files)
LEGACY_HISTORICAL_BASE = "https://www.cftc.gov/files/dea/history/deacot"

# CFTC market codes for the broad futures universe (Legacy report).
# Codes are verified against the live report during the fetch step (Step 6):
# each root must resolve to > 0 rows or it is reported as a mapping gap.
COT_LEGACY_INSTRUMENTS = {
    "ES": "13874A", "NQ": "209742", "YM": "124603",
    "ZN": "043602", "ZB": "020601", "ZF": "044601", "ZT": "042601",
    "CL": "067651", "NG": "023651", "HO": "022651", "RB": "111659",
    "GC": "088691", "SI": "084691", "HG": "085692", "PL": "076651",
    "ZC": "002602", "ZW": "001602", "ZS": "005602", "ZL": "007601", "ZM": "026603",
    "LE": "057642", "HE": "054642",
}

# Legacy report column names vary across annual archives: the "short-format"
# historical files (e.g. TFF-style, current archives) use underscore names;
# older "long-format" archives use spaced names with (All) suffixes. Both
# variants are mapped defensively to the same canonical output names.
_LEGACY_CODE_COLS = ["CFTC_Contract_Market_Code", "CFTC Contract Market Code"]
_LEGACY_COLMAP = {
    "Report_Date_as_YYYY-MM-DD": "report_date_raw",
    "As of Date in Form YYYY-MM-DD": "report_date_raw",
    "Commercial_Positions-Long_All": "commercial_long",
    "Commercial Positions-Long (All)": "commercial_long",
    "Commercial_Positions-Short_All": "commercial_short",
    "Commercial Positions-Short (All)": "commercial_short",
    "Noncommercial_Positions-Long_All": "noncommercial_long",
    "Noncommercial Positions-Long (All)": "noncommercial_long",
    "Noncommercial_Positions-Short_All": "noncommercial_short",
    "Noncommercial Positions-Short (All)": "noncommercial_short",
}
_LEGACY_OUT = ["report_date", "commercial_long", "commercial_short",
               "noncommercial_long", "noncommercial_short"]


def _read_legacy_frame(raw: bytes) -> pl.DataFrame:
    """Read a Legacy report CSV, falling back to unquoted parsing for older
    archives (2010-2014) that contain unescaped quote characters in name fields."""
    try:
        return pl.read_csv(io.BytesIO(raw), ignore_errors=True, infer_schema_length=0)
    except pl.exceptions.ComputeError:
        df = pl.read_csv(io.BytesIO(raw), ignore_errors=True, infer_schema_length=0,
                         quote_char=None, truncate_ragged_lines=True)
        return df.rename({c: c.strip().strip('"') for c in df.columns})


def parse_legacy_csv(raw: bytes, contract_market_code: str) -> pl.DataFrame:
    """Parse a CFTC Legacy Futures-Only CSV, filter to one contract code."""
    df = _read_legacy_frame(raw)
    code_col = next((c for c in _LEGACY_CODE_COLS if c in df.columns), None)
    if code_col is None:
        return pl.DataFrame(schema={c: pl.Utf8 for c in _LEGACY_OUT})
    df = df.filter(pl.col(code_col).cast(pl.Utf8).str.strip_chars() == contract_market_code)
    df = df.rename({k: v for k, v in _LEGACY_COLMAP.items() if k in df.columns})
    numeric_cols = [c for c in ("commercial_long", "commercial_short",
                                 "noncommercial_long", "noncommercial_short") if c in df.columns]
    if numeric_cols:
        df = df.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in numeric_cols])
    if "report_date_raw" in df.columns:
        df = df.with_columns(
            pl.col("report_date_raw").cast(pl.Utf8).str.strptime(
                pl.Date, format="%Y-%m-%d", strict=False
            ).dt.strftime("%Y-%m-%d").alias("report_date")
        )
    cols = [c for c in _LEGACY_OUT if c in df.columns]
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
            nonempty = [d for d in all_rows if d.height > 0]
            if nonempty:
                combined = pl.concat(nonempty)
                self.write_instrument(instrument, combined)
                summary[instrument] = f"wrote {combined.height} rows"
            else:
                summary[instrument] = "no data"
        return summary

    def fetch_year_legacy(self, year: int) -> bytes:
        """Download one year's Legacy Futures-Only historical archive ZIP (returns CSV bytes)."""
        url = f"{LEGACY_HISTORICAL_BASE}{year}.zip"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = [n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))]
        return zf.read(names[0])

    def fetch_all_legacy(self, start_year: int = 2010, end_year: int = 2026,
                         *, skip_existing: bool = True) -> dict:
        """Fetch the Legacy Futures-Only report for the broad universe and write per-root parquet."""
        summary = {}
        year_cache: dict[int, bytes] = {}
        for root, code in COT_LEGACY_INSTRUMENTS.items():
            out = self._root / "cot" / root / "legacy_weekly.parquet"
            if skip_existing and out.exists():
                summary[root] = "skipped"
                continue
            rows = []
            for year in range(start_year, end_year + 1):
                try:
                    raw = year_cache.get(year) or self.fetch_year_legacy(year)
                    year_cache[year] = raw
                    parsed = parse_legacy_csv(raw, code)
                    if parsed.height:
                        rows.append(parsed)
                except Exception as e:
                    logger.warning(f"  {root} {year} (legacy): {e}")
            if rows:
                combined = pl.concat(rows).sort("report_date")
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(out.suffix + ".tmp")
                combined.write_parquet(tmp)
                os.replace(tmp, out)
                summary[root] = f"wrote {combined.height} rows"
            else:
                summary[root] = "no data (verify contract code)"
        return summary
