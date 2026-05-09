"""Convert downloaded Databento dbn.zst files to canonical parquet under H:/Stock_Data/.

Reads each section's staged dbn.zst files from
<storage>/futures_dbn_staging/<section>/ and writes parquet files to:

  Section A_v   -> futures_1min/symbol={ROOT}/year={Y}/month={M}/data.parquet
  Section A_n_diag -> futures_1min_oi_roll/symbol={ROOT}/...
  Section B     -> futures_per_contract/root={ROOT}/year={Y}/data.parquet
  Section C     -> futures_options/root={ROOT}/year={Y}/data.parquet
  Section D     -> futures_definitions/year={Y}/month={M}/data.parquet
  Section E     -> futures_statistics/root={ROOT}/year={Y}/data.parquet
  Section F     -> futures_mbp1/symbol={ROOT}/year={Y}/month={M}/day={D}/data.parquet

Idempotent: skips a target parquet if it already exists.

Usage:
    python scripts/data/databento_batch_convert.py
    python scripts/data/databento_batch_convert.py --section A_v
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import polars as pl
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

import databento as db

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

STAGING_ROOT = get_local_storage_dir() / "futures_dbn_staging"
DATA_ROOT = get_local_storage_dir()

CANONICAL_DTYPE = pl.Datetime("us", "UTC")
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def _read_dbn(path: Path) -> pl.DataFrame:
    """Read a dbn.zst file via the databento library, return a polars DataFrame."""
    store = db.DBNStore.from_file(str(path))
    pdf = store.to_df()
    if pdf.empty:
        return pl.DataFrame()
    return pl.from_pandas(pdf.reset_index())


def _normalize_ts(df: pl.DataFrame, src_col: str = "ts_event") -> pl.DataFrame:
    """Cast Databento ts column to canonical timestamp [us, UTC]."""
    if src_col in df.columns:
        df = df.rename({src_col: "timestamp"})
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").cast(CANONICAL_DTYPE))
    return df


def _strip_continuous_suffix(symbol: str) -> str:
    """ES.v.0 -> ES, GC.n.0 -> GC."""
    return re.sub(r"\.[vnc]\.\d+$", "", symbol)


def _root_from_raw(raw_symbol: str) -> str | None:
    """Extract the family root from a raw CME symbol like 'ESH5'.

    CME symbols are typically <ROOT><MONTH_CODE><YEAR_DIGIT(S)>.
    Returns None for spreads / non-standard.
    """
    if "-" in raw_symbol or " " in raw_symbol:
        return None
    # Walk from end stripping digits, then strip month code letter
    s = raw_symbol.rstrip("0123456789")
    if not s or len(s) < 2:
        return None
    return s[:-1] if s[-1].isalpha() else s


def _write_parquet(df: pl.DataFrame, path: Path) -> int:
    if df.is_empty():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", compression_level=3)
    return len(df)


def _iter_dbn_files(section_dir: Path) -> Iterator[Path]:
    yield from sorted(section_dir.rglob("*.dbn.zst"))


def _ym_from_filename(path: Path) -> tuple[int, int] | None:
    """Parse YYYYMMDD start-date out of a Databento batch filename.

    Format: glbx-mdp3-YYYYMMDD-YYYYMMDD.<schema>.dbn.zst
    Returns (year, month) of the file's start date, or None on failure.

    For schemas where event timestamps can predate the file's date range
    (definitions and statistics, which include "as of" snapshots referencing
    older modifications), filename-derived month is the correct partition key.
    """
    parts = path.name.split("-")
    if len(parts) < 4:
        return None
    start = parts[2]  # YYYYMMDD
    try:
        return int(start[:4]), int(start[4:6])
    except (IndexError, ValueError):
        return None


def convert_a(section: str) -> None:
    """Section A_v / A_n_diag: continuous ohlcv-1m, per-(symbol, month) files.

    File naming from Databento: glbx-mdp3-YYYYMMDD-YYYYMMDD.{SYMBOL}.ohlcv-1m.dbn.zst
    where SYMBOL has dots collapsed (e.g. ES.v.0 -> ES-v-0). The dataframe
    contains a `symbol` column we can trust.
    """
    subdir = "futures_1min" if section == "A_v" else "futures_1min_oi_roll"
    out_root = DATA_ROOT / subdir
    section_dir = STAGING_ROOT / section
    if not section_dir.exists():
        logger.warning(f"  {section}: staging dir missing")
        return

    total_rows = 0
    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section {section}: {len(files)} dbn files")
    for i, f in enumerate(files):
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: read failed {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)

        # Each file has one symbol due to split_symbols=True
        if "symbol" not in df.columns:
            logger.warning(f"  {f.name}: no symbol column, skipping")
            continue
        sym_full = df["symbol"][0]
        root = _strip_continuous_suffix(sym_full)
        # File covers one month per split_duration="month"
        ts_min = df["timestamp"].min()
        year, month = ts_min.year, ts_min.month
        cols = [c for c in OHLCV_COLS if c in df.columns]
        out = (
            out_root / f"symbol={root}" / f"year={year}" / f"month={month}"
            / "data.parquet"
        )
        if out.exists():
            continue
        total_rows += _write_parquet(df.select(cols), out)
        if (i + 1) % 200 == 0:
            logger.info(f"  ... {i+1}/{len(files)} files, {total_rows:,} rows so far")

    logger.info(f"Section {section}: {total_rows:,} total rows written")


def convert_b_c_e(section: str) -> None:
    """Sections B/C/E: parent symbology, monthly files all-symbols mixed.

    Each dbn file is one month of all symbols (contracts at per-instrument level).
    Preserve as monthly files; consumers filter by `symbol` column at read time.

    Storage:
      B (ohlcv-1m per-contract):  futures_per_contract_1min/year={Y}/month={M}/data.parquet
      C (ohlcv-1m options):       futures_options_1min/year={Y}/month={M}/data.parquet
      E (statistics events):      futures_statistics/year={Y}/month={M}/data.parquet
    """
    section_dirs = {
        "B": "futures_per_contract_1min",
        "C": "futures_options_1min",
        "E": "futures_statistics",
    }
    out_root = DATA_ROOT / section_dirs[section]
    section_dir = STAGING_ROOT / section
    if not section_dir.exists():
        logger.warning(f"  {section}: staging dir missing")
        return

    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section {section}: {len(files)} dbn files")
    # For B (ohlcv-1m), partitioning by ts_event is correct since bar timestamps
    # match the file's date range. For E (statistics) and any schema with
    # historical event references, partition by filename instead.
    use_filename_partition = section in ("E",)
    total_rows = 0
    for i, f in enumerate(files):
        if use_filename_partition:
            ym = _ym_from_filename(f)
            if ym is None:
                logger.warning(f"  {f.name}: cannot parse YM from filename, skipping")
                continue
            year, month = ym
            out = out_root / f"year={year}" / f"month={month}" / "data.parquet"
            if out.exists():
                continue
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: read failed {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)
        if not use_filename_partition:
            ts_min = df["timestamp"].min()
            year, month = ts_min.year, ts_min.month
            out = out_root / f"year={year}" / f"month={month}" / "data.parquet"
            if out.exists():
                continue
        total_rows += _write_parquet(df, out)
        if (i + 1) % 50 == 0:
            logger.info(f"  ... {i+1}/{len(files)} files, {total_rows:,} rows so far")
    logger.info(f"Section {section}: {total_rows:,} total rows")


def convert_d() -> None:
    """Section D: definitions, monthly all-symbols. Partition by filename
    (definitions reference earlier ts_event from contract creation; ts_event
    is unreliable as a partition key)."""
    out_root = DATA_ROOT / "futures_definitions"
    section_dir = STAGING_ROOT / "D"
    if not section_dir.exists():
        logger.warning("  D: staging dir missing")
        return

    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section D: {len(files)} dbn files")
    total_rows = 0
    for i, f in enumerate(files):
        ym = _ym_from_filename(f)
        if ym is None:
            logger.warning(f"  {f.name}: cannot parse YM from filename, skipping")
            continue
        year, month = ym
        out = out_root / f"year={year}" / f"month={month}" / "data.parquet"
        if out.exists():
            continue
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)
        total_rows += _write_parquet(df, out)
        if (i + 1) % 20 == 0:
            logger.info(f"  ... {i+1}/{len(files)} files, {total_rows:,} rows so far")
    logger.info(f"Section D: {total_rows:,} rows")


def convert_f() -> None:
    """Section F: mbp-1, daily files per symbol.

    File naming: glbx-mdp3-YYYYMMDD.{SYMBOL}.mbp-1.dbn.zst
    """
    out_root = DATA_ROOT / "futures_mbp1"
    section_dir = STAGING_ROOT / "F"
    if not section_dir.exists():
        logger.warning("  F: staging dir missing")
        return

    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section F: {len(files)} dbn files")
    total_rows = 0
    for i, f in enumerate(files):
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)
        if "symbol" not in df.columns:
            continue
        sym_full = df["symbol"][0]
        root = _strip_continuous_suffix(sym_full)
        ts_min = df["timestamp"].min()
        year, month, day = ts_min.year, ts_min.month, ts_min.day
        out = (
            out_root / f"symbol={root}" / f"year={year}" / f"month={month}"
            / f"day={day}" / "data.parquet"
        )
        if out.exists():
            continue
        total_rows += _write_parquet(df, out)
        if (i + 1) % 50 == 0:
            logger.info(f"  ... {i+1}/{len(files)}")
    logger.info(f"Section F: {total_rows:,} rows")


def convert_trades(section: str = "Trades_ES_MES") -> None:
    """Trades schema converter: per (symbol, month) restricted-window output.

    Filters to 19:00-21:00 UTC (3pm-4pm ET) post-download as Databento batch
    doesn't natively support time-of-day windowing.

    Storage: futures_trades_window/symbol={ROOT}/year={Y}/month={M}/data.parquet
    """
    out_root = DATA_ROOT / "futures_trades_window"
    section_dir = STAGING_ROOT / section
    if not section_dir.exists():
        logger.warning(f"  {section}: staging dir missing")
        return
    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section {section}: {len(files)} dbn files")
    total_rows = 0
    for i, f in enumerate(files):
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: read failed {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)
        # Each file is one (symbol, month) due to split_symbols=True
        if "symbol" not in df.columns:
            logger.warning(f"  {f.name}: no symbol column, skipping")
            continue
        sym_full = df["symbol"][0]
        root = _strip_continuous_suffix(sym_full)
        # Filter to 19:00-21:00 UTC window
        df = df.with_columns(pl.col("timestamp").dt.hour().alias("_hour"))
        df = df.filter((pl.col("_hour") >= 19) & (pl.col("_hour") < 21))
        df = df.drop("_hour")
        if df.is_empty():
            continue
        ts_min = df["timestamp"].min()
        year, month = ts_min.year, ts_min.month
        out = (
            out_root / f"symbol={root}" / f"year={year}" / f"month={month}"
            / "data.parquet"
        )
        if out.exists():
            continue
        total_rows += _write_parquet(df, out)
        if (i + 1) % 200 == 0:
            logger.info(f"  ... {i+1}/{len(files)} files, {total_rows:,} rows so far")
    logger.info(f"Section {section}: {total_rows:,} total rows")


def convert_status(section: str = "Status_universe") -> None:
    """Status schema converter: flat by year.

    Storage: futures_status/year={Y}/data.parquet
    """
    out_root = DATA_ROOT / "futures_status"
    section_dir = STAGING_ROOT / section
    if not section_dir.exists():
        logger.warning(f"  {section}: staging dir missing")
        return
    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section {section}: {len(files)} dbn files")
    by_year: dict[int, list[pl.DataFrame]] = {}
    for f in files:
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: read failed {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)
        ts_min = df["timestamp"].min()
        by_year.setdefault(ts_min.year, []).append(df)
    total_rows = 0
    for year, parts in by_year.items():
        merged = pl.concat(parts).sort("timestamp")
        out = out_root / f"year={year}" / "data.parquet"
        if out.exists():
            continue
        total_rows += _write_parquet(merged, out)
    logger.info(f"Section {section}: {total_rows:,} total rows")


def convert_b_ed_daily() -> None:
    """Section B_ED_daily: Eurodollar (ED.FUT) per-contract daily OHLCV.

    Storage: futures_per_contract_daily/root=ED/year={Y}/data.parquet
    """
    out_root = DATA_ROOT / "futures_per_contract_daily" / "root=ED"
    section_dir = STAGING_ROOT / "B_ED_daily"
    if not section_dir.exists():
        logger.warning("  B_ED_daily: staging dir missing")
        return
    files = list(_iter_dbn_files(section_dir))
    logger.info(f"Section B_ED_daily: {len(files)} dbn files")
    by_year: dict[int, list[pl.DataFrame]] = {}
    for f in files:
        try:
            df = _read_dbn(f)
        except Exception as e:
            logger.error(f"  {f.name}: read failed {e}")
            continue
        if df.is_empty():
            continue
        df = _normalize_ts(df)
        ts_min = df["timestamp"].min()
        by_year.setdefault(ts_min.year, []).append(df)
    total_rows = 0
    for year, parts in by_year.items():
        merged = pl.concat(parts).sort("timestamp")
        out = out_root / f"year={year}" / "data.parquet"
        if out.exists():
            continue
        total_rows += _write_parquet(merged, out)
    logger.info(f"Section B_ED_daily: {total_rows:,} total rows")


CONVERTERS = {
    "A_v": lambda: convert_a("A_v"),
    "A_n_diag": lambda: convert_a("A_n_diag"),
    "B": lambda: convert_b_c_e("B"),
    "C": lambda: convert_b_c_e("C"),
    "D": convert_d,
    "E": lambda: convert_b_c_e("E"),
    "F": convert_f,
    "Trades_ES_MES": lambda: convert_trades("Trades_ES_MES"),
    "Status_continuous": lambda: convert_status("Status_continuous"),
    "Status_parent": lambda: convert_status("Status_parent"),
    "B_ED_daily": convert_b_ed_daily,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--section",
        default="all",
        help="A_v, A_n_diag, B, C, D, E, F or 'all' (default)",
    )
    args = parser.parse_args()

    if args.section == "all":
        sections = list(CONVERTERS.keys())
    else:
        sections = [args.section]
    for s in sections:
        if s not in CONVERTERS:
            logger.error(f"Unknown section: {s}")
            return 1
        logger.info(f"=== Converting Section {s} ===")
        CONVERTERS[s]()

    return 0


if __name__ == "__main__":
    sys.exit(main())
