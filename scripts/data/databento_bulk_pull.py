"""Execute the Databento bulk pull plan.

Sections (per docs/.../DATABENTO_BULK_PULL_PLAN.md):
  A. Continuous OHLCV-1m, 54 symbols (.v.0) + GC (.n.0) diagnostic
  B. Per-contract daily (.FUT), 53 families
  C. Options on futures daily (.OPT), 13 families
  D. Definitions (parent), 66 families
  E. Statistics (parent), 53 families
  F. MBP-1 last 6 months (free under Standard subscription), 4 equity index syms

Output under <storage_root>/futures_*/.

CLI:
    python scripts/data/databento_bulk_pull.py --section A
    python scripts/data/databento_bulk_pull.py --section B
    python scripts/data/databento_bulk_pull.py --section all
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

import databento as db

from src.data.acquisition.plugins.databento_futures import (
    BULK_PULL_UNIVERSE_V,
    BULK_PULL_START,
    DatabentoFuturesPlugin,
    DATASET,
)
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Section B: per-contract daily, all .FUT family parents
ALL_FUT_PARENTS = [
    "ES.FUT", "NQ.FUT", "YM.FUT", "RTY.FUT",
    "MES.FUT", "MNQ.FUT", "M2K.FUT", "MYM.FUT",
    "CL.FUT", "NG.FUT", "HO.FUT", "RB.FUT", "BZ.FUT", "MCL.FUT", "MNG.FUT",
    "GC.FUT", "SI.FUT", "HG.FUT", "PL.FUT", "MGC.FUT", "SIL.FUT",
    "ZT.FUT", "ZF.FUT", "ZN.FUT", "TN.FUT", "ZB.FUT", "UB.FUT",
    "SR3.FUT", "SR1.FUT",
    "10Y.FUT", "30Y.FUT", "5YY.FUT", "2YY.FUT",
    "6E.FUT", "6J.FUT", "6B.FUT", "6A.FUT", "6C.FUT",
    "6S.FUT", "6N.FUT", "6M.FUT",
    "ZC.FUT", "ZS.FUT", "ZW.FUT", "KE.FUT", "ZL.FUT", "ZM.FUT", "LE.FUT", "HE.FUT",
    "BTC.FUT", "MBT.FUT", "ETH.FUT", "MET.FUT",
]

# Section C: options on futures daily, 13 families
ALL_OPT_PARENTS = [
    "ES.OPT", "NQ.OPT", "RTY.OPT",
    "CL.OPT", "NG.OPT",
    "GC.OPT", "SI.OPT",
    "ZN.OPT", "ZB.OPT",
    "6E.OPT", "6J.OPT",
    "ZC.OPT", "ZS.OPT",
]

DEFAULT_END = "2026-02-22"
MBP1_FREE_START = "2025-08-22"
MBP1_FREE_END = "2026-02-22"


def _client() -> db.Historical:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")
    return db.Historical(api_key)


def _save_df(df: pd.DataFrame, path: Path) -> int:
    """Write a DataFrame to parquet, casting timestamp to [us, UTC]. Returns rows."""
    if df.empty:
        return 0
    if "ts_event" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"ts_event": "timestamp"})
    elif df.index.name == "ts_event" and "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"ts_event": "timestamp"})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype(
            "datetime64[us, UTC]"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return len(df)


def section_a(storage_root: Path, end_date: str = DEFAULT_END) -> None:
    """Section A: continuous OHLCV-1m for 54 symbols at .v.0, plus GC.n.0 diagnostic."""
    logger.info("=" * 70)
    logger.info(f"SECTION A: Continuous OHLCV-1m .v.0 for {len(BULK_PULL_UNIVERSE_V)} symbols")
    logger.info("=" * 70)

    plugin_v = DatabentoFuturesPlugin(
        output_dir=storage_root,
        schema="ohlcv-1m",
        roll_rule="v",
        num_threads=6,
    )
    result_v = plugin_v.download(
        symbols=BULK_PULL_UNIVERSE_V,
        start_date=BULK_PULL_START,
        end_date=end_date,
    )
    logger.info(
        f"Section A .v.0 complete: {result_v.succeeded}/{result_v.total_symbols} "
        f"symbols, {result_v.total_rows:,} rows in {result_v.elapsed_seconds:.0f}s"
    )

    logger.info("Section A diagnostic: GC.n.0 (open-interest roll)")
    plugin_n = DatabentoFuturesPlugin(
        output_dir=storage_root,
        schema="ohlcv-1m",
        roll_rule="n",
        storage_subdir="futures_1min_oi_roll",
        num_threads=2,
    )
    result_n = plugin_n.download(
        symbols=["GC"],
        start_date=BULK_PULL_START,
        end_date=end_date,
    )
    logger.info(
        f"Section A GC.n.0 diagnostic: {result_n.succeeded}/{result_n.total_symbols} "
        f"({result_n.total_rows:,} rows)"
    )


def section_b(storage_root: Path, end_date: str = DEFAULT_END) -> None:
    """Section B: per-contract daily OHLCV-1d for all .FUT families."""
    logger.info("=" * 70)
    logger.info(f"SECTION B: Per-contract daily for {len(ALL_FUT_PARENTS)} families")
    logger.info("=" * 70)

    out_root = storage_root / "futures_per_contract"
    client = _client()
    total_rows = 0
    for family in ALL_FUT_PARENTS:
        root = family.replace(".FUT", "")
        out_path = out_root / f"root={root}" / "data.parquet"
        if out_path.exists():
            logger.info(f"  {family}: already exists, skipping")
            continue
        try:
            t0 = time.time()
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema="ohlcv-1d",
                symbols=[family],
                stype_in="parent",
                start=BULK_PULL_START,
                end=end_date,
            )
            df = data.to_df()
            rows = _save_df(df, out_path)
            total_rows += rows
            logger.info(f"  {family}: {rows:,} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"  {family}: FAILED {type(e).__name__}: {e}")

    logger.info(f"Section B total: {total_rows:,} rows")


def section_c(storage_root: Path, end_date: str = DEFAULT_END) -> None:
    """Section C: options on futures daily for 13 .OPT families."""
    logger.info("=" * 70)
    logger.info(f"SECTION C: Options daily for {len(ALL_OPT_PARENTS)} families")
    logger.info("=" * 70)

    out_root = storage_root / "futures_options"
    client = _client()
    total_rows = 0
    for family in ALL_OPT_PARENTS:
        root = family.replace(".OPT", "")
        out_path = out_root / f"root={root}" / "data.parquet"
        if out_path.exists():
            logger.info(f"  {family}: already exists, skipping")
            continue
        try:
            t0 = time.time()
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema="ohlcv-1d",
                symbols=[family],
                stype_in="parent",
                start=BULK_PULL_START,
                end=end_date,
            )
            df = data.to_df()
            rows = _save_df(df, out_path)
            total_rows += rows
            logger.info(f"  {family}: {rows:,} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"  {family}: FAILED {type(e).__name__}: {e}")

    logger.info(f"Section C total: {total_rows:,} rows")


def section_d(storage_root: Path, end_date: str = DEFAULT_END) -> None:
    """Section D: definitions for all .FUT and .OPT family parents."""
    logger.info("=" * 70)
    logger.info("SECTION D: Definitions (FUT + OPT families)")
    logger.info("=" * 70)

    out_root = storage_root / "futures_definitions"
    client = _client()
    all_parents = ALL_FUT_PARENTS + ALL_OPT_PARENTS
    total_rows = 0
    for family in all_parents:
        kind = "FUT" if ".FUT" in family else "OPT"
        root = family.replace(f".{kind}", "")
        out_path = out_root / f"kind={kind}" / f"root={root}" / "data.parquet"
        if out_path.exists():
            logger.info(f"  {family}: already exists, skipping")
            continue
        try:
            t0 = time.time()
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema="definition",
                symbols=[family],
                stype_in="parent",
                start=BULK_PULL_START,
                end=end_date,
            )
            df = data.to_df()
            rows = _save_df(df, out_path)
            total_rows += rows
            logger.info(f"  {family}: {rows:,} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"  {family}: FAILED {type(e).__name__}: {e}")

    logger.info(f"Section D total: {total_rows:,} rows")


def section_e(storage_root: Path, end_date: str = DEFAULT_END) -> None:
    """Section E: statistics (settlement / OI / volume) for all .FUT families."""
    logger.info("=" * 70)
    logger.info(f"SECTION E: Statistics for {len(ALL_FUT_PARENTS)} families")
    logger.info("=" * 70)

    out_root = storage_root / "futures_statistics"
    client = _client()
    total_rows = 0
    for family in ALL_FUT_PARENTS:
        root = family.replace(".FUT", "")
        out_path = out_root / f"root={root}" / "data.parquet"
        if out_path.exists():
            logger.info(f"  {family}: already exists, skipping")
            continue
        try:
            t0 = time.time()
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema="statistics",
                symbols=[family],
                stype_in="parent",
                start=BULK_PULL_START,
                end=end_date,
            )
            df = data.to_df()
            rows = _save_df(df, out_path)
            total_rows += rows
            logger.info(f"  {family}: {rows:,} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"  {family}: FAILED {type(e).__name__}: {e}")

    logger.info(f"Section E total: {total_rows:,} rows")


def section_f(storage_root: Path) -> None:
    """Section F: last 6 months of MBP-1 for ES/MES/NQ/MNQ continuous .v.0.

    Free under the active Standard subscription. No PAYG. Range hardcoded to
    the verified-free window 2025-08-22 to 2026-02-22.
    """
    logger.info("=" * 70)
    logger.info("SECTION F: MBP-1 last 6 months (FREE), 4 equity index symbols")
    logger.info("=" * 70)

    out_root = storage_root / "futures_mbp1"
    client = _client()
    syms = ["ES.v.0", "MES.v.0", "NQ.v.0", "MNQ.v.0"]
    for sym in syms:
        root = sym.replace(".v.0", "")
        out_path = out_root / f"symbol={root}" / "data.parquet"
        if out_path.exists():
            logger.info(f"  {sym}: already exists, skipping")
            continue
        try:
            t0 = time.time()
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema="mbp-1",
                symbols=[sym],
                stype_in="continuous",
                start=MBP1_FREE_START,
                end=MBP1_FREE_END,
            )
            df = data.to_df()
            rows = _save_df(df, out_path)
            logger.info(f"  {sym}: {rows:,} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"  {sym}: FAILED {type(e).__name__}: {e}")


SECTION_FUNCS = {
    "A": section_a,
    "B": section_b,
    "C": section_c,
    "D": section_d,
    "E": section_e,
    "F": section_f,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--section",
        required=True,
        help="One of A,B,C,D,E,F or 'all' or comma-separated like 'D,E'",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="Output root (defaults to local_storage_dir from settings)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END,
        help="End date for sections A-E (default 2026-02-22)",
    )
    args = parser.parse_args()

    storage_root = args.storage_root or get_local_storage_dir()
    storage_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage root: {storage_root}")

    if args.section.lower() == "all":
        sections = ["D", "E", "A", "B", "C", "F"]
    else:
        sections = [s.strip().upper() for s in args.section.split(",")]

    for s in sections:
        if s not in SECTION_FUNCS:
            logger.error(f"Unknown section: {s}")
            return 1

    for s in sections:
        if s == "F":
            SECTION_FUNCS[s](storage_root)
        else:
            SECTION_FUNCS[s](storage_root, end_date=args.end_date)

    return 0


if __name__ == "__main__":
    sys.exit(main())
