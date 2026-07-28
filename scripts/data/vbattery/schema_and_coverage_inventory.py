"""Schema-variant + session-coverage inventory over `options_combined` (V4 / V9 input).

Reads PARQUET FOOTER METADATA ONLY -- no row data -- for every
`root=*/year=*/month=*/data.parquet` partition, and records per partition:
the column count, the ordered column-name tuple, key dtypes, row count, and the
min/max `timestamp` taken from row-group statistics.

CRITICAL: the column ORDER is not stable across partitions. Measured variants
include a 21-column layout whose first column is `symbol`, a 20-column layout
whose first column is `timestamp`, and a 16-column layout. Indexing statistics
by POSITION (e.g. `row_group(i).column(0)`) therefore reads the wrong column on
most partitions and silently produces nonsense coverage numbers. This script
resolves the `timestamp` column BY NAME per file. That bug is the reason this
script exists as a checked-in artifact rather than an ad-hoc query.

Session coverage is computed against the NYSE calendar, counting only sessions
inside each partition's observed [min_ts, max_ts] span, so a partition that is
TRUNCATED (stops mid-month) is distinguishable from one that is merely short.

This is a DATA-PROPERTY measurement. No strategy P&L is computed anywhere.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas_market_calendars as mcal
import polars as pl
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUTPUT_DIR = os.path.join("output", "vbattery", "schema_coverage")

_TRUNCATION_THRESHOLD = 0.90


def _store_base() -> str:
    return os.path.join(get_local_storage_dir(), "options", "options_combined")


def _timestamp_span(md) -> tuple[str | None, str | None]:
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    if "timestamp" not in names:
        return None, None
    idx = names.index("timestamp")

    lows, highs = [], []
    for rg in range(md.num_row_groups):
        stats = md.row_group(rg).column(idx).statistics
        if stats is not None and stats.min is not None:
            lows.append(str(stats.min))
            highs.append(str(stats.max))
    if not lows:
        return None, None
    return min(lows), max(highs)


def inventory_partition(path: str, sessions: set[str]) -> dict:
    root = re.search(r"root=([^\\/]+)", path).group(1)
    year = int(re.search(r"year=(\d+)", path).group(1))
    month = int(re.search(r"month=(\d+)", path).group(1))

    parquet_file = pq.ParquetFile(path)
    md = parquet_file.metadata
    schema = parquet_file.schema_arrow
    names = [schema.field(i).name for i in range(len(schema))]

    low, high = _timestamp_span(md)
    first_day = low[:10] if low else None
    last_day = high[:10] if high else None

    expected_month = sorted(d for d in sessions if d.startswith(f"{year}-{month:02d}"))
    if first_day and last_day:
        expected_span = [d for d in expected_month if first_day <= d <= last_day]
    else:
        expected_span = []

    return {
        "root": root,
        "year": year,
        "month": month,
        "n_columns": md.num_columns,
        "columns": "|".join(names),
        "first_column": names[0] if names else None,
        "has_symbol_column": "symbol" in names,
        "timestamp_dtype": str(schema.field("timestamp").type) if "timestamp" in names else None,
        "expiration_dtype": str(schema.field("expiration").type) if "expiration" in names else None,
        "volume_dtype": str(schema.field("volume").type) if "volume" in names else None,
        "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "file_bytes": os.path.getsize(path),
        "first_session": first_day,
        "last_session": last_day,
        "sessions_expected_in_month": len(expected_month),
        "sessions_expected_in_span": len(expected_span),
        "month_span_ratio": (
            len(expected_span) / len(expected_month) if expected_month else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4/V9 schema + coverage inventory")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()

    calendar = mcal.get_calendar("XNYS")
    schedule = calendar.schedule(start_date=args.start, end_date=args.end)
    sessions = {d.strftime("%Y-%m-%d") for d in schedule.index}

    paths = sorted(
        glob.glob(os.path.join(_store_base(), "root=*", "year=*", "month=*", "data.parquet"))
    )
    logger.info(f"[*] inventorying {len(paths)} partitions (footer metadata only)")

    with RunStatus("options_schema_coverage_inventory", meta={"partitions": len(paths)}) as status:
        rows = []
        for i, path in enumerate(paths):
            try:
                rows.append(inventory_partition(path, sessions))
            except Exception as exc:
                logger.error(f"[-] failed on {path}: {exc}")
                rows.append({"root": None, "columns": f"ERROR: {exc}"})
            if (i + 1) % 500 == 0:
                status.heartbeat(note=f"{i + 1}/{len(paths)}")
                logger.info(f"    {i + 1}/{len(paths)}")

    frame = pl.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "partition_inventory.parquet")
    frame.write_parquet(out)
    logger.info(f"[+] wrote {out} ({len(frame)} partitions)")

    variants = (
        frame.group_by(["n_columns", "first_column", "has_symbol_column"])
        .agg(
            pl.len().alias("n_partitions"),
            pl.col("root").n_unique().alias("n_roots"),
            pl.col("year").min().alias("year_min"),
            pl.col("year").max().alias("year_max"),
        )
        .sort("n_partitions", descending=True)
    )
    logger.info("[*] SCHEMA VARIANTS:")
    for row in variants.iter_rows(named=True):
        logger.info(
            f"    ncols={row['n_columns']} first={row['first_column']} "
            f"symbol={row['has_symbol_column']} -> {row['n_partitions']} partitions, "
            f"{row['n_roots']} roots, years {row['year_min']}-{row['year_max']}"
        )
    variants.write_parquet(os.path.join(OUTPUT_DIR, "schema_variants.parquet"))

    truncated = frame.filter(pl.col("month_span_ratio") < _TRUNCATION_THRESHOLD)
    logger.info(f"[!] TRUNCATED partitions (<{_TRUNCATION_THRESHOLD:.0%} of month spanned): {len(truncated)}")
    truncated.write_parquet(os.path.join(OUTPUT_DIR, "truncated_partitions.parquet"))


if __name__ == "__main__":
    main()
