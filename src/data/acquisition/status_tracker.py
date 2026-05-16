"""Per-pass status tracker for the SIP redownload pipeline.

Joins manifest entries with on-disk parquet stats to classify each symbol
as good / broken / incomplete / failed / pending and writes a single-row-per-symbol
CSV to _manifests/<subdir>.status.csv.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.acquisition.manifest import DownloadManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_status(
    manifest_entry: Optional[dict],
    on_disk_rows: int,
    validation_error: Optional[str],
) -> str:
    """Classify a single symbol given manifest entry + on-disk rows + validation result."""
    if manifest_entry is None:
        return "pending"

    if validation_error:
        return "broken"

    status = manifest_entry.get("status")
    if status == "failed":
        return "failed"
    if status == "pending" or status == "in_progress":
        return "pending"
    if status == "complete":
        if on_disk_rows == 0:
            return "incomplete"
        return "good"
    return "pending"


def _scan_symbol_partitions(
    base_dir: Path, subdir: str, symbol: str
) -> dict:
    """Return on-disk stats for a single symbol."""
    sym_dir = base_dir / subdir / f"symbol={symbol}"
    if not sym_dir.exists():
        return {
            "rows": 0, "first_date": None, "last_date": None,
            "trading_days": 0, "partitions": 0,
        }

    partition_paths = list(sym_dir.glob("year=*/month=*/data.parquet"))
    if not partition_paths:
        return {
            "rows": 0, "first_date": None, "last_date": None,
            "trading_days": 0, "partitions": 0,
        }

    total_rows = 0
    first_ts = None
    last_ts = None
    all_dates: set = set()
    for p in partition_paths:
        df = pd.read_parquet(p, columns=["timestamp"])
        total_rows += len(df)
        if df.empty:
            continue
        ts = pd.to_datetime(df["timestamp"])
        local_first = ts.min()
        local_last = ts.max()
        first_ts = local_first if first_ts is None or local_first < first_ts else first_ts
        last_ts = local_last if last_ts is None or local_last > last_ts else last_ts
        all_dates.update(ts.dt.date.unique())

    return {
        "rows": total_rows,
        "first_date": first_ts.date() if first_ts is not None else None,
        "last_date": last_ts.date() if last_ts is not None else None,
        "trading_days": len(all_dates),
        "partitions": len(partition_paths),
    }


def rebuild_tracker(
    base_dir: Path,
    subdir: str,
    universe: list[str],
    validation_errors: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Rebuild full tracker rows by joining manifest + on-disk + validation."""
    validation_errors = validation_errors or {}
    manifest = DownloadManifest(base_dir, subdir)

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for symbol in universe:
        entry = manifest.get_entry(symbol)
        disk = _scan_symbol_partitions(base_dir, subdir, symbol)
        validation_error = validation_errors.get(symbol)
        status = compute_status(entry, disk["rows"], validation_error)
        rows.append({
            "symbol": symbol,
            "status": status,
            "rows": disk["rows"],
            "first_date": disk["first_date"],
            "last_date": disk["last_date"],
            "trading_days": disk["trading_days"],
            "partitions": disk["partitions"],
            "download_error": entry.get("error") if entry else None,
            "validation_error": validation_error,
            "last_updated": now_iso,
        })
    return rows


def write_tracker_csv(path: Path, rows: list[dict]) -> None:
    """Atomic CSV write -- .tmp then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    logger.info(f"Tracker CSV written: {path} ({len(rows)} rows)")
